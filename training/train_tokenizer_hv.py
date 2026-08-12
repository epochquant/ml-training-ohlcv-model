import os
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import comet_ml

from training.config_hv import ConfigHV
from training.dataset_hv import HighVolatilityDataset
from model.high_volatility import HighVolatilityTokenizer
from training.utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """Same shape as training.train_tokenizer.create_dataloaders, but backed by
    HighVolatilityDataset (Area A normalized windows + Area B regime vectors)."""
    print(f"[Rank {rank}] Creating high-volatility distributed dataloaders...")
    train_dataset = HighVolatilityDataset('train')
    valid_dataset = HighVolatilityDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False
    )
    print(f"[Rank {rank}] Dataloaders created. Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, device, config, save_dir, logger, rank, world_size):
    """Tokenizer training loop -- identical structure to
    training.train_tokenizer.train_model. The tokenizer only ever sees `x`
    (the regime vector and time stamp are consumed downstream by the
    predictor stage, not the tokenizer)."""
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size * config['accumulation_steps']
        print(f"[Rank {rank}] BATCHSIZE (per GPU): {config['batch_size']}")
        print(f"[Rank {rank}] Effective total batch size: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['tokenizer_learning_rate'],
        weight_decay=config['adam_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config['tokenizer_learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.03,
        div_factor=10
    )

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global_train = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (ori_batch_x, _, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)

            current_batch_total_loss = 0.0
            for j in range(config['accumulation_steps']):
                start_idx = j * (ori_batch_x.shape[0] // config['accumulation_steps'])
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config['accumulation_steps'])
                batch_x = ori_batch_x[start_idx:end_idx]

                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs

                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2

                loss_scaled = loss / config['accumulation_steps']
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0 and (batch_idx_global_train + 1) % config['log_interval'] == 0:
                avg_loss = current_batch_total_loss / config['accumulation_steps']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )
            if rank == 0 and logger:
                avg_loss = current_batch_total_loss / config['accumulation_steps']
                logger.log_metric('train_hv_tokenizer_loss_batch', avg_loss, step=batch_idx_global_train)
                logger.log_metric('train_hv_vqvae_vq_loss_each_batch', bsq_loss.item(), step=batch_idx_global_train)
                logger.log_metric('train_hv_recon_loss_pre_each_batch', recon_loss_pre.item(), step=batch_idx_global_train)
                logger.log_metric('train_hv_recon_loss_each_batch', recon_loss_all.item(), step=batch_idx_global_train)
                logger.log_metric('hv_tokenizer_learning_rate', optimizer.param_groups[0]["lr"], step=batch_idx_global_train)

            batch_idx_global_train += 1

        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0
        with torch.no_grad():
            for ori_batch_x, _, _ in val_loader:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)

                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)

        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_count_tensor = torch.tensor(val_sample_count_rank, device=device)
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_count_tensor.item() if val_count_tensor.item() > 0 else 0

        if rank == 0:
            print(f"\n--- [HV] Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_hv_tokenizer_loss_epoch', avg_val_loss, epoch=epoch_idx)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                model.module.save_pretrained(save_path)
                print(f"Best HV tokenizer model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                if logger:
                    logger.log_model("best_model", save_path)

        dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    return model, dt_result


def main(config: dict):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['tokenizer_save_folder_name'])

    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        if config['use_comet']:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag('high-volatility-model')
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    dist.barrier()

    model = HighVolatilityTokenizer.from_pretrained(config['pretrained_tokenizer_path'])
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"HV Tokenizer Model Size: {get_model_size(model.module)}")

    _, dt_result = train_model(model, device, config, save_dir, comet_logger, rank, world_size)

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('HV tokenizer training finished. Summary file saved.')
        if comet_logger:
            comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with `torchrun`.")

    config_instance = ConfigHV()
    main(config_instance.__dict__)
