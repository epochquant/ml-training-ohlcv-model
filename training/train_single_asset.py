import os
import gc
import argparse
import random
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.kronos import Kronos, KronosTokenizer
from data_loader import load_from_json_folder

class DataFrameKlineDataset(Dataset):
    """
    A Dataset class similar to CustomKlineDataset but accepts a pre-loaded pandas DataFrame
    directly, avoiding the need to write to and read from a CSV file.
    """
    def __init__(self, df, lookback_window=512, predict_window=48, clip=5.0, seed=100):
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        self.py_rng = random.Random(seed)
        
        # Preprocess time features
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month
        
        self.data = df[self.feature_list + self.time_feature_list].copy()
        
        if self.data.isnull().any().any():
            self.data = self.data.fillna(method='ffill')
            
        self.n_samples = len(self.data) - self.window + 1
        print(f"Data length: {len(self.data)}, Available sequences: {self.n_samples}")
    
    def set_epoch_seed(self, epoch):
        self.py_rng.seed(self.seed + epoch)
        self.current_epoch = epoch
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        max_start = len(self.data) - self.window
        if max_start <= 0:
            raise ValueError("Data length insufficient to create samples")
            
        epoch = getattr(self, 'current_epoch', 0)
        # Pseudo-random sampling over epochs to vary the batches
        start_idx = (idx * 9973 + (epoch + 1) * 104729) % (max_start + 1)
        end_idx = start_idx + self.window
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        x = window_data[self.feature_list].values.astype(np.float32)
        x_stamp = window_data[self.time_feature_list].values.astype(np.float32)
        
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)
        
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        
        return x_tensor, x_stamp_tensor

def train_model(model, tokenizer, dataloader, epochs, lr, accumulation_steps, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(dataloader),
        epochs=epochs,
        pct_start=0.03,
        div_factor=10
    )

    model.train()
    
    for epoch in range(epochs):
        print(f"\\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        dataloader.dataset.set_epoch_seed(epoch * 10000)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(progress_bar):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            
            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
            
        print(f"Epoch {epoch+1} average loss: {epoch_loss / len(dataloader):.4f}")
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default="JELLYJELLYUSDT", help="Crypto symbol for naming outputs")
    parser.add_argument('--data-dir', type=str, default="finetune_ohlcv_json", help="Directory containing JSON files")
    parser.add_argument('--epochs', type=int, default=3, help="Training epochs")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for CPU memory safety")
    parser.add_argument('--accum-steps', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=4e-5, help="Learning rate")
    parser.add_argument('--limit-data', type=int, default=0, help="Limit number of rows for testing (0 = no limit)")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Starting local fine-tuning on: {device}")
    
    # Set PyTorch thread count to prevent locking up the OS
    torch.set_num_threads(4)

    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    df = load_from_json_folder(args.data_dir)
    if df.empty:
        print("No data loaded. Exiting.")
        return
        
    if args.limit_data > 0:
        df = df.iloc[-args.limit_data:]
        print(f"Limited data to {args.limit_data} rows.")

    dataset = DataFrameKlineDataset(df, lookback_window=512, predict_window=48)
    if len(dataset) == 0:
        print("Not enough data to create sequences. Exiting.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Load Models
    print("Loading base model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base").to(device)

    # Freeze tokenizer, only train base model (predictor)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False

    # 3. Train
    print("Starting training...")
    train_model(model, tokenizer, dataloader, args.epochs, args.lr, args.accum_steps, device)

    # 4. Save
    output_dir = f"./kronos-finetuned-{args.symbol}"
    model_save_path = os.path.join(output_dir, "model")
    tokenizer_save_path = os.path.join(output_dir, "tokenizer")
    
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tokenizer_save_path, exist_ok=True)
    
    print(f"Saving fine-tuned model to {model_save_path}...")
    model.save_pretrained(model_save_path)
    
    print(f"Saving tokenizer to {tokenizer_save_path}...")
    tokenizer.save_pretrained(tokenizer_save_path)
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
