import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import gcsfs
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Ensure repository root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.kronos import Kronos, KronosTokenizer
from model.short_classifier import KronosShortClassifier
from training.focal_loss import BinaryFocalLoss



class ShortReversalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 400, clip: float = 5.0):
        self.seq_len = seq_len
        self.clip = clip

        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_ratio']
        time_cols = ['minute', 'hour', 'weekday', 'day', 'month']

        # Precompute timestamp features if missing
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            if 'minute' not in df.columns:
                df['minute'] = df['timestamps'].dt.minute
                df['hour'] = df['timestamps'].dt.hour
                df['weekday'] = df['timestamps'].dt.weekday
                df['day'] = df['timestamps'].dt.day
                df['month'] = df['timestamps'].dt.month

        for col in feature_cols + time_cols:
            if col not in df.columns:
                df[col] = 0.0

        self.features = df[feature_cols].values.astype(np.float32)
        self.time_features = df[time_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.float32)

        # Valid indices where a full lookback window exists
        self.valid_indices = np.arange(self.seq_len, len(df))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len

        x = self.features[start_idx:end_idx].copy()
        x_stamp = self.time_features[start_idx:end_idx].copy()
        label = self.labels[end_idx - 1]  # Target is the label at the final historical candle

        # Z-score normalization per window
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -self.clip, self.clip)

        return (
            torch.from_numpy(x_norm),
            torch.from_numpy(x_stamp),
            torch.tensor(label, dtype=torch.float32)
        )


def train_epoch(model, tokenizer, dataloader, optimizer, criterion, device, epoch, total_epochs, log_interval=10):
    model.train()
    total_loss = 0.0

    for batch_idx, (x, x_stamp, y) in enumerate(dataloader):
        x = x.to(device)
        x_stamp = x_stamp.to(device)
        y = y.to(device)

        # Quantize inputs using KronosTokenizer
        with torch.no_grad():
            x_tokens = tokenizer.encode(x, half=True)
            s1_ids, s2_ids = x_tokens[0], x_tokens[1]

        optimizer.zero_grad()
        logits = model(s1_ids, s2_ids, stamp=x_stamp)
        loss = criterion(logits, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = loss.item()
            log_msg = (f"[Epoch {epoch}/{total_epochs}, Step {batch_idx+1}/{len(dataloader)}] "
                       f"LR: {lr:.6f}, Loss: {avg_loss:.4f}")
            print(log_msg)

    return total_loss / len(dataloader.dataset)


def evaluate(model, tokenizer, dataloader, criterion, device, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, x_stamp, y in dataloader:
            x = x.to(device)
            x_stamp = x_stamp.to(device)
            y = y.to(device)

            x_tokens = tokenizer.encode(x, half=True)
            s1_ids, s2_ids = x_tokens[0], x_tokens[1]

            logits = model(s1_ids, s2_ids, stamp=x_stamp)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits).squeeze(-1)

            total_loss += loss.item() * len(y)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    if len(dataloader.dataset) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.5

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / max(1, len(dataloader.dataset))
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return avg_loss, precision, recall, f1, auc


def extract_dataset_features(dataloader, model, tokenizer, device):
    model.eval()
    tokenizer.eval()
    all_features = []
    all_labels = []

    print("Pre-extracting features from frozen backbone...")
    total_steps = len(dataloader)
    with torch.no_grad():
        for batch_idx, (x, x_stamp, y) in enumerate(dataloader):
            x = x.to(device)
            x_stamp = x_stamp.to(device)
            
            # Quantize inputs using KronosTokenizer
            x_tokens = tokenizer.encode(x, half=True)
            s1_ids, s2_ids = x_tokens[0], x_tokens[1]
            
            # Extract features from backbone
            features = model.extract_features(s1_ids, s2_ids, stamp=x_stamp) # Shape: [B, d_model]
            
            all_features.append(features.cpu())
            all_labels.append(y.cpu())
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_steps:
                print(f"  [Extraction Progress] Step {batch_idx+1}/{total_steps}")
                
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def train_epoch_features(model, dataloader, optimizer, criterion, device, epoch, total_epochs, log_interval=10):
    model.train()
    total_loss = 0.0

    for batch_idx, (features, y) in enumerate(dataloader):
        features = features.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model.classifier(features)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = loss.item()
            log_msg = (f"[Epoch {epoch}/{total_epochs}, Step {batch_idx+1}/{len(dataloader)}] "
                       f"LR: {lr:.6f}, Loss: {avg_loss:.4f}")
            print(log_msg)

    return total_loss / len(dataloader.dataset)


def evaluate_features(model, dataloader, criterion, device, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for features, y in dataloader:
            features = features.to(device)
            y = y.to(device)

            logits = model.classifier(features)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits).squeeze(-1)

            total_loss += loss.item() * len(y)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    if len(dataloader.dataset) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.5

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / max(1, len(dataloader.dataset))
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return avg_loss, precision, recall, f1, auc


def upload_file_to_gcs(local_path: str, gcs_uri: str) -> bool:
    """Uploads a local file to GCS using gcsfs with CLI fallbacks (gcloud storage / gsutil)."""
    if not gcs_uri or not os.path.exists(local_path):
        return False
    
    clean_gcs_uri = gcs_uri.rstrip("/")
    if not clean_gcs_uri.startswith("gs://"):
        clean_gcs_uri = f"gs://{clean_gcs_uri}"
        
    target_uri = f"{clean_gcs_uri}/{os.path.basename(local_path)}"
    print(f"Uploading {local_path} -> {target_uri}...")

    # Method 1: Try gcsfs
    try:
        fs = gcsfs.GCSFileSystem()
        fs.put(local_path, target_uri)
        print(f"  [OK] Successfully uploaded via gcsfs: {target_uri}")
        return True
    except Exception as e:
        print(f"  [Notice] gcsfs upload failed ({e}). Trying CLI fallbacks...")

    # Method 2: Try gcloud storage
    try:
        subprocess.run(f"gcloud storage cp \"{local_path}\" \"{target_uri}\"", shell=True, check=True)
        print(f"  [OK] Successfully uploaded via gcloud storage: {target_uri}")
        return True
    except Exception:
        pass

    # Method 3: Try gsutil
    try:
        subprocess.run(f"gsutil cp \"{local_path}\" \"{target_uri}\"", shell=True, check=True)
        print(f"  [OK] Successfully uploaded via gsutil: {target_uri}")
        return True
    except Exception:
        print(f"  [ERROR] All GCS upload methods failed for {local_path}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train Kronos Short Reversal Classifier")
    parser.add_argument('--dataset', type=str, required=True, help="Path to labeled CSV dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--log_interval', type=int, default=10, help="Logging interval in steps")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--save_dir', type=str, default="./output_models", help="Directory to save model weights")
    parser.add_argument('--freeze_backbone', type=str, default="True", help="Freeze Kronos backbone (True/False)")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")
    parser.add_argument('--pretrained_kronos', type=str, default="NeoQuasar/Kronos-base", help="Pretrained Kronos path")
    parser.add_argument('--pretrained_tokenizer', type=str, default="NeoQuasar/Kronos-Tokenizer-base", help="Pretrained Tokenizer path")
    parser.add_argument('--gcs_output_dir', type=str, default="", help="GCS URI to upload output model (e.g. gs://bucket/short_models/)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 0. Handle GCS downloads if running on Vertex AI
    if args.dataset.startswith("gs://"):
        print(f"Downloading dataset from GCS: {args.dataset}")
        local_dataset = "/tmp/dataset.csv"
        fs = gcsfs.GCSFileSystem()
        fs.get(args.dataset, local_dataset)
        args.dataset = local_dataset

    if args.pretrained_kronos.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        gcs_clean = args.pretrained_kronos.rstrip("/")
        if fs.exists(gcs_clean):
            print(f"Downloading Kronos backbone from GCS: {args.pretrained_kronos}")
            local_kronos = "/tmp/pretrained_kronos"
            os.makedirs(local_kronos, exist_ok=True)
            fs.get(gcs_clean, local_kronos, recursive=True)
            if not os.path.exists(os.path.join(local_kronos, "config.json")):
                subdirs = [os.path.join(local_kronos, d) for d in os.listdir(local_kronos) if os.path.isdir(os.path.join(local_kronos, d))]
                if len(subdirs) == 1:
                    local_kronos = subdirs[0]
            args.pretrained_kronos = local_kronos
        else:
            print(f"Warning: GCS path '{args.pretrained_kronos}' not found. Falling back to 'NeoQuasar/Kronos-base'.")
            args.pretrained_kronos = "NeoQuasar/Kronos-base"

    if args.pretrained_tokenizer.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        gcs_clean = args.pretrained_tokenizer.rstrip("/")
        if fs.exists(gcs_clean):
            print(f"Downloading Tokenizer from GCS: {args.pretrained_tokenizer}")
            local_tokenizer = "/tmp/pretrained_tokenizer"
            os.makedirs(local_tokenizer, exist_ok=True)
            fs.get(gcs_clean, local_tokenizer, recursive=True)
            if not os.path.exists(os.path.join(local_tokenizer, "config.json")):
                subdirs = [os.path.join(local_tokenizer, d) for d in os.listdir(local_tokenizer) if os.path.isdir(os.path.join(local_tokenizer, d))]
                if len(subdirs) == 1:
                    local_tokenizer = subdirs[0]
            args.pretrained_tokenizer = local_tokenizer
        else:
            print(f"Warning: GCS path '{args.pretrained_tokenizer}' not found. Falling back to 'NeoQuasar/Kronos-Tokenizer-base'.")
            args.pretrained_tokenizer = "NeoQuasar/Kronos-Tokenizer-base"

    # 1. Load Data
    print(f"Loading dataset from {args.dataset}...")
    df = pd.read_csv(args.dataset)
    
    # Train / Validation Split (80 / 20 chronologically)
    # Ensure validation set gets enough rows if dataset is small
    if len(df) > 1000:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)
    else:
        # For small sample datasets, train on whole dataset and evaluate on train set
        train_df = df
        val_df = df

    train_dataset = ShortReversalDataset(train_df, seq_len=400)
    val_dataset = ShortReversalDataset(val_df, seq_len=400)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Train positive targets (Label=1): {train_df['label'].sum()} ({train_df['label'].mean()*100:.2f}%)")


    # 2. Load Pretrained Kronos Backbone & Tokenizer
    print("Loading Kronos pretrained foundation models...")
    tokenizer = KronosTokenizer.from_pretrained(args.pretrained_tokenizer).to(device)
    kronos_base = Kronos.from_pretrained(args.pretrained_kronos).to(device)

    freeze_backbone = args.freeze_backbone.lower() in ("true", "1", "yes")

    # 3. Create Short Classifier
    model = KronosShortClassifier(kronos_base, d_model=kronos_base.d_model, dropout=0.2, freeze_backbone=freeze_backbone).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)

    os.makedirs(args.save_dir, exist_ok=True)
    best_save_path = os.path.join(args.save_dir, "kronos_short_classifier.pt")
    latest_save_path = os.path.join(args.save_dir, "kronos_short_classifier_latest.pt")
    best_f1 = -1.0  # Start at -1.0 so Epoch 1 is always saved as initial baseline

    # If backbone is frozen, pre-extract features for 99%+ speedup and cost reduction
    if freeze_backbone:
        train_features, train_labels = extract_dataset_features(train_loader, model, tokenizer, device)
        val_features, val_labels = extract_dataset_features(val_loader, model, tokenizer, device)
        
        train_feat_dataset = TensorDataset(train_features, train_labels)
        val_feat_dataset = TensorDataset(val_features, val_labels)
        
        train_feat_loader = DataLoader(train_feat_dataset, batch_size=args.batch_size, shuffle=True)
        val_feat_loader = DataLoader(val_feat_dataset, batch_size=args.batch_size, shuffle=False)

    print("\nStarting Training...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        if freeze_backbone:
            train_loss = train_epoch_features(model, train_feat_loader, optimizer, criterion, device, epoch, args.epochs, args.log_interval)
            val_loss, precision, recall, f1, auc = evaluate_features(model, val_feat_loader, criterion, device)
        else:
            train_loss = train_epoch(model, tokenizer, train_loader, optimizer, criterion, device, epoch, args.epochs, args.log_interval)
            val_loss, precision, recall, f1, auc = evaluate(model, tokenizer, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}% | "
              f"F1: {f1*100:.1f}% | ROC-AUC: {auc:.4f}")

        # Instant Checkpoint & Instant GCS Backup per Epoch to prevent data loss on crash/preemption
        torch.save(model.state_dict(), latest_save_path)
        if args.gcs_output_dir:
            upload_file_to_gcs(latest_save_path, args.gcs_output_dir)

        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_save_path)
            print(f"  [★] Saved new best model checkpoint to {best_save_path}")
            if args.gcs_output_dir:
                upload_file_to_gcs(best_save_path, args.gcs_output_dir)

    print("=" * 70)
    print(f"Training Complete! Best Validation F1-Score: {best_f1*100:.2f}%")

    # Final backup check to ensure best or latest checkpoint is in GCS
    if args.gcs_output_dir:
        if os.path.exists(best_save_path):
            upload_file_to_gcs(best_save_path, args.gcs_output_dir)
        elif os.path.exists(latest_save_path):
            upload_file_to_gcs(latest_save_path, args.gcs_output_dir)


if __name__ == '__main__':
    main()
