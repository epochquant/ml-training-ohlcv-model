import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import gcsfs
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


def train_epoch(model, tokenizer, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, x_stamp, y in dataloader:
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


def main():
    parser = argparse.ArgumentParser(description="Train Kronos Short Reversal Classifier")
    parser.add_argument('--dataset', type=str, required=True, help="Path to labeled CSV dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--save_dir', type=str, default="./output_models", help="Directory to save model weights")
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
        print(f"Downloading Kronos backbone from GCS: {args.pretrained_kronos}")
        local_kronos = "/tmp/pretrained_kronos"
        os.makedirs(local_kronos, exist_ok=True)
        fs = gcsfs.GCSFileSystem()
        fs.get(args.pretrained_kronos.rstrip("/") + "/*", local_kronos + "/")
        args.pretrained_kronos = local_kronos

    if args.pretrained_tokenizer.startswith("gs://"):
        print(f"Downloading Tokenizer from GCS: {args.pretrained_tokenizer}")
        local_tokenizer = "/tmp/pretrained_tokenizer"
        os.makedirs(local_tokenizer, exist_ok=True)
        fs = gcsfs.GCSFileSystem()
        fs.get(args.pretrained_tokenizer.rstrip("/") + "/*", local_tokenizer + "/")
        args.pretrained_tokenizer = local_tokenizer

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Train positive targets (Label=1): {train_df['label'].sum()} ({train_df['label'].mean()*100:.2f}%)")


    # 2. Load Pretrained Kronos Backbone & Tokenizer
    print("Loading Kronos pretrained foundation models...")
    tokenizer = KronosTokenizer.from_pretrained(args.pretrained_tokenizer).to(device)
    kronos_base = Kronos.from_pretrained(args.pretrained_kronos).to(device)

    # 3. Create Short Classifier
    model = KronosShortClassifier(kronos_base, d_model=kronos_base.d_model, dropout=0.2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)

    os.makedirs(args.save_dir, exist_ok=True)
    best_save_path = os.path.join(args.save_dir, "kronos_short_classifier.pt")
    best_f1 = 0.0

    print("\nStarting Training...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, criterion, device)
        val_loss, precision, recall, f1, auc = evaluate(model, tokenizer, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}% | "
              f"F1: {f1*100:.1f}% | ROC-AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_save_path)
            print(f"  [★] Saved new best model checkpoint to {best_save_path}")

    print("=" * 70)
    print(f"Training Complete! Best Validation F1-Score: {best_f1*100:.2f}%")

    if args.gcs_output_dir:
        gcs_uri = args.gcs_output_dir.rstrip("/")
        print(f"\nUploading best model checkpoint to GCS: {gcs_uri}")
        fs = gcsfs.GCSFileSystem()
        fs.put(best_save_path, f"{gcs_uri}/kronos_short_classifier.pt")
        print("Upload complete!")


if __name__ == '__main__':
    main()
