import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from training.config_hv import ConfigHV
from model.high_volatility.normalization import normalize_window
from model.high_volatility.regime import compute_regime_vector


class HighVolatilityDataset(Dataset):
    """High-volatility counterpart of training.dataset.QlibDataset.

    QlibDataset feeds the model raw, unnormalized OHLCV values with a
    zero-padded 6th column (training/dataset.py:57-61) -- for this variant
    only, each sampled window is instead normalized via
    model.high_volatility.normalization.normalize_window (Area A) and paired
    with a per-window regime vector from model.high_volatility.regime
    (Area B), broadcast across the window so it can be added at every
    timestep by HighVolatilityKronos. training/dataset.py itself is untouched.
    """

    def __init__(self, split='train'):
        self.config = ConfigHV()
        self.split = split

        df = pd.read_csv(self.config.dataset_path)

        if 'timestamps' not in df.columns:
            raise ValueError("CSV must contain a 'timestamps' column.")

        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month

        data_values = df[self.config.feature_list].values
        time_values = df[self.config.time_feature_list].values

        split_idx = int(len(data_values) * 0.8)
        if split == 'train':
            self.data = data_values[:split_idx]
            self.time_data = time_values[:split_idx]
            self.n_samples = self.config.n_train_iter * self.config.batch_size
        else:
            self.data = data_values[split_idx:]
            self.time_data = time_values[split_idx:]
            self.n_samples = self.config.n_valid_iter * self.config.batch_size

        self.seq_len = self.config.max_context
        if len(self.data) < self.seq_len:
            raise ValueError(f"Not enough data in {split} split.")

    def __len__(self):
        return self.n_samples

    def set_epoch_seed(self, seed):
        np.random.seed(seed)

    def __getitem__(self, idx):
        max_start = len(self.data) - self.seq_len - 1
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + self.seq_len

        # --- PRICE DATA (Matrix Padding 5 -> 6, then Area A normalization) ---
        window = self.data[start_idx:end_idx].astype(np.float64)
        pad_column = np.zeros((self.seq_len, 1), dtype=np.float64)
        window_padded = np.concatenate([window, pad_column], axis=1)

        x_norm, _ = normalize_window(
            window_padded,
            mode=self.config.normalization_mode,
            clip=self.config.clip,
            soft_clip=self.config.soft_clip,
        )

        # --- REGIME VECTOR (Area B), held constant across the window ---
        regime_vec = compute_regime_vector(
            window_padded,
            atr_span=self.config.atr_window,
            trend_span=self.config.trend_window,
            vol_span=self.config.vol_window,
        )
        regime_window = np.repeat(regime_vec[None, :], self.seq_len, axis=0)

        # --- TIME DATA ---
        window_time = self.time_data[start_idx:end_idx]

        x_tensor = torch.from_numpy(x_norm)
        time_tensor = torch.tensor(window_time.astype(np.float32), dtype=torch.float32)
        regime_tensor = torch.from_numpy(regime_window)

        return x_tensor, time_tensor, regime_tensor
