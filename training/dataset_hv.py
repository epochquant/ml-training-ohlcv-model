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
        self.taker_values = None
        if 'takerBuyBaseAssetVolume' in df.columns:
            self.taker_values = df['takerBuyBaseAssetVolume'].values.astype(np.float64)

        split_idx = int(len(data_values) * 0.8)
        if split == 'train':
            self.data = data_values[:split_idx]
            self.time_data = time_values[:split_idx]
            if self.taker_values is not None:
                self.taker_data = self.taker_values[:split_idx]
            else:
                self.taker_data = None
            self.n_samples = self.config.n_train_iter * self.config.batch_size

            # Identify high volatility & breakout/reversal indices for stratified sampling
            self.breakout_indices = []
            if self.config.stratified_sampling and len(self.data) > self.config.max_context + 60:
                closes = self.data[:, 3]
                highs = self.data[:, 1]
                lows = self.data[:, 2]
                vols = self.data[:, 4]
                # Look for forward 15m pump >= 1.5% or volume spike + green candle
                for idx_check in range(self.config.max_context, len(self.data) - 60, 5):
                    c_now = closes[idx_check]
                    c_future = closes[idx_check + 15]
                    if (c_future - c_now) / (c_now + 1e-8) > 0.015:
                        start_cand = max(0, idx_check - self.config.max_context + 1)
                        self.breakout_indices.append(start_cand)
        else:
            self.data = data_values[split_idx:]
            self.time_data = time_values[split_idx:]
            if self.taker_values is not None:
                self.taker_data = self.taker_values[split_idx:]
            else:
                self.taker_data = None
            self.n_samples = self.config.n_valid_iter * self.config.batch_size
            self.breakout_indices = []

        self.seq_len = self.config.max_context
        if len(self.data) < self.seq_len:
            raise ValueError(f"Not enough data in {split} split.")

    def __len__(self):
        return self.n_samples

    def set_epoch_seed(self, seed):
        np.random.seed(seed)

    def __getitem__(self, idx):
        max_start = len(self.data) - self.seq_len - 1
        # Stratified sampling: 35% chance to sample from identified breakout/reversal windows
        if self.split == 'train' and len(self.breakout_indices) > 0 and np.random.rand() < 0.35:
            base_start = np.random.choice(self.breakout_indices)
            jitter = np.random.randint(-20, 20)
            start_idx = int(np.clip(base_start + jitter, 0, max_start))
        else:
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

        # Include taker volume if available
        if self.taker_data is not None:
            taker_win = self.taker_data[start_idx:end_idx, None]
            window_for_regime = np.concatenate([window, pad_column, taker_win], axis=1)
            taker_idx = 6
        else:
            window_for_regime = window_padded
            taker_idx = None

        # --- REGIME VECTOR (Area B), held constant across the window ---
        regime_vec = compute_regime_vector(
            window_for_regime,
            taker_idx=taker_idx,
            atr_span=self.config.atr_window,
            trend_span=self.config.trend_window,
            macro_trend_span=self.config.macro_trend_window,
            vol_span=self.config.vol_window,
            n_features=self.config.n_regime_features,
        )
        regime_window = np.repeat(regime_vec[None, :], self.seq_len, axis=0)

        # --- TIME DATA ---
        window_time = self.time_data[start_idx:end_idx]

        x_tensor = torch.from_numpy(x_norm)
        time_tensor = torch.tensor(window_time.astype(np.float32), dtype=torch.float32)
        regime_tensor = torch.from_numpy(regime_window)

        return x_tensor, time_tensor, regime_tensor
