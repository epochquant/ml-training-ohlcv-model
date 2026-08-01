import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

# Ensure repository root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.kronos import Kronos, KronosTokenizer
from model.short_classifier import KronosShortClassifier
from src.data.label_short_signals import compute_indicators



class ShortSignalPredictor:
    """
    Inference utility for predicting short reversal signals on new OHLCV candle data.
    """

    def __init__(self, 
                 model_path: str = "./output_models/kronos_short_classifier.pt",
                 pretrained_kronos: str = "NeoQuasar/Kronos-base",
                 pretrained_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base",
                 seq_len: int = 400,
                 clip: float = 5.0,
                 device: str = None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.clip = clip

        self.tokenizer = KronosTokenizer.from_pretrained(pretrained_tokenizer).to(self.device)
        kronos_base = Kronos.from_pretrained(pretrained_kronos).to(self.device)
        
        self.model = KronosShortClassifier(kronos_base, d_model=kronos_base.d_model).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained Short Classifier weights from {model_path}")
        else:
            print(f"Warning: Model weights file '{model_path}' not found. Using untrained weights.")

        self.model.eval()

    def predict_dataframe(self, df: pd.DataFrame, threshold: float = 0.80) -> dict:
        """
        Takes a pandas DataFrame containing at least 400 candles.
        Returns a dictionary with short probability, signal flag, and metadata.
        """
        if len(df) < self.seq_len:
            raise ValueError(f"Input DataFrame must contain at least {self.seq_len} candles. Got {len(df)}.")

        df = df.tail(self.seq_len).copy().reset_index(drop=True)
        df = compute_indicators(df)

        # Standard timestamp features
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df['minute'] = df['timestamps'].dt.minute
            df['hour'] = df['timestamps'].dt.hour
            df['weekday'] = df['timestamps'].dt.weekday
            df['day'] = df['timestamps'].dt.day
            df['month'] = df['timestamps'].dt.month

        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_ratio']
        time_cols = ['minute', 'hour', 'weekday', 'day', 'month']

        for c in feature_cols + time_cols:
            if c not in df.columns:
                df[c] = 0.0

        x = df[feature_cols].values.astype(np.float32)
        x_stamp = df[time_cols].values.astype(np.float32)

        # Normalization per window
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -self.clip, self.clip)

        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(self.device)  # [1, 400, 6]
        stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(self.device)  # [1, 400, 5]

        with torch.no_grad():
            x_tokens = self.tokenizer.encode(x_tensor, half=True)
            s1_ids, s2_ids = x_tokens[0], x_tokens[1]

            logits = self.model(s1_ids, s2_ids, stamp=stamp_tensor)
            prob = torch.sigmoid(logits).item()

        is_short_signal = prob >= threshold

        latest_candle = df.iloc[-1]
        return {
            "short_probability": round(prob, 4),
            "short_probability_pct": f"{prob * 100:.1f}%",
            "is_short_signal": is_short_signal,
            "signal_action": "OPEN_SHORT_POSITION" if is_short_signal else "NO_SIGNAL",
            "threshold_used": threshold,
            "latest_close": float(latest_candle['close']),
            "latest_high": float(latest_candle['high']),
            "suggested_take_profit": f"{float(latest_candle['close']) * 0.965:.4f} (-3.5%)",
            "suggested_stop_loss": f"{float(latest_candle['high']) * 1.008:.4f} (+0.8%)"
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Short Reversal Signal on candle JSON/CSV file")
    parser.add_argument('--file', type=str, required=True, help="Path to JSON or CSV candle file")
    parser.add_argument('--model_weights', type=str, default="./output_models/kronos_short_classifier.pt")
    parser.add_argument('--threshold', type=float, default=0.80, help="Confidence threshold (default: 0.80)")
    args = parser.parse_args()

    predictor = ShortSignalPredictor(model_path=args.model_weights)

    if args.file.endswith('.json'):
        with open(args.file, 'r') as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        if 'openTime' in df.columns:
            df['timestamps'] = pd.to_datetime(df['openTime'], unit='ms')
    else:
        df = pd.read_csv(args.file)

    result = predictor.predict_dataframe(df, threshold=args.threshold)
    print("\n" + "=" * 60)
    print("SHORT REVERSAL INFERENCE RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


if __name__ == '__main__':
    main()
