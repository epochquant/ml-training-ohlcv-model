import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Ensure repository root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.kronos import Kronos, KronosTokenizer
from model.high_volatility.short_classifier import HighVolatilityShortClassifier
from training.train_short_classifier_hv import ShortReversalDatasetHV, evaluate, EXTRA_FEATURE_COLS
from training.focal_loss import BinaryFocalLoss


def run_trade_simulation_hv(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.80, seq_len: int = 400,
                             k_win_drop: float = 2.0, k_win_risk: float = 0.8):
    """ATR-relative counterpart of training.evaluate_short_classifier.run_trade_simulation.

    Same short-trade mechanics, but the win condition scales with each
    candle's own ATR% (from label_short_signals_hv.py::compute_indicators_hv)
    instead of a flat 2.5% drop / 1.0% risk, consistent with the ATR-relative
    label definition used to train this classifier.
    """
    valid_df = df.iloc[seq_len:].reset_index(drop=True)
    highs = valid_df['high'].values
    closes = valid_df['close'].values
    atr_pct = valid_df['atr_pct'].values if 'atr_pct' in valid_df.columns else np.full(len(valid_df), 1.0)
    n = len(valid_df)

    trades = []

    for i in range(n - 12):
        if probs[i] >= threshold:
            entry_price = closes[i]
            future_closes = closes[i + 1: i + 13]
            future_highs = highs[i + 1: i + 13]

            min_future_close = np.min(future_closes)
            max_future_high = np.max(future_highs)

            max_drop_pct = (entry_price - min_future_close) / entry_price * 100.0
            max_risk_pct = (max_future_high - entry_price) / entry_price * 100.0

            candle_atr_pct = max(atr_pct[i], 1e-5)
            is_win = (max_drop_pct >= k_win_drop * candle_atr_pct) and (max_risk_pct <= k_win_risk * candle_atr_pct)
            trades.append({
                'entry_index': i,
                'entry_price': entry_price,
                'max_drop_pct': max_drop_pct,
                'max_risk_pct': max_risk_pct,
                'win': is_win
            })

    if not trades:
        return 0, 0.0, 0.0, 0.0

    trade_df = pd.DataFrame(trades)
    win_rate = trade_df['win'].mean() * 100.0
    avg_drop = trade_df['max_drop_pct'].mean()
    avg_risk = trade_df['max_risk_pct'].mean()

    return len(trades), win_rate, avg_drop, avg_risk


def main():
    parser = argparse.ArgumentParser(description="Evaluate the high-volatility Kronos Short Reversal Classifier & Simulate Trades")
    parser.add_argument('--dataset', type=str, required=True, help="Path to labeled test CSV (label_short_signals_hv.py output)")
    parser.add_argument('--model_weights', type=str, default="./output_models_hv/kronos_short_classifier_hv.pt", help="Path to trained weights")
    parser.add_argument('--pretrained_kronos', type=str, default="NeoQuasar/Kronos-base")
    parser.add_argument('--pretrained_tokenizer', type=str, default="NeoQuasar/Kronos-Tokenizer-base")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading test data from {args.dataset}...")
    df = pd.read_csv(args.dataset)
    test_dataset = ShortReversalDatasetHV(df, seq_len=400)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Loading Kronos backbone and high-volatility classifier model...")
    tokenizer = KronosTokenizer.from_pretrained(args.pretrained_tokenizer).to(device)
    kronos_base = Kronos.from_pretrained(args.pretrained_kronos).to(device)

    model = HighVolatilityShortClassifier(
        kronos_base, d_model=kronos_base.d_model, n_extra_features=len(EXTRA_FEATURE_COLS)
    ).to(device)
    if os.path.exists(args.model_weights):
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
        print(f"Loaded weights from {args.model_weights}")
    else:
        print(f"Warning: Weights file {args.model_weights} not found. Running with initial weights.")

    criterion = BinaryFocalLoss()
    avg_loss, precision, recall, f1, auc = evaluate(model, tokenizer, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("HIGH-VOLATILITY CLASSIFIER PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  Test Loss:     {avg_loss:.4f}")
    print(f"  Precision:     {precision * 100:.2f}%")
    print(f"  Recall:        {recall * 100:.2f}%")
    print(f"  F1-Score:      {f1 * 100:.2f}%")
    print(f"  ROC-AUC:       {auc:.4f}")
    print("=" * 60)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, x_stamp, extra, _ in test_loader:
            x, x_stamp, extra = x.to(device), x_stamp.to(device), extra.to(device)
            x_tokens = tokenizer.encode(x, half=True)
            logits = model(x_tokens[0], x_tokens[1], extra, stamp=x_stamp)
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    print("\nSIMULATED SHORT TRADE PERFORMANCE AT DIFFERENT CONFIDENCE THRESHOLDS (ATR-relative win condition)")
    print("-" * 75)
    print(f"{'Threshold':<12} | {'Trades Fired':<14} | {'Win Rate (%)':<14} | {'Avg Drawdown (%)':<16}")
    print("-" * 75)

    for th in [0.50, 0.65, 0.75, 0.80, 0.85, 0.90]:
        num_trades, win_rate, avg_drop, avg_risk = run_trade_simulation_hv(df, all_probs, threshold=th)
        print(f"{th*100:5.0f}%        | {num_trades:<14} | {win_rate:12.1f}% | {avg_drop:14.2f}%")
    print("-" * 75)


if __name__ == '__main__':
    main()
