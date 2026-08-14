import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import torch

from model.high_volatility.regime import compute_regime_vector, RegimeEmbedding
from model.high_volatility.normalization import normalize_window, denormalize_continuation
from model.high_volatility.model import HighVolatilityKronos, load_hv_kronos_from_base
from model.high_volatility.predictor import HighVolatilityPredictor
from model.high_volatility.tokenizer import HighVolatilityTokenizer


def test_regime_vector_computation():
    # Synthetic random OHLCV window (512 steps, 6 columns: open, high, low, close, volume, amount)
    np.random.seed(42)
    seq_len = 400
    prices = 100.0 + np.cumsum(np.random.randn(seq_len) * 0.5)
    highs = prices + np.abs(np.random.randn(seq_len) * 0.3)
    lows = prices - np.abs(np.random.randn(seq_len) * 0.3)
    opens = (highs + lows) / 2.0
    closes = prices
    vols = np.abs(np.random.randn(seq_len) * 1000.0 + 500.0)
    amts = vols * closes
    taker_vols = vols * 0.55

    x_raw = np.stack([opens, highs, lows, closes, vols, amts, taker_vols], axis=1)

    # 8-feature regime vector
    r8 = compute_regime_vector(x_raw, taker_idx=6, n_features=8)
    assert r8.shape == (8,)
    assert not np.isnan(r8).any()
    assert 0.0 <= r8[6] <= 1.0  # taker ratio bounded
    assert 0.0 <= r8[7] <= 1.0  # lower wick ratio bounded

    # 3-feature backward compatibility
    r3 = compute_regime_vector(x_raw, n_features=3)
    assert r3.shape == (3,)
    assert not np.isnan(r3).any()


def test_regime_embedding_zero_init():
    d_model = 256
    n_features = 8
    regime_emb = RegimeEmbedding(n_features=n_features, d_model=d_model)

    dummy_input = torch.randn(2, 50, n_features)
    output = regime_emb(dummy_input)
    assert output.shape == (2, 50, d_model)
    # Output should be exactly zero initially due to zero-initialized final layer
    assert torch.allclose(output, torch.zeros_like(output))


def test_normalization_roundtrip():
    np.random.seed(42)
    x = np.random.uniform(5.0, 10.0, size=(100, 6))

    # Logreturn mode
    norm_log, stats_log = normalize_window(x, mode="logreturn")
    denorm_log = denormalize_continuation(norm_log[-10:], stats_log)
    assert denorm_log.shape == (10, 6)
    assert not np.isnan(denorm_log).any()

    # Robust mode
    norm_rob, stats_rob = normalize_window(x, mode="robust")
    denorm_rob = denormalize_continuation(norm_rob[-10:], stats_rob)
    assert denorm_rob.shape == (10, 6)
    assert not np.isnan(denorm_rob).any()


def test_akeusdt_support_retest_regime():
    # Verify that the 8-feature regime correctly extracts the support bounce and macro trend
    # on AKEUSDT historical candle slice at 2026-08-14 03:31
    import json

    data_file = Path(__file__).parent / "data" / "AKEUSDT_1m_2026-08-10_2026-08-14.json"
    fallback_file = Path(__file__).parent / "data" / "akeusdt_sample.json"

    if data_file.exists():
        with open(data_file, 'r') as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        df['dt'] = pd.to_datetime(df['openTime'], unit='ms')
        target_time = pd.Timestamp('2026-08-14 03:31:00')
        idx = df[df['dt'] == target_time].index[0]
        window = df.iloc[idx-399:idx+1].copy().reset_index(drop=True)
    elif fallback_file.exists():
        with open(fallback_file, 'r') as f:
            raw = json.load(f)
        window = pd.DataFrame(raw)
    else:
        pytest.fail(f"Required test candle data not found in {data_file}")

    for col in ['open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'takerBuyBaseAssetVolume']:
        if col in window.columns:
            window[col] = window[col].astype(float)

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'takerBuyBaseAssetVolume']
    x_raw = window[feature_cols].values

    r_vec = compute_regime_vector(x_raw, taker_idx=6, n_features=8)

    # 1. atr_pct > 0.01 (High volatility)
    assert r_vec[0] > 0.01
    # 2. overext_short is negative due to the 10-min flush
    assert r_vec[1] < 0
    # 3. overext_macro is close to 0 (Price sits on EMA 200)
    assert abs(r_vec[2]) < 0.5
    # 4. ret_240m (4h macro return) is strongly positive (> 10%)
    assert r_vec[4] > 0.10
    # 5. lower_wick_ratio is large (> 0.40, strong absorption shadow)
    assert r_vec[7] > 0.40


def test_hv_kronos_forward_and_embedding():
    # Instantiate a mini HighVolatilityKronos
    model = HighVolatilityKronos(
        s1_bits=6,
        s2_bits=6,
        n_layers=2,
        d_model=64,
        n_heads=2,
        ff_dim=128,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
        n_regime_features=8,
    )
    model.eval()

    batch_size = 2
    seq_len = 32
    s1_ids = torch.randint(0, 64, (batch_size, seq_len))
    s2_ids = torch.randint(0, 64, (batch_size, seq_len))
    stamp = torch.stack([
        torch.randint(0, 60, (batch_size, seq_len)),
        torch.randint(0, 24, (batch_size, seq_len)),
        torch.randint(0, 7, (batch_size, seq_len)),
        torch.randint(1, 28, (batch_size, seq_len)),
        torch.randint(1, 12, (batch_size, seq_len)),
    ], dim=-1).float()
    regime = torch.randn(batch_size, seq_len, 8)

    with torch.no_grad():
        s1_logits, s2_logits = model(s1_ids, s2_ids, stamp=stamp, regime=regime)
        assert s1_logits.shape == (batch_size, seq_len, 64)
        assert s2_logits.shape == (batch_size, seq_len, 64)

        s1_dec, ctx = model.decode_s1(s1_ids, s2_ids, stamp=stamp, regime=regime)
        assert s1_dec.shape == (batch_size, seq_len, 64)
        assert ctx.shape == (batch_size, seq_len, 64)
