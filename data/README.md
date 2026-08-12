# Data Directory

Training data **is not committed to this repository**. It lives in the GCS bucket:

```
gs://epochquant-training/
├── raw/           # Raw Binance JSON kline dumps, organized by symbol
│   └── bnbusdt/
│       └── bnbusdt_1m_2024.json
├── processed/     # Cleaned CSV files ready for training
│   └── bnbusdt_1m.csv
└── models/        # Saved model checkpoints
    └── bnbusdt_1m_run1/
```

## Column Schema (processed CSV)

| Column | Type | Description |
|--------|------|-------------|
| `timestamps` | datetime | Candle open time (UTC) |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | float | Base asset volume |
| `amount` | float | Quote asset volume |

## Uploading Data to GCS

```bash
# Single file
gsutil cp my_data.csv gs://epochquant-training/processed/bnbusdt_1m.csv

# Entire folder
gsutil -m cp -r ./raw/bnbusdt/ gs://epochquant-training/raw/bnbusdt/
```

## Converting Binance JSON to CSV

```bash
python data/convert_json_to_csv.py \
    --input-dir data/raw/bnbusdt/ \
    --output gs://epochquant-training/processed/bnbusdt_1m.csv \
    --symbol BNBUSDT
```

## Loading in Code

```python
from data.data_loader import load_dataset

# From GCS (default in cloud)
df = load_dataset("gs://epochquant-training/processed/bnbusdt_1m.csv")

# From local file (offline dev)
df = load_dataset("data/processed/bnbusdt_1m.csv")

# From local JSON folder
df = load_dataset("data/raw/bnbusdt/")
```

See the root [README.md](../README.md#step-7-train-the-high-volatility-model-variant-optional) for `launch_high_volatility_job.py` usage (the high-volatility `high-volatility-model` training job launcher).
