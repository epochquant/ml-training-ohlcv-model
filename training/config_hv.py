import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root (one level up from training/)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class ConfigHV:
    """High-volatility counterpart of training.config.Config.

    Same env-var-driven shape (consumed by training/train_tokenizer_hv.py and
    training/train_predictor_hv.py exactly like Config feeds
    training/train_tokenizer.py / training/train_predictor.py), but namespaced
    under high-volatility/ in GCS and ./output_models_hv locally, and carrying
    the extra Area A/B/D knobs. training/config.py itself is untouched.
    """

    def __init__(self):
        # --- Core Training Variables ---
        self.seed = 42
        self.accumulation_steps = 1
        self.log_steps = 10
        self.log_interval = 10
        self.n_train_iter = 1000
        self.n_valid_iter = 200

        # ---------------------------------------------------------------
        # GCS Data Source — configured via .env at repo root
        # Path convention:
        #   gs://<GCS_BUCKET_NAME>/high-volatility/training-data/<symbol>_<tf>.csv
        # ---------------------------------------------------------------
        self.gcs_bucket = os.getenv("GCS_BUCKET_NAME", "epochquant-training")
        self.gcs_key_file = os.getenv("GCS_KEY_FILE", "").strip() or None
        self.gcs_project = None

        env_dataset_path = os.getenv("DATASET_PATH", "").strip()
        if env_dataset_path:
            self.dataset_path = env_dataset_path
        else:
            self.dataset_path = f"gs://{self.gcs_bucket}/high-volatility/training-data/dogeusdt_1m.csv"

        self.instrument = os.getenv("SYMBOL", "dogeusdt")
        self.dataset_begin_time = "2024-01-01"
        self.dataset_end_time = "2026-06-06"

        # --- Model Architecture ---
        self.lookback_window = 400
        self.predict_window = 60
        self.max_context = 512
        self.feature_list = ["open", "high", "low", "close", "volume"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

        # --- Area A: normalization & clipping (replaces raw z-score + hard clip) ---
        self.normalization_mode = os.getenv("HV_NORMALIZATION_MODE", "logreturn")
        self.soft_clip = os.getenv("HV_SOFT_CLIP", "true").strip().lower() != "false"
        self.clip = 5.0

        # --- Area B: volatility regime embedding ---
        self.n_regime_features = int(os.getenv("HV_N_REGIME_FEATURES", "8"))
        self.atr_window = float(os.getenv("HV_ATR_WINDOW", "14"))
        self.trend_window = float(os.getenv("HV_TREND_WINDOW", "50"))
        self.macro_trend_window = float(os.getenv("HV_MACRO_TREND_WINDOW", "200"))
        self.macro_ret_window = float(os.getenv("HV_MACRO_RET_WINDOW", "240"))
        self.vol_window = float(os.getenv("HV_VOL_WINDOW", "20"))

        # --- Area C: loss weighting & sampling ---
        self.asymmetric_loss_weight = float(os.getenv("HV_ASYMMETRIC_LOSS_WEIGHT", "2.5"))
        self.stratified_sampling = os.getenv("HV_STRATIFIED_SAMPLING", "true").strip().lower() != "false"

        # --- Area D: uncertainty-aware forecasting ---
        self.sample_count = int(os.getenv("HV_SAMPLE_COUNT", "20"))
        self.predict_quantiles = (0.1, 0.5, 0.9)

        # --- Hardware & Optimizers ---
        self.device = "cuda"
        self.epochs = 3
        self.batch_size = 8
        self.num_workers = 2
        self.tokenizer_learning_rate = 1e-5
        self.predictor_learning_rate = 1e-5
        self.adam_weight_decay = 1e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999

        # --- Save Paths ---
        self.save_path = "./output_models_hv"
        self.tokenizer_save_folder_name = "tokenizer_finetuned"
        self.predictor_save_folder_name = "predictor_finetuned"

        # --- Pretrained Base Models (warm-start source, HuggingFace Hub) ---
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"

        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )

        # --- Extras ---
        self.use_comet = False
        self.backtest_benchmark = os.getenv("SYMBOL", "dogeusdt")
