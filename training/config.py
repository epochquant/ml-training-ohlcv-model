import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root (one level up from training/)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class Config:
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
        # Env vars:  GCS_BUCKET_NAME, GCS_DESTINATION_PREFIX, GCS_KEY_FILE
        # Path convention:
        #   gs://<GCS_BUCKET_NAME>/<GCS_DESTINATION_PREFIX>/<symbol>_<tf>.csv
        # ---------------------------------------------------------------
        self.gcs_bucket  = os.getenv("GCS_BUCKET_NAME", "epochquant-training")
        self.gcs_key_file = os.getenv("GCS_KEY_FILE", "").strip() or None
        self.gcs_project = None  # Set to your GCP project ID if needed

        # Active dataset path — GCS URI or local path for offline dev.
        # The prefix must match GCS_DESTINATION_PREFIX in .env so that
        # convert_json_to_csv.py and this config always point to the same folder.
        _prefix = os.getenv("GCS_DESTINATION_PREFIX", "training-data").strip("/")
        
        env_dataset_path = os.getenv("DATASET_PATH", "").strip()
        if env_dataset_path:
            self.dataset_path = env_dataset_path
        else:
            self.dataset_path = f"gs://{self.gcs_bucket}/{_prefix}/bnbusdt_1m_kronos_data.csv"

        self.instrument = os.getenv("SYMBOL", "bnbusdt")
        self.dataset_begin_time = "2024-12-13"
        self.dataset_end_time = "2026-06-06"

        # --- Model Architecture ---
        self.lookback_window = 400
        self.predict_window = 60
        self.max_context = 512
        self.feature_list = ["open", "high", "low", "close", "volume"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

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
        # Use GCS path for cloud training, e.g.:
        #   self.save_path = "gs://epochquant-training/models"
        self.save_path = "./output_models"
        self.tokenizer_save_folder_name = "tokenizer_finetuned"
        self.predictor_save_folder_name = "predictor_finetuned"

        # --- Pretrained Base Models (HuggingFace Hub) ---
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"

        # Fine-tuned model paths (after tokenizer training is done)
        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )

        # --- Extras ---
        self.use_comet = False
        self.backtest_benchmark = os.getenv("SYMBOL", "bnbusdt")
