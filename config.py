"""
Central configuration for the Crop Disease Classifier.
All magic numbers and paths live here — import from this module everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Input / model architecture
# ---------------------------------------------------------------------------
INPUT_SIZE = (224, 224)         # MobileNetV2 native size — best feature quality
NUM_CHANNELS = 3
FEATURE_DIM = 1280              # MobileNetV2 GlobalAveragePooling2D output dim
NUM_CLASSES = 15                # Dataset: Pepper(2) + Potato(3) + Tomato(10)

# ---------------------------------------------------------------------------
# Training splits
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
BATCH_SIZE = 64                 # Images per batch during feature extraction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Feature cache filenames
# Switch FEATURE_SOURCE to "finetuned" after running scripts/finetune.py
# to train XGBoost on the improved fine-tuned features.
FEATURE_SOURCE = "finetuned"          # "224" = frozen MobileNetV2 | "finetuned" = fine-tuned
_size_tag = f"{INPUT_SIZE[0]}"
TRAIN_FEATURES_PATH = FEATURES_DIR / f"train_features_{FEATURE_SOURCE}.npz"
VAL_FEATURES_PATH   = FEATURES_DIR / f"val_features_{FEATURE_SOURCE}.npz"
TEST_FEATURES_PATH  = FEATURES_DIR / f"test_features_{FEATURE_SOURCE}.npz"

# Saved model artifacts
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.json"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
# Fine-tuned feature extractor (saved by scripts/finetune.py).
# When FEATURE_SOURCE == "finetuned" this is loaded at inference time instead
# of rebuilding the frozen MobileNetV2, ensuring the same feature space that
# XGBoost was trained on is used during prediction.
FINETUNED_EXTRACTOR_PATH = MODELS_DIR / "finetuned_feature_extractor.keras"

# ---------------------------------------------------------------------------
# XGBoost baseline hyperparameters
# ---------------------------------------------------------------------------
XGB_BASE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,   # 15 classes: Pepper(2) + Potato(3) + Tomato(10)
    "eval_metric": "mlogloss",
    "tree_method": "hist",      # Fast histogram-based training
    "device": "cpu",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "n_estimators": 361,
    "learning_rate": 0.1377,
    "max_depth": 3,
    "subsample": 0.9625,
    "colsample_bytree": 0.3280,
    "min_child_weight": 1,
    "reg_lambda": 9.0785,
    "reg_alpha": 0.0468,
}

# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------
# LOCAL CPU setting: 20 trials × 3-fold CV = 60 XGBoost runs (~30-40 min)
# With SMOTE the training set is now ~30k samples (larger) so 3-fold CV
# gives more reliable estimates than 2-fold.
OPTUNA_N_TRIALS = 50
OPTUNA_CV_FOLDS = 3

# Optuna search bounds.
# With 1280 raw features and SMOTE-balanced data:
#   - colsample_bytree is the MOST important param (feature subsampling)
#   - n_estimators raised to 600 (more data → more trees needed)
#   - max_depth kept shallow (1280 correlated features → shallow = less overfit)
OPTUNA_SEARCH_SPACE = {
    "n_estimators":      {"low": 200, "high": 600},   # More data → more trees
    "learning_rate":     {"low": 1e-2, "high": 0.3,  "log": True},
    "max_depth":         {"low": 3,    "high": 6},    # Shallow prevents overfit on correlated features
    "subsample":         {"low": 0.6,  "high": 1.0},
    "colsample_bytree":  {"low": 0.3,  "high": 0.8},  # MOST IMPORTANT: lower = more regularisation on 1280 features
    "min_child_weight":  {"low": 1,    "high": 10},
    "reg_lambda":        {"low": 0.5,  "high": 10.0}, # Wider range: more L2 may help with 1280 correlated dims
    "reg_alpha":         {"low": 0.0,  "high": 2.0},
}

# ---------------------------------------------------------------------------
# Augmentation parameters (training split only)
# ---------------------------------------------------------------------------
AUG_ROTATION_FACTOR = 0.15     # ±54°
AUG_ZOOM_FACTOR = 0.10         # ±10%
AUG_BRIGHTNESS_FACTOR = 0.20
AUG_CONTRAST_FACTOR = 0.20
AUG_NOISE_STDDEV = 0.05        # Applied AFTER preprocess_input (scale [-1,1]); 5% noise

# ---------------------------------------------------------------------------
# LangGraph inference workflow
# ---------------------------------------------------------------------------
# Predictions with top-1 confidence ≥ this threshold go through the
# "high confidence" path; below it the workflow asks for a clearer image.
CONFIDENCE_THRESHOLD = 0.60    # 60 % — adjust if too many low-confidence hits
