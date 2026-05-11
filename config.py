"""
Central configuration for the Crop Disease Classifier.
All magic numbers and paths live here — import from this module everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Input / model architecture
# ---------------------------------------------------------------------------
INPUT_SIZE = (128, 128)         # Low-res simulation of field phone cameras
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

# Feature cache filenames (include input size so different runs don't collide)
_size_tag = f"{INPUT_SIZE[0]}"
TRAIN_FEATURES_PATH = FEATURES_DIR / f"train_features_{_size_tag}.npz"
VAL_FEATURES_PATH = FEATURES_DIR / f"val_features_{_size_tag}.npz"
TEST_FEATURES_PATH = FEATURES_DIR / f"test_features_{_size_tag}.npz"

# Saved model artifacts
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.json"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

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
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
}

# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------
# LOCAL CPU setting: 15 trials × 2-fold CV = 30 XGBoost runs (~15-20 min)
# Increase OPTUNA_N_TRIALS to 50 and OPTUNA_CV_FOLDS to 3 if running on
# a machine with more cores or overnight.
OPTUNA_N_TRIALS = 15
OPTUNA_CV_FOLDS = 2

# Optuna search bounds — n_estimators capped at 400 for CPU speed
OPTUNA_SEARCH_SPACE = {
    "n_estimators":      {"low": 100, "high": 400},   # was 200-1000; capped for CPU, "Number of decision trees built in the XGBoost ensemble."
    "learning_rate":     {"low": 1e-2, "high": 0.3,  "log": True},  # raised floor to 0.01, "How much each new tree corrects the previous mistake. Also called eta (η)."
    "max_depth":         {"low": 3,    "high": 6},    # was 3-8; shallower = faster, "How many levels deep each decision tree can grow."
    "subsample":         {"low": 0.6,  "high": 1.0}, #"Fraction of training rows randomly sampled to build each tree."
    "colsample_bytree":  {"low": 0.5,  "high": 1.0}, #"Fraction of features (columns) randomly sampled to build each tree. Most important parameter for MobileNetV2 features."
    "min_child_weight":  {"low": 1,    "high": 10}, #"Minimum number of training samples required in a leaf node to make a split."
    "reg_lambda":        {"low": 0.5,  "high": 5.0}, #"Strength of L2 regularization (Ridge regression) to prevent overfitting."
    "reg_alpha":         {"low": 0.0,  "high": 2.0}, #"Strength of L1 regularization (Lasso regression) to prevent overfitting and encourage sparsity."
}

# ---------------------------------------------------------------------------
# Augmentation parameters (training split only)
# ---------------------------------------------------------------------------
AUG_ROTATION_FACTOR = 0.15     # ±54°
AUG_ZOOM_FACTOR = 0.10         # ±10%
AUG_BRIGHTNESS_FACTOR = 0.20
AUG_CONTRAST_FACTOR = 0.20
AUG_NOISE_STDDEV = 0.05        # Applied AFTER preprocess_input (scale [-1,1]); 5% noise
