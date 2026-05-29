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
    "n_estimators": 412,
    "learning_rate": 0.1895,
    "max_depth": 4,
    "subsample": 0.79,
    "colsample_bytree": 0.49,
    "min_child_weight": 1,
    "reg_lambda": 4.00,
    "reg_alpha": 0.40,
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
# SMOTE oversampling  (applied to training features only)
# ---------------------------------------------------------------------------
# USE_SMOTE = False  (recommended for this project)
# SMOTE hurts accuracy on 1280-dim MobileNetV2 features because:
#   1. Synthetic interpolated points blur class boundaries in high-dim space.
#   2. SMOTE + sample weights are redundant: once every class reaches
#      target_per_class, all weights ≈ 1.0 and the weights do nothing.
# With USE_SMOTE = False, XGBoost's inverse-frequency sample weights handle
# the 21× class imbalance directly — rare classes get up to 9× gradient signal.
USE_SMOTE = False               # ← set True only to experiment with SMOTE

# These values are only used when USE_SMOTE = True
SMOTE_TARGET_PER_CLASS = 2000   # target samples per minority class
SMOTE_K_NEIGHBORS = 5           # k-nearest neighbours for SMOTE synthesis

# ---------------------------------------------------------------------------
# PCA dimensionality reduction  (applied after SMOTE, before XGBoost)
# ---------------------------------------------------------------------------
# Reduces 1280 MobileNetV2 features → PCA_N_COMPONENTS features.
# PCA is fit ONLY on training data, then applied to val and test.
# Saved to disk so inference pipeline can apply the same transformation.
#
# At 128×128 input, MobileNetV2 features are spread across ALL 1280 dims:
#   256 components → 72.5% variance  (too much loss for fine-grained disease ID)
#   512 components → ~88% variance
#   800 components → ~95% variance
#
# Rule of thumb: keep enough components for ≥95% variance.
# Set USE_PCA = False to skip PCA entirely (XGBoost handles 1280 dims fine
# via colsample_bytree; recommended when variance per component is low).
USE_PCA = False                 # ← set True only if you want PCA compression
PCA_N_COMPONENTS = 800          # captures ~95% variance at 128×128 input
PCA_WHITEN = False              # whitening normalises variance; False = faster

# Saved preprocessor artefacts (alongside xgb_model.json)
PCA_PATH    = MODELS_DIR / "pca.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"   # StandardScaler applied before PCA

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
