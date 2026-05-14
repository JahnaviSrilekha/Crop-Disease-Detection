"""
Feature preprocessor: SMOTE oversampling + StandardScaler + PCA.

Pipeline (applied in this exact order):

  Raw features  (N, 1280)  ← from MobileNetV2 GAP cache
        │
        ▼  SMOTE  (training split only)
  Balanced features  (N_balanced, 1280)
        │
        ▼  StandardScaler  (fit on train, transform all splits)
  Scaled features  (N_balanced, 1280)  mean=0, std=1 per feature
        │
        ▼  PCA  (fit on train, transform all splits)
  Reduced features  (N_balanced, 256)  ← ready for XGBoost

Why this order matters:
  - SMOTE must run before PCA (synthesises in original 1280-dim space)
  - StandardScaler must run before PCA (PCA is sensitive to feature scales)
  - PCA fit only on training data to prevent data leakage into val/test

Saved artefacts (for inference):
  outputs/models/scaler.pkl  ← StandardScaler
  outputs/models/pca.pkl     ← fitted PCA
"""

from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SMOTE Oversampling
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_per_class: int = config.SMOTE_TARGET_PER_CLASS,
    k_neighbors: int = config.SMOTE_K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes using SMOTE so every class has at least
    `target_per_class` training samples.

    Classes already at or above `target_per_class` are left untouched
    (no downsampling ever happens here).

    SMOTE synthesises new samples by:
      1. Pick a minority sample A
      2. Find its k nearest neighbours in the same class
      3. Pick a random neighbour B
      4. Create synthetic sample = A + random(0,1) × (B - A)
      → new point lies on the line between two real samples ✅

    Args:
        X_train:          (N, 1280) float32 feature matrix
        y_train:          (N,) int32 labels
        target_per_class: desired minimum samples per class
        k_neighbors:      SMOTE neighbourhood size

    Returns:
        X_resampled: (N_balanced, 1280) — always >= N
        y_resampled: (N_balanced,)
    """
    unique, counts = np.unique(y_train, return_counts=True)

    logger.info("Class distribution BEFORE SMOTE:")
    for cls, cnt in zip(unique, counts):
        bar = "█" * (cnt // 50)
        logger.info("  class %2d: %5d  %s", cls, cnt, bar)
    logger.info("  Imbalance ratio: %.1fx  (max/min = %d/%d)",
                counts.max() / counts.min(), counts.max(), counts.min())

    # Build sampling_strategy dict:
    # only specify target for classes BELOW target_per_class
    sampling_strategy = {
        cls: target_per_class
        for cls, cnt in zip(unique, counts)
        if cnt < target_per_class
    }

    if not sampling_strategy:
        logger.info("All classes already have >= %d samples. Skipping SMOTE.", target_per_class)
        return X_train, y_train

    logger.info("SMOTE will oversample %d classes to %d samples each:",
                len(sampling_strategy), target_per_class)
    for cls, tgt in sampling_strategy.items():
        orig = counts[unique == cls][0]
        logger.info("  class %2d: %d → %d  (+%d synthetic samples)",
                    cls, orig, tgt, tgt - orig)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    unique_new, counts_new = np.unique(y_resampled, return_counts=True)
    logger.info("Class distribution AFTER SMOTE:")
    for cls, cnt in zip(unique_new, counts_new):
        bar = "█" * (cnt // 50)
        logger.info("  class %2d: %5d  %s", cls, cnt, bar)
    logger.info("  Total training samples: %d → %d  (+%d synthetic)",
                len(y_train), len(y_resampled), len(y_resampled) - len(y_train))

    return X_resampled.astype(np.float32), y_resampled.astype(np.int32)


# ---------------------------------------------------------------------------
# StandardScaler + PCA
# ---------------------------------------------------------------------------

def fit_scaler_pca(
    X_train: np.ndarray,
    n_components: int = config.PCA_N_COMPONENTS,
    whiten: bool = config.PCA_WHITEN,
) -> tuple[StandardScaler, PCA]:
    """
    Fit StandardScaler then PCA on training features.

    Why StandardScaler before PCA:
      PCA finds directions of maximum variance.
      If feature_42 has values [0.001–0.002] and feature_891 has [0–500],
      PCA will be dominated by feature_891 purely due to scale.
      StandardScaler makes all 1280 features have mean=0, std=1 first
      → PCA finds truly meaningful directions of variance ✅

    Args:
        X_train:      (N, 1280) training features (after SMOTE)
        n_components: number of PCA components to keep (default 256)
        whiten:       if True, scale each PC to unit variance

    Returns:
        scaler:  fitted StandardScaler
        pca:     fitted PCA
    """
    logger.info("Fitting StandardScaler on %d training samples...", len(X_train))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    logger.info("  Scaling complete. Feature means ≈ 0, stds ≈ 1 ✅")

    logger.info("Fitting PCA: %d → %d components...", X_train.shape[1], n_components)
    pca = PCA(n_components=n_components, whiten=whiten, random_state=config.RANDOM_STATE)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info("  PCA complete.")
    logger.info("  Components kept:        %d / %d", n_components, X_train.shape[1])
    logger.info("  Variance explained:     %.2f%%", explained)
    logger.info("  Variance discarded:     %.2f%%", 100 - explained)
    logger.info("  Compression ratio:      %.1fx  (%d → %d features)",
                X_train.shape[1] / n_components, X_train.shape[1], n_components)

    # Log how many components explain key variance thresholds
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n = int(np.searchsorted(cumvar, threshold)) + 1
        logger.info("  Components for %.0f%% variance: %d", threshold * 100, n)

    return scaler, pca


def apply_scaler_pca(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    split_name: str = "",
) -> np.ndarray:
    """
    Apply a pre-fitted scaler + PCA to any feature matrix.
    Used for val and test splits (only transform, never fit).

    Args:
        X:          (N, 1280) feature matrix
        scaler:     fitted StandardScaler from fit_scaler_pca()
        pca:        fitted PCA from fit_scaler_pca()
        split_name: name for logging (e.g. "val", "test")

    Returns:
        X_pca: (N, n_components) reduced feature matrix
    """
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled).astype(np.float32)
    logger.info("  %s: (%d, %d) → (%d, %d) after Scaler+PCA",
                split_name or "Data",
                X.shape[0], X.shape[1],
                X_pca.shape[0], X_pca.shape[1])
    return X_pca


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_preprocessors(scaler: StandardScaler, pca: PCA) -> None:
    """Save fitted scaler and PCA to disk for use in inference pipeline."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(pca,    config.PCA_PATH)
    logger.info("Scaler saved to %s", config.SCALER_PATH)
    logger.info("PCA    saved to %s", config.PCA_PATH)


def load_preprocessors() -> tuple[StandardScaler, PCA]:
    """Load fitted scaler and PCA from disk."""
    if not config.SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {config.SCALER_PATH}. Run scripts/train.py first."
        )
    if not config.PCA_PATH.exists():
        raise FileNotFoundError(
            f"PCA not found at {config.PCA_PATH}. Run scripts/train.py first."
        )
    scaler = joblib.load(config.SCALER_PATH)
    pca    = joblib.load(config.PCA_PATH)
    logger.info("Scaler loaded from %s", config.SCALER_PATH)
    logger.info("PCA    loaded from %s", config.PCA_PATH)
    return scaler, pca


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def run_preprocessing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None, PCA | None]:
    """
    Run the full SMOTE → (optional Scaler + PCA) pipeline.

    When config.USE_PCA is False (recommended for XGBoost on MobileNetV2
    features), only SMOTE is applied.  The scaler and pca are saved to disk
    only when USE_PCA is True so inference.py can apply the same transform.

    Args:
        X_train, y_train: raw training features and labels  (N, 1280)
        X_val:            raw validation features
        X_test:           raw test features

    Returns:
        X_train_out: balanced training features — always SMOTEd
                     shape (N_balanced, 1280) if USE_PCA=False
                     shape (N_balanced, n_components) if USE_PCA=True
        y_train_out: (N_balanced,) balanced training labels
        X_val_out:   val features (scaled+PCA only when USE_PCA=True)
        X_test_out:  test features (scaled+PCA only when USE_PCA=True)
        scaler:      fitted StandardScaler or None
        pca:         fitted PCA or None
    """
    logger.info("=" * 55)
    logger.info("STEP 1 — SMOTE Oversampling")
    logger.info("=" * 55)
    # Pass config values explicitly so they are read at call time, not at
    # module import time (important: default args are frozen at definition time).
    X_train_bal, y_train_bal = apply_smote(
        X_train, y_train,
        target_per_class=config.SMOTE_TARGET_PER_CLASS,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
    )

    if not config.USE_PCA:
        logger.info("=" * 55)
        logger.info("PCA DISABLED (config.USE_PCA=False)")
        logger.info("  XGBoost will train on raw %d-dim features.", X_train_bal.shape[1])
        logger.info("  Tip: XGBoost handles high-dim input via colsample_bytree.")
        logger.info("=" * 55)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("  Train: %s  (after SMOTE, no PCA)", X_train_bal.shape)
        logger.info("  Val:   %s", X_val.shape)
        logger.info("  Test:  %s", X_test.shape)
        logger.info("=" * 55)
        return X_train_bal, y_train_bal, X_val, X_test, None, None

    logger.info("=" * 55)
    logger.info("STEP 2 — StandardScaler + PCA  (fit on train only)")
    logger.info("=" * 55)
    scaler, pca = fit_scaler_pca(
        X_train_bal,
        n_components=config.PCA_N_COMPONENTS,
        whiten=config.PCA_WHITEN,
    )

    logger.info("=" * 55)
    logger.info("STEP 3 — Applying Scaler + PCA to all splits")
    logger.info("=" * 55)
    X_train_pca = apply_scaler_pca(X_train_bal, scaler, pca, "train")
    X_val_pca   = apply_scaler_pca(X_val,       scaler, pca, "val")
    X_test_pca  = apply_scaler_pca(X_test,      scaler, pca, "test")

    save_preprocessors(scaler, pca)

    logger.info("=" * 55)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("  Train: %s  (after SMOTE + PCA)", X_train_pca.shape)
    logger.info("  Val:   %s  (PCA only)", X_val_pca.shape)
    logger.info("  Test:  %s  (PCA only)", X_test_pca.shape)
    logger.info("=" * 55)

    return X_train_pca, y_train_bal, X_val_pca, X_test_pca, scaler, pca
