"""
Feature extraction using a frozen MobileNetV2 backbone.

Architecture decision — GlobalAveragePooling2D (GAP) vs. Flatten:
  MobileNetV2 at 128×128 input produces a 4×4×1280 final conv block.
  Flatten → 20,480 features → ~4.2 GB RAM for 54k samples (unusable on CPU).
  GAP     →  1,280 features →  ~276 MB RAM for 54k samples (fits comfortably).
  GAP is the standard MobileNetV2 transfer-learning feature vector and loses
  minimal information relative to the spatial flatten.

Feature extraction is run ONCE and cached as compressed .npz files.
All subsequent training / tuning runs load the cache (seconds vs. 15 min).
"""

from pathlib import Path
from typing import Generator

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Build the frozen feature model
# ---------------------------------------------------------------------------

def build_feature_model() -> tf.keras.Model:
    """
    Build a frozen MobileNetV2 + GlobalAveragePooling2D feature extractor.

    The base MobileNetV2 is loaded with ImageNet weights and ALL layers
    (including BatchNorm statistics) are frozen.  The model has 0 trainable
    parameters.

    Returns:
        model — input: (B, 128, 128, 3) in [-1, 1]
                output: (B, 1280) feature vectors
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*config.INPUT_SIZE, config.NUM_CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze ALL weights including BatchNorm γ/β

    gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=gap, name="mobilenetv2_feature_extractor")

    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    total = sum(tf.size(v).numpy() for v in model.variables)
    logger.info(
        "Feature model built — trainable params: %d / %d (should be 0 / %d)",
        trainable, total, total,
    )
    assert trainable == 0, "Feature model must have 0 trainable parameters"

    return model


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model: tf.keras.Model,
    generator: Generator,
    n_batches: int,
    desc: str = "Extracting",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all batches from generator through model and collect (X, y).

    Uses model.predict_on_batch() which avoids TF graph recompilation
    overhead vs. calling model(batch) inside a Python loop.

    Args:
        model:      The frozen feature extractor (output shape (B, 1280)).
        generator:  Yields (batch_images_np, batch_labels_np).
        n_batches:  Total number of batches (for the progress bar).
        desc:       Progress bar description string.

    Returns:
        X — float32 array of shape (N, 1280)
        y — int array of shape (N,)
    """
    all_features = []
    all_labels = []

    for batch_images, batch_labels in tqdm(generator, total=n_batches, desc=desc):
        feats = model.predict_on_batch(batch_images)   # (B, 1280)
        all_features.append(feats)
        all_labels.append(batch_labels)

    X = np.vstack(all_features).astype(np.float32)    # (N, 1280)
    y = np.concatenate(all_labels).astype(np.int32)   # (N,)

    logger.info("%s complete — X: %s, y: %s", desc, X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_features(X: np.ndarray, y: np.ndarray, filepath: Path) -> None:
    """Save (X, y) as a compressed .npz file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, X=X, y=y)
    size_mb = filepath.stat().st_size / 1e6
    logger.info("Features saved to %s (%.1f MB)", filepath, size_mb)


def load_features(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from a compressed .npz file."""
    data = np.load(filepath)
    X, y = data["X"], data["y"]
    logger.info("Features loaded from %s — X: %s, y: %s", filepath, X.shape, y.shape)
    return X, y


def features_cached(filepath: Path) -> bool:
    """Return True if the .npz cache exists."""
    return filepath.exists()
