"""
Data pipeline: scan PlantVillage dataset, build stratified splits,
apply augmentation, and yield image batches for feature extraction.

PIL is used for image loading instead of OpenCV because PIL automatically
handles EXIF orientation (critical for field phone photos) and reads RGB
directly (no BGR channel-swap needed).
"""

import math
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config
from src.utils import get_logger

logger = get_logger(__name__)

# Image file extensions accepted from PlantVillage
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def build_class_index(data_dir: Path) -> dict[str, int]:
    """
    Walk data_dir and return a sorted {class_name: int} mapping.
    Sorting guarantees the same integer assignment every run.
    """
    class_names = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir()
    )
    if not class_names:
        raise FileNotFoundError(
            f"No subdirectories found in {data_dir}. "
            "Did you download and unzip the PlantVillage dataset?"
        )
    return {name: idx for idx, name in enumerate(class_names)}


def scan_dataset(data_dir: Path, class_index: dict[str, int]) -> pd.DataFrame:
    """
    Collect all image paths under data_dir and return a DataFrame with
    columns [filepath, label_str, label_int].

    Logs per-class counts so class imbalance is visible immediately.
    """
    rows = []
    for class_name, label_int in class_index.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            logger.warning("Class directory not found: %s", class_dir)
            continue
        for p in class_dir.iterdir():
            if p.suffix in _IMG_EXTENSIONS:
                rows.append(
                    {"filepath": str(p), "label_str": class_name, "label_int": label_int}
                )

    df = pd.DataFrame(rows)
    logger.info("Dataset scanned: %d images across %d classes", len(df), df["label_int"].nunique())

    counts = df.groupby("label_str").size().sort_values()
    logger.info("Class size range: min=%d, max=%d", counts.min(), counts.max())
    return df


# ---------------------------------------------------------------------------
# Stratified train / val / test split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    train: float = config.TRAIN_SPLIT,
    val: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into (train_df, val_df, test_df) with stratification on label_int.
    This guarantees proportional class representation even for small classes.
    """
    test_frac = 1.0 - train
    train_df, temp_df = train_test_split(
        df, test_size=test_frac, stratify=df["label_int"], random_state=seed
    )
    # val_frac of the remaining data equals the target val proportion
    val_frac = val / test_frac
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_frac), stratify=temp_df["label_int"], random_state=seed
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df)
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Augmentation pipeline (training split only)
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(target_size: tuple[int, int] = config.INPUT_SIZE) -> tf.keras.Sequential:
    """
    Return a tf.keras.Sequential that resizes, augments, and applies
    MobileNetV2 preprocessing (scales pixels to [-1, 1]).

    Applied ONLY to the training split during feature extraction.
    Val/test use the passthrough pipeline (resize + preprocess only).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(*target_size),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(config.AUG_ROTATION_FACTOR),
            tf.keras.layers.RandomZoom(config.AUG_ZOOM_FACTOR),
            tf.keras.layers.RandomBrightness(config.AUG_BRIGHTNESS_FACTOR),
            tf.keras.layers.RandomContrast(config.AUG_CONTRAST_FACTOR),
            # preprocess_input first (scales 0-255 → [-1, 1]),
            # then GaussianNoise with stddev in [0,1] as Keras 3 requires
            tf.keras.layers.Lambda(
                tf.keras.applications.mobilenet_v2.preprocess_input
            ),
            tf.keras.layers.GaussianNoise(config.AUG_NOISE_STDDEV),
        ],
        name="augmentation_train",
    )


def build_preprocessing_pipeline(target_size: tuple[int, int] = config.INPUT_SIZE) -> tf.keras.Sequential:
    """
    Resize + MobileNetV2 preprocess only — for val and test splits.
    No random augmentation.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(*target_size),
            tf.keras.layers.Lambda(
                tf.keras.applications.mobilenet_v2.preprocess_input
            ),
        ],
        name="preprocessing_eval",
    )


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def image_generator(
    file_list: list[str],
    labels: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    preprocess_fn: Optional[tf.keras.Sequential] = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield (batch_images, batch_labels) tuples.

    Images are loaded with PIL (RGB, EXIF-aware) and converted to float32
    numpy arrays before being passed through preprocess_fn.

    Args:
        file_list:      List of absolute image paths (strings).
        labels:         Integer label array aligned with file_list.
        batch_size:     Number of images per batch.
        preprocess_fn:  A tf.keras.Sequential (augmentation or passthrough).
                        If None, raw resized float32 arrays are returned.
    """
    n = len(file_list)
    target_h, target_w = config.INPUT_SIZE

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_files = file_list[start:end]
        batch_labels = labels[start:end]

        batch_images = []
        for fpath in batch_files:
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize((target_w, target_h), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32)
                batch_images.append(arr)
            except Exception as exc:
                logger.warning("Skipping corrupt image %s: %s", fpath, exc)
                # Replace with a zero image to keep batch shapes consistent
                batch_images.append(np.zeros((target_h, target_w, 3), dtype=np.float32))

        batch_np = np.stack(batch_images, axis=0)  # (B, H, W, 3)

        if preprocess_fn is not None:
            batch_np = preprocess_fn(batch_np, training=True).numpy()

        yield batch_np, batch_labels


def num_batches(n_samples: int, batch_size: int = config.BATCH_SIZE) -> int:
    """Return the number of batches for n_samples images."""
    return math.ceil(n_samples / batch_size)
