"""
Fine-tune MobileNetV2 top layers on plant disease images.

Unfreezes the top layers of the frozen MobileNetV2 backbone and trains them
on the PlantVillage dataset with a small learning rate.  After fine-tuning,
re-extracts features from the updated backbone and saves new .npz caches so
scripts/train.py can train XGBoost on the improved features.

Usage:
  python scripts/finetune.py
  python scripts/finetune.py --unfreeze-from 120 --epochs 15 --lr 5e-5
  python scripts/finetune.py --unfreeze-from 100 --epochs 10 --lr 1e-4

Output:
  outputs/features/train_features_finetuned.npz
  outputs/features/val_features_finetuned.npz
  outputs/features/test_features_finetuned.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf

import config
from src.data_pipeline import (
    build_augmentation_pipeline,
    build_class_index,
    build_preprocessing_pipeline,
    image_generator,
    num_batches,
    scan_dataset,
    stratified_split,
)
from src.feature_extractor import extract_features, save_features
from src.utils import get_logger, setup_output_dirs

logger = get_logger("finetune")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 top layers on plant disease images."
    )
    p.add_argument(
        "--unfreeze-from", type=int, default=100,
        help="Unfreeze MobileNetV2 layers from this index onward (default: 100).",
    )
    p.add_argument(
        "--epochs", type=int, default=10,
        help="Maximum number of fine-tuning epochs (default: 10).",
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate — keep small to avoid catastrophic forgetting (default: 1e-4).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for fine-tuning (default: 32).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def build_tf_dataset(
    file_list: list,
    labels: np.ndarray,
    batch_size: int,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset of (image_tensor, label) pairs.

    Preprocessing (preprocess_input scaling to [-1,1]) is applied inside the
    fine-tune model so the same model can be used for inference without a
    separate preprocessing step.

    Args:
        file_list:  List of image file paths.
        labels:     Integer label array aligned with file_list.
        batch_size: Number of samples per batch.
        augment:    If True, apply random augmentation (training split only).

    Returns:
        Batched, prefetched tf.data.Dataset.
    """
    aug_layers = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(config.AUG_ROTATION_FACTOR),
        tf.keras.layers.RandomZoom(config.AUG_ZOOM_FACTOR),
        tf.keras.layers.RandomBrightness(config.AUG_BRIGHTNESS_FACTOR),
        tf.keras.layers.RandomContrast(config.AUG_CONTRAST_FACTOR),
    ])

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config.INPUT_SIZE)
        img = tf.cast(img, tf.float32)
        if augment:
            img = aug_layers(img, training=True)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_list, labels.astype(np.int32)))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.shuffle(buffer_size=2000, seed=config.RANDOM_STATE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Fine-tune model
# ---------------------------------------------------------------------------

def build_finetune_model(
    unfreeze_from: int,
    num_classes: int,
    lr: float,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build a MobileNetV2 model with a classification head for fine-tuning.

    Bottom layers (indices 0 to unfreeze_from-1) stay frozen — they hold
    universal low-level features (edges, colours, textures) that are already
    correct for any image domain.

    Top layers (unfreeze_from onward) are unfrozen and retrained on plant
    disease images with a very small learning rate.

    Args:
        unfreeze_from: Layer index from which to unfreeze (default 100).
        num_classes:   Number of output classes (15 for this dataset).
        lr:            Learning rate — keep ≤1e-4 to avoid catastrophic forgetting.

    Returns:
        ft_model:  Full model (base + head) for fine-tuning.
        base:      MobileNetV2 base (used to extract features after fine-tuning).
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*config.INPUT_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze bottom layers, unfreeze top layers
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False

    n_trainable  = sum(1 for l in base.layers if l.trainable)
    n_frozen     = len(base.layers) - n_trainable
    logger.info(
        "MobileNetV2 layers — frozen: %d  trainable: %d / %d  (unfreeze_from=%d)",
        n_frozen, n_trainable, len(base.layers), unfreeze_from,
    )

    # Build model: raw pixels → preprocess → base → GAP → Dense head
    inputs  = tf.keras.Input(shape=(*config.INPUT_SIZE, 3))
    x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x       = base(x, training=False)     # training=False → BN uses stored statistics
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    ft_model = tf.keras.Model(inputs, outputs, name="mobilenetv2_finetune")
    ft_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return ft_model, base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_output_dirs()

    # ── Scan and split dataset ─────────────────────────────────────────────
    logger.info("Scanning dataset at %s ...", config.DATA_DIR)
    class_index = build_class_index(config.DATA_DIR)
    df          = scan_dataset(config.DATA_DIR, class_index)
    train_df, val_df, test_df = stratified_split(df)

    logger.info(
        "Split — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── tf.data datasets ───────────────────────────────────────────────────
    train_ds = build_tf_dataset(
        train_df["filepath"].tolist(), train_df["label_int"].values,
        batch_size=args.batch_size, augment=True,
    )
    val_ds = build_tf_dataset(
        val_df["filepath"].tolist(), val_df["label_int"].values,
        batch_size=args.batch_size, augment=False,
    )

    # ── Build and fine-tune ────────────────────────────────────────────────
    ft_model, base = build_finetune_model(
        unfreeze_from=args.unfreeze_from,
        num_classes=config.NUM_CLASSES,
        lr=args.lr,
    )
    ft_model.summary(line_length=90)

    callbacks = [
        # Stop if val_accuracy stops improving for 3 epochs; restore best weights
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        # Halve learning rate if val_loss plateaus for 2 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Fine-tuning for up to %d epochs  (lr=%.0e, batch_size=%d, unfreeze_from=%d)...",
        args.epochs, args.lr, args.batch_size, args.unfreeze_from,
    )
    history = ft_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history["val_accuracy"])
    logger.info("Best validation accuracy during fine-tuning: %.4f", best_val_acc)

    # ── Extract improved features using the fine-tuned base ────────────────
    # Strip the Dense head — keep input → base → GAP as the feature extractor
    feature_extractor = tf.keras.Model(
        inputs=ft_model.input,
        outputs=ft_model.get_layer("global_average_pooling2d").output,
        name="finetuned_feature_extractor",
    )
    logger.info(
        "Feature extractor output shape: %s", feature_extractor.output_shape
    )

    aug_fn = build_augmentation_pipeline()
    pre_fn = build_preprocessing_pipeline()

    splits = [
        (train_df, config.FEATURES_DIR / "train_features_finetuned.npz", aug_fn, "Train"),
        (val_df,   config.FEATURES_DIR / "val_features_finetuned.npz",   pre_fn, "Val"),
        (test_df,  config.FEATURES_DIR / "test_features_finetuned.npz",  pre_fn, "Test"),
    ]

    logger.info("Re-extracting features with fine-tuned model...")
    for split_df, out_path, preprocess_fn, desc in splits:
        gen = image_generator(
            split_df["filepath"].tolist(),
            split_df["label_int"].values,
            batch_size=config.BATCH_SIZE,
            preprocess_fn=preprocess_fn,
        )
        X, y = extract_features(
            feature_extractor,
            gen,
            num_batches(len(split_df)),
            desc=desc,
        )
        save_features(X, y, out_path)
        logger.info("  %s features → %s  shape=%s", desc, out_path.name, X.shape)

    logger.info("=" * 60)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("New feature caches saved to %s", config.FEATURES_DIR)
    logger.info("To train XGBoost on fine-tuned features, temporarily update")
    logger.info("config.py TRAIN/VAL/TEST_FEATURES_PATH to *_finetuned.npz,")
    logger.info("then run:  python scripts/train.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
