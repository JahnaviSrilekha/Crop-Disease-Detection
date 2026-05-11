"""
CLI: Extract MobileNetV2 features for all three splits and cache to .npz.

This is the most time-consuming step (~15 min on CPU for 54k images).
After the first run, cached files are reused by scripts/train.py automatically.

Usage:
  python scripts/extract_features.py           # Skip if cache exists
  python scripts/extract_features.py --force   # Re-extract even if cached
  python scripts/extract_features.py --subsample 0.1  # Use 10% of data (for testing)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
from src.feature_extractor import (
    build_feature_model,
    extract_features,
    features_cached,
    save_features,
)
from src.utils import build_label_encoder, get_logger, save_label_encoder, setup_output_dirs

logger = get_logger("extract_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract MobileNetV2 features for PlantVillage.")
    p.add_argument("--force", action="store_true", help="Re-extract even if .npz cache exists.")
    p.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0 < x <= 1.0). Default: 1.0 (full dataset).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_output_dirs()

    # Check cache
    all_cached = all(
        features_cached(p)
        for p in (config.TRAIN_FEATURES_PATH, config.VAL_FEATURES_PATH, config.TEST_FEATURES_PATH)
    )
    if all_cached and not args.force:
        logger.info(
            "All feature caches already exist. Use --force to re-extract.\n"
            "  Train: %s\n  Val:   %s\n  Test:  %s",
            config.TRAIN_FEATURES_PATH, config.VAL_FEATURES_PATH, config.TEST_FEATURES_PATH,
        )
        return

    # Scan dataset
    logger.info("Scanning dataset at %s ...", config.DATA_DIR)
    class_index = build_class_index(config.DATA_DIR)
    df = scan_dataset(config.DATA_DIR, class_index)

    # Optional subsampling for quick tests
    if args.subsample < 1.0:
        df = df.sample(frac=args.subsample, random_state=config.RANDOM_STATE).reset_index(drop=True)
        logger.info("Subsampled to %d images (%.0f%% of full dataset)", len(df), args.subsample * 100)

    # Stratified split
    train_df, val_df, test_df = stratified_split(df)

    # Save label encoder alongside features so train.py can pick it up
    label_names = sorted(class_index.keys())
    le = build_label_encoder(label_names)
    save_label_encoder(le, config.LABEL_ENCODER_PATH)
    logger.info("Label encoder saved to %s", config.LABEL_ENCODER_PATH)

    # Build frozen feature model
    logger.info("Building frozen MobileNetV2 feature extractor...")
    model = build_feature_model()

    aug_fn = build_augmentation_pipeline()
    pre_fn = build_preprocessing_pipeline()

    # ---- Training split (with augmentation) --------------------------------
    logger.info("Extracting TRAIN features (%d images)...", len(train_df))
    train_gen = image_generator(
        train_df["filepath"].tolist(),
        train_df["label_int"].values,
        batch_size=config.BATCH_SIZE,
        preprocess_fn=aug_fn,
    )
    X_train, y_train = extract_features(
        model, train_gen, num_batches(len(train_df)), desc="Train"
    )
    save_features(X_train, y_train, config.TRAIN_FEATURES_PATH)

    # ---- Validation split --------------------------------------------------
    logger.info("Extracting VAL features (%d images)...", len(val_df))
    val_gen = image_generator(
        val_df["filepath"].tolist(),
        val_df["label_int"].values,
        batch_size=config.BATCH_SIZE,
        preprocess_fn=pre_fn,
    )
    X_val, y_val = extract_features(
        model, val_gen, num_batches(len(val_df)), desc="Val"
    )
    save_features(X_val, y_val, config.VAL_FEATURES_PATH)

    # ---- Test split --------------------------------------------------------
    logger.info("Extracting TEST features (%d images)...", len(test_df))
    test_gen = image_generator(
        test_df["filepath"].tolist(),
        test_df["label_int"].values,
        batch_size=config.BATCH_SIZE,
        preprocess_fn=pre_fn,
    )
    X_test, y_test = extract_features(
        model, test_gen, num_batches(len(test_df)), desc="Test"
    )
    save_features(X_test, y_test, config.TEST_FEATURES_PATH)

    logger.info("Feature extraction complete. All .npz caches written.")


if __name__ == "__main__":
    main()
