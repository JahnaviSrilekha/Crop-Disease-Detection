"""
CLI: End-to-end training pipeline.

Loads cached features → SMOTE oversampling → StandardScaler + PCA →
trains XGBoost baseline → optionally runs Optuna tuning → evaluates on
test set → saves model and all output artefacts.

Usage:
  python scripts/train.py                        # Full run with Optuna tuning
  python scripts/train.py --skip-tuning          # Baseline only (fast)
  python scripts/train.py --n-trials 20          # Fewer Optuna trials
  python scripts/train.py --subsample 0.1        # Use 10% feature cache (for testing)
  python scripts/train.py --skip-preprocessing   # Skip SMOTE+PCA (use raw 1280-dim features)

Requires feature caches to exist. Run scripts/extract_features.py first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.classifier import (
    load_model,
    retrain_best,
    run_tuning,
    save_model,
    train_baseline,
)
from src.evaluate import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_optuna_history,
    plot_per_class_f1,
    save_classification_report,
)
from src.feature_extractor import load_features
from src.preprocessor import run_preprocessing
from src.utils import get_logger, load_label_encoder, setup_output_dirs

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost crop disease classifier.")
    p.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna hyperparameter search; use baseline XGB params.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=config.OPTUNA_N_TRIALS,
        help=f"Number of Optuna trials (default: {config.OPTUNA_N_TRIALS}).",
    )
    p.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of cached features to use (0 < x <= 1.0). Default: 1.0.",
    )
    p.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip SMOTE + PCA; train XGBoost on raw 1280-dim features.",
    )
    return p.parse_args()


def _check_caches() -> None:
    missing = [
        p for p in (config.TRAIN_FEATURES_PATH, config.VAL_FEATURES_PATH, config.TEST_FEATURES_PATH)
        if not p.exists()
    ]
    if missing:
        logger.error(
            "Missing feature caches:\n%s\nRun scripts/extract_features.py first.",
            "\n".join(f"  {p}" for p in missing),
        )
        sys.exit(1)


def _subsample(X, y, frac: float):
    import numpy as np
    if frac >= 1.0:
        return X, y
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(X), size=int(len(X) * frac), replace=False)
    return X[idx], y[idx]


def main() -> None:
    args = parse_args()
    setup_output_dirs()
    _check_caches()

    # ---- Load features ------------------------------------------------------
    logger.info("Loading feature caches...")
    X_train, y_train = load_features(config.TRAIN_FEATURES_PATH)
    X_val, y_val = load_features(config.VAL_FEATURES_PATH)
    X_test, y_test = load_features(config.TEST_FEATURES_PATH)

    if args.subsample < 1.0:
        X_train, y_train = _subsample(X_train, y_train, args.subsample)
        X_val, y_val = _subsample(X_val, y_val, args.subsample)
        X_test, y_test = _subsample(X_test, y_test, args.subsample)
        logger.info(
            "Subsampled — train: %d, val: %d, test: %d",
            len(X_train), len(X_val), len(X_test),
        )

    # Load label encoder (written by extract_features.py)
    if not config.LABEL_ENCODER_PATH.exists():
        logger.error(
            "Label encoder not found at %s. Run scripts/extract_features.py first.",
            config.LABEL_ENCODER_PATH,
        )
        sys.exit(1)
    le = load_label_encoder(config.LABEL_ENCODER_PATH)
    label_names = list(le.classes_)
    logger.info("Classes: %d labels loaded", len(label_names))

    # ---- SMOTE + StandardScaler + PCA ---------------------------------------
    # NOTE: SMOTE and sample weights (in classifier.py) are REDUNDANT.
    # SMOTE balances classes → weights become ~1.0 → weights do nothing.
    # Skipping SMOTE lets the natural 21x imbalance stand so sample weights
    # are meaningful (rare class samples get up to 21x higher gradient signal).
    if args.skip_preprocessing:
        logger.info(
            "Skipping SMOTE + PCA (--skip-preprocessing). "
            "Sample weights in XGBoost will handle class imbalance directly."
        )
    else:
        logger.info("Running SMOTE oversampling + StandardScaler + PCA...")
        X_train, y_train, X_val, X_test, _scaler, _pca = run_preprocessing(
            X_train, y_train, X_val, X_test
        )
        logger.info(
            "Preprocessing complete — train: %s, val: %s, test: %s",
            X_train.shape, X_val.shape, X_test.shape,
        )

    # ---- Baseline training --------------------------------------------------
    logger.info("Training baseline XGBoost model...")
    baseline_model = train_baseline(X_train, y_train, X_val, y_val)

    if args.skip_tuning:
        final_model = baseline_model
        logger.info("Skipping Optuna tuning (--skip-tuning flag set).")
    else:
        # ---- Optuna tuning --------------------------------------------------
        logger.info("Running Optuna hyperparameter search (%d trials)...", args.n_trials)
        study = run_tuning(X_train, y_train, n_trials=args.n_trials)

        # Save Optuna history plot
        plot_optuna_history(study, config.PLOTS_DIR / "optuna_history.png")

        # Retrain with best params on full training set
        logger.info("Retraining with best Optuna params...")
        final_model = retrain_best(X_train, y_train, X_val, y_val, study.best_params)

    # ---- Save model ---------------------------------------------------------
    save_model(final_model, config.XGB_MODEL_PATH)

    # ---- Evaluate on test set -----------------------------------------------
    logger.info("Evaluating on test set...")
    metrics = compute_all_metrics(final_model, X_test, y_test, label_names)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:   {metrics['weighted_f1']:.4f}")
    print(f"  Top-3 Acc:     {metrics['top3_accuracy']:.4f}")
    print("=" * 50 + "\n")

    y_pred = final_model.predict(X_test)

    plot_confusion_matrix(
        y_test, y_pred, label_names,
        config.PLOTS_DIR / "confusion_matrix.png",
    )
    plot_per_class_f1(
        metrics["per_class"], label_names,
        config.PLOTS_DIR / "per_class_f1.png",
    )
    save_classification_report(
        final_model, X_test, y_test, label_names,
        config.REPORTS_DIR / "classification_report.txt",
    )

    logger.info("Training complete. All outputs saved to %s", config.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
