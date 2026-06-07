"""
CLI: End-to-end training pipeline.

Loads cached features → trains XGBoost baseline → optionally runs Optuna
tuning → evaluates on test set → saves model and all output artefacts.

XGBoost's inverse-frequency sample weights handle the 21× class imbalance
directly — rare classes receive proportionally stronger gradient signal.

Usage:
  python scripts/train.py                  # Full run with Optuna tuning
  python scripts/train.py --skip-tuning   # Baseline only (fast)
  python scripts/train.py --n-trials 20   # Override number of Optuna trials
  python scripts/train.py --subsample 0.1 # Use 10% of cache (quick smoke-test)

Requires feature caches to exist. Run scripts/finetune.py first.
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
    plot_confusion_matrix_plotly, #added for interactive confusion matrix
    plot_per_class_f1_plotly, #added for interactive per-class F1 plot
)
from src.feature_extractor import load_features
from src.utils import get_logger, load_label_encoder, setup_output_dirs, update_config_xgb_params

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
    return p.parse_args()


def _check_caches() -> None:
    missing = [
        p for p in (config.TRAIN_FEATURES_PATH, config.VAL_FEATURES_PATH, config.TEST_FEATURES_PATH)
        if not p.exists()
    ]
    if missing:
        logger.error(
            "Missing feature caches:\n%s\nRun scripts/finetune.py first.",
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

    # ---- Load label encoder -------------------------------------------------
    if not config.LABEL_ENCODER_PATH.exists():
        logger.error(
            "Label encoder not found at %s. Run scripts/finetune.py first.",
            config.LABEL_ENCODER_PATH,
        )
        sys.exit(1)
    le = load_label_encoder(config.LABEL_ENCODER_PATH)
    label_names = list(le.classes_)
    logger.info("Classes: %d labels loaded", len(label_names))

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

        # Persist best params back into config.py so the next run uses them as baseline
        updated = update_config_xgb_params(study.best_params)
        logger.info(
            "config.py XGB_BASE_PARAMS updated with Optuna best — %s",
            ", ".join(f"{k}={v}" for k, v in updated.items()),
        )

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

       # NEW: Your interactive hoverable dashboards
    plot_confusion_matrix_plotly(
        y_test, y_pred, label_names,
        config.PLOTS_DIR / "confusion_matrix.png",
    )
    plot_per_class_f1_plotly(
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
