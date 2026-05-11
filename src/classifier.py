"""
XGBoost classifier: baseline training, Optuna hyperparameter tuning,
and model persistence.

Multi-class strategy: XGBoost natively supports 15-class classification
(Pepper×2, Potato×3, Tomato×10) with objective="multi:softprob" — no
one-vs-rest decomposition needed.
predict_proba() returns a (N, 15) probability matrix enabling top-k output.

Model serialisation: XGBoost's native JSON format (save_model / load_model)
is version-stable across XGBoost releases.  Pickle ties the model to a
specific XGBoost version and Python environment.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
from src.utils import get_logger

logger = get_logger(__name__)

# Silence Optuna's default noisy internal logs — we use our own callback instead
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
) -> xgb.XGBClassifier:
    """
    Train an XGBClassifier with the base hyperparameters and early stopping
    on validation mlogloss.

    Args:
        X_train, y_train: Training features (N, 1280) and integer labels.
        X_val, y_val:     Validation features for early stopping.
        params:           Override XGB_BASE_PARAMS (optional).

    Returns:
        Fitted XGBClassifier.
    """
    if params is None:
        params = config.XGB_BASE_PARAMS.copy()

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    logger.info("Baseline val macro F1: %.4f", val_f1)
    return model


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def make_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = config.OPTUNA_CV_FOLDS,
) -> Callable[[optuna.Trial], float]:
    """
    Return an Optuna objective function (closure) that performs stratified
    cross-validation and returns mean macro F1.

    Search space rationale:
    - colsample_bytree [0.5, 1.0]: MobileNetV2 GAP features are correlated
      (from the same conv channels); column subsampling reduces variance.
    - reg_lambda / reg_alpha: L2/L1 regularisation is important with 1280
      potentially correlated features.
    - max_depth capped at 8: deeper trees overfit with 1280 features.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    space = config.OPTUNA_SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": config.NUM_CLASSES,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": -1,
            "random_state": config.RANDOM_STATE,
            "n_estimators": trial.suggest_int(
                "n_estimators", space["n_estimators"]["low"], space["n_estimators"]["high"]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                space["learning_rate"]["low"],
                space["learning_rate"]["high"],
                log=space["learning_rate"].get("log", False),
            ),
            "max_depth": trial.suggest_int(
                "max_depth", space["max_depth"]["low"], space["max_depth"]["high"]
            ),
            "subsample": trial.suggest_float(
                "subsample", space["subsample"]["low"], space["subsample"]["high"]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                space["colsample_bytree"]["low"],
                space["colsample_bytree"]["high"],
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                space["min_child_weight"]["low"],
                space["min_child_weight"]["high"],
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", space["reg_lambda"]["low"], space["reg_lambda"]["high"]
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", space["reg_alpha"]["low"], space["reg_alpha"]["high"]
            ),
        }

        # Log the hyperparameters being tried for this trial
        logger.info(
            "Trial #%d starting | n_est=%d  lr=%.4f  depth=%d  "
            "subsample=%.2f  col_bt=%.2f  mcw=%d  λ=%.2f  α=%.2f",
            trial.number,
            params["n_estimators"],
            params["learning_rate"],
            params["max_depth"],
            params["subsample"],
            params["colsample_bytree"],
            params["min_child_weight"],
            params["reg_lambda"],
            params["reg_alpha"],
        )

        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf,
            scoring="f1_macro",
            n_jobs=1,           # XGBoost already uses all cores internally
        )
        mean_f1 = float(scores.mean())
        std_f1  = float(scores.std())

        # Log the result immediately after CV completes
        logger.info(
            "Trial #%d result   | macro F1 = %.4f ± %.4f  (%.2f%%)",
            trial.number, mean_f1, std_f1, mean_f1 * 100,
        )
        return mean_f1

    return objective


def run_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = config.OPTUNA_N_TRIALS,
) -> optuna.Study:
    """
    Run Optuna hyperparameter search over n_trials.

    Uses TPE sampler (default) with MedianPruner (warmup=10 trials).
    Returns the completed study so the caller can inspect plots and params.
    """
    # MedianPruner warmup capped at half of n_trials so pruning
    # still activates even when n_trials is small (e.g. 10-15)
    warmup = max(3, n_trials // 3)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=warmup)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    objective = make_optuna_objective(X_train, y_train)

    def _log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Print a best-so-far summary line after every completed trial."""
        best = study.best_trial
        logger.info(
            "─── After trial #%d/%d | Best so far: Trial #%d → %.4f (%.2f%%) ───",
            trial.number, n_trials - 1,
            best.number, best.value, best.value * 100,
        )

    logger.info("=" * 60)
    logger.info("Optuna search: %d trials × %d-fold CV = %d XGBoost runs",
                n_trials, config.OPTUNA_CV_FOLDS, n_trials * config.OPTUNA_CV_FOLDS)
    logger.info("Hyperparameter search space:")
    for k, v in config.OPTUNA_SEARCH_SPACE.items():
        logger.info("  %-20s [%s, %s]", k, v["low"], v["high"])
    logger.info("=" * 60)

    study.optimize(objective, n_trials=n_trials,
                   callbacks=[_log_callback], show_progress_bar=False)

    best = study.best_trial
    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("Best trial  : #%d", best.number)
    logger.info("Best macro F1: %.4f (%.2f%%)", best.value, best.value * 100)
    logger.info("Best hyperparameters:")
    for k, v in best.params.items():
        logger.info("  %-20s %s", k, v)
    logger.info("=" * 60)
    return study


def retrain_best(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict,
) -> xgb.XGBClassifier:
    """
    Retrain on the full training set using the best hyperparameters found
    by Optuna.  Validation set is used only for evaluation, not early stopping.
    """
    params = {
        "objective": "multi:softprob",
        "num_class": config.NUM_CLASSES,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": -1,
        "random_state": config.RANDOM_STATE,
        **best_params,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    logger.info("Tuned model val macro F1: %.4f", val_f1)
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: xgb.XGBClassifier, path: Path) -> None:
    """Save using XGBoost's native JSON format (version-stable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("XGBoost model saved to %s", path)


def load_model(path: Path) -> xgb.XGBClassifier:
    """Load an XGBClassifier from XGBoost JSON format."""
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    logger.info("XGBoost model loaded from %s", path)
    return model
