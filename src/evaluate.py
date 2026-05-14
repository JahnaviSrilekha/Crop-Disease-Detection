"""
Model evaluation: metrics, confusion matrix, per-class F1 chart,
and classification report persistence.

Metric choices:
- Macro F1   — headline metric; every disease class counts equally regardless
               of how many training images it has.
- Weighted F1 — secondary; reflects impact on the most common crop/disease pairs.
- Top-3 accuracy — practical for farmer UI: "here are the 3 most likely diseases".
- Per-class breakdown — identifies the hardest classes for targeted improvement.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Compute accuracy, macro/weighted F1, top-3 accuracy, and per-class stats.

    Returns a dict with keys:
        accuracy, macro_f1, weighted_f1, top3_accuracy, per_class (dict)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)   # (N, 15) — Pepper×2, Potato×3, Tomato×10

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    top3_accuracy = _top_k_accuracy(y_test, y_proba, k=3)

    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )

    logger.info("Test accuracy:    %.4f", accuracy)
    logger.info("Test macro F1:    %.4f", macro_f1)
    logger.info("Test weighted F1: %.4f", weighted_f1)
    logger.info("Test top-3 acc:   %.4f", top3_accuracy)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_accuracy": top3_accuracy,
        "per_class": report_dict,
    }


def _top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Fraction of samples where the true class is in the top-k predictions."""
    top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]   # (N, k) — highest k
    correct = sum(
        y_true[i] in top_k_indices[i] for i in range(len(y_true))
    )
    return correct / len(y_true)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_path: Path,
) -> None:
    """
    Save a 15×15 confusion matrix heatmap (Pepper×2, Potato×3, Tomato×10).
    Figure is sized 12×10 inches — readable at 15 classes with annotations.
    """
    cm = confusion_matrix(y_test, y_pred)
    short_names = [_shorten_label(n) for n in label_names]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,         # 15×15 is small enough to show counts in each cell
        fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Crop Disease Classifier (15 classes)", fontsize=14)
    plt.xticks(rotation=45, fontsize=9, ha="right")
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


# ---------------------------------------------------------------------------
# Per-class F1 bar chart
# ---------------------------------------------------------------------------

def plot_per_class_f1(
    report_dict: dict,
    label_names: list[str],
    output_path: Path,
) -> None:
    """
    Horizontal bar chart of per-class F1 scores, sorted ascending
    (worst-performing classes at the top — immediately actionable).
    """
    f1_scores = {
        name: report_dict[name]["f1-score"]
        for name in label_names
        if name in report_dict
    }
    sorted_items = sorted(f1_scores.items(), key=lambda x: x[1])   # ascending
    names = [_shorten_label(k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Colour bars by crop type for quick visual grouping
    colors = [_crop_color(_extract_crop(k)) for k, _ in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(8, len(names) * 0.28)))
    bars = ax.barh(names, values, color=colors)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.9, color="red", linestyle="--", linewidth=0.8, label="F1=0.90")
    ax.set_xlabel("F1 Score (macro)")
    ax.set_title("Per-Class F1 Score — Worst to Best")
    ax.legend(loc="lower right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-class F1 chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def save_classification_report(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    output_path: Path,
) -> str:
    """
    Generate and save a full sklearn classification report as a text file.
    Returns the report string.
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Classification report saved to %s", output_path)
    return report


# ---------------------------------------------------------------------------
# Optuna visualisation
# ---------------------------------------------------------------------------

def plot_optuna_history(study, output_path: Path) -> None:
    """Save Optuna optimisation history (best value per trial)."""
    try:
        import optuna.visualization.matplotlib as ovm
        # plot_optimization_history returns an Axes object, not a Figure.
        # Retrieve the parent Figure via ax.figure before calling savefig.
        ax = ovm.plot_optimization_history(study)
        fig = ax.figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Optuna history plot saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not save Optuna plot: %s", exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shorten_label(name: str) -> str:
    """Shorten a PlantVillage class name for axis labels."""
    return name.replace("___", "_").replace("_", " ")


def _extract_crop(name: str) -> str:
    """Extract the crop name (first token before any underscore sequence)."""
    # Handles: Tomato_..., Potato___..., Pepper__bell___...
    import re
    return re.split(r"_+", name)[0]


# Only the 3 crops that exist in the emmarex/plantdisease dataset
_CROP_COLORS = {
    "Tomato": "#d62728",    # red
    "Potato": "#8c564b",    # brown
    "Pepper": "#ff7f0e",    # orange
}


def _crop_color(crop: str) -> str:
    return _CROP_COLORS.get(crop, "#1f77b4")
