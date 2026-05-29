"""
Shared utilities: logging, output directory setup, label encoding,
and PlantVillage class-name parsing.
"""

import logging
import re
import sys
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder

import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent timestamp format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

def setup_output_dirs() -> None:
    """Create all outputs/ subdirectories if they don't exist."""
    for d in (
        config.FEATURES_DIR,
        config.MODELS_DIR,
        config.PLOTS_DIR,
        config.REPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def build_label_encoder(class_names: list[str]) -> LabelEncoder:
    """Fit and return a LabelEncoder on a sorted list of class names."""
    le = LabelEncoder()
    le.fit(sorted(class_names))
    return le


def save_label_encoder(le: LabelEncoder, path: Path) -> None:
    joblib.dump(le, path)


def load_label_encoder(path: Path) -> LabelEncoder:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# PlantVillage class-name parsing
# ---------------------------------------------------------------------------

# Your actual PlantVillage folder names use INCONSISTENT separators:
#
#   Pepper__bell___Bacterial_spot   → double then triple underscore
#   Pepper__bell___healthy          → double then triple underscore
#   Potato___Early_blight           → triple underscore
#   Potato___Late_blight            → triple underscore
#   Potato___healthy                → triple underscore
#   Tomato_Bacterial_spot           → single underscore
#   Tomato_Early_blight             → single underscore
#   Tomato__Target_Spot             → double underscore
#   Tomato__Tomato_YellowLeaf__Curl_Virus → double underscore
#   Tomato__Tomato_mosaic_virus     → double underscore
#   Tomato_healthy                  → single underscore
#
# Strategy: split on the LONGEST run of underscores to find crop/disease boundary

_MULTI_UNDERSCORE = re.compile(r"_{2,}")   # matches __ or ___ or ____


def parse_class_name(raw_name: str) -> tuple[str, str]:
    """
    Parse an actual PlantVillage folder name into (crop, disease) strings.

    Handles all separator variants found in the dataset:
      Pepper__bell___Bacterial_spot  → ("Pepper bell", "Bacterial spot")
      Potato___Early_blight          → ("Potato", "Early blight")
      Tomato_Bacterial_spot          → ("Tomato", "Bacterial spot")
      Tomato__Target_Spot            → ("Tomato", "Target spot")
      Tomato__Tomato_YellowLeaf__Curl_Virus → ("Tomato", "Tomato yellowleaf curl virus")

    Returns:
        crop    — e.g. "Tomato"
        disease — e.g. "Late blight"
    """
    if _MULTI_UNDERSCORE.search(raw_name):
        # Split on the first multi-underscore sequence
        parts = _MULTI_UNDERSCORE.split(raw_name, maxsplit=1)
        crop = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    else:
        # Single underscores only — first token is crop, rest is disease
        tokens = raw_name.split("_")
        crop = tokens[0]
        disease = " ".join(tokens[1:]) if len(tokens) > 1 else "Unknown"

    disease = disease.capitalize()
    return crop, disease


def friendly_label(raw_name: str) -> str:
    """Return a human-readable label, e.g. 'Tomato — Late blight'."""
    crop, disease = parse_class_name(raw_name)
    return f"{crop} — {disease}"


# ---------------------------------------------------------------------------
# Config auto-update after Optuna
# ---------------------------------------------------------------------------

# Params Optuna tunes that map directly into XGB_BASE_PARAMS.
# Integer params are written without a decimal point.
_XGB_INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight"}


def update_config_xgb_params(best_params: dict) -> dict:
    """
    Write Optuna's best_params back into XGB_BASE_PARAMS inside config.py
    so the next run starts from the best known values.

    Only the keys Optuna tuned are updated.  Fixed keys like objective,
    num_class, device, and random_state are never touched.

    Args:
        best_params: dict returned by ``study.best_params``

    Returns:
        Dict of {param: new_value_string} for the params that were updated.
    """
    config_path = Path(__file__).parent.parent / "config.py"
    text = config_path.read_text()

    updated = {}
    for key, value in best_params.items():
        # Format value: integers without decimal, floats to 4 sig. figures
        if key in _XGB_INT_PARAMS:
            new_val = str(int(round(value)))
        else:
            new_val = f"{float(value):.4f}"

        # Matches  "key": <old_number>  anywhere in the file
        pattern = rf'("{re.escape(key)}":\s*)([\d.e+\-]+)'
        new_text, n_replaced = re.subn(pattern, rf"\g<1>{new_val}", text)
        if n_replaced:
            updated[key] = new_val
            text = new_text

    config_path.write_text(text)
    return updated
