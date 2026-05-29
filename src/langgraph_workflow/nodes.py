"""
LangGraph nodes for the crop-disease inference workflow.

Node execution order:
  validate_image → run_prediction → [confidence_router]
                                       ≥ threshold → final_response
                                       < threshold → low_confidence_response
                                       error       → (graph ends via END edge)

Pipeline loading:
  The MobileNetV2 + XGBoost pipeline is heavy (~200 MB weights).  Loading it
  inside every node call would be extremely slow.  Instead, _pipeline is kept
  as a module-level singleton and initialised lazily on the first prediction.
  Subsequent calls reuse the same in-memory objects.
"""

from __future__ import annotations

from pathlib import Path

import config
from src.langgraph_workflow.state import PlantDiseaseState
from src.utils import get_logger

logger = get_logger("langgraph.nodes")

# Module-level singleton — loaded once, reused on every subsequent call.
_pipeline: dict | None = None
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _get_pipeline() -> dict:
    """Lazily load the inference pipeline on first call."""
    global _pipeline
    if _pipeline is None:
        from src.inference import load_inference_pipeline
        logger.info("Loading inference pipeline (first call — this takes ~10 s)...")
        _pipeline = load_inference_pipeline()
        logger.info("Inference pipeline ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Node 1 — validate_image
# ---------------------------------------------------------------------------

def validate_image(state: PlantDiseaseState) -> PlantDiseaseState:
    """
    Check that the image file exists and has an accepted extension.

    Sets state["status"] to:
      "valid" — file exists and extension is jpg/jpeg/png
      "error" — file missing or unsupported format
    """
    path = Path(state["image_path"])

    if not path.exists():
        logger.warning("Image not found: %s", path)
        return {
            **state,
            "status": "error",
            "message": f"Image not found: {path}\nPlease check the file path and try again.",
        }

    if path.suffix not in _VALID_EXTENSIONS:
        logger.warning("Unsupported image format: %s", path.suffix)
        return {
            **state,
            "status": "error",
            "message": (
                f"Unsupported format '{path.suffix}'. "
                "Please provide a JPG or PNG image."
            ),
        }

    logger.info("Image validated: %s", path.name)
    return {**state, "status": "valid", "message": None}


# ---------------------------------------------------------------------------
# Node 2 — run_prediction
# ---------------------------------------------------------------------------

def run_prediction(state: PlantDiseaseState) -> PlantDiseaseState:
    """
    Run the full MobileNetV2 + XGBoost pipeline on the validated image.

    Populates:
      predictions — top-3 list of {rank, raw_label, crop, disease, confidence}
      confidence  — top-1 confidence score
    """
    from src.inference import predict_disease as _predict

    try:
        pipeline = _get_pipeline()
        predictions = _predict(state["image_path"], pipeline, top_k=3)
        top_confidence = predictions[0]["confidence"]

        logger.info(
            "Prediction complete — top-1: %s — %s (%.1f%%)",
            predictions[0]["crop"],
            predictions[0]["disease"],
            top_confidence * 100,
        )
        return {
            **state,
            "predictions": predictions,
            "confidence": top_confidence,
        }

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return {
            **state,
            "status": "error",
            "predictions": None,
            "confidence": None,
            "message": f"Prediction failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Node 3a — final_response  (high-confidence path)
# ---------------------------------------------------------------------------

def final_response(state: PlantDiseaseState) -> PlantDiseaseState:
    """
    Build a clear, farmer-friendly result message for high-confidence predictions.

    Example output:
        Diagnosis: Tomato — Late blight  (87.3% confidence)

        Top 3 predictions:
          #1  Tomato — Late blight          87.3%
          #2  Tomato — Early blight          8.1%
          #3  Tomato — Healthy               3.2%
    """
    preds = state["predictions"]
    top = preds[0]

    lines = [
        f"Diagnosis: {top['crop']} — {top['disease']}  "
        f"({top['confidence'] * 100:.1f}% confidence)",
        "",
        "Top 3 predictions:",
    ]
    for p in preds:
        label = f"{p['crop']} — {p['disease']}"
        lines.append(f"  #{p['rank']}  {label:<34}  {p['confidence'] * 100:5.1f}%")

    return {**state, "message": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Node 3b — low_confidence_response  (low-confidence path)
# ---------------------------------------------------------------------------

def low_confidence_response(state: PlantDiseaseState) -> PlantDiseaseState:
    """
    Inform the user that confidence is too low for a reliable diagnosis and
    suggest corrective actions.

    Still shows the best guess and top-3 so the farmer has something to work
    with, but makes the uncertainty explicit.
    """
    preds = state["predictions"]
    top = preds[0]
    threshold_pct = config.CONFIDENCE_THRESHOLD * 100

    lines = [
        f"Low confidence prediction ({top['confidence'] * 100:.1f}% < {threshold_pct:.0f}% threshold).",
        "The model is not certain enough to give a reliable diagnosis.",
        "",
        "Tips for a better result:",
        "  • Photograph a single leaf filling most of the frame.",
        "  • Use natural daylight — avoid shadows and flash glare.",
        "  • Keep the camera steady and close (20–30 cm from the leaf).",
        "  • Make sure the affected area (spots, lesions) is in focus.",
        "",
        f"Best guess:  {top['crop']} — {top['disease']}  "
        f"({top['confidence'] * 100:.1f}%)",
        "",
        "Top 3 predictions:",
    ]
    for p in preds:
        label = f"{p['crop']} — {p['disease']}"
        lines.append(f"  #{p['rank']}  {label:<34}  {p['confidence'] * 100:5.1f}%")

    return {**state, "message": "\n".join(lines)}
