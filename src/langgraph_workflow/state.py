"""
Shared state that flows through every node in the LangGraph workflow.

Each field is Optional so nodes can be added incrementally — a node only
fills in the fields it is responsible for and passes the rest through.
"""

from typing import Dict, List, Optional, TypedDict


class PlantDiseaseState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    image_path: str                     # absolute or relative path to the photo

    # ── Set by validate_image ──────────────────────────────────────────────────
    status: str                         # "pending" | "valid" | "error"

    # ── Set by run_prediction ──────────────────────────────────────────────────
    predictions: Optional[List[Dict]]   # top-3 dicts from src.inference.predict_disease
    #   each dict: {rank, raw_label, crop, disease, confidence}
    confidence: Optional[float]         # top-1 confidence score (0 – 1)

    # ── Set by final_response / low_confidence_response ───────────────────────
    message: Optional[str]              # human-readable output shown to the user
