"""
Run a single image through the LangGraph confidence-routing workflow.

Usage:
    python scripts/predict_langgraph.py path/to/leaf.jpg
    python scripts/predict_langgraph.py path/to/leaf.jpg --threshold 0.75
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.langgraph_workflow import build_graph
from src.utils import get_logger

logger = get_logger("predict_langgraph")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify a crop-leaf photo using the LangGraph workflow."
    )
    p.add_argument("image_path", help="Path to the leaf image (JPG or PNG).")
    p.add_argument(
        "--threshold",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=(
            f"Confidence threshold for high vs low confidence routing "
            f"(default: {config.CONFIDENCE_THRESHOLD})."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Allow per-run threshold override without editing config.py
    if args.threshold != config.CONFIDENCE_THRESHOLD:
        config.CONFIDENCE_THRESHOLD = args.threshold
        logger.info("Confidence threshold overridden to %.2f", args.threshold)

    logger.info("Building LangGraph workflow...")
    app = build_graph()

    initial_state = {
        "image_path": str(args.image_path),
        "status": "pending",
        "predictions": None,
        "confidence": None,
        "message": None,
    }

    logger.info("Running workflow for: %s", args.image_path)
    result = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print(result.get("message", "No message returned."))
    print("=" * 50 + "\n")

    # Exit with non-zero code on error so shell scripts can detect failures
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
