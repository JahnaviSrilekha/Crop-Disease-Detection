"""
LangGraph workflow graph for crop disease inference.

Flow:
                    ┌──────────────────┐
                    │  validate_image  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  run_prediction  │
                    └────────┬─────────┘
                             │
                   [confidence_router]
                    /         |         \
              error        ≥ 60%        < 60%
               END     final_response  low_confidence_response
                            │                   │
                           END                 END

Usage:
    from src.langgraph_workflow import build_graph

    app = build_graph()
    result = app.invoke({"image_path": "/path/to/leaf.jpg", "status": "pending"})
    print(result["message"])
"""

import config
from langgraph.graph import END, StateGraph

from src.langgraph_workflow.nodes import (
    final_response,
    low_confidence_response,
    run_prediction,
    validate_image,
)
from src.langgraph_workflow.state import PlantDiseaseState


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def confidence_router(state: PlantDiseaseState) -> str:
    """
    Decide which response node to invoke based on top-1 confidence.

    Returns:
        "error"           — validation or prediction failed  → END
        "high_confidence" — confidence ≥ CONFIDENCE_THRESHOLD → final_response
        "low_confidence"  — confidence <  CONFIDENCE_THRESHOLD → low_confidence_response
    """
    if state.get("status") == "error":
        return "error"

    confidence = state.get("confidence") or 0.0
    if confidence >= config.CONFIDENCE_THRESHOLD:
        return "high_confidence"
    return "low_confidence"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Compile and return the LangGraph application.

    Call once at startup; the returned object is thread-safe and reusable.

    Example:
        app = build_graph()
        result = app.invoke({
            "image_path": "outputs/test_leaf.jpg",
            "status": "pending",
        })
        print(result["message"])
    """
    graph = StateGraph(PlantDiseaseState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("validate_image", validate_image)
    graph.add_node("run_prediction", run_prediction)
    graph.add_node("final_response", final_response)
    graph.add_node("low_confidence_response", low_confidence_response)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("validate_image")

    # ── Edges ─────────────────────────────────────────────────────────────────
    # validate_image always flows into run_prediction
    # (errors from validate_image are surfaced via state["status"]="error",
    #  and confidence_router will route them to END without running XGBoost)
    graph.add_edge("validate_image", "run_prediction")

    # After prediction, branch on confidence
    graph.add_conditional_edges(
        "run_prediction",
        confidence_router,
        {
            "high_confidence": "final_response",
            "low_confidence":  "low_confidence_response",
            "error":           END,
        },
    )

    graph.add_edge("final_response", END)
    graph.add_edge("low_confidence_response", END)

    return graph.compile()
