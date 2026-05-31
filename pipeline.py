# pipeline.py
# -------------------------------------------------------
# This is the HEART of the project — where LangGraph comes in.
#
# Updated pipeline flow:
#
#   [START]
#      ↓
#   [Gatekeeper]  ← is this actually a brain MRI?
#      ↓ yes                ↓ no
#   [Preprocessor]       [END] (with rejection message)
#      ↓
#   [Analysis]
#      ↓ (conditional: skip if error)
#   [Reasoning]
#      ↓
#   [Report Writer]
#      ↓
#   [Critic]
#      ↓
#   [Tumor Conclusion]
#      ↓
#   [END]
# -------------------------------------------------------

from langgraph.graph import StateGraph, END
from agents.state import MRIAnalysisState
from agents.mri_agents import (
    gatekeeper_agent,
    preprocessor_agent,
    analysis_agent,
    reasoning_agent,
    report_writer_agent,
    critic_agent,
    tumor_conclusion_agent,
)


def should_proceed_after_gatekeeper(state: MRIAnalysisState) -> str:
    """
    Conditional edge after the gatekeeper.
    If the image is not a brain MRI → end immediately.
    If it is → proceed to preprocessor.
    """
    if not state.get("is_brain_mri"):
        print("🛑 Gatekeeper rejected image. Pipeline halted.")
        return "end"
    return "preprocessor"


def should_continue_after_analysis(state: MRIAnalysisState) -> str:
    """
    Conditional edge after analysis agent.
    If there's an error → skip reasoning, go straight to report writer.
    """
    if state.get("error"):
        print("⚠️  Error detected, skipping reasoning agent.")
        return "report_writer"
    return "reasoning"


def build_mri_pipeline() -> StateGraph:
    graph = StateGraph(MRIAnalysisState)

    # Register all nodes
    graph.add_node("gatekeeper", gatekeeper_agent)
    graph.add_node("preprocessor", preprocessor_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("report_writer", report_writer_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("tumor_conclusion", tumor_conclusion_agent)

    # Entry point
    graph.set_entry_point("gatekeeper")

    # Gatekeeper → conditional: valid MRI → preprocessor, invalid → END
    graph.add_conditional_edges(
        "gatekeeper",
        should_proceed_after_gatekeeper,
        {
            "preprocessor": "preprocessor",
            "end": END,
        }
    )

    # Preprocessor → Analysis (always)
    graph.add_edge("preprocessor", "analysis")

    # Analysis → conditional: error → report_writer, success → reasoning
    graph.add_conditional_edges(
        "analysis",
        should_continue_after_analysis,
        {
            "reasoning": "reasoning",
            "report_writer": "report_writer",
        }
    )

    # Reasoning → Report Writer → Critic → Tumor Conclusion → END
    graph.add_edge("reasoning", "report_writer")
    graph.add_edge("report_writer", "critic")
    graph.add_edge("critic", "tumor_conclusion")
    graph.add_edge("tumor_conclusion", END)

    return graph.compile()


mri_pipeline = build_mri_pipeline()


def run_analysis(image_b64: str, patient_context: str = "") -> MRIAnalysisState:
    """
    Main entry point: runs the full pipeline on a brain MRI image.
    """
    initial_state: MRIAnalysisState = {
        "input_image_b64": image_b64,
        "patient_context": patient_context if patient_context.strip() else "Not provided.",
        "is_brain_mri": None,
        "gatekeeper_reason": None,
        "image_description": None,
        "image_quality_notes": None,
        "findings": None,
        "regions_of_concern": None,
        "reasoning_chain": None,
        "differential_notes": None,
        "draft_report": None,
        "critique_notes": None,
        "final_report": None,
        "tumor_conclusion": None,
        "error": None,
        "confidence_level": None,
    }

    print("\n🚀 Starting Brain MRI Analysis Pipeline...")
    print("=" * 50)

    final_state = mri_pipeline.invoke(initial_state)

    print("=" * 50)
    print("✅ Pipeline complete!\n")

    return final_state
