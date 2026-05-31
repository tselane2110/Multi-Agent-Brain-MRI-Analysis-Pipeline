# pipeline.py
# -------------------------------------------------------
# This is the HEART of the project — where LangGraph comes in.
#
# LangGraph lets you define a GRAPH of agents:
#   - Nodes = agents (functions that process state)
#   - Edges = the flow between agents (who runs after who)
#
# Think of it like a flowchart where each box is an agent.
# LangGraph handles:
#   - Running agents in the right order
#   - Passing state between them
#   - Allowing conditional branching (if error → skip to end)
# -------------------------------------------------------

from langgraph.graph import StateGraph, END
from state import MRIAnalysisState
from mri_agents import (
    preprocessor_agent,
    analysis_agent,
    reasoning_agent,
    report_writer_agent,
    critic_agent,
)


def should_continue_after_analysis(state: MRIAnalysisState) -> str:
    """
    CONDITIONAL EDGE — this is a key LangGraph feature.
    
    After the analysis agent runs, we check: did it succeed?
    - If there's an error → jump straight to report_writer (which handles errors)
    - If successful → continue to reasoning agent normally
    
    This is how agentic pipelines handle failures gracefully.
    """
    if state.get("error"):
        print("⚠️  Error detected, skipping reasoning agent.")
        return "report_writer"  # skip reasoning, go straight to report
    return "reasoning"          # normal flow


def build_mri_pipeline() -> StateGraph:
    """
    Builds and compiles the LangGraph pipeline.
    
    The graph looks like this:
    
    [START]
       ↓
    [Preprocessor Agent]  ← describes the image
       ↓
    [Analysis Agent]      ← finds anomalies
       ↓ (or skip if error)
    [Reasoning Agent]     ← reasons about findings
       ↓
    [Report Writer Agent] ← writes the report
       ↓
    [Critic Agent]        ← reviews and improves report
       ↓
    [END]
    """
    
    # Step 1: Create a graph with our state type
    # The state type tells LangGraph what data flows through the graph
    graph = StateGraph(MRIAnalysisState)
    
    # Step 2: Add each agent as a "node" in the graph
    # Format: graph.add_node("node_name", function_to_call)
    graph.add_node("preprocessor", preprocessor_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("report_writer", report_writer_agent)
    graph.add_node("critic", critic_agent)
    
    # Step 3: Define the flow (edges between nodes)
    
    # Always start with preprocessor
    graph.set_entry_point("preprocessor")
    
    # After preprocessor → always go to analysis
    graph.add_edge("preprocessor", "analysis")
    
    # After analysis → CONDITIONAL: if error, skip to report_writer
    graph.add_conditional_edges(
        "analysis",                          # from this node
        should_continue_after_analysis,      # call this function to decide
        {
            "reasoning": "reasoning",        # if returns "reasoning" → go to reasoning
            "report_writer": "report_writer" # if returns "report_writer" → go there
        }
    )
    
    # After reasoning → always go to report_writer
    graph.add_edge("reasoning", "report_writer")
    
    # After report_writer → go to critic
    graph.add_edge("report_writer", "critic")
    
    # After critic → END (pipeline complete)
    graph.add_edge("critic", END)
    
    # Step 4: Compile the graph into a runnable pipeline
    return graph.compile()


# Create the pipeline (called once at startup)
mri_pipeline = build_mri_pipeline()


def run_analysis(image_b64: str, patient_context: str = "") -> MRIAnalysisState:
    """
    Main entry point: runs the full pipeline on a brain MRI image.
    
    Args:
        image_b64: base64-encoded MRI image
        patient_context: optional clinical context (age, symptoms, etc.)
    
    Returns:
        The final state after all agents have run
    """
    
    # Initial state — only the input fields are set
    initial_state: MRIAnalysisState = {
        "input_image_b64": image_b64,
        "patient_context": patient_context if patient_context.strip() else "Not provided.",
        "image_description": None,
        "image_quality_notes": None,
        "findings": None,
        "regions_of_concern": None,
        "reasoning_chain": None,
        "differential_notes": None,
        "draft_report": None,
        "critique_notes": None,
        "final_report": None,
        "error": None,
        "confidence_level": None,
    }
    
    print("\n🚀 Starting Brain MRI Analysis Pipeline...")
    print("=" * 50)
    
    # Run the pipeline — LangGraph handles the rest
    final_state = mri_pipeline.invoke(initial_state)
    
    print("=" * 50)
    print("✅ Pipeline complete!\n")
    
    return final_state
