# agents/state.py
# -------------------------------------------------------
# In LangGraph, agents communicate through a SHARED STATE.
# Think of it as a "clipboard" that every agent can read from
# and write to. Each agent adds its findings to this state,
# and the next agent picks up from where the last one left off.
#
# This is one of the core ideas of agentic AI:
#   Agents don't call each other directly — they share state.
# -------------------------------------------------------

from typing import TypedDict, Optional


class MRIAnalysisState(TypedDict):
    """
    The shared state object passed between all agents in the pipeline.
    
    Flow:
      input_image_b64  →  [Preprocessor]  →  image_description
      image_description  →  [Analysis Agent]  →  findings
      findings  →  [Reasoning Agent]  →  reasoning_chain
      reasoning_chain  →  [Report Writer]  →  draft_report
      draft_report  →  [Critic Agent]  →  final_report
    """
    
    # --- Input ---
    input_image_b64: str           # base64 encoded MRI image
    patient_context: Optional[str] # optional: age, symptoms, etc.
    
    # --- Preprocessor Agent output ---
    image_description: Optional[str]   # basic visual description of the MRI
    image_quality_notes: Optional[str] # notes on image quality/artifacts
    
    # --- Analysis Agent output ---
    findings: Optional[str]            # detected anomalies and observations
    regions_of_concern: Optional[str]  # specific brain regions flagged
    
    # --- Reasoning Agent output ---
    reasoning_chain: Optional[str]     # step-by-step medical reasoning
    differential_notes: Optional[str]  # possible interpretations
    
    # --- Report Writer output ---
    draft_report: Optional[str]        # structured radiology-style report
    
    # --- Critic Agent output ---
    critique_notes: Optional[str]      # what the critic flagged
    final_report: Optional[str]        # polished final report
    
    # --- Pipeline metadata ---
    error: Optional[str]               # any error that occurred
    confidence_level: Optional[str]    # overall confidence: HIGH / MEDIUM / LOW
