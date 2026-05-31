# 🧠 Brain MRI Multi-Agent Analysis Pipeline

> A LangGraph-powered agentic AI system for automated brain MRI analysis and report generation.  
> **Academic use only — not for clinical diagnosis.**

---

## What This Project Does

This project implements a **multi-agent pipeline** that:
1. Validates the uploaded image is actually a brain MRI
2. Passes it through 6 specialized AI agents in sequence
3. Produces a structured radiology-style report + a dedicated tumor verdict

Each agent has a single focused responsibility — a core principle of agentic system design.

---

## Architecture

```
[MRI Image + Patient Context]
         ↓
┌─────────────────────────────┐
│  Agent 0: Gatekeeper        │  → Is this a brain MRI? If not → STOP
│  (Groq Llama 4 Scout)       │
└─────────────┬───────────────┘
         ↓ yes            ↓ no → Pipeline halts with rejection message
┌─────────────────────────────┐
│  Agent 1: Preprocessor      │  → Sequence ID, plane, anatomy, image quality
│  (Groq Llama 4 Scout)       │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Agent 2: Analysis          │  → ACR-structured systematic read (10 categories)
│  (Groq Llama 4 Scout)       │
└─────────────┬───────────────┘
              ↓ (conditional: skip to report writer if error)
┌─────────────────────────────┐
│  Agent 3: Reasoning         │  → VINDICATE differential, urgency triage
│  (Groq Llama 3.3 70B)       │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Agent 4: Report Writer     │  → Full ACR-format structured radiology report
│  (Groq Llama 3.3 70B)       │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Agent 5: Critic            │  → QA review: laterality, overconfidence, safety
│  (Groq Llama 3.3 70B)       │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Agent 6: Tumor Conclusion  │  → TUMOR DETECTED / NOT DETECTED / UNCERTAIN
│  (Groq Llama 3.3 70B)       │
└─────────────────────────────┘
         ↓
[Final Report + Tumor Verdict + All Agent Outputs]
```

**Orchestration:** LangGraph `StateGraph`  
**Communication:** Shared typed state (`MRIAnalysisState`)  
**Conditional edges:**
- Gatekeeper rejection → pipeline halts at `END` immediately
- Analysis agent failure → skip reasoning, go straight to report writer

---

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| Agent Orchestration | LangGraph | Free |
| Vision LLM | Groq Llama 4 Scout (17B) | Free (Groq Console) |
| Text LLM | Groq Llama 3.3 70B Versatile | Free (Groq Console) |
| Vision Fallback | Google Gemini 2.0 Flash | Free (AI Studio) |
| Image Processing | OpenCV, Pillow | Free |
| UI | Gradio | Free |

---

## Setup

### 1. Clone / Download the project
```bash
git clone https://github.com/tselane2110/Multi-Agent-Brain-MRI-Analysis-Pipeline
```

### 2. Create a Virtual Environment and Install dependencies
```bash
# Create
python -m venv venv
```
```bash
# Activate — Windows CMD
venv\Scripts\activate.bat
```
```bash
# Activate — Windows PowerShell
venv\Scripts\Activate.ps1
```

```bash
# Install
pip install -r requirements.txt
```

### 3. Get FREE API keys

**Groq (required — powers all 7 agents):**
- Go to https://console.groq.com
- Sign up and create an API key

**Google Gemini (optional — vision fallback only):**
- Go to https://aistudio.google.com/app/apikey
- Only needed if Groq is unavailable

### 4. Set up environment variables
```bash
# Edit .env and paste your API keys
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_gemini_key_here   # optional
```

### 5. Run

**Option A: Gradio Web UI (recommended)**
```bash
python app.py
# Open http://localhost:7860 in your browser
```

**Option B: Command line (for testing)**
```bash
python test_pipeline.py                    # uses a synthetic test image
python test_pipeline.py path/to/mri.jpg   # uses your own MRI image
```

---

## Sample MRI Images for Testing

- **BraTS Dataset**: https://www.med.upenn.edu/cbica/brats/
- **OASIS Brain**: https://www.oasis-brains.org/
- **Radiopaedia**: https://radiopaedia.org (search "brain MRI")
- Any JPEG/PNG brain MRI from Google Images works for demo purposes

---

## Project Structure

```
brain_mri_agent/
├── app.py                  # Gradio web interface (7 output tabs)
├── pipeline.py             # LangGraph graph — nodes, edges, conditional routing
├── test_pipeline.py        # CLI test script
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── state.py            # Shared state TypedDict (all agents read/write here)
│   └── mri_agents.py       # All 7 agent functions
│
└── utils/
    ├── image_utils.py      # Image preprocessing & base64 conversion
    └── llm_setup.py        # LLM initialization (Groq primary, Gemini fallback)
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Multi-agent pipeline | `pipeline.py` — LangGraph StateGraph |
| Shared agent state | `agents/state.py` — TypedDict |
| Conditional edges (×2) | Gatekeeper branch + analysis error branch in `pipeline.py` |
| Input validation agent | `gatekeeper_agent()` — halts pipeline on invalid input |
| Vision + text agents | Agents 0–2 use vision LLM; Agents 3–6 use text LLM |
| ACR structured reporting | `report_writer_agent()` — follows radiology reporting standards |
| VINDICATE differential dx | `reasoning_agent()` — real radiological reasoning framework |
| Self-reflection / critique | `critic_agent()` — QA checklist including laterality audit |
| Dedicated verdict agent | `tumor_conclusion_agent()` — isolated single-question reasoning |
| Graceful error handling | Error propagation through state, pipeline continues to report |
| Multimodal input | Base64 image + text context passed to vision LLM |

---

## Disclaimer

This tool is built for **academic and educational purposes** as part of a Master's-level Agentic AI course. It is **not** validated for clinical use and must never be used for medical diagnosis or treatment decisions. All outputs require verification by a board-certified radiologist and clinical correlation by a qualified physician.
