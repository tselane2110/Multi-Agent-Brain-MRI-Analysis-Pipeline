# 🧠 Brain MRI Multi-Agent Analysis Pipeline

> A LangGraph-powered agentic AI system for automated brain MRI analysis and report generation.  
> **Academic use only — not for clinical diagnosis.**

---

## What This Project Does

This project implements a **multi-agent pipeline** that:
1. Takes a brain MRI image as input
2. Passes it through 5 specialized AI agents in sequence
3. Produces a structured radiology-style report

Each agent has a single focused responsibility — a core principle of agentic system design.

---

## Architecture

```
[MRI Image + Patient Context]
         ↓
┌─────────────────────────────┐
│  Agent 1: Preprocessor      │  → Describes the image (visual grounding)
│  (Groq Llama 4 Scout)      │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Agent 2: Analysis          │  → Detects anomalies, regions of concern
│  (Groq Llama 4 Scout)      │
└─────────────┬───────────────┘
              ↓ (conditional: skip if error)
┌─────────────────────────────┐
│  Agent 3: Reasoning         │  → Medical reasoning, differential diagnosis
│  (Groq Llama 3.3 70B)      │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Agent 4: Report Writer     │  → Structured radiology report
│  (Groq Llama 3.3 70B)      │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Agent 5: Critic            │  → Reviews, critiques, improves report
│  (Groq Llama 3.3 70B)      │
└─────────────────────────────┘
         ↓
[Final Report + All Agent Outputs]
```

**Orchestration:** LangGraph `StateGraph`  
**Communication:** Shared typed state (`MRIAnalysisState`)  
**Conditional flow:** Analysis agent failure → skip reasoning, go to report writer

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
# in cmd:
git clone https://github.com/tselane2110/Multi-Agent-Brain-MRI-Analysis-Pipeline
```

### 2. Create a Virtual Environment and Install dependencies
Creating the virtual environment
```bash
python -m venv venv
```
Activating the virtual environment
```bash
# via windows (cmd):
venv\Scripts\activate.bat
```
```bash
# via windows (powershell)
venv\Scripts\Activate.ps1
```
Installing Dependencies in the virtual environment
```bash
pip install -r requirements.txt
```

### 3. Get FREE API keys

**Groq (required — for both vision and text agents):**
- Go to https://console.groq.com
- Sign up and create an API key

**Google Gemini (optional — fallback only):**
- Go to https://aistudio.google.com/app/apikey
- Click "Create API Key"
- Only needed if Groq is unavailable

### 4. Set up environment variables
```bash
# Edit .env and paste your API keys
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

Free academic MRI images you can use:
- **BraTS Dataset**: https://www.med.upenn.edu/cbica/brats/
- **OASIS Brain**: https://www.oasis-brains.org/
- **Radiopaedia**: https://radiopaedia.org (search "brain MRI")
- Any JPEG/PNG brain MRI from Google Images works for demo purposes

---

## Project Structure

```
brain_mri_agent/
├── app.py                  # Gradio web interface
├── pipeline.py             # LangGraph graph definition (core)
├── test_pipeline.py        # CLI test script
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── state.py            # Shared state TypedDict
│   └── mri_agents.py       # All 5 agent functions
│
└── utils/
    ├── image_utils.py      # Image preprocessing & base64 conversion
    └── llm_setup.py        # Free LLM initialization (Groq primary, Gemini fallback)
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Multi-agent pipeline | `pipeline.py` — LangGraph StateGraph |
| Shared agent state | `agents/state.py` — TypedDict |
| Conditional edges | `pipeline.py` — `should_continue_after_analysis()` |
| Vision + text agents | `agents/mri_agents.py` |
| Self-reflection / critique | `critic_agent()` in `mri_agents.py` |
| Graceful error handling | Error propagation through state |
| Multimodal input | Image + text passed to vision LLM |

---

## Disclaimer

This tool is built for **academic and educational purposes** as part of a Master's-level Agentic AI course. It is **not** validated for clinical use and must never be used for medical diagnosis or treatment decisions. All outputs require verification by a board-certified radiologist and clinical correlation by a qualified physician.
