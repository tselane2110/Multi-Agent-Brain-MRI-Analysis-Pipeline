# app.py
# -------------------------------------------------------
# Gradio UI for the Brain MRI Analysis Pipeline.
# Gradio lets us build a web interface with just Python —
# no HTML/CSS/JS needed. Great for demos and portfolios.
#
# Run this file to launch the app:
#   python app.py
# -------------------------------------------------------

import gradio as gr
from PIL import Image
import traceback

from utils.image_utils import preprocess_for_agent
from pipeline import run_analysis


# ── Helper ──────────────────────────────────────────────

def analyze_mri(image: Image.Image, patient_context: str, progress=gr.Progress()):
    """
    Called when the user clicks "Analyze MRI".
    Runs the full multi-agent pipeline and returns results for display.
    """
    if image is None:
        return (
            "❌ Please upload an MRI image first.",
            "", "", "", "", "", None
        )
    
    try:
        progress(0.1, desc="Preprocessing image...")
        processed_img, image_b64 = preprocess_for_agent(image)
        
        progress(0.2, desc="Starting pipeline... (this takes ~60-90 seconds)")
        
        result = run_analysis(image_b64, patient_context)
        
        progress(1.0, desc="Done!")

        # ── Gatekeeper rejection ──────────────────────────
        if not result.get("is_brain_mri"):
            rejection_msg = (
                f"🚫 IMAGE REJECTED BY GATEKEEPER\n\n"
                f"This image does not appear to be a brain MRI scan.\n\n"
                f"Reason: {result.get('gatekeeper_reason', 'No reason provided.')}\n\n"
                f"Please upload a valid brain MRI image (T1, T2, FLAIR, DWI, etc.)"
            )
            return (rejection_msg, "", "", "", "", "", processed_img)

        error = result.get("error")
        if error:
            return (
                f"⚠️ Pipeline error:\n{error}",
                "", "", "",
                result.get("final_report", "Report not generated."),
                result.get("tumor_conclusion", ""),
                processed_img,
            )

        return (
            result.get("image_description", "Not available"),
            result.get("findings", "Not available"),
            result.get("reasoning_chain", "Not available"),
            result.get("critique_notes", "Not available"),
            result.get("final_report", "Not available"),
            result.get("tumor_conclusion", "Not available"),
            processed_img,
        )
        
    except Exception as e:
        error_msg = f"❌ Unexpected error:\n{traceback.format_exc()}"
        return (error_msg, "", "", "", "", "", None)


# ── UI Layout ────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Brain MRI Analysis Agent",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="""
            .header { text-align: center; margin-bottom: 20px; }
            .disclaimer { 
                background: #fff3cd; 
                border: 1px solid #ffc107; 
                border-radius: 8px; 
                padding: 12px; 
                margin: 10px 0;
                font-size: 0.9em;
            }
            .confidence-badge { font-weight: bold; }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🧠 Brain MRI Multi-Agent Analysis Pipeline</h1>
                <p style="color: #666; font-size: 1.1em;">
                    Powered by LangGraph · Gemini Flash · Academic Research Tool
                </p>
            </div>
            <div class="disclaimer">
                ⚠️ <strong>DISCLAIMER:</strong> This is an academic AI research tool built for 
                a Master's-level Agentic AI course. It is <strong>NOT</strong> intended for 
                clinical diagnosis or medical decision-making. All outputs require verification 
                by a board-certified radiologist.
            </div>
        """)
        
        # Main layout: left = inputs, right = preprocessed image
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload MRI Scan")
                image_input = gr.Image(
                    type="pil",
                    label="Brain MRI Image",
                    height=300,
                )
                
                patient_context = gr.Textbox(
                    label="Patient Context (Optional)",
                    placeholder="e.g., 45-year-old female, presenting with persistent headaches and visual disturbances for 3 weeks.",
                    lines=3,
                )
                
                analyze_btn = gr.Button(
                    "🔬 Run Full Analysis",
                    variant="primary",
                    size="lg",
                )
                
                gr.Markdown("""
                **Pipeline Steps:**
                1. 🛡️ Gatekeeper Agent — validates brain MRI
                2. 🔍 Preprocessor Agent — visual grounding
                3. 🧠 Analysis Agent — anomaly detection  
                4. 💭 Reasoning Agent — medical reasoning
                5. 📝 Report Writer — structured report
                6. 🔬 Critic Agent — quality review
                7. 🧬 Tumor Conclusion — tumor verdict
                
                *Analysis takes ~90-120 seconds*
                """)
                
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Preprocessed Image")
                processed_image = gr.Image(
                    label="After Normalization & Enhancement",
                    height=300,
                )
        
        gr.Markdown("---")
        gr.Markdown("### 📊 Analysis Results")
        
        # Results in tabs
        with gr.Tabs():
            with gr.Tab("📋 Final Report"):
                final_report_output = gr.Textbox(
                    label="AI-Generated Radiology Report",
                    lines=25,
                    show_copy_button=True,
                )

            with gr.Tab("🧬 Tumor Verdict"):
                tumor_output = gr.Textbox(
                    label="Tumor Conclusion Agent Output",
                    lines=15,
                    show_copy_button=True,
                )
            
            with gr.Tab("🔍 Image Description"):
                description_output = gr.Textbox(
                    label="Preprocessor Agent Output",
                    lines=12,
                    show_copy_button=True,
                )
            
            with gr.Tab("🧠 Clinical Findings"):
                findings_output = gr.Textbox(
                    label="Analysis Agent Output",
                    lines=15,
                    show_copy_button=True,
                )
            
            with gr.Tab("💭 Reasoning Chain"):
                reasoning_output = gr.Textbox(
                    label="Reasoning Agent Output",
                    lines=15,
                    show_copy_button=True,
                )
            
            with gr.Tab("🔬 Critic Review"):
                critique_output = gr.Textbox(
                    label="Critic Agent Notes",
                    lines=12,
                    show_copy_button=True,
                )
        
        # Wire up the button
        analyze_btn.click(
            fn=analyze_mri,
            inputs=[image_input, patient_context],
            outputs=[
                description_output,
                findings_output,
                reasoning_output,
                critique_output,
                final_report_output,
                tumor_output,
                processed_image,
            ],
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 20px; color: #888; font-size: 0.85em;">
                Built with LangGraph + Google Gemini Flash + Groq Llama · 
                Master's Agentic AI Course Project · 
                For Academic Use Only
            </div>
        """)
    
    return demo


# ── Launch ───────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Brain MRI Multi-Agent Analysis Pipeline")
    print("=" * 45)
    print("Starting Gradio interface...")
    print("Make sure your .env file has GOOGLE_API_KEY set!")
    print("=" * 45)
    
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,    # set to True to get a public link
        show_error=True,
    )
