# test_pipeline.py
# -------------------------------------------------------
# Run this to test the pipeline WITHOUT the Gradio UI.
# Useful for debugging individual agents.
#
# Usage:
#   python test_pipeline.py                    # uses a synthetic test image
#   python test_pipeline.py path/to/mri.jpg   # uses your own image
# -------------------------------------------------------

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
from image_utils import preprocess_for_agent
from pipeline import run_analysis


def create_synthetic_mri():
    """
    Creates a fake brain MRI-like image for testing.
    This lets you test the pipeline WITHOUT needing a real MRI scan.
    The LLM will analyze it like a real MRI (it won't know it's fake).
    """
    print("📸 Creating synthetic test MRI image...")
    
    img = Image.new("L", (512, 512), color=10)  # dark background
    draw = ImageDraw.Draw(img)
    
    # Draw a rough "brain" oval
    draw.ellipse([80, 60, 430, 450], fill=140, outline=180)
    
    # Draw "ventricles" (darker areas in the center)
    draw.ellipse([200, 180, 280, 270], fill=40)  # left ventricle
    draw.ellipse([230, 190, 320, 280], fill=40)  # right ventricle
    
    # Draw "white matter" areas
    draw.ellipse([100, 150, 200, 300], fill=160)  # left hemisphere
    draw.ellipse([310, 150, 420, 300], fill=160)  # right hemisphere
    
    # Add a subtle "lesion" (bright spot) - the kind of thing agents should detect
    draw.ellipse([320, 220, 360, 255], fill=220, outline=230)
    
    # Add some noise
    img_array = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Save for reference
    img.save("test_mri.jpg")
    print("✅ Synthetic MRI saved as test_mri.jpg")
    return img


def main():
    # Load image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            sys.exit(1)
        print(f"📂 Loading image: {image_path}")
        image = Image.open(image_path)
    else:
        image = create_synthetic_mri()
    
    # Preprocess
    print("🔧 Preprocessing image...")
    processed_img, image_b64 = preprocess_for_agent(image)
    
    # Sample patient context
    patient_context = "45-year-old patient with complaints of persistent headaches and occasional blurred vision for 6 weeks."
    
    # Run the pipeline
    result = run_analysis(image_b64, patient_context)
    
    # Print results
    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    
    if result.get("error"):
        print(f"\n❌ ERROR: {result['error']}")
    else:
        print(f"\n📊 CONFIDENCE LEVEL: {result.get('confidence_level', 'Unknown')}")
        
        print("\n" + "-"*40)
        print("AGENT 1 — IMAGE DESCRIPTION:")
        print("-"*40)
        print(result.get("image_description", "Not available"))
        
        print("\n" + "-"*40)
        print("AGENT 2 — FINDINGS:")
        print("-"*40)
        print(result.get("findings", "Not available"))
        
        print("\n" + "-"*40)
        print("AGENT 3 — REASONING CHAIN:")
        print("-"*40)
        print(result.get("reasoning_chain", "Not available"))
        
        print("\n" + "-"*40)
        print("AGENT 5 — CRITIC NOTES:")
        print("-"*40)
        print(result.get("critique_notes", "Not available"))
        
        print("\n" + "="*60)
        print("FINAL REPORT:")
        print("="*60)
        print(result.get("final_report", "Not available"))
    
    # Save report to file
    report_path = "output_report.txt"
    with open(report_path, "w") as f:
        f.write(result.get("final_report", "No report generated."))
    print(f"\n💾 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
