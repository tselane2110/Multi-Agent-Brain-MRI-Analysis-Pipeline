# utils/image_utils.py
# -------------------------------------------------------
# Handles all image preprocessing before the agents see it.
# Agents work with text + base64 images, so we need to:
#   1. Load and normalize the MRI image
#   2. Convert it to base64 so we can pass it to LLM vision APIs
# -------------------------------------------------------

import base64
import io
import numpy as np
from PIL import Image, ImageEnhance
import cv2


def load_and_preprocess(image_input) -> Image.Image:
    """
    Takes a file path (str) or a PIL Image and returns a
    preprocessed PIL Image ready for the agents.
    
    Preprocessing steps:
      - Convert to grayscale (MRIs are grayscale)
      - Normalize pixel intensity to 0-255 range
      - Enhance contrast so features are more visible to the LLM
      - Resize to a standard size (512x512) for consistent analysis
    """
    # --- Load ---
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        img = image_input  # already a PIL Image

    # --- Convert to grayscale (standard for MRI) ---
    img = img.convert("L")  # "L" = 8-bit grayscale

    # --- Normalize: stretch contrast to full 0-255 range ---
    # This is important because MRI scans can have low contrast
    img_array = np.array(img, dtype=np.float32)
    img_min, img_max = img_array.min(), img_array.max()
    if img_max > img_min:  # avoid division by zero
        img_array = (img_array - img_min) / (img_max - img_min) * 255.0
    img = Image.fromarray(img_array.astype(np.uint8))

    # --- Enhance contrast for LLM visibility ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)   
    
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(1.5)  # 1.5x contrast boost

    # --- Resize to standard size ---
    img = img.resize((512, 512), Image.LANCZOS)

    # Convert back to RGB so the LLM vision model can process it
    # (most vision APIs expect RGB, not grayscale)
    img = img.convert("RGB")

    return img


def image_to_base64(img: Image.Image) -> str:
    """
    Converts a PIL Image to a base64-encoded JPEG string.
    This is how we pass images to LLM vision APIs.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def preprocess_for_agent(image_input) -> tuple[Image.Image, str]:
    """
    Convenience function: preprocesses image and returns
    both the PIL Image (for display) and the base64 string (for LLM).
    """
    processed_img = load_and_preprocess(image_input)
    b64_string = image_to_base64(processed_img)
    return processed_img, b64_string
