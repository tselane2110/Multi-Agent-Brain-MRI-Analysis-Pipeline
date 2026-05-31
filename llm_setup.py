# utils/llm_setup.py
# -------------------------------------------------------
# Sets up our FREE LLM connections.
# PRIMARY: Groq (Llama vision + text) — generous free tier, no daily cap issues
# FALLBACK: Gemini Flash — if you have a fresh API key with quota remaining
# -------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


def get_vision_llm():
    """
    Vision LLM for Agents 1 & 2 (image + text input).
    
    Uses Groq's llama-4-scout — free, has vision, generous rate limits.
    Falls back to Gemini if no Groq key is set.
    
    Get a FREE Groq key at: https://console.groq.com
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        return ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # free vision model on Groq
            groq_api_key=groq_key,
            temperature=0.2,
            max_tokens=1500,
        )

    # Fallback to Gemini (only if you have fresh quota)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No GROQ_API_KEY or GOOGLE_API_KEY found in .env\n"
            "Get a FREE Groq key at: https://console.groq.com  ← recommended\n"
            "Get a FREE Gemini key at: https://aistudio.google.com/app/apikey"
        )
    print("⚠️  Using Gemini for vision (watch your free quota).")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_tokens=1500,
    )


def get_text_llm():
    """
    Text-only LLM for Agents 3, 4, 5.
    
    Uses Groq's Llama 3.3 70B — free, fast, strong reasoning.
    Get a FREE Groq key at: https://console.groq.com
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_key,
            temperature=0.3,
            max_tokens=2000,
        )

    # Fallback to Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No GROQ_API_KEY or GOOGLE_API_KEY found in .env\n"
            "Get a FREE Groq key at: https://console.groq.com"
        )
    print("⚠️  Using Gemini for text agents (watch your free quota).")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_tokens=2000,
    )