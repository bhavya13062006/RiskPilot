"""
LLM provider setup for the RiskPilot summarization agent.

Uses Google Gemini through LangChain.

Local usage:
    GEMINI_API_KEY=your_key_here

The API key is loaded from environment variables / Streamlit secrets.
The key is NEVER stored directly in this file.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


def get_llm(temperature: float = 0.2):
    """
    Return a Gemini LLM if an API key is configured.

    If no API key is available, return None so the
    summarization node can use its rule-based fallback.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
