"""
LLM provider setup for the summarization agent.

Supports Anthropic (Claude) or OpenAI, selected via environment variables so
the same code runs against either provider without edits:

    LLM_PROVIDER=anthropic   ANTHROPIC_API_KEY=...
    LLM_PROVIDER=openai      OPENAI_API_KEY=...

If no key is configured, `get_llm()` returns None and the summarization node
falls back to a deterministic, rule-based memo (see agents/nodes.py). This
keeps the whole pipeline runnable end-to-end — including in CI or a fresh
clone with no keys set — while still demonstrating full LLM-backed output
whenever a key is present.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}


def get_llm(temperature: float = 0.2):
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("LLM_MODEL", DEFAULT_MODEL["anthropic"])
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.getenv("LLM_MODEL", DEFAULT_MODEL["openai"])
        return ChatOpenAI(model=model, temperature=temperature)

    return None  # no key configured -> caller should use the rule-based fallback
