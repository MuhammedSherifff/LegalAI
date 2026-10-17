"""One chat-model constructor, every free provider behind a single prefix.

Format: "provider:model" — chosen because model ids themselves contain '/'.
Default groq:qwen/qwen3.8-27b is the only chat model that passed the full
tool-calling probe (2026-10-17): correct tool pick, no spurious calls,
exactly one graph roundtrip. Gemini 3.5 Flash-Lite passed tool selection
but loops in the graph — reserve it for roles that make one call.
"""

from __future__ import annotations

import functools
import os

from dotenv import load_dotenv

from . import config

load_dotenv(config.PROJECT_ROOT / ".env")

_PROVIDERS = ("groq", "gemini", "openrouter")


@functools.lru_cache(maxsize=None)
def _model_for(provider: str, model: str, temperature: float):
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, api_key=os.getenv("GEMINI_API_KEY")
        )
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    raise ValueError(f"unknown provider {provider!r}, available: {_PROVIDERS}")


def get_llm(spec: str | None = None, temperature: float = 0):
    spec = spec or config.LLM
    try:
        provider, model = spec.split(":", 1)
    except ValueError:
        raise ValueError(f"LLM spec {spec!r} must be 'provider:model'") from None
    llm = _model_for(provider.lower(), model, temperature)
    if llm is None:
        raise ValueError(
            f"{provider} key missing in .env — set {provider.upper()}_API_KEY"
        )
    return llm
