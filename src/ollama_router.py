#!/usr/bin/env python3
"""
BlackRoad Ollama Router
Intercepts @copilot., @lucidia, @blackboxprogramming., and @ollama mentions
and routes every request directly to the local Ollama server.
No external AI provider (OpenAI, Anthropic, GitHub Copilot, etc.) is used.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# All @mention aliases that must be directed to Ollama.
# Trailing dots are optional (e.g. @copilot. or @copilot both work).
OLLAMA_TRIGGERS = re.compile(
    r"@(copilot\.?|lucidia\.?|blackboxprogramming\.?|ollama\.?)",
    re.IGNORECASE,
)


# ─── Public API ───────────────────────────────────────────────────────────────

def is_ollama_request(text: str) -> bool:
    """Return True when *text* contains an @mention that must go to Ollama."""
    return bool(OLLAMA_TRIGGERS.search(text))


def extract_prompt(text: str) -> str:
    """Strip the @mention prefix and return the clean prompt."""
    return OLLAMA_TRIGGERS.sub("", text).strip()


def ask(prompt: str, model: str = DEFAULT_MODEL,
        stream: bool = False) -> dict:
    """
    Send *prompt* to the local Ollama /api/generate endpoint.

    Returns a dict with at least:
      - ``response`` (str) — the model's reply
      - ``model``    (str) — model used
      - ``done``     (bool)

    Raises ``OllamaUnavailableError`` when the server cannot be reached.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt,
                          "stream": stream}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise OllamaUnavailableError(
            f"Ollama not reachable at {OLLAMA_BASE_URL}: {exc}"
        ) from exc


def route(text: str, model: str = DEFAULT_MODEL) -> dict:
    """
    High-level entry point.  Pass any user message containing an @mention.

    * If the message targets Ollama (via any registered alias) the prompt
      is cleaned of the @mention and forwarded to ``ask()``.
    * If no trigger is found a ``ValueError`` is raised — callers should
      not invoke this function for non-Ollama requests.

    Returns the same dict as ``ask()``.
    """
    if not is_ollama_request(text):
        raise ValueError(
            "Message does not contain an Ollama trigger "
            "(@copilot., @lucidia, @blackboxprogramming., @ollama)."
        )
    return ask(extract_prompt(text), model=model)


# ─── Exceptions ───────────────────────────────────────────────────────────────

class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""
