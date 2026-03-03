"""
Tests for BlackRoad Ollama Router.

All requests via @copilot., @lucidia, @blackboxprogramming., or @ollama
must be routed to local Ollama — no external AI provider is invoked.

Network calls are mocked so tests run offline.
"""
from __future__ import annotations

import json
import sys
import types
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import ollama_router as router


# ─── Trigger detection ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "@copilot. write me a poem",
    "@copilot write me a poem",
    "@lucidia explain quantum computing",
    "@lucidia. explain quantum computing",
    "@blackboxprogramming. debug this",
    "@blackboxprogramming debug this",
    "@ollama. summarise this article",
    "@ollama summarise this article",
    "hey @Copilot. what is 2+2",       # case-insensitive
    "question for @LUCIDIA. please help",
])
def test_is_ollama_request_true(text):
    assert router.is_ollama_request(text) is True


@pytest.mark.parametrize("text", [
    "hello world",
    "@openai tell me a joke",
    "@claude explain this",
    "@chatgpt write code",
    "no mention here at all",
])
def test_is_ollama_request_false(text):
    assert router.is_ollama_request(text) is False


# ─── Prompt extraction ────────────────────────────────────────────────────────

def test_extract_prompt_strips_mention():
    assert router.extract_prompt("@copilot. write a haiku") == "write a haiku"


def test_extract_prompt_lucidia():
    assert router.extract_prompt("@lucidia explain this") == "explain this"


def test_extract_prompt_blackboxprogramming():
    assert router.extract_prompt("@blackboxprogramming. debug my code") == "debug my code"


def test_extract_prompt_ollama():
    assert router.extract_prompt("@ollama what is 2+2") == "what is 2+2"


def test_extract_prompt_removes_only_mention():
    result = router.extract_prompt("some text @ollama. do this and that")
    assert "@ollama" not in result
    assert "some text" in result


# ─── ask() — network mocked ───────────────────────────────────────────────────

def _fake_response(payload: dict):
    """Return a mock that urllib.request.urlopen can yield as a context manager."""
    raw = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_ask_sends_to_ollama():
    payload = {"model": "llama3", "response": "Hello!", "done": True}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as mock_open:
        result = router.ask("Hello", model="llama3")
    assert result["response"] == "Hello!"
    assert result["done"] is True
    # Verify the URL targeted localhost Ollama
    call_args = mock_open.call_args
    req = call_args[0][0]
    assert "localhost:11434" in req.full_url or "11434" in req.full_url


def test_ask_raises_on_connection_error():
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(router.OllamaUnavailableError):
            router.ask("Hello")


# ─── route() ─────────────────────────────────────────────────────────────────

def test_route_copilot_goes_to_ollama():
    payload = {"model": "llama3", "response": "I am Ollama", "done": True}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        result = router.route("@copilot. write a limerick")
    assert result["response"] == "I am Ollama"


def test_route_lucidia_goes_to_ollama():
    payload = {"model": "llama3", "response": "Local reply", "done": True}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        result = router.route("@lucidia. what day is it?")
    assert result["response"] == "Local reply"


def test_route_blackboxprogramming_goes_to_ollama():
    payload = {"model": "llama3", "response": "BB reply", "done": True}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        result = router.route("@blackboxprogramming. fix my bug")
    assert result["response"] == "BB reply"


def test_route_ollama_goes_to_ollama():
    payload = {"model": "llama3", "response": "Ollama direct", "done": True}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        result = router.route("@ollama tell me a story")
    assert result["response"] == "Ollama direct"


def test_route_raises_for_non_trigger():
    with pytest.raises(ValueError, match="does not contain an Ollama trigger"):
        router.route("@openai write something")


def test_route_no_external_provider_called():
    """route() must not import or call openai/anthropic/copilot SDKs."""
    payload = {"model": "llama3", "response": "ok", "done": True}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        router.route("@copilot. hello")
    for banned in ("openai", "anthropic", "copilot"):
        assert banned not in sys.modules, f"Banned module '{banned}' was imported"
