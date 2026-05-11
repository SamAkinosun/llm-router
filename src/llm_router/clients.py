"""LLM client interface.

Two implementations:
  - MockClient: deterministic, no network. Used in tests and as the
    default for the example/benchmark code so people can run things
    without an API key.
  - AnthropicClient: thin wrapper around the public REST API. Built so
    that no key is hard-coded; the caller passes one in.

The interface is tiny on purpose. Anything more would couple this
project to a single provider's SDK shape.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class CompletionResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int


class Client(Protocol):
    def complete(self, model: str, prompt: str, max_tokens: int = 1024) -> CompletionResult: ...


class MockClient:
    """Returns a fake completion. Token counts are derived from word count.

    Useful for tests and for letting people see the router work without
    an API key. Behavior is deterministic given the same prompt + model.
    """

    def __init__(self, output_word_count: int = 40):
        self.output_word_count = output_word_count
        self.calls = []  # list of (model, prompt, max_tokens) for inspection

    def complete(self, model: str, prompt: str, max_tokens: int = 1024) -> CompletionResult:
        self.calls.append((model, prompt, max_tokens))
        # Approximate token counts: 1.3 tokens per word is a reasonable rule of thumb.
        input_tokens = max(1, int(len(prompt.split()) * 1.3))
        output_tokens = max(1, int(self.output_word_count * 1.3))
        text = f"[mock {model}] received {input_tokens}-token prompt"
        return CompletionResult(model=model, text=text, input_tokens=input_tokens, output_tokens=output_tokens)


class AnthropicClient:
    """Minimal Anthropic Messages API wrapper using stdlib only.

    Pass `api_key` directly. Do not read it from env in this class; let the
    caller decide where to source it (config, env, secret manager).
    """

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, timeout: float = 60.0):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._timeout = timeout

    def complete(self, model: str, prompt: str, max_tokens: int = 1024) -> CompletionResult:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL,
            data=body,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"Anthropic API error {e.code}: {detail[:300]}") from e

        data = json.loads(raw)
        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return CompletionResult(
            model=data.get("model", model),
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
