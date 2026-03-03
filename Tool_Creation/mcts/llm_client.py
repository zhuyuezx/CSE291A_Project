"""
LLM client for the MCTS optimization loop.

Encapsulates all communication with the LLM backend (Ollama).
Handles request construction, response parsing, and thinking-mode
extraction (Qwen 3.5 puts code in the 'thinking' field).
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Parsed response from an LLM call."""

    content: str          # Main assistant content
    thinking: str         # Thinking/reasoning content (Qwen 3.5 specific)
    full_text: str        # Combined content + thinking
    model: str
    elapsed_sec: float

    @property
    def has_content(self) -> bool:
        return bool(self.content.strip())


class LLMClient:
    """
    Client for calling LLMs via the Ollama REST API.

    Handles:
    - Request building and timeout management
    - Response parsing (content + thinking fields)
    - Retry logic for transient failures

    Example::

        client = LLMClient(model="qwen3.5:35b")
        resp = client.chat("Write a Python function ...")
        print(resp.full_text)
    """

    def __init__(
        self,
        model: str = "qwen3.5:35b",
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
        max_retries: int = 1,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        timeout: int | None = None,
    ) -> LLMResponse:
        """
        Send a single-turn chat message and return the parsed response.

        Args:
            prompt:  User message content.
            system:  Optional system message.
            timeout: Override the default timeout for this call.

        Returns:
            LLMResponse with content, thinking, and timing info.

        Raises:
            requests.RequestException: On network or HTTP errors.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        effective_timeout = timeout or self.timeout
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.time()
                resp = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=effective_timeout,
                )
                resp.raise_for_status()
                elapsed = time.time() - t0
                data = resp.json()
                msg = data.get("message", {})

                content = msg.get("content", "")
                thinking = msg.get("thinking", "")

                return LLMResponse(
                    content=content,
                    thinking=thinking,
                    full_text=(content + "\n" + thinking).strip(),
                    model=self.model,
                    elapsed_sec=round(elapsed, 1),
                )

            except (requests.RequestException, KeyError) as e:
                last_error = e
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # exponential backoff

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempt(s): {last_error}"
        ) from last_error

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, base_url={self.base_url!r})"
