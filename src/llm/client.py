# src/llm/client.py
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI


class LLMClient:
    _call_counter = 0

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: int = 180,
        log_dir: str | None = None,
        role: str = "unknown",
    ):
        self.model = model
        self.temperature = temperature
        self.role = role
        self.log_dir = log_dir
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        LLMClient._call_counter += 1
        call_id = LLMClient._call_counter
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"    [LLM:{self.role}] Call #{call_id} → {self.model} (temp={self.temperature})")
        print(f"    [LLM:{self.role}] System prompt: {len(system_prompt)} chars")
        print(f"    [LLM:{self.role}] User prompt:   {len(user_prompt)} chars")

        t0 = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        elapsed = time.time() - t0
        content = response.choices[0].message.content

        print(f"    [LLM:{self.role}] Response: {len(content)} chars in {elapsed:.1f}s")

        # Save to log file
        if self.log_dir:
            log_entry = {
                "call_id": call_id,
                "timestamp": ts,
                "role": self.role,
                "model": self.model,
                "temperature": self.temperature,
                "elapsed_seconds": round(elapsed, 2),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": content,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                } if response.usage else None,
            }
            log_path = Path(self.log_dir) / f"{ts}_{self.role}_{call_id}.json"
            with open(log_path, "w") as f:
                json.dump(log_entry, f, indent=2)
            print(f"    [LLM:{self.role}] Saved log → {log_path}")

        return content

    @classmethod
    def from_config(cls, config: dict, role: str, log_dir: str | None = None) -> LLMClient:
        section = config[role]
        return cls(
            base_url=section["base_url"],
            model=section["model"],
            api_key=section.get("api_key", ""),
            temperature=section.get("temperature", 0.7),
            max_retries=section.get("max_retries", 3),
            timeout=section.get("timeout", 180),
            log_dir=log_dir,
            role=role,
        )
