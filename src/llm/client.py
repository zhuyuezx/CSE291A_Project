# src/llm/client.py
from __future__ import annotations

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: int = 180,
    ):
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    @classmethod
    def from_config(cls, config: dict, role: str) -> LLMClient:
        section = config[role]
        return cls(
            base_url=section["base_url"],
            model=section["model"],
            api_key=section.get("api_key", ""),
            temperature=section.get("temperature", 0.7),
            max_retries=section.get("max_retries", 3),
            timeout=section.get("timeout", 180),
        )
