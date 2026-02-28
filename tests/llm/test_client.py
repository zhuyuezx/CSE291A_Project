# tests/llm/test_client.py
import os
from src.config import load_config
from src.llm.client import LLMClient


def test_load_config():
    config = load_config("conf.yaml")
    assert "TRACE_ANALYZER" in config
    assert "model" in config["TRACE_ANALYZER"]


def test_client_creation():
    """Client should be creatable even without a valid API key."""
    client = LLMClient(
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key="test-key",
        temperature=0.7,
    )
    assert client.model == "test-model"
