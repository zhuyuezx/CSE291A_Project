# src/config.py
from __future__ import annotations

import os

import yaml


def load_config(path: str = "conf.yaml") -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve environment variables in api_key fields
    for section in config.values():
        if isinstance(section, dict) and "api_key" in section:
            key = section["api_key"]
            if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
                env_var = key[2:-1]
                section["api_key"] = os.environ.get(env_var, key)

    return config
