"""Register VLM backends here. Add a provider = one file + one line below."""

from __future__ import annotations

from rotation_counter.providers.claude import ClaudeProvider
from rotation_counter.providers.gemini import GeminiProvider
from rotation_counter.providers.openai import OpenAIProvider

PROVIDERS = {
    "gemini": GeminiProvider(),
    "openai": OpenAIProvider(),
    "claude": ClaudeProvider(),
}


def get_provider(name: str):
    key = name.lower().strip()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Choose: {', '.join(sorted(PROVIDERS))}")
    return PROVIDERS[key]


def provider_names() -> list[str]:
    return sorted(PROVIDERS)
