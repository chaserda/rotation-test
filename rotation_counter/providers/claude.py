"""Anthropic Claude vision provider."""

from __future__ import annotations

import base64
import json
import os
import time

from rotation_counter.vlm import BATCH_PROMPT, LABEL_SCHEMA, normalize_labels


class ClaudeProvider:
    name = "claude"
    batch_size = 2
    pause_seconds = 0.75

    def __init__(self) -> None:
        self._client = None

    def default_model(self) -> str:
        return os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Set CLAUDE_API_KEY or ANTHROPIC_API_KEY in .env")
            self._client = Anthropic(api_key=key)
        return self._client

    def classify_batch(
        self,
        frames: list[bytes],
        model: str,
        *,
        prompt: str | None = None,
    ) -> list[str]:
        from anthropic import RateLimitError

        client = self._get_client()
        n = len(frames)
        text = prompt or BATCH_PROMPT.format(n=n)

        content: list[dict] = [{"type": "text", "text": text}]
        for i, jpeg in enumerate(frames):
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(jpeg).decode("ascii"),
                    },
                }
            )

        tools = [
            {
                "name": "batch_labels",
                "description": "Return one front/back/side label per image, in order.",
                "input_schema": LABEL_SCHEMA,
            }
        ]

        for attempt in range(5):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max(128, n * 24),
                    tools=tools,
                    tool_choice={"type": "tool", "name": "batch_labels"},
                    messages=[{"role": "user", "content": content}],
                )
                for block in response.content:
                    if getattr(block, "type", None) == "tool_use":
                        data = block.input
                        if isinstance(data, str):
                            data = json.loads(data)
                        return normalize_labels(data["orientations"], n)
                raise ValueError("Claude response missing tool_use block")
            except RateLimitError:
                if attempt == 4:
                    raise
                wait = 16 * (attempt + 1)
                print(f"Claude rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
                if attempt == 4:
                    raise
                print(f"Bad response, retrying... ({e})")
                time.sleep(2 * (attempt + 1))

        raise RuntimeError("Claude classification failed")
