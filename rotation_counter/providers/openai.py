# OpenAI vision provider.

from __future__ import annotations

import base64
import json
import os
import time

from rotation_counter.vlm import BATCH_PROMPT, LABEL_SCHEMA, normalize_labels


class OpenAIProvider:
    name = "openai"
    batch_size = 3
    pause_seconds = 1.0

    def __init__(self) -> None:
        self._client = None

    # gpt-4o-mini often keeps a low free-tier RPD even after adding credits.
    def default_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY in .env")
            self._client = OpenAI(api_key=key)
        return self._client

    def classify_batch(
        self,
        frames: list[bytes],
        model: str,
        *,
        prompt: str | None = None,
    ) -> list[str]:
        from openai import RateLimitError

        client = self._get_client()
        n = len(frames)
        text = prompt or BATCH_PROMPT.format(n=n)

        content: list[dict] = [{"type": "text", "text": text}]
        for i, jpeg in enumerate(frames):
            b64 = base64.b64encode(jpeg).decode("ascii")
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                }
            )

        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=max(96, n * 12),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "batch_labels",
                            "strict": True,
                            "schema": LABEL_SCHEMA,
                        },
                    },
                    messages=[{"role": "user", "content": content}],
                )
                raw = response.choices[0].message.content or ""
                data = json.loads(raw)
                return normalize_labels(data["orientations"], n)
            except RateLimitError as e:
                msg = str(e).lower()
                if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                    raise RuntimeError(
                        "OpenAI quota/billing exhausted. "
                        "Check billing or use --provider gemini."
                    ) from e
                if attempt == 5:
                    raise
                wait = 20 * (attempt + 1)
                print(f"OpenAI rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
                if attempt == 5:
                    raise
                print(f"Bad response, retrying... ({e})")
                time.sleep(2 * (attempt + 1))

        raise RuntimeError("OpenAI classification failed")
