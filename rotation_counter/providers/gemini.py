"""Google Gemini vision provider."""

from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel, Field

from rotation_counter.vlm import BATCH_PROMPT, normalize_labels


class BatchLabels(BaseModel):
    orientations: list[str] = Field(
        description='Exactly one of "front", "back", "side" per image, in order'
    )


class GeminiProvider:
    name = "gemini"
    batch_size = 6
    pause_seconds = 0.5

    def default_model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    def classify_batch(
        self,
        frames: list[bytes],
        model: str,
        *,
        prompt: str | None = None,
    ) -> list[str]:
        from google import genai
        from google.genai import errors, types

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")

        client = genai.Client(api_key=key)
        n = len(frames)
        text = prompt or BATCH_PROMPT.format(n=n)
        parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]

        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[text] + parts,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=BatchLabels,
                        max_output_tokens=max(96, n * 12),
                    ),
                )
                data = json.loads(response.text)
                return normalize_labels(data["orientations"], n)
            except errors.ClientError as e:
                if e.code != 429 or attempt == 4:
                    raise
                wait = 16 * (attempt + 1)
                print(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
                if attempt == 4:
                    raise
                print(f"Bad response, retrying... ({e})")
                time.sleep(2 * (attempt + 1))

        raise RuntimeError("Gemini classification failed")
