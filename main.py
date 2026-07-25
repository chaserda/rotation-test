#!/usr/bin/env python3
"""
Count full 360° rotations in a video.

  python main.py videos/5rotationsTest.mp4
  python main.py videos/5rotationsTest.mp4 --provider openai
  python main.py videos/1.5rotationsTest.mp4 --provider claude --batch
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from rotation_counter.count import count_rotations
from rotation_counter.extract import extract_frames
from rotation_counter.providers import get_provider, provider_names
from rotation_counter.vlm import classify_batched, classify_parallel, close_open_lap

load_dotenv()


def analyze(video_path: str, provider_name: str, *, batch: bool = False) -> int:
    provider = get_provider(provider_name)
    model = provider.default_model()

    print(f"[*] Provider: {provider.name}")
    print(f"[*] Model: {model}")
    print(f"[*] Video: {video_path}")

    frames = extract_frames(video_path)
    print(f"[*] Extracted {len(frames)} frames")

    if batch:
        print("[*] Batch mode: fewer API calls, less precise")
        labels = classify_batched(frames, provider, model)
    else:
        # Default: one face-first call per frame, in parallel.
        labels = classify_parallel(frames, provider, model)

    print(f"[*] Labels: {labels}")

    labels = close_open_lap(frames, labels, provider, model)
    print(f"[*] Labels after finalize: {labels}")

    return count_rotations(labels)


def main() -> None:
    names = provider_names()
    parser = argparse.ArgumentParser(description="Count full rotations in a video")
    parser.add_argument("video", nargs="?", default="videos/5rotationsTest.mp4")
    parser.add_argument(
        "--provider",
        choices=names,
        default=os.getenv("VLM_PROVIDER", "gemini"),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Cheaper sequential batches (slower wall-clock, less precise)",
    )
    args = parser.parse_args()

    result = analyze(args.video, args.provider, batch=args.batch)
    print(f"\n>>> FINAL ROTATION COUNT: {result} <<<")
    print(result)


if __name__ == "__main__":
    main()
