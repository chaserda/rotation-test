#!/usr/bin/env python3
# Count full 360° rotations in a video.
#
#   python main.py videos/5rotationsTest.mp4
#   python main.py videos/5rotationsTest.mp4 --provider openai
#   python main.py videos/1.5rotationsTest.mp4 --provider claude

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from rotation_counter.count import measure_spin
from rotation_counter.extract import extract_frames
from rotation_counter.providers import get_provider, provider_names
from rotation_counter.vlm import classify_parallel, close_open_lap, fix_opening

load_dotenv()


# Extract frames, classify with a VLM, finalize open laps, return spin result.
def analyze(video_path: str, provider_name: str):
    provider = get_provider(provider_name)
    model = provider.default_model()

    print(f"[*] Provider: {provider.name}")
    print(f"[*] Model: {model}")
    print(f"[*] Video: {video_path}")

    frames = extract_frames(video_path)
    print(f"[*] Extracted {len(frames)} frames")

    labels = classify_parallel(frames, provider, model)
    print(f"[*] Labels: {labels}")

    labels = fix_opening(frames, labels, provider, model)
    labels = close_open_lap(frames, labels, provider, model)
    print(f"[*] Labels after finalize: {labels}")

    return measure_spin(labels)


# CLI entrypoint.
def main() -> None:
    names = provider_names()
    parser = argparse.ArgumentParser(description="Count full rotations in a video")
    parser.add_argument("video", nargs="?", default="videos/5rotationsTest.mp4")
    parser.add_argument(
        "--provider",
        choices=names,
        default=os.getenv("VLM_PROVIDER", "gemini"),
    )
    args = parser.parse_args()

    spin = analyze(args.video, args.provider)
    print(
        f"\n>>> FULL ROTATIONS: {spin.full_rotations}  |  "
        f"DEGREES: {spin.degrees}° <<<"
    )
    print(spin.full_rotations)
    print(spin.degrees)


if __name__ == "__main__":
    main()
