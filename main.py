from __future__ import annotations

import json
import os
import sys
import time

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

# Loading the environment variables.
load_dotenv()

# Pydantic model for the response from Gemini.
class BatchLabels(BaseModel):
    orientations: list[str] = Field(
        description='Exactly one of "front", "back", "side" per image, in order'
    )

# Getting the client for the Gemini API.
def get_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    return genai.Client(api_key=key)

# Extracting frames from the video.
def extract_frames(video_path: str, fps_target: float = 4.0, max_width: int = 640):
    """OpenCV only decodes / samples / resizes — no tracking."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(original_fps / fps_target)))

    frames: list[types.Part] = []
    idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            h, w = image.shape[:2]
            if w > max_width:
                scale = max_width / w
                image = cv2.resize(image, (max_width, int(h * scale)))
            ok_enc, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok_enc:
                raise RuntimeError("Failed to encode frame")
            frames.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
        idx += 1

    cap.release()
    return frames

# Asking Gemini to classify the frames into front, back, or side.
def classify_batch(client: genai.Client, parts: list[types.Part], model: str) -> list[str]:
    n = len(parts)
    prompt = (
        "Classify each image of a person who may be rotating.\n"
        "For EVERY image return exactly one of:\n"
        '- "front": face toward the camera (even if body is twisted)\n'
        '- "back": back of head/shirt toward the camera\n'
        '- "side": left/right profile\n'
        "If the face is aimed at the camera, prefer front over side.\n"
        f"There are {n} images in order.\n"
        f'Return JSON {{"orientations":[...]}} with exactly {n} labels.'
    )

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt] + parts,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=BatchLabels,
                    max_output_tokens=max(96, n * 12),
                ),
            )
            labels = [str(x).strip().lower() for x in json.loads(response.text)["orientations"]]
            if len(labels) != n:
                raise ValueError(f"Expected {n} labels, got {len(labels)}")

            out = []
            for label in labels:
                if label in {"left", "right", "profile"}:
                    label = "side"
                if label not in {"front", "back", "side"}:
                    raise ValueError(f"Bad label: {label}")
                out.append(label)
            return out
        except errors.ClientError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(16 * (attempt + 1))
        except (ValueError, json.JSONDecodeError, KeyError, TypeError):
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("Classification failed")

# Classifying all frames in small batches (large dumps under-sample temporally).
def classify_all(
    client: genai.Client,
    parts: list[types.Part],
    model: str,
    batch_size: int = 6,
) -> list[str]:
    """Classify all frames in small batches (large dumps under-sample temporally)."""
    labels: list[str] = []
    for start in range(0, len(parts), batch_size):
        batch = parts[start : start + batch_size]
        print(f"[*] Classifying frames {start}-{start + len(batch) - 1}...")
        labels.extend(classify_batch(client, batch, model))
        if start + batch_size < len(parts):
            time.sleep(1.0)
    return labels

# Counting the full rotations.
def count_rotations(orientations: list[str]) -> int:
    """
    Count full rotations only.

    A full rotation = leave front, see back, return to front.
    Ending on back after a half-turn (1.5) does not count as another rotation.
    """
    if len(orientations) < 3:
        return 0

    count = 0
    left_front = False
    seen_back = False
    ever_front = orientations[0] == "front"
    prev = orientations[0]

    for current in orientations[1:]:
        if current == prev:
            continue  # ignore repeats while lingering on same pose

        if prev == "front" and current != "front":
            left_front = True
            seen_back = current == "back"
        elif current == "back":
            seen_back = True
        elif current == "front":
            # First arrival at front (e.g. started on back) is not a completed lap.
            if ever_front and left_front and seen_back:
                count += 1
            ever_front = True
            left_front = False
            seen_back = False

        prev = current

    return count

# Analyzing the rotations in the video.
def analyze_rotations(video_path: str) -> int:
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    print(f"[*] Video: {video_path}")
    frames = extract_frames(video_path)
    print(f"[*] Extracted {len(frames)} frames")

    orientations = classify_all(client, frames, model)
    print(f"[*] Labels: {orientations}")

    rotations = count_rotations(orientations)
    return rotations

# Running the script
if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "videos/5rotationsTest.mp4"
    result = analyze_rotations(video)
    print(f"\n>>> FINAL ROTATION COUNT: {result} <<<")
    print(result)
