# Shared prompts + parallel classify / open-lap recheck (any provider).

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rotation_counter.count import has_open_lap

ALLOWED = frozenset({"front", "back", "side"})

BATCH_PROMPT = (
    "Classify each image of a person who may be rotating.\n"
    "Decide using the HEAD/FACE first, not the shoulders or lean.\n"
    "For EVERY image return exactly one of:\n"
    '- "front": face is visible and oriented toward the camera '
    "(eyes/nose toward lens). Use front even if shoulders are twisted, "
    "the person is leaning, or arms are uneven.\n"
    '- "back": back of the head / no face visible\n'
    '- "side": clear profile — nose points left or right of the frame, '
    "and the face is NOT toward the camera\n"
    "Rule: if you can see the face looking at the camera, answer front.\n"
    "There are {n} images in order.\n"
    'Return JSON {{"orientations":[...]}} with exactly {n} labels.'
)

RECHECK_PROMPT = (
    "Look at the person's HEAD only.\n"
    "Can you see their face looking toward the camera?\n"
    '- If yes (eyes/face toward lens): "front"\n'
    '- If you mainly see the back of the head: "back"\n'
    '- If it is a pure left/right profile with nose pointing sideways: "side"\n'
    "Ignore shoulder angle, lean, and arm position.\n"
    'Return JSON {"orientations":["front|back|side"]} with exactly 1 label.'
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "orientations": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(ALLOWED)},
        }
    },
    "required": ["orientations"],
    "additionalProperties": False,
}


# Normalize/validate model labels; map left/right/profile → side.
def normalize_labels(raw: list, expected: int) -> list[str]:
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} labels, got {len(raw)}")
    out: list[str] = []
    for item in raw:
        label = str(item).strip().lower()
        if label in {"left", "right", "profile"}:
            label = "side"
        if label not in ALLOWED:
            raise ValueError(f"Bad label: {label}")
        out.append(label)
    return out


# Parallel worker count from CLASSIFY_WORKERS (default 8).
def _workers(default: int = 8) -> int:
    raw = os.getenv("CLASSIFY_WORKERS")
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


# Classify every frame alone in parallel (face-first).
def classify_parallel(
    frames: list[bytes],
    provider,
    model: str,
    *,
    workers: int | None = None,
) -> list[str]:
    if not frames:
        return []

    n_workers = workers or _workers()
    print(
        f"[*] Classifying {len(frames)} frames in parallel "
        f"(workers={n_workers}, face-first)..."
    )

    labels: list[str | None] = [None] * len(frames)

    def one(index: int) -> tuple[int, str]:
        label = provider.classify_batch(
            [frames[index]], model, prompt=RECHECK_PROMPT
        )[0]
        return index, label

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(one, i) for i in range(len(frames))]
        done = 0
        for fut in as_completed(futures):
            index, label = fut.result()
            labels[index] = label
            done += 1
            print(f"    [{done}/{len(frames)}] frame {index}: {label}")

    if any(label is None for label in labels):
        raise RuntimeError("Parallel classify missed one or more frames")
    return list(labels)  # type: ignore[arg-type]


# Sequential batched classify (cheaper API usage, less precise).
def classify_batched(
    frames: list[bytes],
    provider,
    model: str,
    *,
    batch_size: int | None = None,
) -> list[str]:
    if not frames:
        return []

    size = batch_size or provider.batch_size
    labels: list[str] = []
    for start in range(0, len(frames), size):
        batch = frames[start : start + size]
        print(f"[*] Classifying frames {start}-{start + len(batch) - 1}...")
        labels.extend(provider.classify_batch(batch, model))
        if start + size < len(frames):
            time.sleep(provider.pause_seconds)
    return labels


# If a lap is open and the clip does not end on back, re-check the last
# few non-back frames (in parallel).
def close_open_lap(
    frames: list[bytes],
    orientations: list[str],
    provider,
    model: str,
    lookback: int = 4,
) -> list[str]:
    if not has_open_lap(orientations):
        return orientations
    if orientations[-1] == "back":
        print("[*] Open lap ends on back — leaving incomplete.")
        return orientations

    updated = list(orientations)
    start = max(0, len(frames) - lookback)
    indices = [i for i in range(start, len(frames)) if updated[i] != "back"]
    if not indices:
        return updated

    print(f"[*] Open lap — parallel face-first recheck frames {indices}...")

    def one(index: int) -> tuple[int, str]:
        label = provider.classify_batch(
            [frames[index]], model, prompt=RECHECK_PROMPT
        )[0]
        return index, label

    with ThreadPoolExecutor(max_workers=min(_workers(), len(indices))) as pool:
        for fut in as_completed([pool.submit(one, i) for i in indices]):
            index, label = fut.result()
            if label != updated[index]:
                print(f"    frame {index}: {updated[index]} -> {label}")
                updated[index] = label

    return updated
