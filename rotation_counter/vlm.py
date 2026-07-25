"""Shared prompts + batch classify / open-lap recheck (any provider)."""

from __future__ import annotations

import time

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


def classify_all(frames: list[bytes], provider, model: str) -> list[str]:
    labels: list[str] = []
    for start in range(0, len(frames), provider.batch_size):
        batch = frames[start : start + provider.batch_size]
        print(f"[*] Classifying frames {start}-{start + len(batch) - 1}...")
        labels.extend(provider.classify_batch(batch, model))
        if start + provider.batch_size < len(frames):
            time.sleep(provider.pause_seconds)
    return labels


def close_open_lap(
    frames: list[bytes],
    orientations: list[str],
    provider,
    model: str,
    lookback: int = 4,
) -> list[str]:
    """Re-check ending frames if a lap is open and not stuck on back."""
    if not has_open_lap(orientations):
        return orientations
    if orientations[-1] == "back":
        print("[*] Open lap ends on back — leaving incomplete.")
        return orientations

    updated = list(orientations)
    start = max(0, len(frames) - lookback)
    print(f"[*] Open lap — face-first recheck frames {start}-{len(frames) - 1}...")

    for i in range(start, len(frames)):
        if updated[i] == "back":
            continue
        try:
            label = provider.classify_batch(
                [frames[i]], model, prompt=RECHECK_PROMPT
            )[0]
        except Exception as e:
            print(f"    frame {i}: recheck failed ({e})")
            continue
        if label != updated[i]:
            print(f"    frame {i}: {updated[i]} -> {label}")
            updated[i] = label
            if label == "front":
                break
        time.sleep(provider.pause_seconds)
    return updated
