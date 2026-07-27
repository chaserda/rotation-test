# Deterministic full-rotation counting from front/back/side labels.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpinResult:
    full_rotations: int
    degrees: int
    open_lap: bool


# One FSM pass: full laps + whether a lap is still open + last distinct label.
# Full rotation = leave front → see back → return to front.
# Soft start: if the VLM never labeled an opening front, the first front
# after seeing back still counts as completing a lap (common flash-lite miss).
def analyze_labels(orientations: list[str]) -> tuple[int, bool, str | None]:
    if not orientations:
        return 0, False, None

    count = 0
    left_front = False
    seen_back = False
    ever_front = orientations[0] == "front"
    in_front = ever_front
    prev = orientations[0]

    for current in orientations[1:]:
        if current == prev:
            continue

        if prev == "front" and current != "front":
            left_front = True
            in_front = False
            seen_back = current == "back"
        elif current == "back":
            seen_back = True
            in_front = False
        elif current == "front":
            if seen_back and (left_front or not ever_front):
                count += 1
            ever_front = True
            left_front = False
            seen_back = False
            in_front = True

        prev = current

    open_lap = left_front and seen_back and not in_front
    return count, open_lap, prev


# True if we left front, saw back, and never returned to front.
def has_open_lap(orientations: list[str]) -> bool:
    return analyze_labels(orientations)[1]


# full * 360, plus 180/270 for an open lap ending on back/side.
def measure_spin(orientations: list[str]) -> SpinResult:
    full, open_lap, last = analyze_labels(orientations)
    degrees = full * 360
    if open_lap:
        if last == "back":
            degrees += 180
        elif last == "side":
            degrees += 270
    return SpinResult(full_rotations=full, degrees=degrees, open_lap=open_lap)
