# Deterministic full-rotation counting from front/back/side labels.

from __future__ import annotations

from dataclasses import dataclass

# Facing angle relative to camera (skateboard-style yaw bins).
# "side" is a quarter-turn; we don't distinguish left vs right.
FACING_DEGREES = {
    "front": 0,
    "side": 90,
    "back": 180,
}


# Camera-relative facing for one label (0 / 90 / 180).
def facing_degrees(label: str) -> int:
    return FACING_DEGREES[label]


# Collapse consecutive duplicate labels.
def dedupe_runs(orientations: list[str]) -> list[str]:
    if not orientations:
        return []
    out = [orientations[0]]
    for label in orientations[1:]:
        if label != out[-1]:
            out.append(label)
    return out


# Forward spin steps (does not reset after 360).
_FORWARD_STEP = {
    ("front", "side"): 90,
    ("side", "back"): 90,
    ("back", "side"): 90,
    ("side", "front"): 90,
    ("front", "back"): 180,
    ("back", "front"): 180,
}


# Unwrapped yaw per frame: 0 → 90 → 180 → 270 → 360 → 450 → 540 …
# Keeps climbing across laps instead of wrapping back to 0° at each front.
def cumulative_degrees(orientations: list[str]) -> list[int]:
    if not orientations:
        return []

    total = FACING_DEGREES[orientations[0]]
    out = [total]
    prev = orientations[0]

    for label in orientations[1:]:
        if label == prev:
            out.append(total)
            continue
        total += _FORWARD_STEP.get((prev, label), 90)
        out.append(total)
        prev = label

    return out


# Full rotation = leave front, see back, return to / pass front.
# Only a front label closes a lap. Ending on side or back does not count.
def count_rotations(orientations: list[str]) -> int:
    if len(orientations) < 3:
        return 0

    count = 0
    left_front = False
    seen_back = False
    ever_front = orientations[0] == "front"
    prev = orientations[0]

    for current in orientations[1:]:
        if current == prev:
            continue

        if prev == "front" and current != "front":
            left_front = True
            seen_back = current == "back"
        elif current == "back":
            seen_back = True
        elif current == "front":
            if ever_front and left_front and seen_back:
                count += 1
            ever_front = True
            left_front = False
            seen_back = False

        prev = current

    return count


# True if we left front, saw back, and never returned to front.
def has_open_lap(orientations: list[str]) -> bool:
    left_front = False
    seen_back = False
    in_front = bool(orientations) and orientations[0] == "front"
    prev = orientations[0] if orientations else None

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
            left_front = False
            seen_back = False
            in_front = True
        prev = current

    return left_front and seen_back and not in_front


# Full laps + skateboard-style total degrees (180 / 360 / 540 / ...).
@dataclass(frozen=True)
class SpinResult:
    full_rotations: int
    degrees: int
    open_lap: bool

    # Exact degree total as a skateboard-style name (no rounding up).
    @property
    def trick(self) -> str:
        return str(self.degrees)


# Degrees from the counter, not raw label-path length.
# full_rotations * 360
# + 180 if an open lap ends on back
# + 270 if an open lap ends on side (past back, not yet front)
# This avoids counting "lead-in" frames before the first established front
# as extra degrees (which wrongly turned 1 full into 720°).
def measure_spin(orientations: list[str]) -> SpinResult:
    full = count_rotations(orientations)
    open_lap = has_open_lap(orientations)
    degrees = full * 360

    if open_lap:
        last = dedupe_runs(orientations)[-1]
        if last == "back":
            degrees += 180
        elif last == "side":
            degrees += 270

    return SpinResult(
        full_rotations=full,
        degrees=degrees,
        open_lap=open_lap,
    )
