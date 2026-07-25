"""Deterministic full-rotation counting from front/back/side labels."""

from __future__ import annotations


def count_rotations(orientations: list[str]) -> int:
    """
    Full rotation = leave front, see back, return to / pass front.

    Only a front label closes a lap. Ending on side or back does not count.
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


def has_open_lap(orientations: list[str]) -> bool:
    """True if we left front, saw back, and never returned to front."""
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
