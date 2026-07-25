"""Unit tests for the checkpoint counter (no API calls)."""

from __future__ import annotations

import unittest

from rotation_counter.count import count_rotations, has_open_lap


class TestCountRotations(unittest.TestCase):
    def test_three_full(self) -> None:
        seq = ["front"]
        for _ in range(3):
            seq += ["side", "back", "side", "front"]
        self.assertEqual(count_rotations(seq), 3)

    def test_one_point_five_ends_back(self) -> None:
        seq = ["front", "side", "back", "side", "front", "side", "back"]
        self.assertEqual(count_rotations(seq), 1)
        self.assertTrue(has_open_lap(seq))

    def test_open_side_not_counted(self) -> None:
        # Must NOT invent a completion from trailing side alone.
        seq = ["front", "side", "back", "side", "front", "side", "back", "side"]
        self.assertEqual(count_rotations(seq), 1)

    def test_opening_front_not_counted(self) -> None:
        self.assertEqual(count_rotations(["front", "side", "back"]), 0)


if __name__ == "__main__":
    unittest.main()
