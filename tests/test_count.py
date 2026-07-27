# Unit tests for the checkpoint counter (no API calls).

from __future__ import annotations

import unittest

from rotation_counter.count import has_open_lap, measure_spin


class TestCountRotations(unittest.TestCase):
    def test_three_full(self) -> None:
        seq = ["front"]
        for _ in range(3):
            seq += ["side", "back", "side", "front"]
        spin = measure_spin(seq)
        self.assertEqual(spin.full_rotations, 3)
        self.assertEqual(spin.degrees, 1080)
        self.assertFalse(spin.open_lap)

    def test_one_point_five_ends_back(self) -> None:
        seq = ["front", "side", "back", "side", "front", "side", "back"]
        spin = measure_spin(seq)
        self.assertEqual(spin.full_rotations, 1)
        self.assertTrue(spin.open_lap)
        self.assertTrue(has_open_lap(seq))
        self.assertEqual(spin.degrees, 540)

    def test_one_full_is_360(self) -> None:
        seq = ["front", "side", "back", "side", "front"]
        spin = measure_spin(seq)
        self.assertEqual(spin.full_rotations, 1)
        self.assertEqual(spin.degrees, 360)

    # Gemini often misses the opening front (labels side instead).
    # side → back → side → front should still count as 1 full.
    def test_missed_opening_front_still_counts(self) -> None:
        seq = [
            "side",
            "side",
            "side",
            "side",
            "back",
            "back",
            "side",
            "side",
            "front",
            "side",
            "side",
            "side",
            "back",
            "back",
        ]
        spin = measure_spin(seq)
        self.assertEqual(spin.full_rotations, 1)
        self.assertEqual(spin.degrees, 540)
        self.assertTrue(spin.open_lap)

    def test_lead_in_then_two_fronts(self) -> None:
        seq = [
            "side",
            "back",
            "side",
            "front",
            "side",
            "back",
            "side",
            "front",
        ]
        spin = measure_spin(seq)
        self.assertEqual(spin.full_rotations, 2)
        self.assertEqual(spin.degrees, 720)

    def test_open_side_not_counted_as_full(self) -> None:
        seq = ["front", "side", "back", "side", "front", "side", "back", "side"]
        spin = measure_spin(seq)
        self.assertEqual(spin.full_rotations, 1)
        self.assertEqual(spin.degrees, 630)

    def test_opening_front_not_counted(self) -> None:
        spin = measure_spin(["front", "side", "back"])
        self.assertEqual(spin.full_rotations, 0)
        self.assertEqual(spin.degrees, 180)
        self.assertTrue(spin.open_lap)


if __name__ == "__main__":
    unittest.main()
