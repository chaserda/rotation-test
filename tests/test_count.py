# Unit tests for the checkpoint counter (no API calls).

from __future__ import annotations

import unittest

from rotation_counter.count import (
    count_rotations,
    cumulative_degrees,
    has_open_lap,
    measure_spin,
)


class TestCountRotations(unittest.TestCase):
    def test_three_full(self) -> None:
        seq = ["front"]
        for _ in range(3):
            seq += ["side", "back", "side", "front"]
        self.assertEqual(count_rotations(seq), 3)
        spin = measure_spin(seq)
        self.assertEqual(spin.degrees, 1080)
        self.assertEqual(spin.trick, "1080")

    def test_one_point_five_ends_back(self) -> None:
        seq = ["front", "side", "back", "side", "front", "side", "back"]
        self.assertEqual(count_rotations(seq), 1)
        self.assertTrue(has_open_lap(seq))
        spin = measure_spin(seq)
        self.assertEqual(spin.degrees, 540)
        self.assertEqual(spin.trick, "540")

    def test_one_full_is_360_not_720(self) -> None:
        seq = ["front", "side", "back", "side", "front"]
        self.assertEqual(count_rotations(seq), 1)
        spin = measure_spin(seq)
        self.assertEqual(spin.degrees, 360)
        self.assertEqual(spin.trick, "360")

    # Path length looks like 720°, but only one counted full rotation.
    # First front establishes facing; second front completes 1 full.
    # Degrees must follow the counter (360), not raw path length.
    def test_lead_in_before_first_front_not_extra_degrees(self) -> None:
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
        self.assertEqual(count_rotations(seq), 1)
        self.assertEqual(cumulative_degrees(seq)[-1], 720)
        spin = measure_spin(seq)
        self.assertEqual(spin.degrees, 360)

    # Open lap ending on side: 360 + 270 = 630 (do not round up to 720).
    def test_open_side_not_counted_as_full(self) -> None:
        seq = ["front", "side", "back", "side", "front", "side", "back", "side"]
        self.assertEqual(count_rotations(seq), 1)
        spin = measure_spin(seq)
        self.assertEqual(spin.degrees, 630)
        self.assertEqual(spin.trick, "630")

    def test_opening_front_not_counted(self) -> None:
        self.assertEqual(count_rotations(["front", "side", "back"]), 0)
        spin = measure_spin(["front", "side", "back"])
        self.assertEqual(spin.degrees, 180)
        self.assertEqual(spin.trick, "180")


if __name__ == "__main__":
    unittest.main()
