import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine import core_shot_clock_candidate as candidate
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


class ShotClockTests(unittest.TestCase):

    def test_shot_clock_configuration(self):
        era = candidate.Era("2023-24", 99, 0.4, 0.5, True)
        self.assertEqual(era.shot_clock, 24)
        self.assertEqual(replace(era, shot_clock=8).shot_clock, 8)
        for value in (0, -1, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(era, shot_clock=value)

    def test_clock_countdown_and_expiration(self):
        clock = candidate.PossessionClock(24)
        self.assertFalse(clock.advance(6))
        self.assertEqual(clock.remaining, 18)
        self.assertTrue(clock.advance(18))
        self.assertEqual(clock.remaining, 0)
        self.assertTrue(clock.advance(5))
        self.assertEqual(clock.remaining, 0)

    def test_negative_elapsed_time_rejected(self):
        clock = candidate.PossessionClock(24)
        with self.assertRaises(ValueError):
            clock.advance(-1)
        self.assertEqual(clock.remaining, 24)

    def test_clock_reset(self):
        clock = candidate.PossessionClock(24)
        clock.advance(20)
        clock.reset(14)
        self.assertEqual(clock.remaining, 14)
        self.assertFalse(clock.advance(4))
        self.assertEqual(clock.remaining, 10)

    def test_invalid_clock_reset_rejected(self):
        clock = candidate.PossessionClock(24)
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    clock.reset(value)
            self.assertEqual(clock.remaining, 24)

    def test_offensive_rebound_reset_rule(self):
        era = candidate.Era("2017-18", 99, 0.4, 0.5, True)
        self.assertEqual(candidate.offensive_rebound_reset(era), 24)
        self.assertEqual(
            candidate.offensive_rebound_reset(
                replace(era, key="2018-19")
            ), 14
        )
        self.assertEqual(
            candidate.offensive_rebound_reset(
                replace(era, key="2023-24", shot_clock=8)
            ), 8
        )

    def test_experimental_game_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            con = connect(str(Path(directory) / "test.db"))
            try:
                initialize_demo(con)
                players, eras, teams = load(con)
                a = teams["Bay City Legends"]
                b = teams["Chicago Stampede"]
                era = replace(eras["1995-96"], shot_clock=8)
                first = candidate.simulate_game(
                    a, b, players, era, 19
                )
                second = candidate.simulate_game(
                    a, b, players, era, 19
                )
                self.assertEqual(first, second)
            finally:
                con.close()


if __name__ == "__main__":
    unittest.main()
