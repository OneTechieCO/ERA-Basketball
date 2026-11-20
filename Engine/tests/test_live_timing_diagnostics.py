"""Diagnostic timing of the live possession rules. The live engine is not modified."""
import unittest

from era_engine import core
from era_engine.live_timing_diagnostics import (
    APPROXIMATIONS,
    PROMOTION,
    diagnose_game,
    fictional_matchup,
    intelligent_pass_chance,
    offensive_rebound_reset,
    probability_make,
    run_live_timing,
    shot_weights,
    timing_gaps,
)
from era_engine.seed import ERAS


class LiveTimingDiagnosticTests(unittest.TestCase):
    def test_decisions_are_the_live_functions(self):
        self.assertIs(intelligent_pass_chance, core.intelligent_pass_chance)
        self.assertIs(probability_make, core.probability_make)
        self.assertIs(shot_weights, core.shot_weights)
        self.assertIs(offensive_rebound_reset, core.offensive_rebound_reset)
        self.assertFalse(PROMOTION["promoted"])
        self.assertIn("between_possessions", APPROXIMATIONS)
        self.assertIn("free_throws", APPROXIMATIONS)

    def test_one_game_keeps_the_clocks_separate_and_non_negative(self):
        home, away, roster = fictional_matchup()
        game = diagnose_game(ERAS[-1], 7, home, away, roster)
        again = diagnose_game(ERAS[-1], 7, home, away, roster)
        self.assertEqual(game["possessions"], again["possessions"])
        self.assertEqual(game["durations"], again["durations"])
        self.assertIsNone(game["pace_input"])
        self.assertEqual(game["regulation_seconds"], 2880)
        self.assertEqual(game["quarter_lengths"], [720, 720, 720, 720])
        self.assertEqual(game["period_breaks"], 3)
        self.assertEqual(game["period_ending_possessions"], 4)
        self.assertAlmostEqual(sum(game["categories"].values()), 2880, places=4)
        self.assertTrue(all(value >= 0 for value in game["durations"]))
        self.assertEqual(game["roster"], "fictional_demo")
        continued = [possession for possession in game["possession_log"] if possession["continued_after_offensive_rebound"]]
        self.assertGreater(len(continued), 0)
        self.assertTrue(any(
            possession["game_clock_elapsed"]
            > possession["shot_clock_limit"] - possession["shot_clock_remaining"] + 1.0
            for possession in continued
        ))
        fouls = [possession for possession in game["possession_log"] if possession["end_reason"] == "shooting_foul"]
        self.assertGreater(len(fouls), 0)
        self.assertTrue(all(possession["dead_ball_seconds"] == 0.0 for possession in fouls))
        self.assertTrue(all(possession["between_possession_seconds"] == 0.0 for possession in game["possession_log"]))

    def test_five_eras_are_measured_without_using_pace_as_the_possession_count(self):
        rows = run_live_timing(ERAS)
        self.assertEqual([row["era"] for row in rows], [era.key for era in ERAS])
        for row, era in zip(rows, ERAS):
            self.assertEqual(row["seeded_pace"], era.pace)
            self.assertIsNone(row["pace_input"])
            self.assertGreater(row["possessions_per_team_mean"], 40)
            self.assertNotEqual(round(row["possessions_per_team_mean"]), round(era.pace))
            self.assertEqual(row["quarter_ending_possessions_per_game"], 4)
            self.assertGreater(row["passes_per_possession"], 0)
            self.assertAlmostEqual(row["offensive_rebound_rate"], 0.24, delta=0.05)
            self.assertAlmostEqual(sum(row["category_seconds_per_game"].values()), 2880, places=3)
            gaps = timing_gaps(row)
            self.assertTrue(gaps["clock_accounts_for_the_period"])
            self.assertEqual(gaps["between_possession_seconds"], 0.0)
            self.assertEqual(gaps["dead_ball_seconds"], 0.0)
            self.assertGreater(gaps["setup_seconds"], 0)
            self.assertGreater(gaps["shot_clock_left_at_release"], 0)


if __name__ == "__main__":
    unittest.main()
