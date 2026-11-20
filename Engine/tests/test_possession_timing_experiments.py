"""Clock-phase experiment. The live engine is not modified."""
import unittest

from era_engine.game_clock_experiments import REGULATION_SECONDS, GameClock, ShotClock, setup_seconds
from era_engine.possession_timing_experiments import (
    NO_PLAY_BY_PLAY,
    OREB_RECOVERY_SECONDS,
    PROMOTION,
    SETUP_AUDIT,
    TRANSITION_PUSH_SECONDS,
    begin_possession,
    compare_with_live_diagnostic,
    diagnose_game,
    fictional_matchup,
    recover_offensive_rebound,
)
from era_engine.seed import ERAS


class PossessionClockRuleTests(unittest.TestCase):
    def test_a_throw_in_does_not_run_and_the_advance_is_not_stacked(self):
        game = GameClock()
        shot = ShotClock()
        before = game.quarter_remaining
        started = begin_possession(game, shot, "made_basket_inbound")
        self.assertEqual(started["stopped"]["game_clock_elapsed"], 0.0)
        self.assertEqual(started["running"]["kind"], "half_court_advance")
        self.assertEqual(started["running"]["game_clock_elapsed"], setup_seconds(24))
        self.assertEqual(game.quarter_remaining, before - setup_seconds(24))
        self.assertEqual(shot.remaining, 24 - setup_seconds(24))
        self.assertNotIn("transition_push", (started["running"]["kind"],))

        game = GameClock()
        shot = ShotClock()
        transition = begin_possession(game, shot, "transition")
        self.assertIsNone(transition["stopped"])
        self.assertEqual(transition["running"]["kind"], "transition_push")
        self.assertEqual(transition["running"]["game_clock_elapsed"], TRANSITION_PUSH_SECONDS)
        self.assertNotEqual(transition["running"]["kind"], "half_court_advance")
        self.assertIn("double_count_if_stacked", SETUP_AUDIT)
        self.assertIn("Play-by-play", NO_PLAY_BY_PLAY)
        self.assertFalse(PROMOTION["promoted"])

    def test_offensive_rebound_moves_only_the_game_clock_and_then_resets_the_shot_clock(self):
        game = GameClock()
        shot = ShotClock()
        shot.remaining = 4
        elapsed_before = game.elapsed
        recovery = recover_offensive_rebound(game, shot, 14)
        self.assertEqual(recovery["shot_clock_elapsed"], 0.0)
        self.assertEqual(recovery["game_clock_elapsed"], OREB_RECOVERY_SECONDS)
        self.assertFalse(recovery["game_clock_moved_backward"])
        self.assertEqual(game.elapsed, elapsed_before + OREB_RECOVERY_SECONDS)
        self.assertEqual(shot.remaining, 14)
        self.assertTrue(recovery["continued"])

        game.quarter_remaining = 0.4
        shot.remaining = 9
        cut = recover_offensive_rebound(game, shot, 14)
        self.assertFalse(cut["continued"])
        self.assertEqual(game.quarter_remaining, 0.0)
        self.assertEqual(shot.remaining, 9)
        self.assertTrue(game.start_next_quarter())
        self.assertEqual(game.quarter_remaining, 720)

    def test_a_period_cuts_the_advance_and_does_not_keep_the_unused_time(self):
        game = GameClock()
        shot = ShotClock()
        game.quarter_remaining = 2.0
        started = begin_possession(game, shot, "quarter_throw_in")
        self.assertFalse(started["running"]["finished"])
        self.assertEqual(started["running"]["reason"], "period_expired")
        self.assertEqual(started["running"]["game_clock_elapsed"], 2.0)
        self.assertEqual(shot.remaining, 22.0)
        self.assertEqual(game.quarter_remaining, 0.0)
        self.assertTrue(game.start_next_quarter())
        self.assertEqual(game.quarter_remaining, 720)


class PossessionTimingResultTests(unittest.TestCase):
    def test_a_game_does_not_double_count_bring_up_time(self):
        home, away, roster = fictional_matchup()
        game = diagnose_game(ERAS[0], 7, home, away, roster)
        again = diagnose_game(ERAS[0], 7, home, away, roster)
        self.assertEqual(game["durations"], again["durations"])
        self.assertIsNone(game["pace_input"])
        self.assertEqual(game["quarter_lengths"], [720, 720, 720, 720])
        self.assertEqual(game["period_breaks"], 3)
        self.assertEqual(game["period_ending_possessions"], 4)
        self.assertAlmostEqual(sum(game["durations"]), REGULATION_SECONDS, places=3)
        modes = {possession["start_mode"] for possession in game["possession_log"]}
        self.assertIn("transition", modes)
        self.assertIn("made_basket_inbound", modes)
        self.assertEqual(game["possession_log"][0]["start_mode"], "opening_jump")
        for possession in game["possession_log"]:
            categories = possession["categories"]
            self.assertFalse(categories.get("half_court_advance") and categories.get("transition_push"))
            self.assertNotIn("setup", categories)
            self.assertGreaterEqual(possession["game_clock_elapsed"], 0.0)
            self.assertEqual(possession["stopped_running_seconds"], 0.0)
            if possession["start_mode"] == "transition":
                self.assertNotIn("half_court_advance", categories)
            if possession["start_mode"] == "made_basket_inbound":
                self.assertNotIn("transition_push", categories)
                self.assertEqual(possession["counts"].get("stopped_throw_in"), 1)

    def test_five_eras_stay_away_from_a_fitted_pace(self):
        rows = compare_with_live_diagnostic(ERAS)
        self.assertEqual([row["era"] for row in rows], [era.key for era in ERAS])
        for row, era in zip(rows, ERAS):
            self.assertIsNone(row["pace_input"])
            self.assertFalse(row["promoted"])
            self.assertGreater(row["possessions_per_team_mean"], 40)
            self.assertNotEqual(round(row["possessions_per_team_mean"]), round(era.pace))
            self.assertEqual(row["quarter_endings_per_game"], 4)
            self.assertGreater(row["transition_possession_share"], 0.0)
            self.assertLess(row["transition_possession_share"], 1.0)
            self.assertGreater(row["transition_shortening_seconds"], 0.0)
            self.assertGreater(row["shot_clock_left_at_release"], 8.0)
            self.assertGreater(row["median_seconds"], 0.0)
            self.assertIn("transition", row["mean_seconds_by_start"])
            self.assertIn("made_basket_inbound", row["mean_seconds_by_start"])


if __name__ == "__main__":
    unittest.main()
