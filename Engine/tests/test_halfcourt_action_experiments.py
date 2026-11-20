"""Half-court action layer. The live shoot-or-pass formula stays in core.py."""
import unittest
from pathlib import Path

from era_engine.core import Player, intelligent_pass_chance
from era_engine.halfcourt_action_experiments import (
    LATE_CLOCK_NO_PREP_SECONDS,
    OBSERVED_CONTEXT,
    PROMOTION,
    UNCALIBRATED_ACTION_SECONDS,
    action_duration_sensitivity,
    diagnose_game,
    fictional_matchup,
    preparatory_action_chance,
    run_halfcourt,
)
from era_engine.playbyplay_timing import ROOT, sample_summary
from era_engine.seed import ERAS


def _player(**overrides):
    values = {
        "player_id": "T",
        "name": "Test",
        "origin_era": "2010s",
        "offense": 80,
        "defense": 70,
        "passing": 70,
        "rebounding": 70,
        "iq": 70,
        "stamina": 80,
        "usage": 0.19,
        "shot_skills": {"rim": 75, "paint": 75, "mid": 75, "three": 75},
        "tendencies": {"rim": 0.25, "paint": 0.25, "mid": 0.25, "three": 0.25},
    }
    values.update(overrides)
    return Player(**values)


class HalfcourtActionTests(unittest.TestCase):
    def test_preparatory_layer_is_separate_from_the_live_decision(self):
        shooter = _player(iq=95, passing=70, shot_skills={"rim": 90, "paint": 90, "mid": 90, "three": 90})
        creator = _player(player_id="C", iq=70, passing=95)
        pressured = _player(player_id="D", defense=95)
        ordinary = _player(player_id="O", defense=70)
        self.assertEqual(preparatory_action_chance(shooter, ordinary, LATE_CLOCK_NO_PREP_SECONDS), 0.0)
        self.assertGreater(
            preparatory_action_chance(creator, pressured, 18),
            preparatory_action_chance(creator, ordinary, 18),
        )
        self.assertLess(
            preparatory_action_chance(shooter, ordinary, 18),
            preparatory_action_chance(_player(), ordinary, 18),
        )
        self.assertIsNot(preparatory_action_chance, intelligent_pass_chance)
        self.assertFalse(OBSERVED_CONTEXT["used_as_a_duration"])
        self.assertTrue(all(value == 1.0 for value in UNCALIBRATED_ACTION_SECONDS.values()))
        self.assertFalse(PROMOTION["promoted"])

    def test_one_game_does_not_stack_the_bring_up_or_rewind_the_game_clock(self):
        home, away, roster = fictional_matchup()
        game = diagnose_game(ERAS[-1], 7, home, away, roster)
        again = diagnose_game(ERAS[-1], 7, home, away, roster)
        self.assertEqual(game["durations"], again["durations"])
        self.assertIsNone(game["pace_input"])
        self.assertAlmostEqual(sum(game["durations"]), 2880, places=3)
        self.assertEqual(sum(1 for possession in game["possession_log"] if possession["period_ending"]), 4)
        for possession in game["possession_log"]:
            categories = possession["categories"]
            self.assertFalse(categories.get("half_court_advance") and categories.get("transition_push"))
            if possession["start_mode"] == "transition":
                self.assertNotIn("dribble_evaluate", categories)
                self.assertNotIn("hold", categories)
                self.assertNotIn("screen_action", categories)
            self.assertNotIn("throw_in", categories)
        self.assertTrue(any(possession["offensive_rebounds"] > 0 for possession in game["possession_log"]))

    def test_duration_scale_changes_time_without_being_a_pace_target(self):
        rows = action_duration_sensitivity(ERAS[-1], seeds=(7, 19))
        by_scale = {row["action_scale"]: row["mean_seconds"] for row in rows}
        self.assertLess(by_scale[0.5], by_scale[1.0])
        self.assertLess(by_scale[1.0], by_scale[2.0])
        self.assertTrue(all(row["fitted_to_observed_mean"] is False for row in rows))
        self.assertLess(by_scale[2.0] - by_scale[0.5], 6.0)

    def test_five_eras_and_the_held_out_games_stay_separate(self):
        rows = run_halfcourt(ERAS, seeds=(7, 19, 42, 101))
        self.assertEqual([row["era"] for row in rows], [era.key for era in ERAS])
        for row in rows:
            self.assertGreater(row["possessions_per_team_mean"], 40)
            self.assertEqual(row["quarter_endings_per_game"], 4)
            self.assertIsNone(row["pace_input"])
            self.assertGreater(row["p90_seconds"], row["median_seconds"])
        original = set(sample_summary(ROOT)["game_ids"])
        validation_root = Path(__file__).resolve().parents[1] / "research_playbyplay" / "validation"
        validation = sample_summary(validation_root)
        self.assertEqual(validation["role"], "validation")
        self.assertTrue(original.isdisjoint(validation["game_ids"]))
        self.assertEqual(validation["games"], 4)
        self.assertIsNone(validation["pass_duration"])
        self.assertGreater(validation["duration"]["count"], 400)


if __name__ == "__main__":
    unittest.main()
