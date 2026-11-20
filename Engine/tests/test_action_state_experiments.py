"""Variant B2. Actions can change the next look. The live make formula stays put."""
import unittest
from pathlib import Path

from era_engine.action_state_experiments import (
    FITTED_TO_HELD_OUT,
    OPENNESS_ASSUMPTIONS,
    PROMOTION,
    SAFEGUARD_ACTIONS,
    apply_action_effect,
    b2_continue_chance,
    b2_opportunity_report,
    compare_models,
    diagnose_game,
    historical_comparison,
    live_openness,
    screen_success_chance,
    shot_openness,
    variant_b_endings,
)
from era_engine.core import Player, clamp, probability_make
from era_engine.live_timing_diagnostics import fictional_matchup
from era_engine.playbyplay_timing import ROOT, sample_summary
from era_engine.seed import ERAS, demo_players


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


class ActionStateTests(unittest.TestCase):
    def test_variant_b_stops_at_the_cap_without_changing_the_shot(self):
        row = compare_models([ERAS[-2]], seeds=(7, 19))[0]
        endings = variant_b_endings(row["B"])
        self.assertGreater(endings["cap_share"], 0.25)
        self.assertEqual(endings["chain_share"][4], 0.0)
        self.assertGreater(endings["favorable_share"], endings["urgent_share"])
        self.assertFalse(endings["actions_change_shot_input"])
        self.assertLess(abs(endings["mean_look_change"]), 0.05)

    def test_actions_change_openness_and_the_make_formula_stays_live(self):
        shooter = _player(iq=92, passing=88)
        defender = _player(player_id="D", defense=95, iq=78)
        opened, press = apply_action_effect("screen_action", shooter, defender, 0.0, 0.4, True)
        closed, heavier = apply_action_effect("screen_action", shooter, defender, 0.0, 0.4, False)
        self.assertAlmostEqual(opened - 0.0, OPENNESS_ASSUMPTIONS["screen_success"])
        self.assertLess(closed, opened)
        self.assertGreater(heavier, press)
        self.assertEqual(shot_openness(shooter, defender, None, 0.0), clamp(live_openness(shooter, defender, None), -0.8, 1.0))
        self.assertGreater(shot_openness(shooter, defender, None, 0.22), shot_openness(shooter, defender, None, 0.0))
        self.assertIs(probability_make, probability_make)
        self.assertFalse(OPENNESS_ASSUMPTIONS["changes_make_formula"])
        self.assertGreater(screen_success_chance(shooter, _player(player_id="W", defense=70)), screen_success_chance(shooter, defender))
        smart = _player(iq=95, usage=0.33)
        self.assertLess(b2_continue_chance(smart, defender, 18, 0.22, 0.0, 1), b2_continue_chance(smart, defender, 18, 0.0, 0.4, 1))
        self.assertEqual(b2_continue_chance(smart, defender, 8, 0.22, 0.0, 1), 0.0)
        self.assertGreater(b2_continue_chance(smart, defender, 16, 0.0, 0.4, 3), 0.0)
        self.assertEqual(b2_continue_chance(smart, defender, 16, 0.0, 0.4, SAFEGUARD_ACTIONS), 0.0)
        self.assertGreater(SAFEGUARD_ACTIONS, 3)

    def test_b2_games_are_not_fitted_and_can_run_past_three_actions(self):
        home, away, roster = fictional_matchup()
        era = ERAS[-2]
        game = diagnose_game(era, 7, home, away, roster)
        self.assertEqual(game["durations"], diagnose_game(era, 7, home, away, roster)["durations"])
        self.assertAlmostEqual(sum(game["durations"]), 2880, places=3)
        self.assertFalse(game["fitted_to_held_out"])
        rows = compare_models(ERAS, seeds=(7, 19))
        sample = rows[-2]
        report = b2_opportunity_report(sample["B2"])
        self.assertGreater(report["chain_share"][4], 0.0)
        self.assertLess(report["safeguard_share"], 0.05)
        self.assertEqual(report["make_formula"], "probability_make")
        self.assertGreater(sample["B2"]["transition_early_finish_share"], 0.5)
        self.assertLess(max(player["attempt_share"] for player in sample["B2"]["shot_distribution"].values()), 0.5)
        history = historical_comparison(sample["B2"])
        self.assertFalse(history["parameters_fit_to_validation"])
        self.assertFalse(history["fictional_roster_calibrated"])
        self.assertGreater(history["excluded_exploration"], 0)
        exploration = set(sample_summary(ROOT)["game_ids"])
        validation = set(sample_summary(Path(__file__).resolve().parents[1] / "research_playbyplay" / "validation")["game_ids"])
        self.assertTrue(exploration.isdisjoint(validation))
        self.assertTrue(all(row["observed_2015_16_applied"] is False for row in rows))
        self.assertFalse(FITTED_TO_HELD_OUT)
        self.assertFalse(PROMOTION["promoted"])


if __name__ == "__main__":
    unittest.main()
