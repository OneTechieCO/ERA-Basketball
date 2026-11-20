"""Action-selection experiments. The live shot and pass formulas stay in core.py."""
import unittest
from pathlib import Path

from era_engine.action_decision_experiments import (
    APPLIES_2015_16_TO_OTHER_ERAS,
    FITTED_TO_HELD_OUT,
    FAVORABLE_LOOK,
    PROMOTION,
    URGENT_SECONDS,
    baseline_decision_profile,
    compare_models,
    diagnose_game,
    fictional_matchup,
    observed_duration_sample,
    offensive_state,
    validation_report,
    variant_a_action_weights,
    variant_a_continue_chance,
    variant_b_action_weights,
)
from era_engine.core import Player, intelligent_pass_chance
from era_engine.playbyplay_timing import ROOT
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


class ActionDecisionTests(unittest.TestCase):
    def test_baseline_preparatory_chance_is_small_and_not_a_finish_decision(self):
        home, _away, roster = fictional_matchup()
        profile = baseline_decision_profile(roster, home)
        preparatory = [row["preparatory"] for row in profile]
        self.assertLess(sum(preparatory) / len(preparatory), 0.40)
        self.assertTrue(all(row["offered_on_transition"] is False for row in profile))
        self.assertTrue(all(row["offered_once_per_halfcourt_trip"] for row in profile))
        for row in profile:
            self.assertAlmostEqual(row["dribble_given_preparatory"], row["hold_given_preparatory"])
            self.assertAlmostEqual(row["pass_given_finish"] + row["shoot_given_finish"], 1.0)
        self.assertGreater(intelligent_pass_chance(demo_players()[0], demo_players()[6], 0.0, 0, 18), 0.12)

    def test_variant_a_continues_for_pressure_and_stops_for_a_good_look(self):
        ordinary = _player(player_id="O", defense=73)
        lock = _player(player_id="L", defense=95)
        shooter = _player(
            player_id="S", iq=95, usage=0.33, passing=70,
            shot_skills={"rim": 90, "paint": 90, "mid": 90, "three": 90},
            tendencies={"rim": 0.1, "paint": 0.1, "mid": 0.2, "three": 0.6},
        )
        low_iq = _player(player_id="Q", iq=70, shot_skills=shooter.shot_skills, tendencies=shooter.tendencies)
        poor = _player(player_id="P", iq=70, shot_skills={"rim": 55, "paint": 55, "mid": 55, "three": 55})
        poor_smart = _player(player_id="H", iq=95, shot_skills=poor.shot_skills, tendencies=poor.tendencies)
        creator = _player(player_id="C", passing=95, iq=78)
        reader = _player(player_id="R", passing=70, iq=95)
        self.assertEqual(variant_a_continue_chance(shooter, ordinary, URGENT_SECONDS, 0), 0.0)
        self.assertLess(variant_a_continue_chance(shooter, ordinary, 18, 0), variant_a_continue_chance(poor, lock, 18, 0))
        self.assertLess(variant_a_continue_chance(shooter, ordinary, 18, 0), variant_a_continue_chance(low_iq, ordinary, 18, 0))
        self.assertGreater(variant_a_continue_chance(poor_smart, lock, 18, 0), variant_a_continue_chance(poor, lock, 18, 0))
        self.assertGreater(variant_a_action_weights(creator, lock)["screen_action"], variant_a_action_weights(reader, lock)["screen_action"])
        self.assertGreater(variant_a_action_weights(reader, lock)["dribble_evaluate"], variant_a_action_weights(creator, ordinary)["dribble_evaluate"])

    def test_variant_b_states_separate_development_from_a_finish(self):
        players = demo_players()
        star, role, ordinary, lock = players[0], players[1], players[3], players[6]
        self.assertEqual(offensive_state(role, lock, URGENT_SECONDS, 0, None), "urgent")
        self.assertEqual(offensive_state(star, ordinary, 18, 0, None), "favorable")
        self.assertEqual(offensive_state(role, lock, 18, 0, None), "developing")
        self.assertEqual(offensive_state(role, ordinary, 18, 1, "screen_action"), "favorable")
        self.assertEqual(offensive_state(star, lock, 18, 1, "screen_action"), "favorable")
        self.assertEqual(offensive_state(role, lock, 18, 1, "screen_action"), "developing")
        self.assertEqual(offensive_state(role, lock, 18, 3, None), "favorable")
        self.assertNotIn("shoot", variant_b_action_weights(role, lock))
        creator = _player(player_id="C", passing=95)
        holder = _player(player_id="H", passing=70)
        self.assertGreater(variant_b_action_weights(creator, lock)["pass"], variant_b_action_weights(holder, lock)["pass"])
        self.assertGreater(FAVORABLE_LOOK, 0.9)

    def test_games_keep_transition_shots_and_do_not_chase_the_sample(self):
        home, away, roster = fictional_matchup()
        era = ERAS[-2]
        base = compare_models([era], seeds=(7, 19))[0]
        again = diagnose_game(era, 7, home, away, roster, "B")
        self.assertEqual(again["durations"], diagnose_game(era, 7, home, away, roster, "B")["durations"])
        self.assertAlmostEqual(sum(again["durations"]), 2880, places=3)
        self.assertIsNone(again["pace_input"])
        self.assertFalse(again["fitted_to_held_out"])
        self.assertLess(abs(base["A"]["mean_seconds"] - base["baseline"]["mean_seconds"]), 1.5)
        self.assertGreater(base["B"]["sequence_share"], base["baseline"]["sequence_share"])
        self.assertLess(base["B"]["early_release_share"], base["baseline"]["early_release_share"])
        self.assertGreater(base["B"]["transition_early_finish_share"], 0.5)
        self.assertLess(base["B"]["mean_seconds"], 14.0)
        for model in (base["baseline"], base["A"], base["B"]):
            self.assertEqual(model["quarter_endings_per_game"], 4)
            self.assertLess(max(player["attempt_share"] for player in model["shot_distribution"].values()), 0.5)
            self.assertFalse(model["uses_2015_16_as_other_era_fact"])

    def test_validation_sample_is_not_a_tuning_target(self):
        exploration = observed_duration_sample(ROOT)
        validation = observed_duration_sample(Path(__file__).resolve().parents[1] / "research_playbyplay" / "validation")
        self.assertTrue(set(exploration["game_ids"]).isdisjoint(validation["game_ids"]))
        self.assertGreater(exploration["excluded_share"], 0.05)
        self.assertGreater(validation["excluded_share"], 0.05)
        row = compare_models([ERAS[-2]], seeds=(7,))[0]
        report = validation_report(row["B"], exploration, validation)
        self.assertFalse(report["parameters_fit_to_validation"])
        self.assertIn("2015-16", report["era_scope"])
        self.assertFalse(FITTED_TO_HELD_OUT)
        self.assertFalse(APPLIES_2015_16_TO_OTHER_ERAS)
        self.assertFalse(PROMOTION["promoted"])
        rows = compare_models(ERAS, seeds=(7, 19))
        self.assertEqual([item["era"] for item in rows], [era.key for era in ERAS])
        self.assertTrue(all(item["observed_2015_16_applied"] is False for item in rows))


if __name__ == "__main__":
    unittest.main()
