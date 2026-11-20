"""Variant B3 keeps a possession state. The live make formula stays in core.py."""
import json
import unittest
from pathlib import Path

from era_engine.action_causal_experiments import (
    FITTED_TO_SAMPLE,
    PROMOTION,
    RESERVED_GAMES_USED_FOR_ASSESSMENT,
    SAFEGUARD_ACTIONS,
    PossessionState,
    action_openness,
    action_weights,
    apply_hold,
    apply_pass,
    apply_screen,
    diagnose_game,
    leading_action,
    must_finish,
)
from era_engine.core import Player, probability_make
from era_engine.live_timing_diagnostics import fictional_matchup
from era_engine.seed import ERAS, demo_players


def _poor():
    return Player(
        "P", "Poor", "2010s", 70, 70, 70, 70, 70, 80, 0.19,
        {"rim": 55, "paint": 55, "mid": 55, "three": 55},
        {"rim": 0.25, "paint": 0.25, "mid": 0.25, "three": 0.25},
    )


class ActionCausalTests(unittest.TestCase):
    def test_action_state_persists_and_a_failure_does_not_create_a_shot(self):
        players = demo_players()
        poor, lock, teammate = _poor(), players[6], players[3]
        state = PossessionState(poor, lock)
        offense = [poor, teammate]
        defense = [lock, players[0]]
        before = action_openness(state.separation, state.pressure)
        self.assertNotEqual(leading_action(action_weights(state, offense, defense, 18)), "shoot")
        apply_screen(state, "failed")
        self.assertEqual(state.separation, 0.0)
        self.assertLessEqual(action_openness(state.separation, state.pressure), before)
        self.assertNotEqual(leading_action(action_weights(state, offense, defense, 18)), "shoot")
        self.assertGreater(state.pressure, 0.0)
        opened = PossessionState(poor, lock)
        apply_screen(opened, "opening")
        self.assertEqual(opened.separation, 1.0)
        action_weights(opened, offense, defense, 18)
        self.assertEqual(opened.separation, 1.0)
        quality = opened.quality()
        apply_hold(opened)
        self.assertEqual(opened.quality(), quality)
        self.assertEqual(opened.ballhandler.player_id, poor.player_id)

    def test_pass_and_switch_change_who_the_next_decision_uses(self):
        players = demo_players()
        star, lock, other = players[0], players[6], players[5]
        offense, defense = players[:5], players[5:]
        state = PossessionState(star, lock)
        before = leading_action(action_weights(state, offense, defense, 18))
        apply_screen(state, "switch", other)
        self.assertEqual(state.defender.player_id, other.player_id)
        self.assertNotEqual(leading_action(action_weights(state, offense, defense, 18)), before)
        apply_pass(state, players[1], players[8])
        self.assertEqual(state.ballhandler.player_id, players[1].player_id)
        self.assertEqual(state.separation, 0.0)
        self.assertEqual(state.defender.player_id, players[8].player_id)
        clean = PossessionState(star, players[3])
        self.assertEqual(leading_action(action_weights(clean, offense, defense, 18)), "shoot")
        self.assertTrue(must_finish(SAFEGUARD_ACTIONS, 18))
        self.assertTrue(must_finish(1, 8))
        self.assertFalse(must_finish(3, 16))
        self.assertTrue(must_finish(0, 18, putback=True))
        self.assertEqual(probability_make.__module__, "era_engine.core")

    def test_games_keep_the_clock_legal_and_the_accounting_closed(self):
        home, away, roster = fictional_matchup()
        era = ERAS[-2]
        game = diagnose_game(era, 7, home, away, roster)
        self.assertEqual(game["durations"], diagnose_game(era, 7, home, away, roster)["durations"])
        self.assertAlmostEqual(sum(game["durations"]), 2880, places=3)
        self.assertFalse(game["fitted_to_sample"])
        makes = fga = points = assists = 0
        for possession in game["possession_log"]:
            makes += possession["counts"].get("make", 0)
            fga += possession["counts"].get("fga", 0)
            points += possession["counts"].get("points", 0)
            assists += possession["counts"].get("assist", 0)
            self.assertGreaterEqual(possession["minimum_shot_clock"], 0.0)
            for event in possession["history"]:
                self.assertGreaterEqual(event["shot_clock"], 0.0)
                self.assertLessEqual(event["development_actions"], SAFEGUARD_ACTIONS)
        self.assertLessEqual(makes, fga)
        self.assertLessEqual(assists, makes)
        self.assertGreaterEqual(points, makes)
        manifest = json.loads(Path("research_playbyplay/reserved_b3/manifest.json").read_text())
        self.assertFalse(manifest["used_for_b3_assessment"])
        self.assertTrue(RESERVED_GAMES_USED_FOR_ASSESSMENT is False)
        self.assertFalse(FITTED_TO_SAMPLE)
        self.assertFalse(PROMOTION["promoted"])
        known = {
            "0021500003", "0021500290", "0021500605", "0021501227",
            "0021500164", "0021500504", "0021500822", "0021501112",
        }
        reserved = {game["game_id"] for game in manifest["games"]}
        self.assertTrue(reserved.isdisjoint(known))
        self.assertEqual(len(reserved), 4)


if __name__ == "__main__":
    unittest.main()
