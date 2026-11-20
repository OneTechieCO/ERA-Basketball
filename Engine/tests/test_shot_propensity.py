"""Shooting propensity replaces the usage term. It does not assign a shot quota."""
import math
import sqlite3
import unittest
from dataclasses import replace

from era_engine.core import (
    SHOT_ROLE_PASS_LIMIT,
    intelligent_pass_chance,
    selection_weight,
    simulate_game,
)
from era_engine.historical_players import HistoricalDataError, _fga_per_100_possessions, attempt_rate_gap
from era_engine.seed import ERAS, demo_players


def _chance(player, remaining=18.0):
    defender = replace(player, player_id="D", defense=73, shot_propensity=None)
    return intelligent_pass_chance(player, defender, 0.0, 0, remaining)


class ShotPropensityTests(unittest.TestCase):
    def setUp(self):
        self.players = demo_players()
        self.star = self.players[0]
        self.era = next(era for era in ERAS if era.key == "2015-16")

    def test_an_unset_propensity_keeps_the_usage_term(self):
        self.assertIsNone(self.star.shot_propensity)
        low_usage = replace(self.star, usage=0.12)
        self.assertLess(_chance(self.star), _chance(low_usage))
        self.assertGreater(selection_weight(self.star), selection_weight(low_usage))

    def test_a_set_propensity_does_not_also_use_usage(self):
        high = replace(self.star, usage=0.33, shot_propensity=0.0)
        low = replace(self.star, usage=0.10, shot_propensity=0.0)
        self.assertEqual(_chance(high), _chance(low))
        self.assertGreater(selection_weight(high), selection_weight(low))
        self.assertGreater(_chance(high), _chance(self.star))
        seeking = replace(high, shot_propensity=8.0)
        passing = replace(high, shot_propensity=-6.0)
        self.assertEqual(selection_weight(high), selection_weight(seeking))
        self.assertEqual(selection_weight(high), selection_weight(passing))
        self.assertLess(_chance(seeking), _chance(high))
        self.assertGreater(_chance(passing), _chance(high))
        self.assertGreaterEqual(_chance(replace(high, shot_propensity=80.0)), 0.12)
        self.assertLessEqual(_chance(replace(high, shot_propensity=-80.0)), 0.82)
        self.assertLess(
            _chance(high) - _chance(seeking),
            SHOT_ROLE_PASS_LIMIT + 1e-9,
        )

    def test_shot_quality_and_the_clock_still_move_the_decision(self):
        rated = replace(self.star, shot_propensity=1.0)
        weak = replace(
            rated,
            shot_skills={zone: 48 for zone in rated.shot_skills},
            tendencies={zone: 0.25 for zone in rated.tendencies},
        )
        self.assertGreater(_chance(weak), _chance(rated))
        self.assertGreater(_chance(rated, remaining=20), _chance(rated, remaining=6))

    def test_a_non_finite_propensity_is_rejected(self):
        with self.assertRaises(ValueError):
            _chance(replace(self.star, shot_propensity=math.nan))

    def test_the_attempt_gap_follows_event_mix_and_shrinks_a_short_sample(self):
        matched = attempt_rate_gap(1600, 2700, 2000, 8000, 10000)
        shooter = attempt_rate_gap(1900, 2700, 2000, 8000, 10000)
        passer = attempt_rate_gap(1200, 2700, 2000, 8000, 10000)
        one_game = attempt_rate_gap(6, 7, 6, 8000, 10000)
        self.assertLess(abs(matched["shot_propensity"]), 0.05)
        self.assertGreater(shooter["shot_propensity"], 0.0)
        self.assertLess(passer["shot_propensity"], 0.0)
        self.assertLess(abs(one_game["shot_propensity"]), 2.0)
        self.assertGreater(one_game["fga_per_36"], 20)
        self.assertNotEqual(matched["shot_propensity"], matched["shot_event_fraction"])
        with self.assertRaises(HistoricalDataError):
            attempt_rate_gap(10, 0, 10, 100, 200)
        with self.assertRaises(HistoricalDataError):
            attempt_rate_gap(math.nan, 100, 20, 100, 200)
        with self.assertRaises(HistoricalDataError):
            attempt_rate_gap(30, 100, 10, 100, 200)

    def test_possessions_are_used_only_from_a_row_that_also_has_attempts(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE nba_team_season_research
               (season_id TEXT, team_id INT, fga REAL, possessions REAL, possessions_measure TEXT)"""
        )
        connection.execute(
            "INSERT INTO nba_team_season_research VALUES ('2015-16', 1, NULL, 8303, 'season_total')"
        )
        self.assertIsNone(_fga_per_100_possessions(connection, "2015-16", 1)["fga_per_100_team_possessions"])
        connection.execute(
            "INSERT INTO nba_team_season_research VALUES ('2015-16', 1, 7159, 8303, 'season_total')"
        )
        found = _fga_per_100_possessions(connection, "2015-16", 1)
        self.assertAlmostEqual(found["fga_per_100_team_possessions"], 7159 / 8303 * 100)
        connection.close()

    def test_explicit_propensity_stays_deterministic_and_balances_the_score(self):
        home = self.players[:5]
        away = self.players[5:]
        from era_engine.core import Team
        team_a = Team("Bay City Legends", tuple(player.player_id for player in home), "perimeter")
        team_b = Team("Chicago Stampede", tuple(player.player_id for player in away), "balanced")
        roster = {player.player_id: player for player in self.players}
        seeking = {pid: replace(player, shot_propensity=6.0 if pid == home[0].player_id else 0.0) for pid, player in roster.items()}
        first = simulate_game(team_a, team_b, seeking, self.era, 7)
        second = simulate_game(team_a, team_b, seeking, self.era, 7)
        self.assertEqual(first, second)
        self.assertNotEqual(first["score"], simulate_game(team_a, team_b, roster, self.era, 7)["score"])
        self.assertEqual(first["score"][0], sum(first["player_stats"][pid].get("pts", 0) for pid in team_a.player_ids))
        self.assertAlmostEqual(sum(first["live_seconds"].values()), first["game_clock_seconds"], delta=1e-6)
        for event in first["events"]:
            self.assertGreaterEqual(event["shot_clock_remaining"], 0.0)
            self.assertGreaterEqual(event["quarter_remaining"], 0.0)
            self.assertTrue(math.isfinite(event["elapsed_seconds"]))
        self.assertNotIn("free_throw", first["live_seconds"])
        self.assertFalse(first["pace_used_to_choose_possessions"])
