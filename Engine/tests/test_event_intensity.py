"""Historical touch weights use one minutes-aware rate. Fictional usage stays."""
import math
import unittest
from dataclasses import replace

from era_engine.core import Team, intelligent_pass_chance, selection_weight, simulate_game
from era_engine.historical_players import (
    HistoricalDataError,
    OFFENSIVE_EVENT_FREE_THROW_WEIGHT,
    load_researched_rosters,
    shrunk_event_intensity,
)
from era_engine.seed import ERAS, demo_players


class EventIntensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bulls, cls.warriors = load_researched_rosters()

    def test_fictional_selection_still_uses_usage(self):
        star = demo_players()[0]
        role = demo_players()[3]
        self.assertIsNone(star.event_intensity)
        self.assertAlmostEqual(
            selection_weight(star),
            max(star.usage, 1e-9) * (70 + star.iq) / 140,
        )
        self.assertGreater(selection_weight(star), selection_weight(role))
        self.assertGreater(selection_weight(replace(role, usage=0.0)), 0.0)

    def test_historical_weights_follow_event_intensity_not_mixed_usage(self):
        curry = next(profile.player for profile in self.warriors.profiles if profile.player_name == "Stephen Curry")
        klay = next(profile.player for profile in self.warriors.profiles if profile.player_name == "Klay Thompson")
        self.assertAlmostEqual(curry.usage, 0.326)
        self.assertEqual(selection_weight(curry) / selection_weight(klay), curry.event_intensity / klay.event_intensity)
        swapped = replace(curry, usage=0.05)
        self.assertEqual(selection_weight(swapped), selection_weight(curry))
        self.assertNotEqual(selection_weight(replace(curry, event_intensity=curry.event_intensity / 2)), selection_weight(curry))

    def test_a_short_minute_sample_moves_toward_the_roster_rate(self):
        season = shrunk_event_intensity(1800, 2700, 9000, 20000)
        one_game = shrunk_event_intensity(8, 7, 9000, 20000)
        self.assertGreater(abs(8 / 7 - 9000 / 20000), abs(one_game["event_intensity"] - one_game["event_intensity_reference"]))
        self.assertLess(abs(season["event_intensity"] - season["events_per_minute"]), 0.05)
        self.assertGreater(one_game["event_intensity"], 0.0)
        with self.assertRaises(HistoricalDataError):
            shrunk_event_intensity(1, 0, 10, 100)
        with self.assertRaises(ValueError):
            selection_weight(replace(demo_players()[0], event_intensity=math.nan))

    def test_intensity_does_not_change_the_pass_decision(self):
        player = demo_players()[0]
        defender = replace(player, player_id="D", defense=73)
        touched = replace(player, event_intensity=0.8, shot_propensity=1.5)
        untouched = replace(player, event_intensity=0.2, shot_propensity=1.5)
        self.assertEqual(
            intelligent_pass_chance(touched, defender, 0.0, 0, 18),
            intelligent_pass_chance(untouched, defender, 0.0, 0, 18),
        )

    def test_stored_intensity_matches_the_stint_formula(self):
        for roster in (self.bulls, self.warriors):
            total_events = 0.0
            total_minutes = 0.0
            rows = []
            for profile in roster.profiles:
                counting = profile.counting
                events = counting["fga"] + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * counting["fta"] + counting["turnovers"]
                rows.append((profile, events, counting["minutes"]))
                total_events += events
                total_minutes += counting["minutes"]
            for profile, events, minutes in rows:
                expected = shrunk_event_intensity(events, minutes, total_events, total_minutes)
                self.assertAlmostEqual(profile.player.event_intensity, expected["event_intensity"])
                self.assertGreater(profile.player.event_intensity, 0.0)
                self.assertIn("not NBA USG%", profile.ratings["event_intensity"].statement)

    def test_off_court_players_record_no_touches_and_the_score_balances(self):
        players = demo_players()
        home = players[:5]
        away = players[5:]
        roster = {player.player_id: replace(player, event_intensity=0.4 if index == 0 else 0.2) for index, player in enumerate(players)}
        era = next(item for item in ERAS if item.key == "2015-16")
        team_a = Team("Bay City Legends", tuple(player.player_id for player in home), "perimeter")
        team_b = Team("Chicago Stampede", tuple(player.player_id for player in away), "balanced")
        plain = {player.player_id: player for player in players}
        first = simulate_game(team_a, team_b, roster, era, 11)
        self.assertEqual(first, simulate_game(team_a, team_b, roster, era, 11))
        self.assertNotEqual(first["score"], simulate_game(team_a, team_b, plain, era, 11)["score"])
        self.assertEqual(first["score"][0], sum(first["player_stats"][pid].get("pts", 0) for pid in team_a.player_ids))
        for event in first["events"]:
            offense = event.get("offense")
            if offense is not None:
                self.assertIn(offense, event["offense_lineup"])
