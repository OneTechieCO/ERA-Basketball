"""Shot-clock behavior of the live engine in era_engine.core."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine.core import (
    Era,
    PossessionClock,
    UnresolvedGameError,
    offensive_rebound_reset,
    simulate_game,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


class LiveShotClockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "test.db"))
        initialize_demo(self.con)
        self.players, self.eras, self.teams = load(self.con)
        self.team_a = self.teams["Bay City Legends"]
        self.team_b = self.teams["Chicago Stampede"]

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_seeded_eras_start_new_possessions_at_24(self):
        for key, era in self.eras.items():
            with self.subTest(era=key):
                self.assertEqual(era.shot_clock, 24)
                game = simulate_game(self.team_a, self.team_b, self.players, era, 7)
                violations = [event for event in game["events"] if event["kind"] == "shot_clock_violation"]
                attempts = sum(stats.get("fga", 0) for stats in game["player_stats"].values())
                self.assertEqual(violations, [])
                self.assertGreater(attempts, 0)

    def test_clock_countdown_expiration_and_reset(self):
        clock = PossessionClock(24)
        self.assertFalse(clock.advance(6))
        self.assertEqual(clock.remaining, 18)
        self.assertTrue(clock.advance(18))
        self.assertEqual(clock.remaining, 0)
        self.assertTrue(clock.advance(5))
        self.assertEqual(clock.remaining, 0)
        with self.assertRaises(ValueError):
            clock.advance(-1)
        self.assertEqual(clock.remaining, 0)
        clock.reset(14)
        self.assertEqual(clock.remaining, 14)
        self.assertFalse(clock.advance(4))
        self.assertEqual(clock.remaining, 10)
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    clock.reset(value)
                self.assertEqual(clock.remaining, 10)

    def test_shot_clock_configuration(self):
        era = Era("2023-24", 99, 0.4, 0.5, True)
        self.assertEqual(era.shot_clock, 24)
        self.assertEqual(replace(era, shot_clock=8).shot_clock, 8)
        for value in (0, -1, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(era, shot_clock=value)

    def test_offensive_rebound_reset_by_era(self):
        self.assertEqual(offensive_rebound_reset(self.eras["1985-86"]), 24)
        self.assertEqual(offensive_rebound_reset(self.eras["1995-96"]), 24)
        self.assertEqual(offensive_rebound_reset(self.eras["2015-16"]), 24)
        self.assertEqual(offensive_rebound_reset(self.eras["2023-24"]), 14)
        pre_change = Era("2017-18", 99, 0.4, 0.5, True)
        post_change = Era("2018-19", 99, 0.4, 0.5, True)
        self.assertEqual(offensive_rebound_reset(pre_change), 24)
        self.assertEqual(offensive_rebound_reset(post_change), 14)
        self.assertEqual(
            offensive_rebound_reset(replace(post_change, shot_clock=8)),
            8,
        )

    def test_live_games_record_the_rebound_reset_they_use(self):
        expected = {"1995-96": 24, "2015-16": 24, "2023-24": 14}
        for key, reset in expected.items():
            with self.subTest(era=key):
                game = simulate_game(self.team_a, self.team_b, self.players, self.eras[key], 7)
                resets = [
                    event["shot_clock_reset"]
                    for event in game["events"]
                    if event.get("offensive_rebound")
                ]
                self.assertGreater(len(resets), 0)
                self.assertEqual(set(resets), {reset})
                for event in game["events"]:
                    if event["kind"] == "miss" and not event["offensive_rebound"]:
                        self.assertNotIn("shot_clock_reset", event)

    def test_short_clock_expires_as_a_turnover(self):
        era = replace(self.eras["1995-96"], shot_clock=1)
        with self.assertRaises(UnresolvedGameError) as first_try:
            simulate_game(self.team_a, self.team_b, self.players, era, 19)
        with self.assertRaises(UnresolvedGameError) as second_try:
            simulate_game(self.team_a, self.team_b, self.players, era, 19)
        first = first_try.exception.result
        second = second_try.exception.result
        self.assertEqual(first, second)
        self.assertFalse(first["completed"])
        self.assertEqual(first["status"], "incomplete")
        self.assertIn("No tie-break point", str(first_try.exception))
        violations = [event for event in first["events"] if event["kind"] == "shot_clock_violation"]
        live_turnovers = [event for event in first["events"] if event["kind"] == "turnover"]
        turnovers = sum(stats.get("tov", 0) for stats in first["player_stats"].values())
        attempts = sum(stats.get("fga", 0) for stats in first["player_stats"].values())
        self.assertGreater(len(violations), 0)
        self.assertEqual(attempts, 0)
        self.assertEqual(first["score"], [0, 0])
        period_endings = [event for event in first["events"] if event["kind"] == "period_expired"]
        self.assertEqual(
            len(violations) + len(live_turnovers) + len(period_endings),
            first["possession_count"],
        )
        self.assertEqual(turnovers, len(violations) + len(live_turnovers))
        for event in violations:
            self.assertEqual(event["kind"], "shot_clock_violation")
            self.assertIn("offense", event)

    def test_late_clock_suppresses_the_existing_pass(self):
        # After the 0.5-second setup, 1.5 seconds remain, so the pass rate is forced to zero.
        pressed = replace(self.eras["1995-96"], shot_clock=2)
        with self.assertRaises(UnresolvedGameError) as pressed_try:
            simulate_game(self.team_a, self.team_b, self.players, pressed, 7)
        pressed_game = pressed_try.exception.result
        self.assertEqual(pressed_game["score"], [0, 0])
        self.assertFalse(pressed_game["completed"])
        pressed_assists = sum(stats.get("ast", 0) for stats in pressed_game["player_stats"].values())
        pressed_violations = [
            event for event in pressed_game["events"] if event["kind"] == "shot_clock_violation"
        ]
        self.assertEqual(pressed_assists, 0)
        self.assertGreater(len(pressed_violations), 0)

        normal = simulate_game(self.team_a, self.team_b, self.players, self.eras["1995-96"], 7)
        normal_assists = sum(stats.get("ast", 0) for stats in normal["player_stats"].values())
        self.assertGreater(normal_assists, 0)

    def test_expired_clock_does_not_award_points(self):
        era = replace(self.eras["2023-24"], shot_clock=4)
        game = simulate_game(self.team_a, self.team_b, self.players, era, 11)
        violations = [event for event in game["events"] if event["kind"] == "shot_clock_violation"]
        self.assertGreater(len(violations), 0)
        for pid, stats in game["player_stats"].items():
            fgm = stats.get("fgm", 0)
            fg3m = stats.get("fg3m", 0)
            ftm = stats.get("ftm", 0)
            self.assertEqual(stats.get("pts", 0), 2 * (fgm - fg3m) + 3 * fg3m + ftm)
        for index, team in enumerate((self.team_a, self.team_b)):
            self.assertEqual(
                game["score"][index],
                sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
            )


if __name__ == "__main__":
    unittest.main()
