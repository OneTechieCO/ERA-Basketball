"""Multiple-pass possessions in the live engine."""
import tempfile
import unittest
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from era_engine.core import (
    PossessionClock,
    UnresolvedGameError,
    ZONES,
    intelligent_pass_chance,
    simulate_game,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


class LivePassingTests(unittest.TestCase):
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

    def _game(self, era="1995-96", seed=7, shot_clock=None):
        selected = self.eras[era]
        if shot_clock is not None:
            selected = replace(selected, shot_clock=shot_clock)
        return simulate_game(self.team_a, self.team_b, self.players, selected, seed)

    def test_clock_cannot_fall_below_zero(self):
        clock = PossessionClock(3)
        self.assertFalse(clock.advance(2))
        self.assertEqual(clock.remaining, 1)
        self.assertTrue(clock.advance(2))
        self.assertEqual(clock.remaining, 0)

    def test_possessions_can_include_multiple_passes(self):
        game = self._game()
        counted = [event["passes"] for event in game["events"]]
        self.assertTrue(counted)
        self.assertGreaterEqual(max(counted), 2)
        self.assertLessEqual(max(counted), 6)
        self.assertTrue(all(0 <= count <= 6 for count in counted))

    def test_pass_turnover_ends_possession_without_a_shot_or_assist(self):
        game = self._game(seed=11)
        by_possession = defaultdict(list)
        for event in game["events"]:
            by_possession[event["possession"]].append(event)
        turnovers = [
            event for event in game["events"] if event["kind"] == "pass_turnover"
        ]
        self.assertGreater(len(turnovers), 0)
        for event in turnovers:
            sequence = by_possession[event["possession"]]
            self.assertIs(sequence[-1], event)
            self.assertNotIn("assister", event)
            self.assertGreaterEqual(event["passes"], 0)
            self.assertNotEqual(event["offense"], event["defender"])
        credited = sum(
            1 for event in game["events"]
            if event["kind"] == "made_shot" and event["assister"]
        )
        assists = sum(stats.get("ast", 0) for stats in game["player_stats"].values())
        self.assertEqual(assists, credited)

    def test_a_make_after_a_pass_credits_that_teammate(self):
        game = self._game(seed=7)
        self.assertEqual(game["score"], [141, 146])
        self.assertEqual(game["possessions_by_team"], [135, 134])
        assisted = 0
        for event in game["events"]:
            if event["kind"] != "made_shot":
                continue
            if event["passes"] == 0:
                self.assertIsNone(event["assister"])
                continue
            self.assertIn(event["assister"], event["offense_lineup"])
            self.assertNotEqual(event["assister"], event["offense"])
            assisted += 1
        for team in (self.team_a, self.team_b):
            makes = sum(game["player_stats"][pid].get("fgm", 0) for pid in team.player_ids)
            assists = sum(game["player_stats"][pid].get("ast", 0) for pid in team.player_ids)
            self.assertLessEqual(assists, makes)
        self.assertEqual(
            assisted,
            sum(stats.get("ast", 0) for stats in game["player_stats"].values()),
        )
        self.assertGreater(assisted, 18 + 16)

    def test_only_the_final_passer_can_be_credited_with_an_assist(self):
        game = self._game()
        for event in game["events"]:
            if event["kind"] != "made_shot":
                self.assertNotIn("assister", event)
                continue
            if event["assister"] is None:
                continue
            self.assertGreaterEqual(event["passes"], 1)
            self.assertNotEqual(event["assister"], event["offense"])
        for team in (self.team_a, self.team_b):
            assists = sum(game["player_stats"][pid].get("ast", 0) for pid in team.player_ids)
            makes = sum(game["player_stats"][pid].get("fgm", 0) for pid in team.player_ids)
            self.assertLessEqual(assists, makes)

    def test_short_clock_blocks_passes(self):
        with self.assertRaises(UnresolvedGameError) as expired_try:
            self._game(shot_clock=1, seed=19)
        expired = expired_try.exception.result
        self.assertFalse(expired["completed"])
        self.assertEqual(expired["status"], "incomplete")
        self.assertGreater(
            sum(event["kind"] == "shot_clock_violation" for event in expired["events"]),
            0,
        )
        self.assertEqual(sum(stats.get("fga", 0) for stats in expired["player_stats"].values()), 0)
        self.assertEqual(expired["score"], [0, 0])
        self.assertTrue(all(event["passes"] == 0 for event in expired["events"]))
        self.assertFalse(any(event["kind"] == "pass_turnover" for event in expired["events"]))
        violations = [event for event in expired["events"] if event["kind"] == "shot_clock_violation"]
        live_turnovers = [event for event in expired["events"] if event["kind"] == "turnover"]
        turnovers = sum(stats.get("tov", 0) for stats in expired["player_stats"].values())
        period_endings = [event for event in expired["events"] if event["kind"] == "period_expired"]
        self.assertEqual(
            len(violations) + len(live_turnovers) + len(period_endings),
            expired["possession_count"],
        )
        self.assertEqual(turnovers, len(violations) + len(live_turnovers))

        with self.assertRaises(UnresolvedGameError) as pressed_try:
            self._game(shot_clock=2)
        pressed = pressed_try.exception.result
        self.assertEqual(pressed["score"], [0, 0])
        self.assertFalse(pressed["completed"])
        self.assertEqual(sum(stats.get("ast", 0) for stats in pressed["player_stats"].values()), 0)
        self.assertTrue(all(event["passes"] == 0 for event in pressed["events"]))

    def test_offensive_rebound_resets_and_continues_the_possession(self):
        expected = {"1995-96": 24, "2023-24": 14}
        for era, reset in expected.items():
            with self.subTest(era=era):
                game = self._game(era=era)
                by_possession = defaultdict(list)
                for event in game["events"]:
                    by_possession[event["possession"]].append(event)
                resets = []
                continued = 0
                for sequence in by_possession.values():
                    for index, event in enumerate(sequence):
                        if not event.get("offensive_rebound"):
                            continue
                        resets.append(event["shot_clock_reset"])
                        self.assertLess(index, len(sequence) - 1)
                        continued += 1
                self.assertGreater(continued, 0)
                self.assertEqual(set(resets), {reset})

    def test_passing_skill_increases_completed_passes_and_lowers_pass_turnover_rate(self):
        def play(passing, iq):
            roster = {
                pid: replace(player, passing=passing, iq=iq)
                for pid, player in self.players.items()
            }
            passes = 0
            failures = 0
            for seed in range(12):
                game = simulate_game(self.team_a, self.team_b, roster, self.eras["1995-96"], seed)
                for event in game["events"]:
                    passes += event["passes"]
                    if event["kind"] == "pass_turnover":
                        failures += 1
            attempts = passes + failures
            return passes, failures / attempts

        elite_passes, elite_rate = play(99, 99)
        poor_passes, poor_rate = play(40, 40)
        self.assertGreater(elite_passes, poor_passes)
        self.assertLess(elite_rate, poor_rate)

    def test_live_engine_uses_the_intelligent_pass_decision(self):
        seen = []

        def spy(shooter, defender, style_bonus, pass_count, clock_remaining):
            seen.append((shooter, defender, style_bonus, pass_count, clock_remaining))
            return intelligent_pass_chance(
                shooter, defender, style_bonus, pass_count, clock_remaining
            )

        with patch("era_engine.core.intelligent_pass_chance", spy):
            simulate_game(self.team_a, self.team_b, self.players, self.eras["2015-16"], 7)
        self.assertGreater(len(seen), 20)
        self.assertGreater(len({remaining for *_, remaining in seen}), 1)
        self.assertGreater(len({defender.defense for _, defender, *_ in seen}), 1)

        shooter, defender, style_bonus, pass_count, remaining = seen[0]
        base = intelligent_pass_chance(shooter, defender, style_bonus, pass_count, remaining)
        weaker_shot = replace(
            shooter,
            shot_skills={zone: 48 for zone in ZONES},
            tendencies={zone: 0.25 for zone in ZONES},
        )
        self.assertGreater(
            intelligent_pass_chance(weaker_shot, defender, style_bonus, pass_count, remaining),
            base,
        )
        self.assertGreater(
            intelligent_pass_chance(
                shooter, replace(defender, defense=defender.defense + 25),
                style_bonus, pass_count, remaining,
            ),
            base,
        )
        self.assertGreater(
            intelligent_pass_chance(shooter, defender, style_bonus, pass_count, remaining + 12),
            intelligent_pass_chance(shooter, defender, style_bonus, pass_count, max(0.0, remaining - 12)),
        )


if __name__ == "__main__":
    unittest.main()
