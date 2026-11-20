"""Regulation game clock in the live engine."""
import tempfile
import unittest
from pathlib import Path

from era_engine.core import (
    RegulationClock,
    TimingAssumptions,
    simulate_game,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


class RegulationClockTests(unittest.TestCase):
    def test_four_quarters_do_not_run_past_zero_or_into_overtime(self):
        clock = RegulationClock(12 * 60, 4)
        self.assertEqual(clock.elapsed, 0.0)
        self.assertEqual(clock.quarter_remaining, 720.0)
        for quarter in range(1, 5):
            self.assertEqual(clock.quarter, quarter)
            self.assertEqual(clock.advance(720), 720)
            self.assertEqual(clock.quarter_remaining, 0.0)
            if quarter < 4:
                self.assertTrue(clock.start_next_quarter())
        self.assertFalse(clock.start_next_quarter())
        self.assertTrue(clock.regulation_complete)
        self.assertEqual(clock.elapsed, 2880.0)
        self.assertEqual(clock.advance(5), 0.0)
        self.assertEqual(clock.quarter_lengths(), [720.0, 720.0, 720.0, 720.0])
        with self.assertRaises(ValueError):
            clock.advance(-1)

    def test_a_quarter_cannot_change_while_time_remains(self):
        clock = RegulationClock(720, 4)
        with self.assertRaises(ValueError):
            clock.start_next_quarter()
        self.assertEqual(clock.quarter, 1)


class LiveGameClockTests(unittest.TestCase):
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

    def test_regulation_is_2880_seconds_over_four_quarters(self):
        game = simulate_game(self.team_a, self.team_b, self.players, self.eras["2015-16"], 7)
        again = simulate_game(self.team_a, self.team_b, self.players, self.eras["2015-16"], 7)
        self.assertEqual(game, again)
        periods = game["overtime"]["periods"]
        self.assertEqual(game["game_clock"], "overtime" if periods else "regulation")
        self.assertTrue(game["completed"])
        self.assertEqual(game["status"], "final")
        self.assertNotEqual(game["score"][0], game["score"][1])
        self.assertFalse(game["overtime"]["tie_broken_by_rule"])
        self.assertFalse(game["overtime"]["unresolved_tie"])
        self.assertFalse(game["overtime"]["timeouts_modeled"])
        self.assertEqual(game["overtime"]["fixture_points"], [])
        self.assertEqual(game["regulation_score"], game["period_scores"][3]["running_score"])
        self.assertEqual(game["game_clock_seconds"], 2880.0 + 300.0 * periods)
        self.assertEqual(game["quarters_completed"], 4)
        self.assertEqual(game["quarter_lengths"], [720.0, 720.0, 720.0, 720.0])
        self.assertEqual(game["regulation_seconds"], 2880.0)
        self.assertAlmostEqual(sum(game["live_seconds"].values()), game["game_clock_seconds"], delta=1e-3)
        self.assertFalse(game["timing_calibrated"])
        self.assertFalse(game["pace_used_to_choose_possessions"])
        self.assertEqual(game["timing_assumptions"]["calibrated"], False)
        self.assertEqual(game["possession_count"], sum(game["possessions_by_team"]))
        self.assertEqual(game["possessions_per_team"], game["possessions_by_team"][0])

    def test_clocks_stay_non_negative_and_possessions_stay_inside_one_quarter(self):
        for era in self.eras.values():
            game = simulate_game(self.team_a, self.team_b, self.players, era, 19)
            with self.subTest(era=era.key):
                self.assertEqual(game["regulation_seconds"], era.game_minutes * 60)
                quarters = {}
                for event in game["events"]:
                    self.assertGreaterEqual(event["shot_clock_remaining"], 0.0)
                    self.assertGreaterEqual(event["quarter_remaining"], 0.0)
                    self.assertLessEqual(event["shot_clock_remaining"], era.shot_clock)
                    self.assertIn(event["quarter"], (1, 2, 3, 4))
                    quarters.setdefault(event["possession"], set()).add(event["quarter"])
                self.assertTrue(all(len(found) == 1 for found in quarters.values()))
                ending = {
                    event["possession"]: event["quarter"]
                    for event in game["events"]
                    if event["period_ending"]
                }
                self.assertEqual(len(ending), 4)
                self.assertEqual(set(ending.values()), {1, 2, 3, 4})
                self.assertEqual(game["period_ending_possessions"], 4)

    def test_shot_clock_reset_does_not_rewind_the_game_clock(self):
        game = simulate_game(self.team_a, self.team_b, self.players, self.eras["2023-24"], 7)
        by_possession = {}
        for event in game["events"]:
            by_possession.setdefault(event["possession"], []).append(event)
        continued = 0
        for sequence in by_possession.values():
            for index, event in enumerate(sequence):
                if not event.get("offensive_rebound"):
                    continue
                self.assertEqual(event["shot_clock_reset"], 14)
                follower = sequence[index + 1]
                self.assertGreaterEqual(event["quarter_remaining"], follower["quarter_remaining"])
                self.assertEqual(event["quarter"], follower["quarter"])
                if follower["kind"] != "period_expired":
                    self.assertLessEqual(follower["shot_clock_remaining"], event["shot_clock_reset"])
                    continued += 1
        self.assertGreater(continued, 0)

    def test_free_throws_do_not_add_game_clock_time(self):
        saw_free_throws = False
        for seed in range(8):
            game = simulate_game(self.team_a, self.team_b, self.players, self.eras["1995-96"], seed)
            self.assertEqual(game["regulation_seconds"], 2880.0)
            self.assertNotIn("free_throw", game["live_seconds"])
            attempts = sum(stats.get("fta", 0) for stats in game["player_stats"].values())
            if attempts:
                saw_free_throws = True
                break
        self.assertTrue(saw_free_throws)

    def test_timing_assumptions_are_configurable_and_not_a_pace_target(self):
        era = self.eras["2015-16"]
        baseline = simulate_game(self.team_a, self.team_b, self.players, era, 7)
        longer_gather = simulate_game(
            self.team_a,
            self.team_b,
            self.players,
            era,
            7,
            timing=TimingAssumptions(rebound_gather_seconds=5.0),
        )
        self.assertEqual(longer_gather["regulation_seconds"], 2880.0)
        self.assertFalse(longer_gather["timing_calibrated"])
        self.assertNotEqual(baseline["possession_count"], longer_gather["possession_count"])
        self.assertGreater(longer_gather["live_seconds"].get("rebound_gather", 0.0), baseline["live_seconds"].get("rebound_gather", 0.0))

    def test_score_and_possession_accounting_stay_closed(self):
        game = simulate_game(self.team_a, self.team_b, self.players, self.eras["2005-06"], 42)
        for index, team in enumerate((self.team_a, self.team_b)):
            points = sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids)
            self.assertEqual(game["score"][index], points)
        for stats in game["player_stats"].values():
            self.assertLessEqual(stats.get("fgm", 0), stats.get("fga", 0))
            self.assertLessEqual(stats.get("fg3m", 0), stats.get("fg3a", 0))
            self.assertLessEqual(stats.get("ftm", 0), stats.get("fta", 0))
            fgm = stats.get("fgm", 0)
            fg3m = stats.get("fg3m", 0)
            self.assertEqual(stats.get("pts", 0), 2 * (fgm - fg3m) + 3 * fg3m + stats.get("ftm", 0))
        kinds = {"made_shot", "miss", "shooting_foul", "turnover", "pass_turnover", "shot_clock_violation", "period_expired"}
        self.assertTrue(all(event["kind"] in kinds for event in game["events"]))
        self.assertGreater(game["possession_count"], 0)
        self.assertLessEqual(abs(game["possessions_by_team"][0] - game["possessions_by_team"][1]), 1)


if __name__ == "__main__":
    unittest.main()
