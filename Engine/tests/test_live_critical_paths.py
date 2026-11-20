"""Correctness tests for uncovered live-engine paths in era_engine.core.

These tests use the published clock, rotation, and timing rules. They do not
add an experimental possession model.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine.cli import main
from era_engine.core import (
    Era,
    PossessionClock,
    RegulationClock,
    Team,
    TimingAssumptions,
    _charge_live,
    estimate_season,
    game_clock_stopped_for_start,
    made_basket_stops_clock,
    shot_weights,
    simulate_game,
    simulate_series,
    weighted_pick,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


def _six_player_team(prefix, template):
    ids = tuple(f"{prefix}{index}" for index in range(6))
    roster = {
        player_id: replace(template, player_id=player_id, name=player_id, stamina=80)
        for player_id in ids
    }
    return Team(prefix, ids, "balanced"), roster


class ClockContractTests(unittest.TestCase):
    def test_made_basket_at_zero_does_not_stop_the_clock_again(self):
        cases = (
            (1, 0.0, False),
            (3, -1.0, False),
            (4, 0.0, False),
            (1, 60.0, True),
            (2, 30.0, True),
            (3, 60.01, False),
            (4, 120.0, True),
            (6, 90.0, True),
            (4, 120.01, False),
        )
        for period, remaining, stopped in cases:
            with self.subTest(period=period, remaining=remaining):
                self.assertEqual(made_basket_stops_clock(period, remaining), stopped)

    def test_legal_possession_starts_and_an_unknown_start(self):
        self.assertFalse(game_clock_stopped_for_start("live_ball", True))
        self.assertFalse(game_clock_stopped_for_start("made_basket_inbound", False))
        self.assertTrue(game_clock_stopped_for_start("made_basket_inbound", True))
        for start in ("period_start", "jump_ball", "dead_ball_inbound"):
            with self.subTest(start=start):
                self.assertTrue(game_clock_stopped_for_start(start, False))
        with self.assertRaises(ValueError) as caught:
            game_clock_stopped_for_start("tip", False)
        self.assertIn("unknown possession start", str(caught.exception))

    def test_shared_charge_stops_on_the_shorter_clock(self):
        game = RegulationClock(12)
        shot = PossessionClock(8)
        with self.assertRaises(ValueError):
            _charge_live(game, shot, -1)
        self.assertEqual(game.quarter_remaining, 12)
        self.assertEqual(shot.remaining, 8)

        elapsed, reason = _charge_live(game, shot, 5)
        self.assertEqual((elapsed, reason), (5.0, None))
        self.assertEqual(shot.remaining, 3)
        self.assertEqual(game.quarter_remaining, 7)

        elapsed, reason = _charge_live(game, shot, 10)
        self.assertEqual(reason, "shot_clock")
        self.assertEqual(elapsed, 3.0)
        self.assertEqual(shot.remaining, 0.0)
        self.assertEqual(game.quarter_remaining, 4)

        period = RegulationClock(2)
        longer_shot = PossessionClock(10)
        elapsed, reason = _charge_live(period, longer_shot, 5)
        self.assertEqual(reason, "period")
        self.assertEqual(elapsed, 2.0)
        self.assertEqual(longer_shot.remaining, 8.0)
        self.assertEqual(period.quarter_remaining, 0.0)

    def test_quarter_lengths_before_regulation_is_complete(self):
        clock = RegulationClock(10, 4)
        clock.advance(4)
        self.assertEqual(clock.quarter_lengths(), [4.0, 0.0, 0.0, 0.0])
        clock.advance(6)
        self.assertTrue(clock.start_next_quarter())
        clock.advance(1)
        self.assertEqual(clock.quarter_lengths(), [10.0, 1.0, 0.0, 0.0])
        self.assertFalse(clock.regulation_complete)

    def test_clock_construction_and_a_second_overtime_start_are_rejected(self):
        for quarter_length, quarters, overtime_length in ((0, 4, 300), (-5, 4, 300), (720, 0, 300), (720, 4, 0), (720, 4, -1)):
            with self.subTest(quarter_length=quarter_length, quarters=quarters, overtime_length=overtime_length):
                with self.assertRaises(ValueError):
                    RegulationClock(quarter_length, quarters, overtime_length=overtime_length)
        clock = RegulationClock(10, 4)
        for _ in range(3):
            clock.advance(10)
            self.assertTrue(clock.start_next_quarter())
        clock.advance(10)
        self.assertEqual(clock.start_overtime(), 1)
        with self.assertRaises(ValueError) as caught:
            clock.start_overtime()
        self.assertIn("already running", str(caught.exception))
        self.assertTrue(clock.in_overtime)
        self.assertEqual(clock.overtime_period, 1)
        self.assertEqual(clock.overtime_remaining, 300.0)

    def test_setup_charge_is_at_most_one_quarter_of_a_fresh_shot_clock(self):
        timing = TimingAssumptions()
        for limit in (1, 8, 14, 24):
            with self.subTest(limit=limit):
                setup = timing.setup_seconds(limit)
                self.assertGreater(setup, 0)
                self.assertLess(setup, limit)
        self.assertEqual(timing.setup_seconds(24), 6.0)
        self.assertEqual(timing.setup_seconds(1), 0.25)

    def test_non_positive_shot_weights_are_rejected(self):
        import random

        for weights in ({}, {"rim": 0.0}, {"rim": -2.0, "paint": -1.0}):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    weighted_pick(random.Random(1), weights)

    def test_invalid_era_bounds_are_rejected_and_the_edges_remain_legal(self):
        era = Era("2015-16", 96.0, 0.33, 0.32, True)
        for changes in (
            {"physicality": -0.01},
            {"physicality": 1.01},
            {"three_distance_factor": 0},
            {"three_distance_factor": -1},
            {"game_minutes": 0},
            {"game_minutes": -4},
            {"game_minutes": 48.5},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(era, **changes)
        self.assertEqual(replace(era, physicality=0).physicality, 0)
        self.assertEqual(replace(era, physicality=1).physicality, 1)
        self.assertEqual(replace(era, game_minutes=1).game_minutes, 1)


class LiveCriticalPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "test.db"))
        initialize_demo(self.con)
        self.players, self.eras, self.teams = load(self.con)
        self.team_a = self.teams["Bay City Legends"]
        self.team_b = self.teams["Chicago Stampede"]
        self.era = self.eras["2015-16"]

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_both_clocks_start_together_on_the_legal_touch(self):
        game = simulate_game(
            self.team_a,
            self.team_b,
            self.players,
            self.era,
            7,
            equalize_periods=("regulation",),
        )
        again = simulate_game(
            self.team_a,
            self.team_b,
            self.players,
            self.era,
            7,
            equalize_periods=("regulation",),
        )
        self.assertEqual(game, again)
        setup = TimingAssumptions().setup_seconds(self.era.shot_clock)
        self.assertEqual(game["possession_starts"]["stopped_inbound_shot_clock_seconds"], 0.0)
        self.assertIn("legally touched", game["possession_starts"]["simplification"])
        by_possession = {}
        for event in game["events"]:
            by_possession.setdefault(event["possession"], []).append(event)
        saw_stopped_make = False
        saw_running_make = False
        saw_period = False
        saw_jump = False
        saw_dead = False
        saw_live = False
        for sequence in by_possession.values():
            first = sequence[0]
            self.assertAlmostEqual(first["setup_game_clock_seconds"], first["setup_shot_clock_seconds"])
            self.assertGreater(first["setup_game_clock_seconds"], 0.0)
            self.assertLessEqual(first["setup_game_clock_seconds"], setup + 1e-9)
            self.assertLessEqual(
                first["shot_clock_remaining"],
                self.era.shot_clock - first["setup_shot_clock_seconds"] + 1e-6,
            )
            if first["game_clock_stopped_at_start"]:
                self.assertIn(
                    first["possession_start"],
                    ("period_start", "jump_ball", "dead_ball_inbound", "made_basket_inbound"),
                )
            else:
                self.assertIn(first["possession_start"], ("live_ball", "made_basket_inbound"))
            if first["possession_start"] == "period_start":
                saw_period = True
            elif first["possession_start"] == "jump_ball":
                saw_jump = True
            elif first["possession_start"] == "dead_ball_inbound":
                saw_dead = True
            elif first["possession_start"] == "live_ball":
                saw_live = True
        for number, sequence in by_possession.items():
            if number + 1 not in by_possession:
                continue
            last = sequence[-1]
            if last["kind"] != "made_shot" or last["period_ending"]:
                continue
            nxt = by_possession[number + 1][0]
            self.assertEqual(nxt["possession_start"], "made_basket_inbound")
            self.assertAlmostEqual(nxt["setup_game_clock_seconds"], nxt["setup_shot_clock_seconds"])
            if last["clock_stopped"]:
                self.assertTrue(nxt["game_clock_stopped_at_start"])
                saw_stopped_make = True
            else:
                self.assertFalse(nxt["game_clock_stopped_at_start"])
                saw_running_make = True
        self.assertTrue(saw_period)
        self.assertTrue(saw_jump)
        self.assertTrue(saw_dead)
        self.assertTrue(saw_live)
        self.assertTrue(saw_stopped_make)
        self.assertTrue(saw_running_make)
        self.assertAlmostEqual(sum(game["live_seconds"].values()), game["game_clock_seconds"], delta=1e-3)
        self.assertGreater(game["live_seconds"].get("setup", 0.0), 0.0)
        periods = {}
        for event in game["events"]:
            periods.setdefault(event["period"], event)
        self.assertEqual(
            {event["possession_start"] for period, event in periods.items() if period.startswith("Q")},
            {"period_start"},
        )
        self.assertEqual(
            {event["possession_start"] for period, event in periods.items() if period.startswith("OT")},
            {"jump_ball"},
        )

    def test_a_pass_that_outruns_the_shot_clock_is_a_turnover(self):
        timing = TimingAssumptions(completed_pass_seconds=8.0)
        pressed = replace(self.era, shot_clock=10)
        game = simulate_game(self.team_a, self.team_b, self.players, pressed, 0, timing=timing)
        again = simulate_game(self.team_a, self.team_b, self.players, pressed, 0, timing=timing)
        self.assertEqual(game, again)
        violations = [
            event for event in game["events"]
            if event["kind"] == "shot_clock_violation" and event["passes"] > 0
        ]
        self.assertGreater(len(violations), 0)
        for event in violations:
            self.assertEqual(event["shot_clock_remaining"], 0.0)
            self.assertGreater(event["quarter_remaining"], 0.0)
            self.assertIsNotNone(event["offense"])
            self.assertNotIn("points", event)
            self.assertGreater(game["player_stats"][event["offense"]].get("tov", 0), 0)
        attempts = sum(stats.get("fga", 0) for stats in game["player_stats"].values())
        self.assertGreater(attempts, 0)
        for index, team in enumerate((self.team_a, self.team_b)):
            self.assertEqual(
                game["score"][index],
                sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
            )

    def test_a_pass_can_end_the_quarter_while_the_shot_clock_still_has_time(self):
        game = simulate_game(self.team_a, self.team_b, self.players, self.era, 0)
        by_possession = {}
        for event in game["events"]:
            by_possession.setdefault(event["possession"], []).append(event)
        horn_passes = [
            event for event in game["events"]
            if event["kind"] == "period_expired" and event["passes"] > 0
        ]
        self.assertGreaterEqual(len(horn_passes), 1)
        for event in horn_passes:
            sequence = by_possession[event["possession"]]
            self.assertEqual(event["quarter_remaining"], 0.0)
            self.assertGreater(event["shot_clock_remaining"], 0.0)
            self.assertNotIn("made_shot", {item["kind"] for item in sequence})
            self.assertNotIn("points", event)
        self.assertEqual(game["quarter_lengths"], [720.0, 720.0, 720.0, 720.0])

    def test_a_made_basket_on_the_horn_does_not_also_stop_the_clock(self):
        timing = TimingAssumptions(release_low_seconds=2.0, release_high_seconds=2.0)
        game = simulate_game(self.team_a, self.team_b, self.players, self.era, 19, timing=timing)
        again = simulate_game(self.team_a, self.team_b, self.players, self.era, 19, timing=timing)
        self.assertEqual(game, again)
        by_possession = {}
        for event in game["events"]:
            by_possession.setdefault(event["possession"], []).append(event)
        horn_makes = [
            event for event in game["events"]
            if event["kind"] == "made_shot" and event["quarter_remaining"] == 0.0
        ]
        self.assertGreaterEqual(len(horn_makes), 1)
        for make in horn_makes:
            self.assertFalse(make["clock_stopped"])
            self.assertGreater(make["points"], 0)
            self.assertFalse(made_basket_stops_clock(make["quarter"], make["quarter_remaining"]))
            following = by_possession.get(make["possession"] + 1)
            if following is not None:
                self.assertEqual(following[0]["possession_start"], "period_start")
                self.assertTrue(following[0]["game_clock_stopped_at_start"])
        for index, team in enumerate((self.team_a, self.team_b)):
            self.assertEqual(
                game["score"][index],
                sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
            )

    def test_track_events_false_keeps_the_result_and_drops_the_log(self):
        tracked = simulate_game(self.team_a, self.team_b, self.players, self.era, 7)
        quiet = simulate_game(self.team_a, self.team_b, self.players, self.era, 7, track_events=False)
        self.assertNotEqual(tracked["events"], [])
        self.assertEqual(quiet["events"], [])
        self.assertEqual(quiet["score"], tracked["score"])
        self.assertEqual(quiet["player_stats"], tracked["player_stats"])
        self.assertEqual(quiet["possession_count"], tracked["possession_count"])
        self.assertEqual(quiet["regulation_score"], tracked["regulation_score"])
        self.assertEqual(quiet["game_clock_seconds"], tracked["game_clock_seconds"])

    def test_a_six_player_roster_uses_equal_time_without_minute_targets(self):
        home, home_players = _six_player_team("H", self.players["ERA_0001"])
        away, away_players = _six_player_team("A", self.players["ERA_0006"])
        roster = {**home_players, **away_players}
        game = simulate_game(home, away, roster, self.era, 3)
        again = simulate_game(home, away, roster, self.era, 3)
        self.assertEqual(game, again)
        self.assertEqual(game["rotation_mode"], ["scheduled", "scheduled"])
        periods = game["overtime"]["periods"]
        self.assertEqual(game["regulation_seconds"], 2880.0)
        self.assertGreater(game["substitution_counts"][0], 0)
        self.assertGreater(game["substitution_counts"][1], 0)
        for event in game["events"]:
            self.assertEqual(len(event["offense_lineup"]), 5)
            self.assertEqual(len(set(event["offense_lineup"])), 5)
            self.assertEqual(len(event["defense_lineup"]), 5)
        for team in (home, away):
            minutes = {pid: game["player_stats"][pid]["min"] for pid in team.player_ids}
            self.assertAlmostEqual(sum(minutes.values()), 240.0 + 25.0 * periods, delta=1e-3)
            self.assertTrue(all(value > 0.0 for value in minutes.values()))
            self.assertTrue(all(game["player_stats"][pid]["gp"] == 1 for pid in team.player_ids))

    def test_post_coaching_raises_rim_and_paint_weights_without_changing_the_others(self):
        player = self.players["ERA_0001"]
        balanced = shot_weights(player, self.era, "balanced", 0.0)
        post = shot_weights(player, self.era, "post", 0.0)
        self.assertAlmostEqual(post["rim"], balanced["rim"] * 1.15)
        self.assertAlmostEqual(post["paint"], balanced["paint"] * 1.18)
        self.assertAlmostEqual(post["mid"], balanced["mid"])
        self.assertAlmostEqual(post["three"], balanced["three"])
        home = replace(self.team_a, coach_style="post")
        away = replace(self.team_b, coach_style="post")
        game = simulate_game(home, away, self.players, self.era, 7)
        again = simulate_game(home, away, self.players, self.era, 7)
        self.assertEqual(game, again)
        self.assertTrue(game["completed"])
        self.assertGreater(sum(stats.get("fga", 0) for stats in game["player_stats"].values()), 0)
        for index, team in enumerate((home, away)):
            self.assertEqual(
                game["score"][index],
                sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
            )

    def test_five_players_with_full_minute_targets_stay_in_overtime(self):
        targets = (
            {pid: 48.0 for pid in self.team_a.player_ids},
            {pid: 48.0 for pid in self.team_b.player_ids},
        )
        game = simulate_game(
            self.team_a,
            self.team_b,
            self.players,
            self.era,
            7,
            rotation_targets=targets,
            equalize_periods=("regulation",),
        )
        listed = {tuple(self.team_a.player_ids), tuple(self.team_b.player_ids)}
        self.assertEqual(game["rotation_mode"], ["scheduled", "scheduled"])
        self.assertEqual(game["substitution_counts"], [0, 0])
        self.assertEqual(game["overtime"]["periods"], 1)
        self.assertEqual(game["regulation_score"], [139, 139])
        self.assertEqual(game["score"], [154, 146])
        self.assertEqual(len(game["overtime"]["jump_balls"]), 1)
        for event in game["events"]:
            self.assertIn(tuple(event["offense_lineup"]), listed)
            self.assertIn(tuple(event["defense_lineup"]), listed)
        for team in (self.team_a, self.team_b):
            minutes = [game["player_stats"][pid]["min"] for pid in team.player_ids]
            self.assertTrue(all(abs(value - 53.0) < 1e-6 for value in minutes))

    def test_zero_rebound_gather_still_credits_one_rebound_and_no_gather_time(self):
        game = simulate_game(
            self.team_a,
            self.team_b,
            self.players,
            self.era,
            7,
            timing=TimingAssumptions(rebound_gather_seconds=0.0),
        )
        misses = [event for event in game["events"] if event["kind"] == "miss"]
        self.assertGreater(len(misses), 0)
        for event in misses:
            self.assertIn(event["rebounder"], event["offense_lineup"] + event["defense_lineup"])
        self.assertEqual(game["live_seconds"].get("rebound_gather", 0.0), 0.0)
        for stats in game["player_stats"].values():
            self.assertEqual(stats.get("reb", 0), stats.get("oreb", 0) + stats.get("dreb", 0))
        periods = game["overtime"]["periods"]
        self.assertEqual(game["game_clock_seconds"], 2880.0 + 300.0 * periods)

    def test_invalid_game_inputs_raise_before_a_result_is_returned(self):
        missing = {pid: player for pid, player in self.players.items() if pid != "ERA_0001"}
        with self.assertRaises(ValueError) as caught:
            simulate_game(self.team_a, self.team_b, missing, self.era, 7)
        self.assertIn("Unknown player", str(caught.exception))

        duplicate = list(self.team_b.player_ids)
        duplicate[1] = duplicate[0]
        with self.assertRaises(ValueError) as caught:
            simulate_game(self.team_a, replace(self.team_b, player_ids=tuple(duplicate)), self.players, self.era, 7)
        self.assertIn("Duplicate player on team B", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            simulate_game(self.team_a, self.team_b, self.players, self.era, 7, equalize_periods=("Q1",))
        self.assertIn("equalize_periods", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            simulate_game(self.team_a, self.team_b, self.players, self.era, 7, rotation_targets=({"ERA_0001": 48.0},))
        self.assertIn("one budget for each team", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            simulate_game(
                self.team_a,
                self.team_b,
                self.players,
                self.era,
                7,
                rotation_targets=({"NOT_ON_THE_TEAM": 48.0}, None),
            )
        self.assertIn("not on the team", str(caught.exception))

        home, home_players = _six_player_team("H", self.players["ERA_0001"])
        away, away_players = _six_player_team("A", self.players["ERA_0006"])
        with self.assertRaises(ValueError) as caught:
            simulate_game(
                home,
                away,
                {**home_players, **away_players},
                replace(self.era, game_minutes=40),
                7,
            )
        self.assertIn("48-minute", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            simulate_game(
                self.team_a,
                self.team_b,
                self.players,
                self.era,
                7,
                timing=TimingAssumptions(setup_cap_seconds=-1),
            )
        self.assertIn("cannot be negative", str(caught.exception))

        for games in (0, 83, -1):
            with self.subTest(games=games):
                with self.assertRaises(ValueError):
                    simulate_series(self.team_a, self.team_b, self.players, self.era, games=games)


class ProductionEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "test.db"))
        initialize_demo(self.con)
        self.players, self.eras, self.teams = load(self.con)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_estimate_season_accepts_two_runs_and_rejects_the_bounds(self):
        for runs in (1, 0, 5001):
            with self.subTest(runs=runs):
                with self.assertRaises(ValueError):
                    estimate_season(
                        self.teams["Bay City Legends"],
                        self.teams["Chicago Stampede"],
                        self.players,
                        self.eras["2015-16"],
                        runs=runs,
                    )
        first = estimate_season(
            self.teams["Bay City Legends"],
            self.teams["Chicago Stampede"],
            self.players,
            self.eras["2015-16"],
            runs=2,
            seed=1,
        )
        second = estimate_season(
            self.teams["Bay City Legends"],
            self.teams["Chicago Stampede"],
            self.players,
            self.eras["2015-16"],
            runs=2,
            seed=1,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["runs"], 2)
        self.assertIn("NOT real-world", first["note"])
        expected = set(self.teams["Bay City Legends"].player_ids + self.teams["Chicago Stampede"].player_ids)
        self.assertEqual(set(first["players"]), expected)
        for summary in first["players"].values():
            self.assertLessEqual(summary["simulation_p05"], summary["points_per_game_mean"])
            self.assertLessEqual(summary["points_per_game_mean"], summary["simulation_p95"])

    def test_cli_init_simulate_and_project_use_temporary_files(self):
        directory = Path(self.tmp.name)
        database = directory / "cli.db"
        results = directory / "results.json"
        projection = directory / "projection.json"

        init_output = self._run_cli("init", str(database))
        self.assertIn("Created sample DB", init_output)
        self.assertTrue(database.is_file())

        simulate_output = self._run_cli(
            "simulate",
            str(database),
            "--games",
            "1",
            "--seed",
            "7",
            "--output",
            str(results),
        )
        self.assertIn(str(results), simulate_output)
        payload = json.loads(results.read_text())
        self.assertEqual(payload["games"], 1)
        self.assertEqual(payload["era"], "1995-96")
        self.assertIn("simulation_id", payload)
        self.assertIn("ERA_0001", payload["averages"])

        project_output = self._run_cli(
            "project",
            str(database),
            "--runs",
            "2",
            "--output",
            str(projection),
        )
        self.assertIn(str(projection), project_output)
        projected = json.loads(projection.read_text())
        self.assertEqual(projected["runs"], 2)
        self.assertIn("ERA_0001", projected["players"])
        self.assertTrue(projected["note"])

    def _run_cli(self, *args):
        buffer = io.StringIO()
        previous = sys.argv
        sys.argv = ["era_engine.cli", *args]
        try:
            with contextlib.redirect_stdout(buffer):
                main()
        finally:
            sys.argv = previous
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
