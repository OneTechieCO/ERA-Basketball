"""Overtime and legal substitution windows in the live engine."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine.core import (
    JUMP_BALL_MODEL,
    LATE_MAKE_STOP_FINAL_PERIOD,
    LATE_MAKE_STOP_QUARTERS_1_TO_3,
    OVERTIME_PERIOD_SECONDS,
    OVERTIME_RUNAWAY_LIMIT,
    RegulationClock,
    Team,
    UnresolvedGameError,
    jump_ball_winner,
    simulate_game,
    simulate_series,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


def _bench_roster(prefix, template):
    ids = tuple(f"{prefix}{index}" for index in range(8))
    roster = {}
    for index, player_id in enumerate(ids):
        roster[player_id] = replace(
            template,
            player_id=player_id,
            name=player_id,
            stamina=25 if index == 0 else 90,
        )
    targets = {player_id: 36.0 for player_id in ids[:5]}
    targets[ids[5]] = 30.0
    targets[ids[6]] = 30.0
    targets[ids[7]] = 0.0
    return Team(prefix, ids, "balanced"), roster, targets


def _team_five(event, prefix):
    for lineup in (event["offense_lineup"], event["defense_lineup"]):
        if lineup[0].startswith(prefix):
            return tuple(lineup)
    raise AssertionError(event["possession"])


class OvertimeClockTests(unittest.TestCase):
    def test_overtime_is_opt_in_and_cannot_run_below_zero(self):
        clock = RegulationClock(720, 4)
        for _ in range(4):
            self.assertEqual(clock.advance(720), 720)
            if clock.quarter < 4:
                self.assertTrue(clock.start_next_quarter())
        self.assertTrue(clock.regulation_complete)
        self.assertEqual(clock.advance(5), 0.0)
        self.assertEqual(clock.elapsed, 2880.0)
        with self.assertRaises(ValueError):
            RegulationClock(720, 4).start_overtime()
        self.assertEqual(clock.start_overtime(), 1)
        self.assertEqual(clock.advance(500), 300.0)
        self.assertEqual(clock.overtime_remaining, 0.0)
        self.assertEqual(clock.period_remaining(), 0.0)
        self.assertFalse(clock.in_overtime)
        self.assertEqual(clock.completed_overtime, 1)
        self.assertEqual(clock.elapsed, 3180.0)
        self.assertEqual(clock.advance(5), 0.0)
        self.assertEqual(clock.quarter_lengths(), [720.0, 720.0, 720.0, 720.0])
        self.assertEqual(clock.start_overtime(), 2)
        self.assertEqual(clock.advance(10), 10.0)
        self.assertEqual(clock.period_remaining(), 290.0)
        self.assertEqual(clock.elapsed, 3190.0)

    def test_jump_ball_uses_rebounding_and_does_not_draw_a_random_number(self):
        home = replace(self_player(), player_id="H", rebounding=80)
        away = replace(home, player_id="A", rebounding=70)
        roster = {"H": home, "A": away, "H2": replace(home, player_id="H2", rebounding=80)}
        self.assertEqual(jump_ball_winner(("H",), ("A",), roster), 0)
        self.assertEqual(jump_ball_winner(("A",), ("H",), roster), 1)
        low = replace(home, player_id="M", rebounding=80)
        high = replace(home, player_id="Z", rebounding=80)
        tied = {"M": low, "Z": high}
        self.assertEqual(jump_ball_winner(("M",), ("Z",), tied), 0)
        self.assertEqual(jump_ball_winner(("Z",), ("M",), tied), 1)
        self.assertIn("does not draw a random number", JUMP_BALL_MODEL)


def self_player():
    from era_engine.seed import demo_players
    return demo_players()[0]


class LiveOvertimeTests(unittest.TestCase):
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

    def test_a_regulation_game_ends_without_overtime(self):
        game = simulate_game(self.team_a, self.team_b, self.players, self.era, 0)
        again = simulate_game(self.team_a, self.team_b, self.players, self.era, 0)
        self.assertEqual(game, again)
        self.assertEqual(game["score"], [120, 131])
        self.assertEqual(game["regulation_score"], [120, 131])
        self.assertEqual(game["overtime"]["periods"], 0)
        self.assertEqual(game["game_clock"], "regulation")
        self.assertEqual(game["game_clock_seconds"], 2880.0)
        self.assertEqual(game["regulation_seconds"], 2880.0)
        self.assertEqual(game["period_ending_possessions"], 4)
        self.assertEqual([row["period"] for row in game["period_scores"]], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(game["period_scores"][-1]["running_score"], [120, 131])
        self.assertEqual(game["overtime"]["fixture_points"], [])
        self.assertFalse(game["overtime"]["tie_broken_by_rule"])
        self.assertFalse(game["overtime"]["unresolved_tie"])
        self._assert_closed_book(game, periods=0)
        for team in (self.team_a, self.team_b):
            minutes = [game["player_stats"][pid]["min"] for pid in team.player_ids]
            self.assertTrue(all(abs(value - 48.0) < 1e-6 for value in minutes))
            self.assertAlmostEqual(sum(minutes), 240.0, delta=1e-4)

    def test_a_tied_regulation_plays_one_overtime_until_there_is_a_winner(self):
        game = simulate_game(
            self.team_a, self.team_b, self.players, self.era, 7,
            equalize_periods=("regulation",),
        )
        self.assertEqual(game["regulation_score"], [139, 139])
        self.assertEqual(game["overtime"]["periods"], 1)
        self.assertEqual(game["score"], [154, 146])
        self.assertTrue(game["completed"])
        self.assertNotEqual(game["score"][0], game["score"][1])
        self.assertEqual(game["game_clock"], "overtime")
        self.assertEqual(game["period_scores"][-1]["period"], "OT1")
        self.assertNotEqual(game["period_scores"][-1]["points"][0], game["period_scores"][-1]["points"][1])
        self.assertEqual(game["overtime"]["jump_balls"][0]["period"], "OT1")
        self.assertEqual(len(game["overtime"]["jump_balls"]), 1)
        self.assertFalse(game["overtime"]["tie_broken_by_rule"])
        self.assertFalse(game["overtime"]["timeouts_modeled"])
        self.assertEqual(
            [(row["period"], row["points"]) for row in game["overtime"]["fixture_points"]],
            [("regulation", 16)],
        )
        self._assert_closed_book(game, periods=1)
        self._assert_fresh_period_clocks(game)
        for team in (self.team_a, self.team_b):
            minutes = [game["player_stats"][pid]["min"] for pid in team.player_ids]
            self.assertTrue(all(abs(value - 53.0) < 1e-6 for value in minutes))
            self.assertAlmostEqual(sum(minutes), 265.0, delta=1e-4)
            self.assertTrue(all(value > 48.0 for value in minutes))

    def test_two_consecutive_overtime_periods_keep_the_clock_and_minutes(self):
        game = simulate_game(
            self.team_a, self.team_b, self.players, self.era, 7,
            equalize_periods=("regulation", "OT1"),
        )
        self.assertEqual(game["overtime"]["periods"], 2)
        self.assertEqual(game["regulation_score"], [139, 139])
        self.assertEqual(game["score"], [168, 166])
        self.assertTrue(game["completed"])
        self.assertEqual(game["game_clock_seconds"], 2880.0 + 2 * OVERTIME_PERIOD_SECONDS)
        labels = [row["period"] for row in game["period_scores"]]
        self.assertEqual(labels, ["Q1", "Q2", "Q3", "Q4", "OT1", "OT2"])
        tied = {row["period"]: row["points"] for row in game["period_scores"]}
        self.assertEqual(tied["OT1"][0], tied["OT1"][1])
        self.assertNotEqual(tied["OT2"][0], tied["OT2"][1])
        self.assertEqual(
            [(row["period"], row["points"]) for row in game["overtime"]["fixture_points"]],
            [("regulation", 16), ("OT1", 8)],
        )
        self.assertFalse(game["overtime"]["tie_broken_by_rule"])
        self.assertFalse(game["overtime"]["unresolved_tie"])
        self._assert_closed_book(game, periods=2)
        self._assert_fresh_period_clocks(game)
        for team in (self.team_a, self.team_b):
            minutes = [game["player_stats"][pid]["min"] for pid in team.player_ids]
            self.assertTrue(all(abs(value - 58.0) < 1e-6 for value in minutes))
            self.assertAlmostEqual(sum(minutes), 290.0, delta=1e-4)

    def test_a_larger_roster_extends_rotations_without_inventing_players(self):
        home, home_players, home_targets = _bench_roster("H", self.players["ERA_0001"])
        away, away_players, away_targets = _bench_roster("A", self.players["ERA_0006"])
        roster = {**home_players, **away_players}
        game = simulate_game(
            home, away, roster, self.era, 7,
            rotation_targets=(home_targets, away_targets),
            equalize_periods=("regulation",),
        )
        periods = game["overtime"]["periods"]
        self.assertGreaterEqual(periods, 1)
        self.assertNotEqual(game["score"][0], game["score"][1])
        self._assert_closed_book(game, periods=periods)
        self._assert_legal_substitutions(game, ("H", "A"))
        for event in game["events"]:
            self.assertEqual(len(event["offense_lineup"]), 5)
            self.assertEqual(len(set(event["offense_lineup"])), 5)
            self.assertEqual(len(event["defense_lineup"]), 5)
            self.assertNotIn("H7", event["offense_lineup"] + event["defense_lineup"])
            self.assertNotIn("A7", event["offense_lineup"] + event["defense_lineup"])
        for index, team in enumerate((home, away)):
            zero = "H7" if index == 0 else "A7"
            minutes = {pid: game["player_stats"][pid]["min"] for pid in team.player_ids}
            self.assertAlmostEqual(sum(minutes.values()), 240.0 + 25.0 * periods, delta=1e-3)
            self.assertEqual(minutes[zero], 0.0)
            self.assertEqual(
                game["score"][index],
                sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
            )
            self.assertTrue(all(value > 0.0 for pid, value in minutes.items() if pid != zero))

    def test_substitutions_do_not_follow_a_live_ball(self):
        home, home_players, home_targets = _bench_roster("H", self.players["ERA_0001"])
        away, away_players, away_targets = _bench_roster("A", self.players["ERA_0006"])
        roster = {**home_players, **away_players}
        game = simulate_game(
            home, away, roster, self.era, 7,
            rotation_targets=(home_targets, away_targets),
        )
        self._assert_legal_substitutions(game, ("H", "A"))
        saw_live_ball = False
        by_possession = {}
        for event in game["events"]:
            by_possession.setdefault(event["possession"], []).append(event)
        numbers = sorted(by_possession)
        for previous, current in zip(numbers, numbers[1:]):
            last = by_possession[previous][-1]
            if last["period_ending"] or last["kind"] in ("shooting_foul", "shot_clock_violation"):
                continue
            self.assertIn(last["kind"], ("turnover", "pass_turnover", "made_shot", "miss"))
            self.assertEqual(_team_five(last, "H"), _team_five(by_possession[current][0], "H"))
            self.assertEqual(_team_five(last, "A"), _team_five(by_possession[current][0], "A"))
            saw_live_ball = True
        self.assertTrue(saw_live_ball)
        self.assertFalse(game["overtime"]["timeouts_modeled"])
        self.assertIn("made field goal", game["overtime"]["made_field_goal_limitation"])

    def test_an_unresolved_tie_does_not_receive_a_winning_point(self):
        era = replace(self.era, shot_clock=1)
        with self.assertRaises(UnresolvedGameError) as caught:
            simulate_game(self.team_a, self.team_b, self.players, era, 19)
        game = caught.exception.result
        self.assertIn("No tie-break point", str(caught.exception))
        self.assertEqual(game["score"], [0, 0])
        self.assertEqual(game["regulation_score"], [0, 0])
        self.assertFalse(game["completed"])
        self.assertEqual(game["status"], "incomplete")
        self.assertEqual(game["overtime"]["periods"], OVERTIME_RUNAWAY_LIMIT)
        self.assertTrue(game["overtime"]["unresolved_tie"])
        self.assertFalse(game["overtime"]["tie_broken_by_rule"])
        self.assertEqual(game["overtime"]["fixture_points"], [])
        self.assertEqual(game["game_clock_seconds"], 2880.0 + OVERTIME_PERIOD_SECONDS * OVERTIME_RUNAWAY_LIMIT)
        self._assert_closed_book(game, periods=OVERTIME_RUNAWAY_LIMIT)
        with self.assertRaises(UnresolvedGameError):
            simulate_series(self.team_a, self.team_b, self.players, era, games=1, seed=19)

    def test_possession_starts_control_the_clock_and_substitutions(self):
        game = simulate_game(
            self.team_a, self.team_b, self.players, self.era, 7,
            equalize_periods=("regulation",),
        )
        by_possession = {}
        for event in game["events"]:
            by_possession.setdefault(event["possession"], []).append(event)
        seen = {name: 0 for name in game["possession_starts"]["states"]}
        running_makes = 0
        stopped_makes = 0
        for number, sequence in by_possession.items():
            first = sequence[0]
            seen[first["possession_start"]] += 1
            self.assertEqual(len({event["possession_start"] for event in sequence}), 1)
            self.assertAlmostEqual(first["setup_game_clock_seconds"], first["setup_shot_clock_seconds"])
            self.assertGreater(first["setup_game_clock_seconds"], 0.0)
            if first["game_clock_stopped_at_start"]:
                self.assertIn(
                    first["possession_start"],
                    ("period_start", "jump_ball", "dead_ball_inbound", "made_basket_inbound"),
                )
            else:
                self.assertIn(first["possession_start"], ("live_ball", "made_basket_inbound"))
            if number + 1 not in by_possession:
                continue
            last = sequence[-1]
            nxt = by_possession[number + 1][0]
            if last["period_ending"]:
                self.assertIn(nxt["possession_start"], ("period_start", "jump_ball"))
                continue
            if last["kind"] == "made_shot":
                limit = LATE_MAKE_STOP_FINAL_PERIOD if last["quarter"] >= 4 else LATE_MAKE_STOP_QUARTERS_1_TO_3
                self.assertEqual(nxt["possession_start"], "made_basket_inbound")
                self.assertEqual(last["clock_stopped"], nxt["game_clock_stopped_at_start"])
                self.assertEqual(set(last["offense_lineup"]), set(nxt["defense_lineup"]))
                self.assertEqual(set(last["defense_lineup"]), set(nxt["offense_lineup"]))
                if last["quarter_remaining"] <= limit:
                    self.assertTrue(last["clock_stopped"])
                    stopped_makes += 1
                else:
                    self.assertFalse(last["clock_stopped"])
                    running_makes += 1
            elif last["kind"] in ("turnover", "pass_turnover") or (
                last["kind"] == "miss" and not last.get("offensive_rebound")
            ):
                self.assertEqual(nxt["possession_start"], "live_ball")
                self.assertFalse(nxt["game_clock_stopped_at_start"])
                self.assertEqual(set(last["offense_lineup"]), set(nxt["defense_lineup"]))
                self.assertEqual(set(last["defense_lineup"]), set(nxt["offense_lineup"]))
            elif last["kind"] in ("shooting_foul", "shot_clock_violation"):
                self.assertEqual(nxt["possession_start"], "dead_ball_inbound")
                self.assertTrue(nxt["game_clock_stopped_at_start"])
                self.assertAlmostEqual(nxt["setup_game_clock_seconds"], nxt["setup_shot_clock_seconds"])
                self.assertGreater(nxt["setup_game_clock_seconds"], 0.0)
        self.assertTrue(all(seen[name] > 0 for name in seen))
        self.assertGreater(running_makes, 0)
        self.assertGreater(stopped_makes, 0)
        self.assertFalse(game["possession_starts"]["made_basket_authorizes_substitution"])
        self.assertFalse(game["possession_starts"]["timeouts_modeled"])
        self.assertEqual(game["substitution_counts"], [0, 0])

    def _assert_closed_book(self, game, periods):
        self.assertEqual(game["game_clock_seconds"], 2880.0 + OVERTIME_PERIOD_SECONDS * periods)
        self.assertEqual(game["regulation_seconds"], 2880.0)
        self.assertAlmostEqual(sum(game["live_seconds"].values()), game["game_clock_seconds"], delta=1e-3)
        self.assertEqual(len(game["period_scores"]), 4 + periods)
        self.assertAlmostEqual(sum(row["seconds"] for row in game["period_scores"]), game["game_clock_seconds"], delta=1e-6)
        points = [0, 0]
        for row in game["period_scores"]:
            points[0] += row["points"][0]
            points[1] += row["points"][1]
            self.assertEqual(row["running_score"], points)
        self.assertEqual(points, game["score"])
        self.assertEqual(game["period_scores"][3]["running_score"], game["regulation_score"])
        self.assertEqual(game["period_ending_possessions"], 4 + periods)
        shot_limit = 24 if game["era"] != self.era.key else self.era.shot_clock
        for event in game["events"]:
            self.assertGreaterEqual(event["shot_clock_remaining"], 0.0)
            self.assertGreaterEqual(event["quarter_remaining"], 0.0)
            self.assertLessEqual(event["shot_clock_remaining"], shot_limit)
            self.assertGreaterEqual(event["quarter"], 1)
            self.assertLessEqual(event["quarter"], 4 + periods)
        rosters = (self.team_a, self.team_b) if game["teams"] == [self.team_a.name, self.team_b.name] else None
        if rosters is not None:
            for index, team in enumerate(rosters):
                self.assertEqual(
                    game["score"][index],
                    sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
                )
                self.assertAlmostEqual(
                    sum(game["player_stats"][pid]["min"] for pid in team.player_ids),
                    240.0 + 25.0 * periods,
                    delta=1e-3,
                )
        for stats in game["player_stats"].values():
            self.assertGreaterEqual(stats.get("min", 0), 0.0)
            self.assertLessEqual(stats.get("fgm", 0), stats.get("fga", 0))
            self.assertLessEqual(stats.get("fg3m", 0), stats.get("fg3a", 0))
            self.assertLessEqual(stats.get("ftm", 0), stats.get("fta", 0))
            fgm = stats.get("fgm", 0)
            fg3m = stats.get("fg3m", 0)
            self.assertEqual(stats.get("pts", 0), 2 * (fgm - fg3m) + 3 * fg3m + stats.get("ftm", 0))

    def _assert_fresh_period_clocks(self, game):
        lengths = {"Q1": 720.0, "Q2": 720.0, "Q3": 720.0, "Q4": 720.0}
        for period in range(1, game["overtime"]["periods"] + 1):
            lengths[f"OT{period}"] = OVERTIME_PERIOD_SECONDS
        seen = set()
        for event in game["events"]:
            period = event["period"]
            if period in seen:
                continue
            seen.add(period)
            length = lengths[period]
            self.assertIn(event["possession_start"], ("period_start", "jump_ball"))
            self.assertTrue(event["game_clock_stopped_at_start"])
            self.assertAlmostEqual(event["setup_game_clock_seconds"], event["setup_shot_clock_seconds"])
            self.assertGreater(event["setup_game_clock_seconds"], 0.0)
            self.assertGreater(event["quarter_remaining"], 0.0)
            self.assertLessEqual(event["quarter_remaining"], length - event["setup_game_clock_seconds"] + 1e-6)
            self.assertGreaterEqual(event["quarter_remaining"], length - 30.0)
            self.assertLessEqual(event["shot_clock_remaining"], 24.0 - event["setup_shot_clock_seconds"] + 1e-6)
            self.assertGreaterEqual(event["shot_clock_remaining"], 0.0)
        self.assertEqual(seen, set(lengths))

    def _assert_legal_substitutions(self, game, prefixes):
        by_possession = {}
        for event in game["events"]:
            by_possession.setdefault(event["possession"], []).append(event)
            self.assertEqual(len({tuple(event["offense_lineup"]) for event in by_possession[event["possession"]]}), 1)
        numbers = sorted(by_possession)
        for previous, current in zip(numbers, numbers[1:]):
            last = by_possession[previous][-1]
            changed = any(
                _team_five(last, prefix) != _team_five(by_possession[current][0], prefix)
                for prefix in prefixes
            )
            if not changed:
                continue
            self.assertTrue(last["period_ending"] or last["kind"] in ("shooting_foul", "shot_clock_violation"))
        for side in game["substitutions"]:
            for sub in side:
                self.assertIn(sub["reason"], ("period_break", "shooting_foul", "shot_clock_violation"))
                self.assertEqual(len(sub["players_in"]), len(sub["players_out"]))


if __name__ == "__main__":
    unittest.main()
