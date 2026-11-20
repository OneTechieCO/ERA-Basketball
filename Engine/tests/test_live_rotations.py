"""Live rotations in simulate_game. Synthetic benches are local to this file."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine.core import Team, simulate_game
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


def _bench_roster(prefix, template):
    """Eight local players. The last one is given a zero-minute target by the caller."""
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


class LiveRotationTests(unittest.TestCase):
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

    def test_five_player_demo_rosters_stay_on_the_court(self):
        era = self.eras["2015-16"]
        game = simulate_game(self.team_a, self.team_b, self.players, era, 7)
        again = simulate_game(self.team_a, self.team_b, self.players, era, 7)
        self.assertEqual(game, again)
        self.assertEqual(game["rotation_mode"], ["full_game", "full_game"])
        self.assertEqual(game["substitution_counts"], [0, 0])
        self.assertTrue(game["rotations_live"])
        self.assertEqual(game["regulation_seconds"], 2880.0)
        self.assertEqual(game["quarters_completed"], 4)
        self.assertTrue(game["completed"])
        elapsed_minutes = game["game_clock_seconds"] / 60.0
        listed = (tuple(self.team_a.player_ids), tuple(self.team_b.player_ids))
        by_possession = {}
        for event in game["events"]:
            self.assertGreaterEqual(event["shot_clock_remaining"], 0.0)
            self.assertEqual(len(event["offense_lineup"]), 5)
            self.assertIn(tuple(event["offense_lineup"]), listed)
            by_possession.setdefault(event["possession"], []).append(event)
        for sequence in by_possession.values():
            self.assertEqual(len({tuple(event["offense_lineup"]) for event in sequence}), 1)
            self.assertEqual(len({tuple(event["defense_lineup"]) for event in sequence}), 1)
            self.assertEqual(len({event["quarter"] for event in sequence}), 1)
        for team in (self.team_a, self.team_b):
            minutes = [game["player_stats"][pid]["min"] for pid in team.player_ids]
            self.assertTrue(all(abs(value - elapsed_minutes) < 1e-6 for value in minutes))
            self.assertAlmostEqual(sum(minutes), elapsed_minutes * 5, delta=1e-4)
            self.assertTrue(all(game["player_stats"][pid]["gp"] == 1 for pid in team.player_ids))
        self.assertEqual(self.players["ERA_0001"].name, "Sterling Cross")

    def test_a_short_roster_is_rejected(self):
        short = replace(self.team_a, player_ids=self.team_a.player_ids[:4])
        with self.assertRaises(ValueError):
            simulate_game(short, self.team_b, self.players, self.eras["2015-16"], 7)

    def test_substitutes_enter_between_possessions_and_minutes_add_to_240(self):
        era = self.eras["2015-16"]
        home, home_players, home_targets = _bench_roster("H", self.players["ERA_0001"])
        away, away_players, away_targets = _bench_roster("A", self.players["ERA_0006"])
        roster = {**home_players, **away_players}
        game = simulate_game(
            home, away, roster, era, 7,
            rotation_targets=(home_targets, away_targets),
        )
        again = simulate_game(
            home, away, roster, era, 7,
            rotation_targets=(home_targets, away_targets),
        )
        self.assertEqual(game, again)
        self.assertEqual(game["rotation_mode"], ["scheduled", "scheduled"])
        self.assertEqual(game["regulation_seconds"], 2880.0)
        self.assertEqual(game["quarters_completed"], 4)
        self.assertEqual(set(event["quarter"] for event in game["events"]), {1, 2, 3, 4})
        by_possession = {}
        for event in game["events"]:
            self.assertEqual(len(event["offense_lineup"]), 5)
            self.assertEqual(len(set(event["offense_lineup"])), 5)
            self.assertEqual(len(event["defense_lineup"]), 5)
            self.assertGreaterEqual(event["shot_clock_remaining"], 0.0)
            self.assertNotIn("H7", event["offense_lineup"])
            self.assertNotIn("A7", event["defense_lineup"])
            self.assertNotIn("H7", event["defense_lineup"])
            self.assertNotIn("A7", event["offense_lineup"])
            by_possession.setdefault(event["possession"], []).append(event)
        for sequence in by_possession.values():
            self.assertEqual(len({tuple(event["offense_lineup"]) for event in sequence}), 1)
            self.assertEqual(len({tuple(event["defense_lineup"]) for event in sequence}), 1)
        for team, zero in ((home, "H7"), (away, "A7")):
            minutes = {pid: game["player_stats"][pid]["min"] for pid in team.player_ids}
            self.assertAlmostEqual(sum(minutes.values()), 240.0, delta=1e-3)
            self.assertEqual(minutes[zero], 0.0)
            self.assertEqual(game["player_stats"][zero]["gp"], 0)
            self.assertTrue(all(value == 0 for stat, value in game["player_stats"][zero].items() if stat != "gp"))
            for pid, value in minutes.items():
                self.assertLessEqual(value, 48.0 + 1e-6)
                if pid != zero:
                    self.assertGreater(value, 0.0)
        self.assertGreater(game["substitution_counts"][0], 0)
        self.assertGreater(game["substitution_counts"][1], 0)
        self.assertTrue(any(sub["elapsed_seconds"] >= 12 * 60 for side in game["substitutions"] for sub in side))
        for side, team in enumerate((home, away)):
            for sub in game["substitutions"][side]:
                sequence = by_possession[sub["possession"]]
                self.assertEqual(sequence[0]["quarter"], sub["quarter"])
                on_court = set(sequence[0]["offense_lineup"] + sequence[0]["defense_lineup"])
                self.assertTrue(set(sub["players_in"]).issubset(on_court))
                self.assertTrue(set(sub["players_out"]).isdisjoint(on_court))
        seen = set()
        fresh_checked = False
        for event in game["events"]:
            on_court = event["offense_lineup"] + event["defense_lineup"]
            fresh = [pid for pid in on_court if pid not in seen]
            if fresh and seen:
                for pid in fresh:
                    self.assertEqual(event["player_fatigue"][pid], 0.0)
                veterans = [pid for pid in on_court if pid not in fresh]
                self.assertTrue(any(event["player_fatigue"][pid] > event["player_fatigue"][fresh[0]] for pid in veterans))
                fresh_checked = True
                break
            seen.update(on_court)
        self.assertTrue(fresh_checked)
        continued = 0
        for sequence in by_possession.values():
            for index, event in enumerate(sequence):
                if not event.get("offensive_rebound"):
                    continue
                follower = sequence[index + 1]
                self.assertEqual(event["offense_lineup"], follower["offense_lineup"])
                self.assertEqual(event["defense_lineup"], follower["defense_lineup"])
                continued += 1
        self.assertGreater(continued, 0)
        self._assert_lineup_changes_are_legal(game, by_possession)

    def _assert_lineup_changes_are_legal(self, game, by_possession):
        def five(event, prefix):
            for lineup in (event["offense_lineup"], event["defense_lineup"]):
                if lineup[0].startswith(prefix):
                    return tuple(lineup)
            raise AssertionError(event["possession"])

        numbers = sorted(by_possession)
        for previous, current in zip(numbers, numbers[1:]):
            changed = (
                five(by_possession[previous][-1], "H") != five(by_possession[current][0], "H")
                or five(by_possession[previous][-1], "A") != five(by_possession[current][0], "A")
            )
            if not changed:
                continue
            last = by_possession[previous][-1]
            self.assertTrue(last["period_ending"] or last["kind"] in ("shooting_foul", "shot_clock_violation"))
        for side in game["substitutions"]:
            for sub in side:
                self.assertIn(sub["reason"], ("period_break", "shooting_foul", "shot_clock_violation"))

    def test_a_reserve_can_record_points(self):
        era = self.eras["1995-96"]
        home, home_players, home_targets = _bench_roster("H", self.players["ERA_0001"])
        away, away_players, away_targets = _bench_roster("A", self.players["ERA_0006"])
        roster = {**home_players, **away_players}
        reserves = ("H5", "H6", "A5", "A6")
        points = {pid: 0 for pid in reserves}
        for seed in range(6):
            game = simulate_game(
                home, away, roster, era, seed,
                rotation_targets=(home_targets, away_targets),
            )
            for index, team in enumerate((home, away)):
                self.assertEqual(
                    game["score"][index],
                    sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
                )
            for pid in reserves:
                points[pid] += game["player_stats"][pid].get("pts", 0)
                self.assertLessEqual(
                    game["player_stats"][pid].get("fgm", 0),
                    game["player_stats"][pid].get("fga", 0),
                )
        self.assertGreater(sum(points.values()), 0)


if __name__ == "__main__":
    unittest.main()
