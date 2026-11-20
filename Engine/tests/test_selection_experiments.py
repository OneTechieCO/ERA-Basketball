"""Invariants for isolated ballhandler and receiver weight experiments."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine.core import Team, selection_weight, simulate_game as simulate_live
from era_engine.selection_experiments import (
    VARIANTS,
    live_ballhandler_weight,
    live_receiver_weight,
    reset_counts,
    simulate_counted_live,
    touch_weight,
    COUNTS,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


class SelectionExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "test.db"))
        initialize_demo(self.con)
        self.players, self.eras, self.teams = load(self.con)
        self.team_a = self.teams["Bay City Legends"]
        self.team_b = self.teams["Chicago Stampede"]
        self.star = self.players["ERA_0001"]
        self.role = self.players["ERA_0004"]
        reset_counts()

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()
        reset_counts()

    def test_usage_and_iq_raise_a_positive_proportional_weight(self):
        self.assertGreater(touch_weight(self.star), touch_weight(self.role))
        self.assertGreater(touch_weight(self.star), 0)
        doubled = replace(self.role, usage=self.role.usage * 2)
        self.assertAlmostEqual(touch_weight(doubled), touch_weight(self.role) * 2)
        smarter = replace(self.role, iq=self.role.iq + 20)
        self.assertGreater(touch_weight(smarter), touch_weight(self.role))
        ratio = touch_weight(self.star) / touch_weight(self.role)
        expected = (self.star.usage * (70 + self.star.iq)) / (self.role.usage * (70 + self.role.iq))
        self.assertAlmostEqual(ratio, expected)
        self.assertLess(ratio, 4)
        self.assertEqual(live_ballhandler_weight(self.star), live_ballhandler_weight(self.role))
        self.assertEqual(live_receiver_weight(self.star), live_receiver_weight(self.role))
        self.assertEqual(selection_weight(self.star), touch_weight(self.star))
        self.assertEqual(selection_weight(self.role), touch_weight(self.role))
        self.assertGreater(selection_weight(replace(self.role, usage=0.0)), 0.0)

    def test_counted_live_matches_the_engine_and_partial_floors_differ(self):
        era = self.eras["1995-96"]
        live = simulate_live(self.team_a, self.team_b, self.players, era, 7)
        self.assertEqual(simulate_counted_live(self.team_a, self.team_b, self.players, era, 7), live)
        integrated = VARIANTS["C"](self.team_a, self.team_b, self.players, era, 7)
        self.assertEqual(integrated, live)
        self.assertEqual(integrated, VARIANTS["C"](self.team_a, self.team_b, self.players, era, 7))
        for name in ("A", "B"):
            with self.subTest(variant=name):
                game = VARIANTS[name](self.team_a, self.team_b, self.players, era, 7)
                self.assertNotEqual(game, live)
                self.assertEqual(game, VARIANTS[name](self.team_a, self.team_b, self.players, era, 7))

    def test_variants_keep_scoring_invariants_and_the_clock(self):
        era = replace(self.eras["2023-24"], shot_clock=1)
        full = self.eras["2015-16"]
        for name, simulate in VARIANTS.items():
            with self.subTest(variant=name):
                game = simulate(self.team_a, self.team_b, self.players, full, 19)
                self.assertTrue(all(0 <= event["passes"] <= 6 for event in game["events"]))
                self.assertGreater(max(event["passes"] for event in game["events"]), 0)
                for stats in game["player_stats"].values():
                    fgm = stats.get("fgm", 0)
                    fg3m = stats.get("fg3m", 0)
                    self.assertLessEqual(fgm, stats.get("fga", 0))
                    self.assertEqual(
                        stats.get("pts", 0),
                        2 * (fgm - fg3m) + 3 * fg3m + stats.get("ftm", 0),
                    )
                with self.assertRaises(RuntimeError) as caught:
                    simulate(self.team_a, self.team_b, self.players, era, 19)
                short = caught.exception.result
                self.assertFalse(short["completed"])
                self.assertEqual(short["status"], "incomplete")
                self.assertEqual(short["score"], [0, 0])
                self.assertIn("No tie-break point", str(caught.exception))
                self.assertGreater(
                    sum(event["kind"] == "shot_clock_violation" for event in short["events"]), 0
                )
                self.assertTrue(all(event["passes"] == 0 for event in short["events"]))
                self.assertEqual(sum(stats.get("fga", 0) for stats in short["player_stats"].values()), 0)

    def test_higher_usage_is_chosen_more_often_without_a_shutout(self):
        era = self.eras["2005-06"]
        reset_counts()
        for name, simulate in VARIANTS.items():
            shares = {pid: 0 for pid in self.team_a.player_ids}
            for idx in range(10):
                game = simulate(self.team_a, self.team_b, self.players, era, 101 + idx * 997)
                for pid in shares:
                    shares[pid] += game["player_stats"][pid].get("fga", 0)
            team_fga = sum(shares.values())
            with self.subTest(variant=name):
                self.assertGreater(min(shares.values()) / team_fga, 0.08)
                self.assertLess(max(shares.values()) / team_fga, 0.45)
        bay = list(self.team_a.player_ids)
        star = COUNTS["A"]["ballhandler"]["ERA_0001"]
        self.assertGreater(star, max(COUNTS["A"]["ballhandler"][pid] for pid in bay if pid != "ERA_0001"))
        star_targets = COUNTS["B"]["receiver"]["ERA_0001"]
        self.assertGreater(star_targets, max(COUNTS["B"]["receiver"][pid] for pid in bay if pid != "ERA_0001"))
        self.assertGreater(
            COUNTS["C"]["ballhandler"]["ERA_0001"],
            max(COUNTS["C"]["ballhandler"][pid] for pid in bay if pid != "ERA_0001"),
        )
        self.assertGreater(
            COUNTS["C"]["receiver"]["ERA_0001"],
            max(COUNTS["C"]["receiver"][pid] for pid in bay if pid != "ERA_0001"),
        )

    def test_production_selection_uses_usage_without_a_shot_quota_or_off_court_touches(self):
        star_id = self.team_a.player_ids[0]
        role_ids = self.team_a.player_ids[1:]
        for era_key, era in self.eras.items():
            fga = {pid: 0 for pid in self.team_a.player_ids}
            for seed in range(10):
                game = simulate_live(self.team_a, self.team_b, self.players, era, seed)
                self.assertEqual(game["game_clock_seconds"], 2880.0 + 300.0 * game["overtime"]["periods"])
                for event in game["events"]:
                    offense = event.get("offense")
                    if offense is not None:
                        self.assertIn(offense, event["offense_lineup"])
                for pid in fga:
                    fga[pid] += game["player_stats"][pid].get("fga", 0)
            team_fga = sum(fga.values())
            star_share = fga[star_id] / team_fga
            with self.subTest(era=era_key):
                self.assertGreater(star_share, max(fga[pid] / team_fga for pid in role_ids))
                self.assertGreater(min(fga[pid] / team_fga for pid in role_ids), 0.08)
                self.assertLess(star_share, 0.45)

        template = self.players["ERA_0002"]
        ids = tuple(f"S{index}" for index in range(8))
        usages = (0.33, 0.28, 0.22, 0.16, 0.12, 0.20, 0.10, 0.40)
        roster = dict(self.players)
        for index, player_id in enumerate(ids):
            roster[player_id] = replace(
                template,
                player_id=player_id,
                name=player_id,
                usage=usages[index],
                iq=(92, 88, 80, 76, 74, 84, 70, 90)[index],
                passing=90 if index == 5 else 70,
            )
        home = Team("Bench", ids, "balanced")
        targets = {ids[index]: 36.0 for index in range(5)}
        targets[ids[5]] = 30.0
        targets[ids[6]] = 30.0
        targets[ids[7]] = 0.0
        away_targets = {pid: 48.0 for pid in self.team_b.player_ids}
        attempts = {pid: 0 for pid in ids}
        minutes = {pid: 0.0 for pid in ids}
        assists = {pid: 0 for pid in ids}
        for seed in (0, 7, 8, 19):
            game = simulate_live(
                home, self.team_b, roster, self.eras["2015-16"], seed,
                rotation_targets=(targets, away_targets),
            )
            self.assertEqual(game["rotation_mode"], ["scheduled", "scheduled"])
            for event in game["events"]:
                self.assertNotIn(ids[7], event["offense_lineup"])
                self.assertNotIn(ids[7], event["defense_lineup"])
                offense = event.get("offense")
                if offense is not None:
                    self.assertIn(offense, event["offense_lineup"])
            for pid in ids:
                attempts[pid] += game["player_stats"][pid].get("fga", 0)
                minutes[pid] += game["player_stats"][pid]["min"]
                assists[pid] += game["player_stats"][pid].get("ast", 0)
        self.assertEqual(minutes[ids[7]], 0.0)
        self.assertEqual(attempts[ids[7]], 0)
        self.assertGreater(assists[ids[5]], 0)
        for pid in ids[:7]:
            self.assertGreater(minutes[pid], 0.0)
            self.assertGreater(attempts[pid], 0)
        high = attempts[ids[0]] / minutes[ids[0]]
        low = attempts[ids[6]] / minutes[ids[6]]
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
