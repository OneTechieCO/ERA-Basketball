"""Invariants for the isolated receiver-selection experiments."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine.core import simulate_game as simulate_live
from era_engine.receiver_experiments import (
    VARIANTS,
    receiver_weight_a,
    receiver_weight_b,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


class ReceiverExperimentTests(unittest.TestCase):
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

    def test_weights_prefer_the_star_without_excluding_teammates(self):
        roster = [self.players[pid] for pid in self.team_a.player_ids]
        star = self.players["ERA_0001"]
        for weight in (receiver_weight_a, receiver_weight_b):
            values = {player.player_id: weight(player) for player in roster}
            self.assertGreater(values[star.player_id], max(
                value for pid, value in values.items() if pid != star.player_id
            ))
            smallest = min(values.values())
            self.assertGreater(smallest, 0)
            self.assertLess(values[star.player_id] / smallest, 8)

    def test_variants_keep_scoring_invariants(self):
        for name, simulate in VARIANTS.items():
            for era_key in ("1995-96", "2023-24"):
                with self.subTest(variant=name, era=era_key):
                    game = simulate(
                        self.team_a, self.team_b, self.players, self.eras[era_key], 7
                    )
                    again = simulate(
                        self.team_a, self.team_b, self.players, self.eras[era_key], 7
                    )
                    self.assertEqual(game, again)
                    self.assertTrue(all(0 <= event["passes"] <= 6 for event in game["events"]))
                    for stats in game["player_stats"].values():
                        fgm = stats.get("fgm", 0)
                        fg3m = stats.get("fg3m", 0)
                        self.assertLessEqual(fgm, stats.get("fga", 0))
                        self.assertLessEqual(fg3m, stats.get("fg3a", 0))
                        self.assertLessEqual(stats.get("ftm", 0), stats.get("fta", 0))
                        self.assertGreaterEqual(fgm - fg3m, 0)
                        self.assertEqual(
                            stats.get("pts", 0),
                            2 * (fgm - fg3m) + 3 * fg3m + stats.get("ftm", 0),
                        )
                    for team in (self.team_a, self.team_b):
                        idx = 0 if team is self.team_a else 1
                        self.assertEqual(
                            game["score"][idx],
                            sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
                        )
                        assists = sum(game["player_stats"][pid].get("ast", 0) for pid in team.player_ids)
                        makes = sum(game["player_stats"][pid].get("fgm", 0) for pid in team.player_ids)
                        self.assertLessEqual(assists, makes)

    def test_short_clock_still_prevents_passes(self):
        era = replace(self.eras["1995-96"], shot_clock=1)
        for name, simulate in VARIANTS.items():
            with self.subTest(variant=name):
                game = simulate(self.team_a, self.team_b, self.players, era, 19)
                self.assertGreater(
                    sum(event["kind"] == "shot_clock_violation" for event in game["events"]),
                    0,
                )
                self.assertTrue(all(event["passes"] == 0 for event in game["events"]))
                self.assertEqual(sum(stats.get("fga", 0) for stats in game["player_stats"].values()), 0)

    def test_variants_are_not_the_live_receiver_rule(self):
        era = self.eras["1995-96"]
        live = simulate_live(self.team_a, self.team_b, self.players, era, 7)
        for simulate in VARIANTS.values():
            other = simulate(self.team_a, self.team_b, self.players, era, 7)
            self.assertNotEqual(live, other)


if __name__ == "__main__":
    unittest.main()
