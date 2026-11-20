"""Scoring invariants for the live engine in era_engine.core.

These checks stay on simulate_game / simulate_series results. They do not
import the experimental shot-clock or passing modules.
"""
import tempfile
import unittest
from pathlib import Path

from era_engine.core import simulate_game, simulate_series
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load

SEEDS = (1, 7, 19, 42, 101)


class CoreInvariantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "test.db"))
        initialize_demo(self.con)
        self.players, self.eras, self.teams = load(self.con)
        self.team_a = self.teams["Bay City Legends"]
        self.team_b = self.teams["Chicago Stampede"]
        self.teams_in_order = (self.team_a, self.team_b)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def _simulated_games(self):
        """Yield live-engine games across every seeded era and several seeds.

        A short series is included so adaptation, which simulate_series applies
        and simulate_game leaves at zero by default, is covered too.
        """
        for era_key, era in self.eras.items():
            for seed in SEEDS:
                game = simulate_game(
                    self.team_a, self.team_b, self.players, era, seed
                )
                yield era_key, seed, "game", game
            series = simulate_series(
                self.team_a,
                self.team_b,
                self.players,
                era,
                games=3,
                seed=SEEDS[0],
            )
            for result in series["game_results"]:
                yield era_key, SEEDS[0] + (result["game"] - 1) * 997, "series", result

    def test_player_points_match_scoring_events(self):
        for era_key, seed, source, game in self._simulated_games():
            for pid, stats in game["player_stats"].items():
                with self.subTest(era=era_key, seed=seed, source=source, player=pid):
                    fgm = stats.get("fgm", 0)
                    fg3m = stats.get("fg3m", 0)
                    ftm = stats.get("ftm", 0)
                    two_point_makes = fgm - fg3m
                    self.assertGreaterEqual(two_point_makes, 0)
                    expected = 2 * two_point_makes + 3 * fg3m + ftm
                    self.assertEqual(stats.get("pts", 0), expected)

    def test_makes_never_exceed_attempts(self):
        for era_key, seed, source, game in self._simulated_games():
            for pid, stats in game["player_stats"].items():
                with self.subTest(era=era_key, seed=seed, source=source, player=pid):
                    self.assertLessEqual(stats.get("fgm", 0), stats.get("fga", 0))
                    self.assertLessEqual(stats.get("fg3m", 0), stats.get("fg3a", 0))
                    self.assertLessEqual(stats.get("ftm", 0), stats.get("fta", 0))

    def test_team_points_equal_sum_of_player_points(self):
        for era_key, seed, source, game in self._simulated_games():
            for idx, team in enumerate(self.teams_in_order):
                with self.subTest(era=era_key, seed=seed, source=source, team=team.name):
                    player_points = sum(
                        game["player_stats"][pid].get("pts", 0) for pid in team.player_ids
                    )
                    self.assertEqual(game["score"][idx], player_points)

    def test_team_assists_never_exceed_made_field_goals(self):
        for era_key, seed, source, game in self._simulated_games():
            for team in self.teams_in_order:
                with self.subTest(era=era_key, seed=seed, source=source, team=team.name):
                    assists = sum(
                        game["player_stats"][pid].get("ast", 0) for pid in team.player_ids
                    )
                    makes = sum(
                        game["player_stats"][pid].get("fgm", 0) for pid in team.player_ids
                    )
                    self.assertLessEqual(assists, makes)


if __name__ == "__main__":
    unittest.main()
