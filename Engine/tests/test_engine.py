import tempfile
import unittest
from pathlib import Path
from era_engine.storage import connect, load
from era_engine.seed import initialize_demo
from era_engine.core import simulate_game, simulate_series, shot_weights

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / 'test.db'))
        initialize_demo(self.con)
        self.players, self.eras, self.teams = load(self.con)
        self.a = self.teams['Bay City Legends']
        self.b = self.teams['Chicago Stampede']
        self.era = self.eras['1995-96']
    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()
    def test_repeatability(self):
        a = simulate_game(self.a, self.b, self.players, self.era, 19)
        b = simulate_game(self.a, self.b, self.players, self.era, 19)
        self.assertEqual(a, b)
    def test_points_match_score(self):
        game = simulate_game(self.a, self.b, self.players, self.era)
        for idx, t in enumerate((self.a, self.b)):
            self.assertEqual(game['score'][idx], sum(game['player_stats'][pid].get('pts', 0) for pid in t.player_ids))
    def test_stat_constraints(self):
        series = simulate_series(self.a, self.b, self.players, self.era, games=10)
        for data in series['game_results']:
            for stats in data['player_stats'].values():
                self.assertLessEqual(stats.get('fgm', 0), stats.get('fga', 0))
                self.assertLessEqual(stats.get('fg3m', 0), stats.get('fg3a', 0))
                self.assertLessEqual(stats.get('fg3m', 0), stats.get('fgm', 0))
                self.assertEqual(stats.get('reb', 0), stats.get('dreb', 0) + stats.get('oreb', 0))
    def test_era_tendencies_differ(self):
        p = self.players['ERA_0001']
        early = shot_weights(p, self.eras['1985-86'], 'balanced', 1)
        recent = shot_weights(p, self.eras['2023-24'], 'balanced', 1)
        self.assertLess(early['three'] / sum(early.values()), recent['three'] / sum(recent.values()))
    def test_db_mapping_separate(self):
        self.assertEqual(self.players['ERA_0001'].name, 'Sterling Cross')
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM nba_research_links').fetchone()[0], 0)
    def test_duplicate_player_on_team(self):
        from dataclasses import replace

        duplicate_ids = list(self.a.player_ids)
        duplicate_ids[1] = duplicate_ids[0]
        invalid_team = replace(self.a, player_ids=duplicate_ids)

        with self.assertRaises(ValueError):
            simulate_game(invalid_team, self.b, self.players, self.era)

    def test_player_on_both_teams(self):
        from dataclasses import replace

        shared_ids = list(self.b.player_ids)
        shared_ids[0] = self.a.player_ids[0]
        invalid_team = replace(self.b, player_ids=shared_ids)

        with self.assertRaises(ValueError):
            simulate_game(self.a, invalid_team, self.players, self.era)

    def test_negative_era_pace(self):
        from dataclasses import replace

        with self.assertRaises(ValueError):
            replace(self.era, pace=-10)

    def test_invalid_three_point_share(self):
        from dataclasses import replace

        with self.assertRaises(ValueError):
            replace(self.era, three_share=1.5)

if __name__ == '__main__':
    unittest.main()

