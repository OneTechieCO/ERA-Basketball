"""Research-database imports. These tests do not call NBA.com and do not rate players."""
import json
import tempfile
import unittest
from pathlib import Path

from era_engine.nba_import import (
    SOURCE_NAME,
    audit_team_player_totals,
    import_league_dash_player_response,
    import_player_career_response,
    import_team_advanced_response,
    import_team_dashboard_response,
    import_team_year_response,
    load_team_rosters,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, create_research_schema


def _career(season_rows, postseason_rows=None):
    headers = [
        "PLAYER_ID", "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN", "PTS",
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
        "AST", "STL", "BLK", "TOV",
    ]
    return {
        "resource": "playercareerstats",
        "parameters": {"PerMode": "Totals", "PlayerID": 893},
        "resultSets": [
            {"name": "SeasonTotalsRegularSeason", "headers": headers, "rowSet": season_rows},
            {
                "name": "SeasonTotalsPostSeason",
                "headers": headers,
                "rowSet": postseason_rows or [],
            },
        ],
    }


class ResearchImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "research.db"))
        initialize_demo(self.con)
        create_research_schema(self.con)
        self.before_players = self.con.execute("SELECT profile_json FROM players ORDER BY player_id").fetchall()

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def _player_row(self, season="1995-96", games=82, minutes=3090, points=2491, turnovers=197):
        return [
            893, season, 1610612741, "CHI", games, minutes, points,
            916, 1850, 0.495, 111, 260, 548, 657, 148, 395, 352, 180, 42, turnovers,
        ]

    def test_regular_season_is_selected_and_postseason_is_not(self):
        raw = _career(
            [self._player_row("1994-95", 17, 668, 457, 35), self._player_row()],
            [self._player_row("1995-96", 18, 733, 552, 42)],
        )
        stored = import_player_career_response(
            self.con, raw, player_id=893, player_name="Michael Jordan",
            season_id="1995-96", retrieved_on="2026-11-11",
        )
        self.con.commit()
        self.assertEqual(stored, 1)
        rows = self.con.execute(
            "SELECT games, minutes, points, turnovers, counting_measure FROM nba_player_season_research"
        ).fetchall()
        self.assertEqual(rows, [(82, 3090, 2491, 197, "season_total")])

    def test_missing_fields_stay_null_and_rates_are_not_copied_into_totals(self):
        headers = ["PLAYER_ID", "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "PTS", "FG_PCT"]
        raw = {
            "resource": "playercareerstats",
            "parameters": {"PerMode": "Totals"},
            "resultSets": [{
                "name": "SeasonTotalsRegularSeason",
                "headers": headers,
                "rowSet": [[201939, "2015-16", 1610612744, "GSW", 79, 2375, 0.504]],
            }],
        }
        import_player_career_response(
            self.con, raw, player_id=201939, player_name="Stephen Curry",
            season_id="2015-16", retrieved_on="2026-11-11",
        )
        self.con.commit()
        row = self.con.execute(
            """SELECT minutes, fga, turnovers, usage_rate, usage_rate_measure, points, counting_measure
               FROM nba_player_season_research"""
        ).fetchone()
        self.assertEqual(row, (None, None, None, None, None, 2375, "season_total"))
        self.assertNotEqual(row[5], 0.504)

    def test_raw_source_is_preserved_and_import_is_repeatable(self):
        raw = _career([self._player_row()])
        for _ in range(2):
            import_player_career_response(
                self.con, raw, player_id=893, player_name="Michael Jordan",
                season_id="1995-96", retrieved_on="2026-11-11",
            )
        self.con.commit()
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM nba_player_season_research").fetchone()[0], 1)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM nba_source_response").fetchone()[0], 1)
        saved = json.loads(self.con.execute("SELECT raw_json FROM nba_source_response").fetchone()[0])
        self.assertEqual(saved, raw)
        fields = json.loads(self.con.execute("SELECT field_names_json FROM nba_source_response").fetchone()[0])
        self.assertIn("FG_PCT", fields)
        self.assertNotIn("USG_PCT", fields)

    def test_team_totals_and_missing_pace_stay_separate(self):
        year = {
            "resource": "teamyearbyyearstats",
            "parameters": {"PerMode": "Totals", "TeamID": 1610612741},
            "resultSets": [{
                "name": "TeamStats",
                "headers": ["TEAM_ID", "YEAR", "GP", "FGA", "PTS", "FG_PCT", "AST", "OREB", "DREB", "STL", "BLK", "TOV"],
                "rowSet": [[1610612741, "1994-95", 82, 1, 1, 0.4, 1, 1, 1, 1, 1, 1],
                           [1610612741, "1995-96", 82, 6892, 8625, 0.478, 2033, 1247, 2411, 745, 345, 1175]],
            }],
        }
        advanced = {
            "resource": "leaguedashteamstats",
            "parameters": {"PerMode": "Totals", "MeasureType": "Advanced", "Season": "1995-96"},
            "resultSets": [{
                "name": "LeagueDashTeamStats",
                "headers": ["TEAM_ID", "TEAM_NAME", "PACE", "POSS"],
                "rowSet": [],
            }],
        }
        self.assertEqual(import_team_year_response(
            self.con, year, team_id=1610612741, team_name="Chicago Bulls",
            season_id="1995-96", retrieved_on="2026-11-11",
        ), 1)
        self.assertEqual(import_team_advanced_response(
            self.con, advanced, team_id=1610612741, team_name="Chicago Bulls",
            season_id="1995-96", retrieved_on="2026-11-11",
        ), 0)
        self.con.commit()
        totals = self.con.execute(
            """SELECT fga, points, minutes, pace, possessions, counting_measure
               FROM nba_team_season_research WHERE endpoint='teamyearbyyearstats'"""
        ).fetchone()
        self.assertEqual(totals, (6892, 8625, None, None, None, "season_total"))
        missing = self.con.execute(
            """SELECT pace, possessions, pace_measure, possessions_measure, points
               FROM nba_team_season_research WHERE endpoint='leaguedashteamstats'"""
        ).fetchone()
        self.assertEqual(missing, (None, None, None, None, None))

    def test_advanced_pace_is_a_rate_and_possessions_are_a_total(self):
        raw = {
            "resource": "leaguedashteamstats",
            "parameters": {"PerMode": "Totals", "MeasureType": "Advanced", "Season": "2015-16"},
            "resultSets": [{
                "name": "LeagueDashTeamStats",
                "headers": ["TEAM_ID", "GP", "MIN", "PACE", "E_PACE", "POSS"],
                "rowSet": [[1610612744, 82, 3976.0, 100.24, 101.6, 8303]],
            }],
        }
        import_team_advanced_response(
            self.con, raw, team_id=1610612744, team_name="Golden State Warriors",
            season_id="2015-16", retrieved_on="2026-11-11",
        )
        self.con.commit()
        row = self.con.execute(
            """SELECT games, minutes, pace, pace_measure, possessions, possessions_measure, points
               FROM nba_team_season_research"""
        ).fetchone()
        self.assertEqual(row, (82, 3976.0, 100.24, "rate", 8303, "season_total", None))

    def test_research_import_does_not_change_fictional_profiles(self):
        import_player_career_response(
            self.con, _career([self._player_row()]), player_id=893, player_name="Michael Jordan",
            season_id="1995-96", retrieved_on="2026-11-11",
        )
        self.con.commit()
        after = self.con.execute("SELECT profile_json FROM players ORDER BY player_id").fetchall()
        self.assertEqual(after, self.before_players)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM nba_research_links").fetchone()[0], 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM players").fetchone()[0], 10)
        self.assertIn("Sterling Cross", after[0][0])
        self.assertEqual(
            self.con.execute("SELECT source_name FROM nba_source_response").fetchone()[0],
            SOURCE_NAME,
        )


def _line(player_id, season, team_id, abbreviation, games, minutes, points, turnovers):
    return [
        player_id, season, team_id, abbreviation, games, minutes, points,
        10, 20, 0.5, 1, 4, 2, 3, 4, 5, 6, 7, 8, turnovers,
    ]


def _write_roster_payloads(root: Path) -> None:
    """Synthetic payloads. They are not historical statistics."""
    careers = root / "careers"
    careers.mkdir()

    def dump(name, payload):
        (root / name).write_text(json.dumps(payload))

    def roster(team_id, season, players):
        return {
            "resource": "commonteamroster",
            "parameters": {"TeamID": team_id, "Season": season},
            "resultSets": [{
                "name": "CommonTeamRoster",
                "headers": ["TeamID", "SEASON", "PLAYER", "PLAYER_ID", "POSITION"],
                "rowSet": [[team_id, season, name, player_id, position] for player_id, name, position in players],
            }],
        }

    def dash(season, team_id, rows):
        headers = [
            "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN", "PTS",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
        ]
        return {
            "resource": "leaguedashplayerstats",
            "parameters": {"PerMode": "Totals", "Season": season, "SeasonType": "Regular Season", "TeamID": team_id},
            "resultSets": [{"name": "LeagueDashPlayerStats", "headers": headers, "rowSet": rows}],
        }

    def dashboard(season, team_id, rows):
        return {
            "resource": "teamplayerdashboard",
            "parameters": {"PerMode": "Totals", "Season": season, "SeasonType": "Regular Season", "TeamID": team_id},
            "resultSets": [{
                "name": "PlayersSeasonTotals",
                "headers": ["PLAYER_ID", "PLAYER_NAME", "GP", "PTS"],
                "rowSet": rows,
            }],
        }

    dump("bulls_1995_roster.json", roster(1610612741, "1995-96", [
        (893, "Michael Jordan", "G"),
        (927, "John Salley", "F"),
        (999, "Missing Player", "G"),
    ]))
    dump("gsw_2015_roster.json", roster(1610612744, "2015-16", [
        (201939, "Stephen Curry", "G"),
    ]))
    dump("bulls_1995_players_dash.json", dash("1995-96", 1610612741, []))
    dump("gsw_2015_players_dash.json", dash("2015-16", 1610612744, [
        [201939, "Stephen Curry", 1610612744, "GSW", 79, 2699.975, 2375, 805, 1598, 402, 886, 363, 400, 68, 362, 430, 527, 169, 15, 262],
        [201574, "Jason Thompson", 1610612761, "TOR", 28, 178.98, 60, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]))
    dump("bulls_1995_dashboard.json", dashboard("1995-96", 1610612741, []))
    dump("gsw_2015_dashboard.json", dashboard("2015-16", 1610612744, [
        [201939, "Stephen Curry", 79, 2375],
        [201574, "Jason Thompson", 28, 60],
    ]))
    (careers / "893.json").write_text(json.dumps(_career([
        _line(893, "1995-96", 1610612741, "CHI", 82, 3090, 2491, 197),
    ])))
    (careers / "927.json").write_text(json.dumps(_career([
        _line(927, "1995-96", 1610612761, "TOR", 25, 400, 149, 20),
        _line(927, "1995-96", 1610612741, "CHI", 17, 120, 36, 4),
        _line(927, "1995-96", 0, "TOT", 42, 520, 185, 24),
    ])))
    (careers / "201939.json").write_text(json.dumps(_career([
        _line(201939, "2015-16", 1610612744, "GSW", 79, 2700, 2375, 262),
    ])))
    (careers / "201574.json").write_text(json.dumps(_career([
        _line(201574, "2015-16", 1610612744, "GSW", 28, 179, 60, 9),
        _line(201574, "2015-16", 1610612761, "TOR", 19, 200, 87, 8),
        _line(201574, "2015-16", 0, "TOT", 47, 379, 147, 17),
    ])))


class TeamRosterImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "research.db"))
        initialize_demo(self.con)
        create_research_schema(self.con)
        self.before_players = self.con.execute("SELECT profile_json FROM players ORDER BY player_id").fetchall()

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_other_team_and_combined_totals_are_not_attributed_to_the_requested_team(self):
        root = Path(self.tmp.name) / "payloads"
        root.mkdir()
        _write_roster_payloads(root)
        import_team_advanced_response(self.con, {
            "resource": "leaguedashteamstats",
            "parameters": {"PerMode": "Totals", "Season": "1995-96"},
            "resultSets": [{"name": "LeagueDashTeamStats", "headers": ["TEAM_ID", "PACE", "POSS"], "rowSet": []}],
        }, team_id=1610612741, team_name="Chicago Bulls", season_id="1995-96", retrieved_on="2026-11-11")
        first = load_team_rosters(self.con, str(root), "2026-11-11")
        second = load_team_rosters(self.con, str(root), "2026-11-11")
        self.assertEqual(first["roster_players"], {"1995-96": 3, "2015-16": 1})
        self.assertEqual(first["dash_rows"], {"1995-96": 0, "2015-16": 1})
        self.assertEqual(first["dash_rejected_other_team"], {"1995-96": 0, "2015-16": 1})
        self.assertEqual(first["dashboard_rows_without_team_id"], {"1995-96": 0, "2015-16": 2})
        self.assertEqual(first["missing_career_files"], [{
            "player_id": 999, "player_name": "Missing Player", "season_id": "1995-96", "team_id": 1610612741,
        }])
        self.assertEqual(second["career_rows"], first["career_rows"])

        chicago = self.con.execute(
            """SELECT player_name, points, turnovers FROM nba_player_season_research
               WHERE team_id=1610612741 AND endpoint='playercareerstats' ORDER BY nba_player_id"""
        ).fetchall()
        self.assertEqual(chicago, [("Michael Jordan", 2491, 197), ("John Salley", 36, 4)])
        combined = self.con.execute(
            """SELECT team_id, team_abbreviation, points FROM nba_player_season_research
               WHERE nba_player_id=927 AND team_abbreviation='TOT'"""
        ).fetchone()
        self.assertEqual(combined, (0, "TOT", 185))
        toronto = self.con.execute(
            """SELECT team_id, points FROM nba_player_season_research
               WHERE nba_player_id=927 AND team_abbreviation='TOR'"""
        ).fetchone()
        self.assertEqual(toronto, (1610612761, 149))
        self.assertIsNone(self.con.execute(
            "SELECT 1 FROM nba_player_season_research WHERE nba_player_id=999"
        ).fetchone())

        warriors = self.con.execute(
            """SELECT nba_player_id, points FROM nba_player_season_research
               WHERE team_id=1610612744 AND endpoint='playercareerstats' ORDER BY nba_player_id"""
        ).fetchall()
        self.assertEqual(warriors, [(201574, 60), (201939, 2375)])
        self.assertIsNone(self.con.execute(
            """SELECT 1 FROM nba_player_season_research
               WHERE endpoint='leaguedashplayerstats' AND nba_player_id=201574"""
        ).fetchone())
        dash_curry = self.con.execute(
            """SELECT minutes, points, counting_measure FROM nba_player_season_research
               WHERE endpoint='leaguedashplayerstats' AND nba_player_id=201939"""
        ).fetchone()
        self.assertEqual(dash_curry, (2699.975, 2375, "season_total"))
        saved_dash = json.loads(self.con.execute(
            """SELECT raw_json FROM nba_source_response
               WHERE endpoint='leaguedashplayerstats' AND season_id='2015-16'"""
        ).fetchone()[0])
        dash_rows = saved_dash["resultSets"][0]["rowSet"]
        self.assertIn(1610612761, [row[2] for row in dash_rows])
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM nba_player_season_research WHERE endpoint='teamplayerdashboard'"
        ).fetchone()[0], 0)
        self.assertEqual(self.con.execute(
            """SELECT pace, possessions FROM nba_team_season_research
               WHERE team_id=1610612741 AND endpoint='leaguedashteamstats'"""
        ).fetchone(), (None, None))
        self.assertIsNone(self.con.execute(
            "SELECT usage_rate FROM nba_player_season_research WHERE nba_player_id=893"
        ).fetchone()[0])
        after = self.con.execute("SELECT profile_json FROM players ORDER BY player_id").fetchall()
        self.assertEqual(after, self.before_players)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM nba_research_links").fetchone()[0], 0)

    def test_dashboard_with_team_id_keeps_only_the_requested_team(self):
        raw = {
            "resource": "teamplayerdashboard",
            "parameters": {"PerMode": "Totals", "Season": "1995-96", "SeasonType": "Regular Season"},
            "resultSets": [{
                "name": "PlayersSeasonTotals",
                "headers": ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "PTS"],
                "rowSet": [
                    [893, "Michael Jordan", 1610612741, "CHI", 82, 2491],
                    [927, "John Salley", 1610612761, "TOR", 25, 149],
                    [927, "John Salley", 0, "TOT", 42, 185],
                ],
            }],
        }
        self.assertEqual(import_team_dashboard_response(
            self.con, raw, team_id=1610612741, team_name="Chicago Bulls",
            season_id="1995-96", retrieved_on="2026-11-11",
        ), 0)
        self.con.commit()
        rows = self.con.execute(
            "SELECT nba_player_id, team_id, team_abbreviation, points FROM nba_player_season_research"
        ).fetchall()
        self.assertEqual(rows, [(893, 1610612741, "CHI", 2491)])

    def test_rebound_total_is_not_split_when_the_source_has_no_split(self):
        headers = ["PLAYER_ID", "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "PTS", "REB"]
        raw = {
            "resource": "playercareerstats",
            "parameters": {"PerMode": "Totals"},
            "resultSets": [{
                "name": "SeasonTotalsRegularSeason",
                "headers": headers,
                "rowSet": [[893, "1995-96", 1610612741, "CHI", 82, 2491, 543]],
            }],
        }
        import_player_career_response(
            self.con, raw, player_id=893, player_name="Michael Jordan",
            season_id="1995-96", retrieved_on="2026-11-11",
        )
        self.con.commit()
        self.assertEqual(self.con.execute(
            "SELECT offensive_rebounds, defensive_rebounds, points FROM nba_player_season_research"
        ).fetchone(), (None, None, 2491))

    def test_audit_reports_differences_without_changing_either_side(self):
        import_team_year_response(self.con, {
            "resource": "teamyearbyyearstats",
            "parameters": {"PerMode": "Totals"},
            "resultSets": [{
                "name": "TeamStats",
                "headers": ["TEAM_ID", "YEAR", "PTS", "TOV", "OREB", "DREB"],
                "rowSet": [[1610612741, "1995-96", 100, 12, 5, 7]],
            }],
        }, team_id=1610612741, team_name="Chicago Bulls", season_id="1995-96", retrieved_on="2026-11-11")
        import_player_career_response(self.con, _career([
            _line(893, "1995-96", 1610612741, "CHI", 82, 3090, 40, 4),
        ]), player_id=893, player_name="Michael Jordan", season_id="1995-96", retrieved_on="2026-11-11")
        import_player_career_response(self.con, _career([
            _line(927, "1995-96", 1610612741, "CHI", 17, 120, 60, 6),
            _line(927, "1995-96", 0, "TOT", 42, 520, 200, 20),
        ]), player_id=927, player_name="John Salley", season_id="1995-96", retrieved_on="2026-11-11")
        import_league_dash_player_response(self.con, {
            "resource": "leaguedashplayerstats",
            "parameters": {"PerMode": "Totals", "Season": "1995-96", "SeasonType": "Regular Season"},
            "resultSets": [{
                "name": "LeagueDashPlayerStats",
                "headers": ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "TOV"],
                "rowSet": [[893, "Michael Jordan", 1610612741, "CHI", 40, 4]],
            }],
        }, team_id=1610612741, team_name="Chicago Bulls", season_id="1995-96", retrieved_on="2026-11-11")
        self.con.commit()
        report = audit_team_player_totals(self.con, team_id=1610612741, season_id="1995-96")
        self.assertEqual(report["player_rows"], 2)
        self.assertEqual(report["fields"]["points"], {"players": 100, "team": 100, "team_minus_players": 0})
        self.assertEqual(report["fields"]["turnovers"], {"players": 10, "team": 12, "team_minus_players": 2})
        self.assertEqual(report["fields"]["minutes"], {"players": 3210, "team": None, "team_minus_players": None})
        unchanged = self.con.execute(
            "SELECT points, turnovers FROM nba_team_season_research"
        ).fetchone()
        self.assertEqual(unchanged, (100, 12))
        player_points = self.con.execute(
            """SELECT SUM(points) FROM nba_player_season_research
               WHERE endpoint='playercareerstats' AND team_id=1610612741"""
        ).fetchone()[0]
        self.assertEqual(player_points, 100)

    def test_roster_season_may_be_the_starting_year(self):
        from era_engine.nba_import import import_roster_response
        raw = {
            "resource": "commonteamroster",
            "parameters": {"Season": "1995-96", "TeamID": 1610612741},
            "resultSets": [{
                "name": "CommonTeamRoster",
                "headers": ["TeamID", "SEASON", "PLAYER", "PLAYER_ID", "POSITION"],
                "rowSet": [
                    [1610612741, "1995", "Michael Jordan", 893, "G"],
                    [1610612741, "2015", "Stephen Curry", 201939, "G"],
                ],
            }],
        }
        players = import_roster_response(
            self.con, raw, team_id=1610612741, team_name="Chicago Bulls",
            season_id="1995-96", retrieved_on="2026-11-11",
        )
        self.assertEqual(players, [(893, "Michael Jordan")])

    def test_missing_roster_file_raises_and_does_not_invent_players(self):
        with self.assertRaises(FileNotFoundError):
            load_team_rosters(self.con, self.tmp.name, "2026-11-11")
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM nba_player_season_research").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
