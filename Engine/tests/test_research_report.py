"""Historical rate reports. These tests do not call the network or run the simulator."""
import shutil
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from era_engine.research_report import (
    build_comparison_report,
    classify_external_observation,
    import_external_directory,
    per_100_possessions,
    per_36_minutes,
    quotient,
    scale,
)
from era_engine.seed import initialize_demo
from era_engine.storage import connect, create_research_schema

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "research_external" / "basketball_reference"
RESEARCH_DB = Path(__file__).resolve().parents[1] / "research.db"


def _insert_player(con, **overrides):
    record = {
        "nba_player_id": 893, "player_name": "Michael Jordan", "season_id": "1995-96",
        "team_id": 1610612741, "team_abbreviation": "CHI", "source_name": "nba.com via nba_api",
        "endpoint": "playercareerstats", "dataset_name": "SeasonTotalsRegularSeason",
        "counting_measure": "season_total", "games": 82, "minutes": 3090, "points": 2491,
        "fgm": 916, "fga": 1850, "fg3m": 111, "fg3a": 260, "ftm": 548, "fta": 657,
        "assists": 352, "offensive_rebounds": 148, "defensive_rebounds": 395,
        "steals": 180, "blocks": 42, "turnovers": 197, "usage_rate": None, "usage_rate_measure": None,
    }
    record.update(overrides)
    columns = list(record)
    con.execute(
        f'''INSERT INTO nba_player_season_research({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})''',
        [record[column] for column in columns],
    )


def _insert_team(con, **overrides):
    record = {
        "team_id": 1610612741, "team_name": "Chicago Bulls", "season_id": "1995-96",
        "source_name": "nba.com via nba_api", "endpoint": "teamyearbyyearstats", "dataset_name": "TeamStats",
        "counting_measure": "season_total", "games": 82, "minutes": None, "points": 100,
        "fgm": 40, "fga": 80, "fg3m": 10, "fg3a": 20, "ftm": 10, "fta": 12,
        "assists": 20, "offensive_rebounds": 5, "defensive_rebounds": 7,
        "steals": 8, "blocks": 4, "turnovers": 12, "pace": None, "pace_measure": None,
        "possessions": None, "possessions_measure": None,
    }
    record.update(overrides)
    columns = list(record)
    con.execute(
        f'''INSERT INTO nba_team_season_research({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})''',
        [record[column] for column in columns],
    )


class ResearchReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "research.db"))
        initialize_demo(self.con)
        create_research_schema(self.con)
        self.before_players = self.con.execute("SELECT profile_json FROM players ORDER BY player_id").fetchall()

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_rates_are_undefined_when_a_denominator_or_component_is_missing(self):
        self.assertIsNone(quotient(10, 0))
        self.assertIsNone(quotient(10, None))
        self.assertIsNone(scale(None, 82, 36))
        self.assertIsNone(per_36_minutes({"minutes": None, "points": 2491}))
        self.assertIsNone(per_100_possessions({"possessions": 8303, "possessions_measure": "rate", "points": 100}))
        record = {"minutes": 3090, "points": 2491, "offensive_rebounds": 148, "defensive_rebounds": None}
        self.assertEqual(per_36_minutes(record)["points"], 2491 / 3090 * 36)
        self.assertIsNone(per_36_minutes(record)["rebounds"])
        _insert_player(
            self.con, fga=None, fg3m=None, fg3a=0, fta=0, minutes=0,
            offensive_rebounds=None, defensive_rebounds=4, points=10,
        )
        _insert_team(self.con)
        report = build_comparison_report(self.con)
        player = report["seasons"][0]["players"][0]
        self.assertIsNone(player["percentages"]["field_goal"])
        self.assertIsNone(player["percentages"]["three_point"])
        self.assertIsNone(player["percentages"]["free_throw"])
        self.assertIsNone(player["percentages"]["three_point_attempt_rate"])
        self.assertIsNone(player["percentages"]["effective_field_goal"])
        self.assertIsNone(player["per_game"]["rebounds"])
        self.assertIsNone(player["per_36_minutes"])
        self.assertIsNone(player["per_100_on_floor_possessions"])

    def test_report_uses_only_the_requested_team_and_does_not_rewrite_sources(self):
        _insert_player(self.con)
        _insert_player(
            self.con, nba_player_id=927, player_name="John Salley", team_abbreviation="CHI",
            games=17, minutes=120, points=36, fgm=12, fga=35, fg3m=0, fg3a=0, ftm=12, fta=20,
            assists=15, offensive_rebounds=20, defensive_rebounds=23, steals=8, blocks=15, turnovers=16,
        )
        _insert_player(
            self.con, nba_player_id=927, player_name="John Salley", team_id=0, team_abbreviation="TOT",
            games=42, minutes=520, points=185, turnovers=55,
        )
        _insert_player(
            self.con, nba_player_id=927, player_name="John Salley", team_id=1610612761, team_abbreviation="TOR",
            season_id="1995-96", games=25, points=149, turnovers=39,
        )
        _insert_team(self.con, points=8625, turnovers=1175, fgm=3293, fga=6892, fg3m=544, fg3a=1349, ftm=1495, fta=2004)
        _insert_team(
            self.con, endpoint="leaguedashteamstats", dataset_name="LeagueDashTeamStats",
            counting_measure=None, points=None, fgm=None, fga=None, fg3m=None, fg3a=None,
            ftm=None, fta=None, assists=None, offensive_rebounds=None, defensive_rebounds=None,
            steals=None, blocks=None, turnovers=None, possessions=8303, possessions_measure="season_total",
            pace=100.24, pace_measure="rate", minutes=3976,
        )
        before = self.con.execute(
            "SELECT points, turnovers, pace, possessions FROM nba_team_season_research ORDER BY endpoint"
        ).fetchall()
        report = build_comparison_report(self.con)
        after = self.con.execute(
            "SELECT points, turnovers, pace, possessions FROM nba_team_season_research ORDER BY endpoint"
        ).fetchall()
        self.assertEqual(after, before)
        chicago = report["seasons"][0]
        self.assertEqual([player["player_name"] for player in chicago["players"]], ["John Salley", "Michael Jordan"])
        salley = chicago["players"][0]
        self.assertEqual(salley["per_game"]["points"], 36 / 17)
        self.assertNotEqual(salley["per_game"]["points"], 185 / 42)
        jordan = chicago["players"][1]
        self.assertEqual(jordan["per_game"]["points"], 2491 / 82)
        self.assertEqual(jordan["per_game"]["minutes"], 3090 / 82)
        self.assertEqual(jordan["percentages"]["field_goal"], 916 / 1850)
        self.assertEqual(jordan["percentages"]["three_point"], 111 / 260)
        self.assertEqual(jordan["percentages"]["free_throw"], 548 / 657)
        self.assertEqual(jordan["percentages"]["three_point_attempt_rate"], 260 / 1850)
        self.assertEqual(jordan["percentages"]["effective_field_goal"], (916 + 0.5 * 111) / 1850)
        self.assertEqual(jordan["per_game"]["rebounds"], (148 + 395) / 82)
        self.assertEqual(jordan["per_36_minutes"]["points"], 2491 / 3090 * 36)
        self.assertEqual(jordan["per_36_minutes"]["turnovers"], 197 / 3090 * 36)
        self.assertIsNone(jordan["nba_usage_rate"])
        counting = chicago["team_counting_source"]
        self.assertEqual(counting["per_game"]["points"], 8625 / 82)
        self.assertEqual(counting["per_game"]["field_goals_attempted"], 6892 / 82)
        self.assertIsNone(counting["per_game"]["minutes"])
        self.assertIsNone(counting["per_100_possessions"])
        self.assertIsNone(counting["pace"])
        advanced = chicago["team_advanced_source"]
        self.assertEqual(advanced["pace"], 100.24)
        self.assertEqual(advanced["pace_measure"], "rate")
        self.assertIsNone(advanced["per_100_possessions"]["points"])
        self.assertEqual(chicago["turnover_gap"]["computed_team_minus_players"], 1175 - (197 + 16))
        self.assertFalse(chicago["turnover_gap"]["reconciled"])
        self.assertEqual(self.con.execute("SELECT turnovers FROM nba_team_season_research WHERE endpoint='teamyearbyyearstats'").fetchone()[0], 1175)

    def test_same_row_possessions_can_make_a_per_100_rate(self):
        rates = per_100_possessions({
            "possessions": 200, "possessions_measure": "season_total", "points": 100,
            "fgm": None, "fga": None, "fg3m": None, "fg3a": None, "ftm": None, "fta": None,
            "assists": None, "offensive_rebounds": 4, "defensive_rebounds": 6,
            "steals": None, "blocks": None, "turnovers": None, "minutes": None,
        })
        self.assertEqual(rates["points"], 50)
        self.assertEqual(rates["rebounds"], 5)
        self.assertIsNone(rates["assists"])

    def test_verified_external_fields_stay_separate_from_nba_nulls(self):
        _insert_player(self.con, nba_player_id=201939, player_name="Stephen Curry", season_id="2015-16",
                       team_id=1610612744, team_abbreviation="GSW", games=79, minutes=2700, points=2375)
        _insert_team(self.con, team_id=1610612744, team_name="Golden State Warriors", season_id="2015-16",
                     pace=None, possessions=None)
        _insert_team(
            self.con, team_id=1610612744, team_name="Golden State Warriors", season_id="2015-16",
            endpoint="leaguedashteamstats", dataset_name="LeagueDashTeamStats", pace=100.24,
            pace_measure="rate", possessions=8303, possessions_measure="season_total", points=None,
            counting_measure=None,
        )
        counts = import_external_directory(self.con, str(EXTERNAL_DIR))
        self.assertEqual(counts, {"verified": 4, "unimported": 0})
        import_external_directory(self.con, str(EXTERNAL_DIR))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM research_external_observation").fetchone()[0], 4)
        report = build_comparison_report(self.con)
        warriors = report["seasons"][1]
        self.assertEqual(warriors["external_pace"]["displayed_value"], float(Decimal("99.3")))
        self.assertFalse(warriors["external_pace"]["merged_into_nba_pace"])
        self.assertEqual(warriors["team_advanced_source"]["pace"], 100.24)
        self.assertIsNone(warriors["team_counting_source"]["pace"])
        curry = warriors["players"][0]
        self.assertEqual(curry["external_on_floor_usage_fraction"], float(Decimal("32.6") / Decimal(100)))
        self.assertIsNone(curry["nba_usage_rate"])
        self.assertIsNone(self.con.execute("SELECT usage_rate FROM nba_player_season_research").fetchone()[0])
        self.assertEqual(
            self.con.execute("SELECT pace FROM nba_team_season_research WHERE endpoint='teamyearbyyearstats'").fetchone()[0],
            None,
        )
        chicago = report["seasons"][0]
        self.assertEqual(chicago["external_pace"]["displayed_value"], float(Decimal("91.1")))
        self.assertIsNone(chicago["team_counting_source"])
        jordan_fraction = next(
            row["normalized_fraction"] for row in report["external_observations"]
            if row["entity_name"] == "Michael Jordan"
        )
        self.assertEqual(jordan_fraction, float(Decimal("33.3") / Decimal(100)))
        raw = self.con.execute(
            "SELECT raw_json FROM research_external_observation WHERE entity_id=893"
        ).fetchone()[0]
        self.assertIn("0.333367650000000000", raw)
        self.assertNotEqual(jordan_fraction, float(Decimal("0.333367650000000000")))
        self.assertEqual(
            self.con.execute("SELECT profile_json FROM players ORDER BY player_id").fetchall(),
            self.before_players,
        )
        self.assertFalse(report["simulator_comparison"]["simulation_was_run"])

    def test_missing_or_mislabeled_external_fields_stay_unimported(self):
        missing = {
            "source_name": "Basketball-Reference.com",
            "url": "https://www.basketball-reference.com/players/j/jordami01.html",
            "retrieved_on": "2026-11-11",
            "season_id": "1995-96",
            "page_season_id": "1995-96",
            "entity_kind": "player",
            "entity_id": 893,
            "entity_name": "Michael Jordan",
            "field_name": "usg_pct",
            "field_label": "USG%",
            "table_id": "advanced",
            "labeled_cells": {"year_id": "1995-96", "team_name_abbr": "CHI"},
            "column_definition": "Usage Percentage",
            "methodology": "basketball_reference_usage_percentage_on_floor_share",
            "source_sort_key": "0.333",
        }
        lead = classify_external_observation(missing)
        self.assertEqual(lead["import_status"], "unimported")
        self.assertIsNone(lead["displayed_value"])
        self.assertIsNone(lead["normalized_fraction"])
        playoffs = dict(missing, table_id="playoffs", labeled_cells={"year_id": "1995-96", "team_name_abbr": "CHI", "usg_pct": "32.9"})
        self.assertEqual(classify_external_observation(playoffs)["import_status"], "unimported")
        other_team = {
            "source_name": "Basketball-Reference.com",
            "url": "https://www.basketball-reference.com/teams/CHI/1996.html",
            "retrieved_on": "2026-11-11",
            "season_id": "1995-96",
            "page_season_id": "1995-96",
            "entity_kind": "team",
            "entity_id": 1610612741,
            "entity_name": "Chicago Bulls",
            "field_name": "pace",
            "field_label": "Pace",
            "table_id": "team_misc",
            "labeled_cells": {"player": "Lg Rank", "pace": "20"},
            "column_definition": "Pace Factor: An estimate of possessions per 48 minutes",
            "methodology": "basketball_reference_pace_factor_possessions_per_48_minutes",
        }
        self.assertEqual(classify_external_observation(other_team)["import_status"], "unimported")
        self.assertIsNone(classify_external_observation(other_team)["displayed_value"])


class StoredResearchReportTests(unittest.TestCase):
    def test_stored_seasons_keep_their_turnover_gaps_and_separate_sources(self):
        if not RESEARCH_DB.is_file():
            self.skipTest("research.db is not in the workspace")
        tmp = tempfile.TemporaryDirectory()
        copied = Path(tmp.name) / "research.db"
        shutil.copy(RESEARCH_DB, copied)
        con = sqlite3.connect(copied)
        before_teams = con.execute(
            "SELECT season_id, endpoint, pace, possessions, points, turnovers FROM nba_team_season_research ORDER BY season_id, endpoint"
        ).fetchall()
        before_usage = con.execute("SELECT usage_rate FROM nba_player_season_research").fetchall()
        import_external_directory(con, str(EXTERNAL_DIR))
        report = build_comparison_report(con)
        after_teams = con.execute(
            "SELECT season_id, endpoint, pace, possessions, points, turnovers FROM nba_team_season_research ORDER BY season_id, endpoint"
        ).fetchall()
        after_usage = con.execute("SELECT usage_rate FROM nba_player_season_research").fetchall()
        self.assertEqual(after_teams, before_teams)
        self.assertEqual(after_usage, before_usage)
        self.assertTrue(all(value is None for (value,) in after_usage))
        chicago, warriors = report["seasons"]
        self.assertEqual(chicago["player_count"], 15)
        self.assertEqual(warriors["player_count"], 16)
        self.assertEqual(chicago["turnover_gap"]["carried_forward_gap"], 41)
        self.assertEqual(warriors["turnover_gap"]["carried_forward_gap"], 26)
        self.assertEqual(chicago["turnover_gap"]["computed_team_minus_players"], 41)
        self.assertEqual(warriors["turnover_gap"]["computed_team_minus_players"], 26)
        self.assertFalse(chicago["turnover_gap"]["reconciled"])
        self.assertFalse(warriors["turnover_gap"]["reconciled"])
        self.assertIsNone(chicago["team_counting_source"]["pace"])
        self.assertIsNone(chicago["team_counting_source"]["possessions"])
        self.assertIsNone(chicago["team_advanced_source"]["pace"])
        self.assertIsNone(chicago["team_advanced_source"]["possessions"])
        self.assertIsNone(chicago["team_counting_source"]["per_100_possessions"])
        self.assertEqual(warriors["team_advanced_source"]["pace"], 100.24)
        self.assertEqual(warriors["team_advanced_source"]["possessions"], 8303)
        self.assertIsNone(warriors["team_advanced_source"]["per_100_possessions"]["points"])
        self.assertEqual(chicago["external_pace"]["displayed_value"], float(Decimal("91.1")))
        self.assertEqual(warriors["external_pace"]["displayed_value"], float(Decimal("99.3")))
        self.assertFalse(chicago["external_pace"]["merged_into_nba_pace"])
        jordan = next(player for player in chicago["players"] if player["nba_player_id"] == 893)
        curry = next(player for player in warriors["players"] if player["nba_player_id"] == 201939)
        self.assertEqual(jordan["per_game"]["points"], 2491 / 82)
        self.assertEqual(curry["per_game"]["points"], 2375 / 79)
        self.assertEqual(jordan["external_on_floor_usage_fraction"], float(Decimal("33.3") / Decimal(100)))
        self.assertEqual(curry["external_on_floor_usage_fraction"], float(Decimal("32.6") / Decimal(100)))
        self.assertIsNone(jordan["per_100_on_floor_possessions"])
        self.assertFalse(report["simulator_comparison"]["simulation_was_run"])
        player_points = con.execute(
            """SELECT SUM(points) FROM nba_player_season_research
               WHERE season_id='1995-96' AND team_id=1610612741 AND endpoint='playercareerstats'"""
        ).fetchone()[0]
        self.assertEqual(player_points, 8625)
        con.close()
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
