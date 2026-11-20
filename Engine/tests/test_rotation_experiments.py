"""Isolated rotation scheduler. These tests do not run the live game or rate players."""
import sqlite3
import unittest
from pathlib import Path

from era_engine.rotation_experiments import (
    INTEGRATION,
    PLAYER_MINUTE_CAP,
    REGULATION_TEAM_MINUTES,
    RotationPlayer,
    describe_season_minutes,
    overtime_requirement,
    project_regulation_targets,
    schedule_regulation,
    season_minutes_from_research,
    validate_regulation,
)

RESEARCH_DB = Path(__file__).resolve().parents[1] / "research.db"


def _players(specs):
    return [RotationPlayer(player_id, target, available, stamina) for player_id, target, available, stamina in specs]


def _first_run(result, player_id):
    run = 0
    for segment in result["segments"]:
        length = segment["end_minute"] - segment["start_minute"]
        if player_id in segment["on_court"]:
            run += length
        elif run:
            break
    return run


class RotationSchedulerTests(unittest.TestCase):
    def test_regulation_rotation_keeps_five_players_and_240_minutes(self):
        players = _players([
            ("A", 36, True, 80),
            ("B", 36, True, 80),
            ("C", 36, True, 80),
            ("D", 36, True, 80),
            ("E", 36, True, 80),
            ("F", 30, True, 80),
            ("G", 30, True, 80),
        ])
        first = schedule_regulation(players, "target_minutes")
        second = schedule_regulation(players, "target_minutes")
        self.assertEqual(first, second)
        validate_regulation(first, players)
        self.assertEqual(sum(first["minutes"].values()), 240)
        self.assertTrue(all(minutes <= 48 for minutes in first["minutes"].values()))
        self.assertEqual(first["minutes"], {player.player_id: 36 if player.player_id < "F" else 30 for player in players})
        for segment in first["segments"]:
            self.assertEqual(len(segment["on_court"]), 5)
            self.assertEqual(len(set(segment["on_court"])), 5)
        for substitution in first["substitutions"]:
            self.assertEqual(len(substitution["players_out"]), len(substitution["players_in"]))
            self.assertFalse(set(substitution["players_out"]) & set(substitution["players_in"]))
        self.assertFalse(first["historical_starting_lineup"])
        self.assertEqual(first["opening_five_basis"], "largest_targets")
        self.assertIsNone(first["overtime"]["schedule"])
        self.assertFalse(first["promoted"])

    def test_unavailable_player_stays_off_the_court(self):
        players = _players([
            ("A", 48, True, 90),
            ("B", 48, True, 90),
            ("C", 48, True, 90),
            ("D", 48, True, 90),
            ("E", 48, True, 90),
            ("F", 0, False, 90),
        ])
        result = schedule_regulation(players)
        self.assertEqual(result["minutes"]["F"], 0)
        self.assertTrue(all("F" not in segment["on_court"] for segment in result["segments"]))
        self.assertEqual(len(players), 6)
        with self.assertRaises(ValueError):
            schedule_regulation(_players([
                ("A", 48, True, 90),
                ("B", 48, True, 90),
                ("C", 48, True, 90),
                ("D", 48, True, 90),
            ]))

    def test_caller_supplied_opening_lineup_is_not_inferred(self):
        players = _players([(chr(65 + index), 40, True, 90) for index in range(6)])
        opening = ("F", "E", "D", "C", "B")
        result = schedule_regulation(players, starting_five=opening)
        self.assertEqual(result["opening_five"], tuple(sorted(opening)))
        self.assertEqual(result["opening_five_basis"], "caller_supplied")
        self.assertFalse(result["historical_starting_lineup"])
        full_time = _players([
            ("A", 48, True, 90),
            ("B", 40, True, 90),
            ("C", 40, True, 90),
            ("D", 40, True, 90),
            ("E", 40, True, 90),
            ("F", 32, True, 90),
        ])
        with self.assertRaises(ValueError):
            schedule_regulation(full_time, starting_five=("B", "C", "D", "E", "F"))

    def test_low_stamina_shortens_a_stint_without_changing_the_minute_total(self):
        players = _players([
            ("A", 36, True, 25),
            ("B", 36, True, 100),
            ("C", 36, True, 100),
            ("D", 36, True, 100),
            ("E", 36, True, 100),
            ("F", 30, True, 100),
            ("G", 30, True, 100),
        ])
        result = schedule_regulation(players, "target_minutes")
        self.assertEqual(_first_run(result, "A"), 4)
        self.assertGreater(_first_run(result, "B"), _first_run(result, "A"))
        self.assertEqual(result["substitutions"][0]["players_out"], ("A",))
        self.assertEqual(result["minutes"]["A"], 36)
        self.assertEqual(sum(result["minutes"].values()), 240)
        self.assertIn("not the live possession-index fatigue", result["fatigue_model"])
        roster = _players([(chr(65 + index), 40, True, 100) for index in range(6)])
        short = schedule_regulation(roster, "short_stints")
        longer = schedule_regulation(roster, "target_minutes")
        self.assertLess(short["substitutions"][0]["minute"], longer["substitutions"][0]["minute"])
        self.assertGreater(len(short["substitutions"]), len(longer["substitutions"]))
        self.assertTrue(all(minutes == 40 for minutes in short["minutes"].values()))
        self.assertTrue(all(minutes == 40 for minutes in longer["minutes"].values()))

    def test_equal_time_splits_the_regulation_budget(self):
        players = _players([(f"P{index}", 0, True, 70) for index in range(7)])
        result = schedule_regulation(players, "equal_time")
        self.assertEqual(sum(result["minutes"].values()), 240)
        self.assertLessEqual(max(result["minutes"].values()) - min(result["minutes"].values()), 1)
        self.assertTrue(all(minutes <= PLAYER_MINUTE_CAP for minutes in result["minutes"].values()))
        with self.assertRaises(ValueError):
            schedule_regulation(_players([("A", 49, True, 70), ("B", 48, True, 70), ("C", 48, True, 70), ("D", 48, True, 70), ("E", 47, True, 70)]))
        with self.assertRaises(ValueError):
            schedule_regulation(_players([(chr(65 + index), 20, True, 70) for index in range(5)]))

    def test_season_minutes_stay_aggregates_and_overtime_is_not_scheduled(self):
        description = describe_season_minutes(
            [
                {"player_id": "star", "player_name": "Star", "games": 82, "minutes": 3090},
                {"player_id": "reserve", "player_name": "Reserve", "games": 17, "minutes": 191},
            ],
            team_games=82,
        )
        star, reserve = description["players"]
        self.assertEqual(star["minutes_per_appearance"], 3090 / 82)
        self.assertEqual(star["minutes_per_team_game"], 3090 / 82)
        self.assertNotEqual(reserve["minutes_per_appearance"], reserve["minutes_per_team_game"])
        self.assertEqual(reserve["minutes_per_appearance"], 191 / 17)
        self.assertEqual(reserve["minutes_per_team_game"], 191 / 82)
        self.assertIsNone(star["single_game_minutes"])
        self.assertIsNone(description["starting_lineup"])
        self.assertIsNone(description["single_game_lineups"])
        self.assertIsNone(description["overtime"]["schedule"])
        self.assertEqual(description["overtime"], overtime_requirement())
        self.assertEqual(description["minutes_above_regulation"], (3090 + 191) - 240 * 82)

    def test_projection_fits_a_regulation_game_without_naming_starters(self):
        description = describe_season_minutes(
            [
                {"player_id": f"P{index}", "player_name": f"P{index}", "games": 82, "minutes": minutes}
                for index, minutes in enumerate((3000, 2500, 2200, 1800, 1500, 900, 700, 400))
            ],
            team_games=82,
        )
        projected = project_regulation_targets(description)
        self.assertFalse(projected["historical_game"])
        self.assertIsNone(projected["starting_lineup"])
        self.assertAlmostEqual(sum(projected["targets"].values()), REGULATION_TEAM_MINUTES)
        self.assertTrue(all(value <= 48 for value in projected["targets"].values()))
        players = [
            RotationPlayer(player_id, target, True, 80)
            for player_id, target in projected["targets"].items()
        ]
        result = schedule_regulation(players)
        self.assertEqual(sum(result["minutes"].values()), 240)
        self.assertTrue(all(minutes <= 48 for minutes in result["minutes"].values()))
        self.assertFalse(result["historical_starting_lineup"])
        with self.assertRaises(ValueError):
            project_regulation_targets(describe_season_minutes(
                [{"player_id": f"P{index}", "games": 10, "minutes": 100} for index in range(4)],
                team_games=10,
            ))

    def test_scheduler_is_used_by_the_live_engine(self):
        self.assertTrue(INTEGRATION["promoted"])
        self.assertEqual(INTEGRATION["live_engine"], "era_engine.core.simulate_game")
        self.assertTrue(INTEGRATION["game_clock_required_before_live_integration"])


class HistoricalMinuteTests(unittest.TestCase):
    def test_stored_seasons_keep_average_minutes_separate_from_single_games(self):
        if not RESEARCH_DB.is_file():
            self.skipTest("research.db is not in the workspace")
        con = sqlite3.connect(f"file:{RESEARCH_DB}?mode=ro", uri=True)
        before = con.execute("SELECT COUNT(*) FROM nba_player_season_research").fetchone()[0]
        bulls = season_minutes_from_research(con, 1610612741, "1995-96")
        warriors = season_minutes_from_research(con, 1610612744, "2015-16")
        after = con.execute("SELECT COUNT(*) FROM nba_player_season_research").fetchone()[0]
        con.close()
        self.assertEqual(before, after)
        self.assertEqual(bulls["player_count"], 15)
        self.assertEqual(warriors["player_count"], 16)
        self.assertEqual(bulls["minutes_above_regulation"], 50)
        self.assertEqual(warriors["minutes_above_regulation"], 201)
        self.assertIsNone(bulls["starting_lineup"])
        self.assertIsNone(warriors["starting_lineup"])
        self.assertIsNone(bulls["overtime"]["historical_overtime_schedule"])
        jordan = next(player for player in bulls["players"] if player["player_name"] == "Michael Jordan")
        pippen = next(player for player in bulls["players"] if player["player_name"] == "Scottie Pippen")
        salley = next(player for player in bulls["players"] if player["player_name"] == "John Salley")
        self.assertEqual(jordan["minutes_per_appearance"], jordan["minutes_per_team_game"])
        self.assertEqual(jordan["minutes"], 3090)
        self.assertNotEqual(pippen["minutes_per_appearance"], pippen["minutes_per_team_game"])
        self.assertEqual(pippen["minutes_per_appearance"], 2825 / 77)
        self.assertEqual(salley["minutes"], 191)
        curry = next(player for player in warriors["players"] if player["player_name"] == "Stephen Curry")
        self.assertEqual(curry["minutes_per_appearance"], 2700 / 79)
        self.assertIsNone(curry["single_game_minutes"])
        projected = project_regulation_targets(bulls)
        self.assertIsNone(projected["starting_lineup"])
        self.assertAlmostEqual(sum(projected["targets"].values()), 240)
        self.assertLessEqual(max(projected["targets"].values()), 48)


if __name__ == "__main__":
    unittest.main()
