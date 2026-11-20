"""App-facing batch contract. The 1,000-game run stays a command."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from era_engine.batch import (
    TIMING_FIRST_TRIP,
    TIMING_ORIGINAL,
    BatchRequestError,
    configuration_catalog,
    main,
    simulate_batch,
)
from era_engine.core import Team, UnresolvedGameError, simulate_game, simulate_series
from era_engine.seed import ERAS, demo_players


def _demo(games, **extra):
    request = {
        "team_a": "Bay City Legends",
        "team_b": "Chicago Stampede",
        "era": "1995-96",
        "games": games,
        "timing_mode": TIMING_ORIGINAL,
    }
    request.update(extra)
    return simulate_batch(request)


def _without_runtime(result):
    copy = json.loads(json.dumps(result))
    copy.pop("runtime")
    return copy


class BatchRequestTests(unittest.TestCase):
    def test_the_series_cap_stays_in_place(self):
        players = {player.player_id: player for player in demo_players()}
        bay = Team("Bay City Legends", tuple(f"ERA_{index:04d}" for index in range(1, 6)), "perimeter")
        stampede = Team("Chicago Stampede", tuple(f"ERA_{index:04d}" for index in range(6, 11)), "balanced")
        with self.assertRaises(ValueError):
            simulate_series(bay, stampede, players, ERAS[1], games=100)

    def test_unsupported_counts_teams_eras_and_seeds_are_rejected(self):
        with self.assertRaises(BatchRequestError) as caught:
            _demo(82)
        self.assertEqual(caught.exception.code, "unsupported_game_count")
        with self.assertRaises(BatchRequestError) as caught:
            _demo(True)
        self.assertEqual(caught.exception.code, "unsupported_game_count")
        with self.assertRaises(BatchRequestError) as caught:
            _demo(1, team_a="Los Angeles Lakers")
        self.assertEqual(caught.exception.code, "unknown_team")
        with self.assertRaises(BatchRequestError) as caught:
            _demo(1, team_a="Chicago Bulls", season_a="2015-16")
        self.assertEqual(caught.exception.code, "unavailable_season")
        with self.assertRaises(BatchRequestError) as caught:
            _demo(1, team_a="Chicago Bulls")
        self.assertEqual(caught.exception.code, "unavailable_season")
        with self.assertRaises(BatchRequestError) as caught:
            _demo(1, season_a="1995-96")
        self.assertEqual(caught.exception.code, "unavailable_season")
        with self.assertRaises(BatchRequestError) as caught:
            _demo(1, era="1994-95")
        self.assertEqual(caught.exception.code, "invalid_era")
        for seed in (-1, True, 1.5, "7"):
            with self.assertRaises(BatchRequestError) as caught:
                _demo(1, seed=seed)
            self.assertEqual(caught.exception.code, "malformed_seed")
        with self.assertRaises(BatchRequestError) as caught:
            simulate_batch({"team_a": "Bay City Legends"}, team_b="Chicago Stampede")
        self.assertEqual(caught.exception.code, "malformed_request")
        with self.assertRaises(BatchRequestError) as caught:
            _demo(1, team_b="Bay City Legends")
        self.assertEqual(caught.exception.code, "invalid_matchup")

    def test_the_2015_16_sample_is_not_applied_to_1995_96(self):
        def unavailable():
            raise AssertionError("the 2015-16 sample was loaded")

        with patch("era_engine.batch.shot_trip_duration_sample", unavailable):
            result = _demo(1, team_a="CHI", season_a="1995-96", team_b="Chicago Stampede")
        self.assertEqual(result["request"]["timing_mode"], TIMING_ORIGINAL)
        self.assertIsNone(result["calibration"]["timing"]["sample_season"])
        pace = next(item["message"] for item in result["calibration"]["warnings"] if item["code"] == "pace_1995_96_uncalibrated")
        self.assertIn("136", pace)
        self.assertIn("91", pace)
        self.assertIn("No 1995-96 play-by-play timestamps are stored.", pace)

        calls = {"sample": 0, "games": 0}

        def count_sample():
            calls["sample"] += 1
            return {"durations": (12.0,)}

        real_game = simulate_game

        def count_game(*args, **kwargs):
            calls["games"] += 1
            return real_game(*args, **kwargs)

        with patch("era_engine.batch.shot_trip_duration_sample", count_sample), patch("era_engine.batch.simulate_game", count_game):
            with self.assertRaises(BatchRequestError) as caught:
                _demo(1, team_a="Chicago Bulls", season_a="1995-96", timing_mode=TIMING_FIRST_TRIP)
        self.assertEqual(caught.exception.code, "timing_sample_not_for_era")
        self.assertEqual(calls, {"sample": 0, "games": 0})

    def test_a_missing_2015_16_sample_is_an_error(self):
        with patch("era_engine.batch.shot_trip_duration_sample", side_effect=FileNotFoundError("missing")):
            with self.assertRaises(BatchRequestError) as caught:
                simulate_batch(
                    team_a="GSW",
                    season_a="2015-16",
                    team_b="Chicago Stampede",
                    era="2015-16",
                    games=1,
                    timing_mode=TIMING_FIRST_TRIP,
                )
        self.assertEqual(caught.exception.code, "timing_sample_missing")
        with patch("era_engine.batch.shot_trip_duration_sample", return_value={"durations": ()}):
            with self.assertRaises(BatchRequestError) as caught:
                simulate_batch(
                    team_a=1610612744,
                    season_a="2015-16",
                    team_b="Chicago Stampede",
                    era="2015-16",
                    games=1,
                    timing_mode=TIMING_FIRST_TRIP,
                )
        self.assertEqual(caught.exception.code, "timing_sample_missing")

    def test_catalog_keeps_cross_era_output_hypothetical(self):
        catalog = {row["id"]: row for row in configuration_catalog()}
        self.assertEqual(catalog["bulls_1995_96_versus_fictional"]["timing_mode"], TIMING_ORIGINAL)
        self.assertEqual(catalog["warriors_2015_16_versus_fictional"]["timing_mode"], TIMING_FIRST_TRIP)
        self.assertEqual(catalog["cross_era"]["output"], "hypothetical_model_projection")
        self.assertFalse(any(row["historical_calibration"] == "complete" for row in catalog.values()))


class BatchResultTests(unittest.TestCase):
    def test_one_and_ten_game_results_are_deterministic_and_balance(self):
        first = _demo(1, seed=7)
        second = _demo(1, seed=7)
        self.assertEqual(_without_runtime(first), _without_runtime(second))
        self._balanced(first)
        self.assertEqual(first["status"], "completed")
        self.assertFalse(first["event_logs_retained"])
        self.assertNotIn("events", first["game_results"][0])
        self.assertEqual(first["calibration"]["output"], "fictional_demo")
        self.assertFalse(first["historical_calibration_complete"])
        self.assertEqual(first["series_limit_preserved"], 82)
        self.assertEqual(first["variant_b3"], "experimental_not_used")
        encoded = json.dumps(first, allow_nan=False)
        self.assertLess(len(encoded.encode()), 200_000)

        ten = _demo(10, seed=3)
        self._balanced(ten)
        self.assertEqual(ten["games_completed"], 10)
        self.assertEqual(ten["seeds"], {"start": 3, "step": 1, "count": 10})
        self.assertEqual([row["seed"] for row in ten["game_results"]], list(range(3, 13)))
        self.assertIs(ten["request"]["seed_defaulted"], False)
        omitted = _demo(1)
        self.assertEqual(omitted["request"]["seed"], 0)
        self.assertIs(omitted["request"]["seed_defaulted"], True)

    def test_one_hundred_games_stay_compact(self):
        result = _demo(100, seed=11)
        self._balanced(result)
        self.assertEqual(result["games_completed"], 100)
        self.assertTrue(all("events" not in row for row in result["game_results"]))
        size = len(json.dumps(result, allow_nan=False).encode())
        self.assertLess(size, 500_000)

    def test_historical_configurations_carry_their_labels(self):
        bulls = simulate_batch(
            team_a="Chicago Bulls",
            season_a="1995-96",
            team_b="Bay City Legends",
            era="1995-96",
            games=1,
            seed=7,
            timing_mode=TIMING_ORIGINAL,
        )
        self._balanced(bulls)
        self.assertEqual(bulls["calibration"]["output"], "historical_roster_versus_fictional_opponent")
        self.assertEqual(bulls["calibration"]["timing"]["classification"], "assumed_timing")
        self.assertEqual(bulls["teams"]["team_a"]["ratings"], "derived_provisional")
        self.assertEqual(bulls["teams"]["team_a"]["players"][0]["stored_season"]["classification"], "verified_counting_totals")
        jordan = next(row for row in bulls["teams"]["team_a"]["players"] if row["name"] == "Michael Jordan")
        self.assertGreater(jordan["stored_season"]["points_per_game"], 20)

        warriors = simulate_batch(
            team_a="Golden State Warriors",
            season_a="2015-16",
            team_b="Chicago Stampede",
            era="2015-16",
            games=1,
            seed=7,
            timing_mode=TIMING_FIRST_TRIP,
        )
        self._balanced(warriors)
        self.assertEqual(warriors["calibration"]["timing"]["classification"], "observed_timing_sample")
        self.assertEqual(warriors["calibration"]["timing"]["sample_season"], "2015-16")
        self.assertTrue(warriors["calibration"]["timing"]["observed_for_requested_era"])
        self.assertFalse(warriors["calibration"]["timing"]["full_timing_model_calibrated"])
        self.assertFalse(warriors["calibration"]["timing"]["passes_in_sample"])
        codes = [item["code"] for item in warriors["calibration"]["warnings"]]
        self.assertIn("timing_sample_observed", codes)
        self.assertNotIn("pace_1995_96_uncalibrated", codes)
        self.assertNotIn("timing_sample_outside_matchup", codes)

        cross = simulate_batch(
            team_a="Chicago Bulls",
            season_a="1995-96",
            team_b="Golden State Warriors",
            season_b="2015-16",
            era="2015-16",
            games=1,
            seed=1,
            timing_mode=TIMING_ORIGINAL,
        )
        self.assertEqual(cross["calibration"]["output"], "hypothetical_model_projection")
        self.assertIn("hypothetical_cross_era", [item["code"] for item in cross["calibration"]["warnings"]])
        self.assertIn("timing_sample_not_selected", [item["code"] for item in cross["calibration"]["warnings"]])

        other_era = _demo(1, era="1985-86")
        self.assertIn("timing_assumed", [item["code"] for item in other_era["calibration"]["warnings"]])

    def test_ten_historical_games_do_not_use_the_other_eras_sample(self):
        seen = []
        real_game = simulate_game

        def spy(*args, **kwargs):
            seen.append(kwargs.get("timing"))
            return real_game(*args, **kwargs)

        with patch("era_engine.batch.simulate_game", spy):
            result = simulate_batch(
                team_a="Chicago Bulls",
                season_a="1995-96",
                team_b="Bay City Legends",
                era="1995-96",
                games=10,
                seed=20,
                timing_mode=TIMING_ORIGINAL,
            )
        self.assertEqual(result["games_completed"], 10)
        self.assertTrue(all(timing is None for timing in seen))
        self._balanced(result)

    def test_an_unresolved_game_is_not_a_completed_result(self):
        def unresolved(*args, **kwargs):
            raise UnresolvedGameError({"overtime": {"periods": 15}, "score": [90, 90]})

        with patch("era_engine.batch.simulate_game", unresolved):
            result = _demo(1, seed=4)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["games_completed"], 0)
        self.assertEqual(result["game_results"], [])
        self.assertEqual(result["wins"]["ties_returned_as_completed"], 0)
        self.assertEqual(result["wins"]["team_a_win_pct"], None)
        self.assertIn("incomplete", result["incomplete_games"][0]["message"])
        self.assertFalse(result["incomplete_games"][0]["returned_as_completed"])

    def test_a_failed_check_is_excluded_from_the_averages(self):
        real_game = simulate_game

        def corrupt(*args, **kwargs):
            game = real_game(*args, **kwargs)
            game["score"] = [game["score"][0] + 9, game["score"][1]]
            return game

        with patch("era_engine.batch.simulate_game", corrupt):
            result = _demo(1, seed=5)
        self.assertEqual(result["games_completed"], 0)
        self.assertEqual(result["games_failed_checks"], 1)
        self.assertEqual(result["game_results"], [])
        self.assertIsNone(result["teams"]["team_a"]["average"]["pts"])

    def test_events_are_included_only_when_requested(self):
        quiet = _demo(1, seed=8)
        loud = _demo(1, seed=8, include_events=True)
        self.assertGreater(len(loud["game_results"][0]["events"]), 10)
        self.assertEqual(quiet["game_results"][0]["score"], loud["game_results"][0]["score"])
        self.assertGreater(len(json.dumps(loud).encode()), len(json.dumps(quiet).encode()))

    def test_the_command_writes_a_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "batch.json"
            code = main([
                "--team-a", "Bay City Legends",
                "--team-b", "Chicago Stampede",
                "--era", "2005-06",
                "--games", "1",
                "--seed", "2",
                "--timing", TIMING_ORIGINAL,
                "--summary",
                "--output", str(destination),
            ])
            self.assertEqual(code, 0)
            summary = json.loads(destination.read_text())
        self.assertEqual(summary["games_completed"], 1)
        self.assertGreater(summary["bytes"], 1000)
        self.assertLess(summary["bytes"], 200_000)
        self.assertIn("timing_assumed", summary["warnings"])
        code = main([
            "--team-a", "Bay City Legends",
            "--team-b", "Chicago Stampede",
            "--era", "1995-96",
            "--games", "50",
            "--timing", TIMING_ORIGINAL,
        ])
        self.assertEqual(code, 2)

    def _balanced(self, result):
        self.assertEqual(result["games_incomplete"], 0)
        self.assertEqual(result["games_failed_checks"], 0)
        self.assertEqual(result["games_completed"], result["games_requested"])
        self.assertEqual(result["wins"]["team_a"] + result["wins"]["team_b"], result["games_completed"])
        self.assertEqual(result["wins"]["ties_returned_as_completed"], 0)
        wins = {"team_a": 0, "team_b": 0}
        for row in result["game_results"]:
            self.assertNotEqual(row["score"][0], row["score"][1])
            side = "team_a" if row["score"][0] > row["score"][1] else "team_b"
            self.assertEqual(row["winner"], result["teams"][side]["identifier"])
            wins[side] += 1
            if row["overtime_periods"] == 0:
                self.assertEqual(row["regulation_score"], row["score"])
        self.assertEqual(wins["team_a"], result["wins"]["team_a"])
        for side in ("team_a", "team_b"):
            team = result["teams"][side]
            players = team["players"]
            for stat in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "min"):
                self.assertAlmostEqual(sum(row["average"][stat] for row in players), team["average"][stat])
            self.assertAlmostEqual(team["average"]["oreb"] + team["average"]["dreb"], team["average"]["reb"])
            self.assertLessEqual(team["average"]["fgm"], team["average"]["fga"])
            self.assertLessEqual(team["average"]["fg3m"], team["average"]["fg3a"])
            self.assertLessEqual(team["average"]["ftm"], team["average"]["fta"])
            self.assertAlmostEqual(sum(row["average"]["fga_share"] for row in players), 1)
            periods = result["overtime"]["mean_periods"]
            self.assertAlmostEqual(team["average"]["min"], 240 + 25 * periods)
            for row in players:
                if row["average"]["min"] == 0:
                    for stat in ("pts", "fga", "fg3a", "fta", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov"):
                        self.assertEqual(row["average"][stat], 0)


if __name__ == "__main__":
    unittest.main()
