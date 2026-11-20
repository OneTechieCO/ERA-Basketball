"""Optional 1995-96 aggregate-pace timing. Original timing stays the default."""
import unittest
from unittest.mock import patch

from era_engine.batch import TIMING_ORIGINAL, TIMING_PROVISIONAL_1995_96, BatchRequestError, simulate_batch
from era_engine.core import (
    PROVISIONAL_1995_96_DEVELOPMENT_MEAN,
    PROVISIONAL_1995_96_PACE_REFERENCE,
    Team,
    TimingAssumptions,
    simulate_game,
)
from era_engine.historical_players import load_researched_rosters
from era_engine.seed import ERAS, demo_players
from era_engine.validation import check_game


def _sides():
    bulls, _warriors = load_researched_rosters()
    players = {player.player_id: player for player in demo_players()}
    players.update(bulls.players)
    bay = Team("Bay City Legends", tuple(f"ERA_{index:04d}" for index in range(1, 6)), "perimeter")
    era = next(item for item in ERAS if item.key == "1995-96")
    return bulls, bay, players, era


class AggregatePaceTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bulls, cls.bay, cls.players, cls.era = _sides()
        cls.rotations = (dict(cls.bulls.rotation_targets), None)
        cls.provisional = TimingAssumptions(development_mean_seconds=PROVISIONAL_1995_96_DEVELOPMENT_MEAN)

    def _play(self, seed, timing=None):
        return simulate_game(
            self.bulls.team, self.bay, self.players, self.era, seed,
            timing=timing, rotation_targets=self.rotations,
        )

    def test_original_timing_does_not_draw_the_provisional_hold(self):
        game = self._play(7)
        self.assertEqual(game["timing_assumptions"]["timing_mode"], "original")
        self.assertIsNone(game["timing_assumptions"]["development_mean_seconds"])
        self.assertNotIn("provisional_development", game["live_seconds"])
        self.assertFalse(game["event_level_timestamps"] if "event_level_timestamps" in game else False)
        self.assertFalse(game["timing_assumptions"]["event_level_timestamps"])
        explicit = self._play(7, TimingAssumptions())
        self.assertEqual(game["score"], explicit["score"])
        self.assertEqual(game["events"], explicit["events"])

    def test_the_hold_is_deterministic_and_legal(self):
        first = self._play(11, self.provisional)
        second = self._play(11, self.provisional)
        self.assertEqual(first, second)
        self.assertEqual(first["timing_assumptions"]["timing_mode"], "provisional_aggregate_pace")
        self.assertEqual(first["timing_assumptions"]["pace_reference"], PROVISIONAL_1995_96_PACE_REFERENCE)
        self.assertEqual(first["timing_assumptions"]["pace_reference_role"], "in_sample_mean_benchmark")
        self.assertFalse(first["timing_assumptions"]["event_level_timestamps"])
        self.assertFalse(first["pace_used_to_choose_possessions"])
        self.assertFalse(first["timing_calibrated"])
        self.assertGreater(first["live_seconds"].get("provisional_development", 0.0), 0.0)
        self.assertNotIn("unobserved_development", first["live_seconds"])
        self.assertAlmostEqual(sum(first["live_seconds"].values()), first["game_clock_seconds"], delta=1e-6)
        self.assertEqual(check_game(first, self.era, self.bulls.team.player_ids, self.bay.player_ids), [])
        original = self._play(11)
        self.assertLess(first["possession_count"], original["possession_count"])
        self.assertNotEqual(first["possessions_by_team"][0], round(PROVISIONAL_1995_96_PACE_REFERENCE))

    def test_a_turnover_skips_the_hold_and_a_rebound_does_not_draw_twice(self):
        turnover = None
        continuation = None
        for seed in range(30):
            game = self._play(seed, self.provisional)
            shots = {}
            for event in game["events"]:
                if event["kind"] in ("turnover", "pass_turnover") and turnover is None:
                    turnover = event
                if event["kind"] in ("made_shot", "miss", "shooting_foul"):
                    shots.setdefault(event["possession"], []).append(event)
            for sequence in shots.values():
                if len(sequence) >= 2 and any(event.get("offensive_rebound") for event in sequence):
                    continuation = sequence
                    break
            if turnover is not None and continuation is not None:
                break
        self.assertIsNotNone(turnover)
        self.assertEqual(turnover.get("unobserved_development_seconds", 0.0), 0.0)
        self.assertGreater(continuation[0].get("unobserved_development_seconds", 0.0), 0.0)
        self.assertEqual(continuation[1].get("unobserved_development_seconds", 0.0), 0.0)

    def test_a_tiny_draw_adds_no_clock_time(self):
        game = self._play(3, TimingAssumptions(development_mean_seconds=1e-8))
        self.assertNotIn("provisional_development", game["live_seconds"])
        self.assertEqual(game["status"], "final")
        self.assertAlmostEqual(sum(game["live_seconds"].values()), game["game_clock_seconds"], delta=1e-6)

    def test_the_sample_and_the_hold_cannot_be_combined(self):
        with self.assertRaises(ValueError):
            simulate_game(
                self.bulls.team, self.bay, self.players, self.era, 1,
                timing=TimingAssumptions(first_trip_targets=(12.0,), development_mean_seconds=5.0),
                rotation_targets=self.rotations,
            )
        with self.assertRaises(ValueError):
            self._play(1, TimingAssumptions(development_mean_seconds=0.0))

    def test_batch_selects_the_mode_only_for_1995_96(self):
        def unavailable():
            raise AssertionError("the 2015-16 sample was loaded")

        with patch("era_engine.batch.shot_trip_duration_sample", unavailable):
            result = simulate_batch(
                team_a="Chicago Bulls",
                season_a="1995-96",
                team_b="Bay City Legends",
                era="1995-96",
                games=1,
                seed=7,
                timing_mode=TIMING_PROVISIONAL_1995_96,
            )
        self.assertEqual(result["request"]["timing_mode"], TIMING_PROVISIONAL_1995_96)
        self.assertEqual(result["calibration"]["timing"]["classification"], "provisional_aggregate_pace")
        self.assertFalse(result["calibration"]["timing"]["event_level_timestamps"])
        self.assertEqual(result["calibration"]["timing"]["pace_reference_role"], "in_sample_mean_benchmark")
        self.assertFalse(result["calibration"]["timing"]["full_timing_model_calibrated"])
        self.assertFalse(result["historical_calibration_complete"])
        message = next(item["message"] for item in result["calibration"]["warnings"] if item["code"] == "provisional_aggregate_pace")
        self.assertIn("91.1", message)
        self.assertIn("in-sample", message)
        self.assertIn("No 1995-96 play-by-play timestamps are stored.", message)
        self.assertEqual(result["schema"], "era-basketball-batch-v1")
        replay = simulate_batch(
            team_a="CHI",
            season_a="1995-96",
            team_b="Bay City Legends",
            era="1995-96",
            games=1,
            seed=7,
            timing_mode=TIMING_PROVISIONAL_1995_96,
        )
        self.assertEqual(result["game_results"], replay["game_results"])
        self.assertEqual(result["wins"], replay["wins"])

        with self.assertRaises(BatchRequestError) as caught:
            simulate_batch(
                team_a="Golden State Warriors",
                season_a="2015-16",
                team_b="Chicago Stampede",
                era="2015-16",
                games=1,
                timing_mode=TIMING_PROVISIONAL_1995_96,
            )
        self.assertEqual(caught.exception.code, "timing_model_not_for_era")

        original = simulate_batch(
            team_a="Chicago Bulls",
            season_a="1995-96",
            team_b="Bay City Legends",
            era="1995-96",
            games=1,
            seed=7,
            timing_mode=TIMING_ORIGINAL,
        )
        self.assertIn("pace_1995_96_uncalibrated", [item["code"] for item in original["calibration"]["warnings"]])
        self.assertGreater(original["teams"]["team_a"]["possessions_per_48"], result["teams"]["team_a"]["possessions_per_48"])


if __name__ == "__main__":
    unittest.main()
