"""Optional first-trip residual. The original timing mode stays the default."""
import json
import tempfile
import unittest
from pathlib import Path

from era_engine.core import TimingAssumptions, simulate_game
from era_engine.playbyplay_timing import ROOT, shot_trip_duration_sample
from era_engine.seed import ERAS, initialize_demo
from era_engine.storage import connect, load

LEGAL_ENDINGS = {
    "period_expired",
    "shot_clock_violation",
    "turnover",
    "pass_turnover",
    "shooting_foul",
    "made_shot",
    "miss",
}
EXPLORATION_IDS = ("0021500003", "0021500290", "0021500605", "0021501227")


class ShotTripSampleTests(unittest.TestCase):
    def test_exploration_sample_is_a_total_not_an_action_time(self):
        sample = shot_trip_duration_sample()
        self.assertEqual(tuple(sample["game_ids"]), EXPLORATION_IDS)
        self.assertEqual(sample["count"], 471)
        self.assertEqual(len(sample["durations"]), 471)
        self.assertTrue(all(0.0 <= value <= 24.0 for value in sample["durations"]))
        self.assertGreater(sum(sample["durations"]) / len(sample["durations"]), 12.0)
        self.assertLess(sum(sample["durations"]) / len(sample["durations"]), 15.0)
        self.assertIn("dribble", sample["what_the_duration_is_not"])
        reserved = {game["game_id"] for game in json.loads(
            (ROOT.parent / "research_playbyplay" / "reserved_b3" / "manifest.json").read_text()
        )["games"]}
        self.assertTrue(reserved.isdisjoint(sample["game_ids"]))


class ProvisionalTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.tmp = tmp
        connection = connect(str(Path(tmp.name) / "demo.db"))
        initialize_demo(connection)
        cls.players, cls.eras, cls.teams = load(connection)
        connection.close()
        cls.home = cls.teams["Bay City Legends"]
        cls.away = cls.teams["Chicago Stampede"]
        cls.era = next(era for era in ERAS if era.key == "2015-16")
        cls.sample = shot_trip_duration_sample()["durations"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _play(self, seed=7, targets=None):
        timing = None if targets is None else TimingAssumptions(first_trip_targets=tuple(targets))
        return simulate_game(self.home, self.away, self.players, self.era, seed, timing=timing)

    def test_original_timing_is_the_default_and_matches_an_explicit_fallback(self):
        default = simulate_game(self.home, self.away, self.players, self.era, 7)
        fallback = simulate_game(
            self.home, self.away, self.players, self.era, 7, timing=TimingAssumptions(),
        )
        self.assertEqual(default["score"], fallback["score"])
        self.assertEqual(default["possessions_by_team"], fallback["possessions_by_team"])
        self.assertEqual(default["events"], fallback["events"])
        self.assertEqual(default["timing_assumptions"]["timing_mode"], "original")
        self.assertNotIn("unobserved_development", default["live_seconds"])
        self.assertFalse(default["pace_used_to_choose_possessions"])

    def test_residual_mode_is_deterministic_and_keeps_legal_clocks(self):
        first = self._play(19, self.sample)
        second = self._play(19, self.sample)
        self.assertEqual(first, second)
        self.assertEqual(first["timing_assumptions"]["timing_mode"], "first_trip_residual")
        self.assertFalse(first["pace_used_to_choose_possessions"])
        self.assertAlmostEqual(sum(first["live_seconds"].values()), first["game_clock_seconds"], delta=1e-6)
        self.assertNotIn("free_throw", first["live_seconds"])
        self.assertGreater(first["live_seconds"].get("unobserved_development", 0.0), 0.0)
        for event in first["events"]:
            self.assertIn(event["kind"], LEGAL_ENDINGS)
            self.assertGreaterEqual(event["shot_clock_remaining"], 0.0)
            self.assertGreaterEqual(event["quarter_remaining"], 0.0)
            self.assertGreaterEqual(event["elapsed_seconds"], 0.0)
            if event.get("unobserved_development_seconds", 0.0) > 0.0:
                self.assertGreater(event["setup_game_clock_seconds"], 0.0)
        self.assertEqual(
            first["score"][0],
            sum(first["player_stats"][pid].get("pts", 0) for pid in self.home.player_ids),
        )
        self.assertAlmostEqual(
            sum(first["player_stats"][pid]["min"] for pid in self.home.player_ids),
            first["game_clock_seconds"] / 60.0 * 5,
            delta=1e-6,
        )

    def test_a_short_target_adds_no_development_and_a_long_target_is_not_a_pace_quota(self):
        original = self._play(7)
        short = self._play(7, (1.0,))
        long = self._play(7, (20.0,))
        self.assertNotIn("unobserved_development", short["live_seconds"])
        self.assertGreater(long["live_seconds"]["unobserved_development"], 0.0)
        self.assertLess(long["possessions_by_team"][0], original["possessions_by_team"][0])
        self.assertNotEqual(long["possessions_by_team"][0], 100)
        self.assertIn("Pace is not an input", long["timing_assumptions"]["unobserved_development"])

    def test_an_offensive_rebound_does_not_draw_a_second_first_trip(self):
        found = None
        for seed in range(40):
            game = self._play(seed, (22.0,))
            shots = {}
            for event in game["events"]:
                if event["kind"] not in ("made_shot", "miss", "shooting_foul"):
                    continue
                shots.setdefault(event["possession"], []).append(event)
            for sequence in shots.values():
                if any(event.get("offensive_rebound") for event in sequence) and len(sequence) >= 2:
                    found = sequence
                    break
            if found is not None:
                break
        self.assertIsNotNone(found)
        self.assertGreater(found[0].get("unobserved_development_seconds", 0.0), 0.0)
        self.assertEqual(found[1].get("unobserved_development_seconds", 0.0), 0.0)

    def test_a_full_shot_clock_total_is_a_shot_not_a_manufactured_violation(self):
        for seed in range(8):
            game = self._play(seed, (24.0,))
            self.assertFalse(any(event["kind"] == "shot_clock_violation" for event in game["events"]))
            self.assertGreater(game["live_seconds"].get("unobserved_development", 0.0), 0.0)
            self.assertAlmostEqual(sum(game["live_seconds"].values()), game["game_clock_seconds"], delta=1e-6)

    def test_empty_or_negative_targets_are_rejected(self):
        with self.assertRaises(ValueError):
            self._play(0, ())
        with self.assertRaises(ValueError):
            self._play(0, (-1.0,))
