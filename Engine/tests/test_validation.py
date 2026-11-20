"""Fast checks for the reliability runner. The 1,000-game workload stays a command."""
import json
import tempfile
import unittest
from pathlib import Path

from era_engine.core import UnresolvedGameError, offensive_rebound_reset, simulate_game
from era_engine.seed import ERAS
from era_engine.validation import (
    SERIES_GAME_LIMIT,
    batch_seeds,
    check_game,
    checkpoints_for,
    demo_matchups,
    main,
)


class ValidationRunnerTests(unittest.TestCase):
    def test_seeds_are_recorded_and_the_series_cap_stays_82(self):
        self.assertEqual(batch_seeds(4), [0, 1, 2, 3])
        self.assertEqual(checkpoints_for(1000), (1, 10, 100, 1000))
        self.assertEqual(checkpoints_for(1), (1,))
        self.assertEqual(SERIES_GAME_LIMIT, 82)
        with self.assertRaises(ValueError):
            batch_seeds(0)

    def test_a_finished_demo_game_passes_the_accounting_checks(self):
        matchup = next(item for item in demo_matchups() if item.era.key == "1995-96")
        game = simulate_game(matchup.team_a, matchup.team_b, matchup.players, matchup.era, 0, track_events=True)
        self.assertEqual(check_game(game, matchup.era, matchup.team_a.player_ids, matchup.team_b.player_ids), [])
        game["score"] = [game["score"][0] + 1, game["score"][1]]
        self.assertIn("player points do not sum to the team score", check_game(game, matchup.era, matchup.team_a.player_ids, matchup.team_b.player_ids))

    def test_off_court_credit_bad_reset_and_a_negative_clock_fail(self):
        matchup = demo_matchups()[0]
        game = simulate_game(matchup.team_a, matchup.team_b, matchup.players, matchup.era, 0, track_events=True)
        ids = (matchup.team_a.player_ids, matchup.team_b.player_ids)
        broken_clock = json.loads(json.dumps(game))
        broken_clock["events"][0]["shot_clock_remaining"] = -1
        self.assertIn("shot_clock_remaining is negative or invalid", check_game(broken_clock, matchup.era, *ids))

        broken_actor = json.loads(json.dumps(game))
        for event in broken_actor["events"]:
            if event.get("offense"):
                event["offense"] = "ERA_9999"
                break
        self.assertIn("an off-court player is credited on offense", check_game(broken_actor, matchup.era, *ids))

        recent = next(item for item in demo_matchups() if item.era.key == "2023-24")
        early = next(item for item in demo_matchups() if item.era.key == "1995-96")
        self.assertEqual(offensive_rebound_reset(recent.era), 14)
        self.assertEqual(offensive_rebound_reset(early.era), 24)
        found = False
        for seed in range(8):
            sample = simulate_game(recent.team_a, recent.team_b, recent.players, recent.era, seed, track_events=True)
            if any(event.get("offensive_rebound") for event in sample["events"]):
                found = True
                self.assertEqual(check_game(sample, recent.era, recent.team_a.player_ids, recent.team_b.player_ids), [])
                for event in sample["events"]:
                    if event.get("offensive_rebound"):
                        event["shot_clock_reset"] = 24
                        break
                self.assertIn(
                    "offensive-rebound reset does not follow the era rule",
                    check_game(sample, recent.era, recent.team_a.player_ids, recent.team_b.player_ids),
                )
                break
        self.assertTrue(found)

    def test_an_unresolved_tie_counts_as_incomplete(self):
        matchup = demo_matchups()[0]

        def unresolved(*args, **kwargs):
            raise UnresolvedGameError({"overtime": {"periods": 15}, "score": [100, 100]})

        from era_engine.validation import run_workload
        result = run_workload(matchup, 1, replay=False, simulate=unresolved)
        self.assertEqual(result["games_completed"], 0)
        self.assertEqual(result["games_incomplete"], 1)
        self.assertEqual(result["failure_examples"][0]["status"], "incomplete")

    def test_the_command_runs_one_game_without_the_thousand_game_workload(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "report.json"
            code = main(["--max-games", "1", "--cross-era-games", "1", "--output", str(destination)])
            report = json.loads(destination.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(report["functional_failures"], 0)
        self.assertEqual(report["series_contract"]["maximum_games"], 82)
        names = {item["name"]: item for item in report["workloads"]}
        self.assertEqual(set(names), {f"demo_{era.key}" for era in ERAS} | {"warriors_2015_16", "bulls_1995_96"})
        self.assertEqual(names["warriors_2015_16"]["timing_mode"], "first_trip_residual")
        self.assertEqual(names["bulls_1995_96"]["timing_mode"], "original")
        self.assertEqual(list(names["demo_1995-96"]["checkpoints"]), ["1"])
        for item in report["cross_era"]:
            self.assertTrue(item["hypothetical"])
            self.assertEqual(item["timing_mode"], "original")
            self.assertEqual(item["games_completed"], 1)
            self.assertEqual(item["replay"]["mismatches"], [])
            self.assertIn("not evidence about which real team would win", item["label"])
        self.assertIn("Not a historical replay", names["warriors_2015_16"]["label"])
        self.assertIsNone(report["historical_comparison"]["bulls_1995_96"]["stored"]["dash_rows"][0]["pace"])
        self.assertFalse(report["historical_comparison"]["warriors_2015_16"]["stored"]["possessions_and_fga_on_same_row"])

    def test_provisional_timing_is_not_reported_as_original(self):
        from era_engine.core import PROVISIONAL_1995_96_DEVELOPMENT_MEAN, TimingAssumptions
        from era_engine.historical_players import load_researched_rosters
        from era_engine.validation import Matchup, run_workload, timing_mode_name

        bulls, _warriors = load_researched_rosters()
        bay = next(item for item in demo_matchups() if item.era.key == "1995-96")
        timing = TimingAssumptions(development_mean_seconds=PROVISIONAL_1995_96_DEVELOPMENT_MEAN)
        self.assertEqual(timing_mode_name(timing), "provisional_aggregate_pace")
        self.assertEqual(timing_mode_name(None), "original")
        result = run_workload(Matchup(
            name="bulls_provisional_label",
            label="label check",
            team_a=bulls.team,
            team_b=bay.team_a,
            players={**bay.players, **bulls.players},
            era=bay.era,
            timing=timing,
            rotation_targets=(dict(bulls.rotation_targets), None),
        ), 1, replay=False)
        self.assertEqual(result["timing_mode"], "provisional_aggregate_pace")
        self.assertEqual(result["games_completed"], 1)
        self.assertEqual(result["games_failed_checks_or_errors"], 0)
