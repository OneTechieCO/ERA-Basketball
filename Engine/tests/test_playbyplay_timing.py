"""Stored play-by-play timing. These tests do not call the network or the live engine."""
import json
import unittest
from pathlib import Path

from era_engine.playbyplay_timing import (
    PROPOSED_HALFCOURT_MODEL,
    ROOT,
    clock_seconds,
    compare_with_engine,
    load_actions,
    load_manifest,
    reconstruct_possessions,
    warriors_sample_summary,
)


class StoredPlayByPlayTests(unittest.TestCase):
    def test_manifest_matches_the_saved_responses(self):
        manifest = load_manifest()
        self.assertEqual(manifest["source_name"], "nba.com via nba_api")
        self.assertEqual(manifest["play_by_play_endpoint"], "playbyplayv3")
        self.assertEqual(manifest["clock_field"], "clock")
        self.assertIsNone(manifest["shot_clock_field"])
        self.assertFalse(manifest["existing_season_tables_modified"])
        self.assertEqual(len(manifest["games"]), 4)
        self.assertEqual(len(manifest["bulls_1995_96_probe"]), 2)
        for game in manifest["games"]:
            actions = load_actions(game["game_id"])
            self.assertGreater(len(actions), 100)
            self.assertIn("clock", actions[0])
            self.assertNotIn("shotClock", actions[0])
            self.assertEqual(actions[0]["period"], 1)
        for game in manifest["bulls_1995_96_probe"]:
            self.assertEqual(load_actions(game["game_id"]), [])
            self.assertEqual(game["actions"], 0)
        finder = json.loads((ROOT / "source" / "leaguegamefinder_gsw_2015_16.json").read_text())
        self.assertIn("resultSets", finder)
        ids = {
            row[4]
            for table in finder["resultSets"]
            if table.get("name") == "LeagueGameFinderResults"
            for row in table["rowSet"]
        }
        for game in manifest["games"]:
            self.assertIn(game["game_id"], ids)

    def test_clock_and_possession_rules_on_a_small_sequence(self):
        self.assertEqual(clock_seconds("PT12M00.00S"), 720)
        self.assertEqual(clock_seconds("PT02M16.00S"), 136)
        actions = [
            {"actionNumber": 1, "clock": "PT12M00.00S", "period": 1, "teamTricode": "", "actionType": "period", "subType": "start", "description": "Start"},
            {"actionNumber": 2, "clock": "PT11M40.00S", "period": 1, "teamTricode": "GSW", "personId": 1, "actionType": "Missed Shot", "subType": "Layup", "description": "MISS"},
            {"actionNumber": 3, "clock": "PT11M38.00S", "period": 1, "teamTricode": "GSW", "personId": 1, "actionType": "Rebound", "subType": "Unknown", "description": "Player REBOUND (Off:1 Def:0)"},
            {"actionNumber": 4, "clock": "PT11M30.00S", "period": 1, "teamTricode": "GSW", "personId": 1, "actionType": "Made Shot", "subType": "Layup", "description": "Layup"},
            {"actionNumber": 5, "clock": "PT11M30.00S", "period": 1, "teamTricode": "NOP", "personId": 2, "actionType": "Foul", "subType": "Shooting", "description": "foul"},
            {"actionNumber": 6, "clock": "PT11M30.00S", "period": 1, "teamTricode": "GSW", "personId": 1, "actionType": "Free Throw", "subType": "Free Throw 1 of 2", "description": "MISS free throw"},
            {"actionNumber": 7, "clock": "PT11M30.00S", "period": 1, "teamTricode": "GSW", "personId": 9, "actionType": "Rebound", "subType": "Unknown", "description": "Team Rebound"},
            {"actionNumber": 8, "clock": "PT11M30.00S", "period": 1, "teamTricode": "GSW", "personId": 1, "actionType": "Free Throw", "subType": "Free Throw 2 of 2", "description": "Free throw"},
            {"actionNumber": 9, "clock": "PT11M10.00S", "period": 1, "teamTricode": "NOP", "personId": 2, "actionType": "Turnover", "subType": "Bad Pass", "description": "Bad Pass"},
        ]
        rows = [row for row in reconstruct_possessions(actions, "synthetic") if row["reliable"]]
        self.assertEqual([row["end_reason"] for row in rows], ["made_shot", "turnover"])
        self.assertEqual(rows[0]["offensive_rebounds"], 1)
        self.assertEqual(rows[0]["duration"], 30)
        self.assertEqual(rows[0]["continuation_seconds"], 8)
        self.assertIsNone(rows[0]["shot_clock_remaining"])
        self.assertEqual(rows[1]["duration"], 20)
        self.assertEqual(rows[1]["offense"], "NOP")

    def test_saved_warriors_sample_is_a_distribution_not_a_pace_target(self):
        summary = warriors_sample_summary()
        self.assertEqual(summary["games"], 4)
        self.assertGreater(summary["possessions"], 400)
        self.assertGreater(summary["duration"]["p50"], 10)
        self.assertLess(summary["duration"]["p50"], 18)
        self.assertGreater(summary["shot_endings"]["mean"], summary["turnover_endings"]["mean"])
        self.assertIsNone(summary["shot_clock_remaining"])
        self.assertFalse(summary["shot_clock_field_present"])
        self.assertFalse(summary["transition_halfcourt_supported"])
        self.assertIsNone(summary["pass_duration"])
        self.assertGreater(summary["offensive_rebounds"], 0)
        self.assertFalse(PROPOSED_HALFCOURT_MODEL["implemented"])
        comparison = compare_with_engine(summary, {"experimental_mean": summary["duration"]["mean"]})
        self.assertFalse(comparison["pace_fitted"])
        self.assertEqual(comparison["models"][0]["minus_observed"], 0)
        self.assertFalse(comparison["transition_compared"])


if __name__ == "__main__":
    unittest.main()
