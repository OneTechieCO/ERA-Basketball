"""Experimental game clock. These tests do not run the live possession engine."""
import unittest

from era_engine.game_clock_experiments import (
    QUARTER_SECONDS,
    REGULATION_QUARTERS,
    REGULATION_SECONDS,
    SHOT_CLOCK_SECONDS,
    GameClock,
    ShotClock,
    apply_live_action,
    apply_stoppage,
    lineup_at,
    play_era_regulation,
    play_regulation_game,
    setup_seconds,
    simulate_seeded_eras,
    summarize_games,
)
from era_engine.rotation_experiments import RotationPlayer, schedule_regulation
from era_engine.seed import ERAS


def _rotation(prefix, count=8, minutes=30.0):
    players = [RotationPlayer(f"{prefix}{index}", minutes, True, 80) for index in range(count)]
    players.append(RotationPlayer(f"{prefix}OUT", 0, False, 80))
    rotation = schedule_regulation(players, "target_minutes")
    rotation["team_name"] = prefix
    return rotation


class GameClockTests(unittest.TestCase):
    def test_regulation_is_four_twelve_minute_quarters(self):
        clock = GameClock()
        self.assertEqual(clock.game_remaining, REGULATION_SECONDS)
        self.assertEqual(clock.quarter_remaining, QUARTER_SECONDS)
        for quarter in range(1, REGULATION_QUARTERS + 1):
            self.assertEqual(clock.quarter, quarter)
            elapsed = clock.advance_live(QUARTER_SECONDS)
            self.assertEqual(elapsed, QUARTER_SECONDS)
            self.assertEqual(clock.quarter_remaining, 0.0)
            if quarter < REGULATION_QUARTERS:
                self.assertTrue(clock.start_next_quarter())
                self.assertEqual(clock.quarter_remaining, QUARTER_SECONDS)
        self.assertFalse(clock.start_next_quarter())
        self.assertTrue(clock.regulation_complete)
        self.assertEqual(clock.game_remaining, 0.0)
        self.assertIsNone(play_regulation_game(1, _rotation("H"), _rotation("A"))["overtime"]["schedule"])

    def test_clocks_do_not_go_negative_and_dead_ball_time_does_not_run(self):
        game = GameClock()
        shot = ShotClock()
        game.quarter_remaining = 2.0
        shot.remaining = 5.0
        action = apply_live_action(game, shot, 6.0, "setup")
        self.assertFalse(action["finished"])
        self.assertEqual(action["reason"], "period_expired")
        self.assertEqual(action["game_clock_elapsed"], 2.0)
        self.assertEqual(game.quarter_remaining, 0.0)
        self.assertEqual(shot.remaining, 3.0)
        self.assertEqual(game.advance_live(4.0), 0.0)
        with self.assertRaises(ValueError):
            game.advance_live(-1)
        with self.assertRaises(ValueError):
            shot.advance(-1)
        stopped = apply_stoppage(game, "free_throw", 4.0)
        self.assertTrue(stopped["stopped_clock"])
        self.assertEqual(stopped["game_clock_elapsed"], 0.0)
        self.assertEqual(game.quarter_remaining, 0.0)
        self.assertEqual(shot.remaining, 3.0)

    def test_shot_clock_reset_does_not_move_the_game_clock_or_cross_a_period(self):
        self.assertEqual(setup_seconds(24), 6.0)
        game = GameClock()
        shot = ShotClock()
        before = game.game_remaining
        apply_live_action(game, shot, setup_seconds(shot.limit), "setup")
        apply_live_action(game, shot, 2.0, "completed_pass")
        apply_live_action(game, shot, 2.0, "release")
        apply_live_action(game, shot, 1.0, "rebound_gather")
        shot.reset(14)
        self.assertEqual(shot.remaining, 14)
        self.assertEqual(game.game_remaining, before - 11.0)
        self.assertGreater(11.0, 0)
        self.assertNotEqual(11.0, SHOT_CLOCK_SECONDS)
        game.quarter_remaining = 3.0
        shot.remaining = 20.0
        unfinished = apply_live_action(game, shot, 6.0, "setup")
        self.assertEqual(unfinished["reason"], "period_expired")
        self.assertTrue(game.start_next_quarter())
        self.assertEqual(game.quarter_remaining, QUARTER_SECONDS)
        self.assertEqual(shot.remaining, 17.0)
        long_game = GameClock()
        long_shot = ShotClock()
        for kind, seconds in (
            ("setup", 6.0),
            ("completed_pass", 2.0),
            ("completed_pass", 2.0),
            ("completed_pass", 2.0),
            ("completed_pass", 2.0),
            ("completed_pass", 2.0),
            ("completed_pass", 2.0),
            ("release", 3.0),
        ):
            self.assertTrue(apply_live_action(long_game, long_shot, seconds, kind)["finished"])
        gather = long_game.advance_live(1.0)
        self.assertEqual(gather, 1.0)
        long_shot.reset(14)
        self.assertEqual(long_shot.remaining, 14)
        continued = apply_live_action(long_game, long_shot, 3.0, "release")
        self.assertTrue(continued["finished"])
        self.assertGreater(REGULATION_SECONDS - long_game.game_remaining, SHOT_CLOCK_SECONDS)
        self.assertEqual(long_shot.remaining, 11)

    def test_a_release_that_does_not_fit_is_a_shot_clock_violation(self):
        game = GameClock()
        shot = ShotClock()
        shot.remaining = 1.0
        action = apply_live_action(game, shot, 3.0, "release")
        self.assertFalse(action["finished"])
        self.assertEqual(action["reason"], "shot_clock_expired")
        self.assertEqual(shot.remaining, 0.0)
        self.assertEqual(action["game_clock_elapsed"], 1.0)
        self.assertGreater(game.quarter_remaining, 0.0)


class RotationAndTimingTests(unittest.TestCase):
    def test_game_clock_reads_the_rotation_without_crossing_live_actions(self):
        home = _rotation("H")
        away = _rotation("A")
        game = play_regulation_game(7, home, away)
        self.assertEqual(game["regulation_seconds"], REGULATION_SECONDS)
        self.assertEqual(game["quarter_lengths"], [QUARTER_SECONDS] * 4)
        self.assertEqual(game["period_ending_possessions"], 4)
        self.assertIsNone(game["pace_input"])
        self.assertFalse(game["promoted"])
        for index, rotation in enumerate((home, away)):
            minutes = game["player_minutes"][index]
            self.assertAlmostEqual(sum(minutes.values()), 240.0, places=6)
            self.assertTrue(all(value <= 48.0 + 1e-6 for value in minutes.values()))
            self.assertEqual(minutes[f"{rotation['team_name']}OUT"], 0.0)
            self.assertGreater(game["substitutions"][index], 0)
        for possession in game["possession_log"]:
            for lineup in possession["lineups"]:
                self.assertEqual(len(set(lineup)), 5)
            self.assertAlmostEqual(
                possession["game_clock_elapsed"],
                sum(action["game_clock_elapsed"] for action in possession["actions"]),
                places=6,
            )
            scheduled = lineup_at(home, possession["elapsed_at_start"])
            self.assertEqual(possession["lineups"][0], scheduled)
        crossed = [
            possession for possession in game["possession_log"]
            if any(
                possession["elapsed_at_start"] < sub["minute"] * 60
                <= possession["elapsed_at_start"] + possession["game_clock_elapsed"]
                for sub in home["substitutions"]
            )
        ]
        self.assertGreater(len(crossed), 0)
        for possession in crossed:
            self.assertEqual(possession["lineups"][0], lineup_at(home, possession["elapsed_at_start"]))
        fouls = [
            possession for possession in game["possession_log"]
            if possession["end_reason"] == "shooting_foul"
        ]
        self.assertGreater(len(fouls), 0)
        for possession in fouls:
            free_throws = [action for action in possession["actions"] if action["kind"] == "free_throw"]
            self.assertTrue(free_throws)
            self.assertTrue(all(action["game_clock_elapsed"] == 0.0 and action["stopped_clock"] for action in free_throws))
        self.assertEqual(play_regulation_game(7, home, away)["possessions"], game["possessions"])

    def test_possession_counts_are_not_taken_from_era_pace(self):
        home = _rotation("H")
        away = _rotation("A")
        seeds = (7, 19, 42, 101, 256, 997, 2026, 4242)
        games = [play_regulation_game(seed, home, away) for seed in seeds]
        summary = summarize_games(games)
        paces = {era.key: era.pace for era in ERAS}
        observed = summary["possessions_per_team_combined_mean"]
        self.assertGreater(observed, 40)
        self.assertNotIn(observed, paces.values())
        for game in games:
            self.assertAlmostEqual(
                game["average_game_clock_per_possession"] * game["possessions"],
                REGULATION_SECONDS,
                places=5,
            )
            self.assertEqual(game["quarter_lengths"], [QUARTER_SECONDS] * 4)
            self.assertTrue(all(possession["game_clock_elapsed"] >= 0 for possession in game["possession_log"]))
            self.assertTrue(all(possession["shot_clock_remaining"] >= 0 for possession in game["possession_log"]))

    def test_seeded_eras_keep_their_pace_as_a_comparison_only(self):
        home = _rotation("H")
        away = _rotation("A")
        seeds = (7, 19, 42, 101, 256, 997, 2026, 4242)
        rows = simulate_seeded_eras(ERAS, home, away, seeds)
        by_era = {row["era"]: row for row in rows}
        self.assertEqual(set(by_era), {era.key for era in ERAS})
        full_reset = [row for row in rows if row["rebound_reset"] == 24]
        self.assertEqual(len(full_reset), 4)
        self.assertEqual(by_era["2023-24"]["rebound_reset"], 14)
        self.assertEqual(
            [round(value, 6) for value in full_reset[0]["possessions_per_team"]],
            [round(value, 6) for value in full_reset[1]["possessions_per_team"]],
        )
        self.assertNotEqual(full_reset[0]["seeded_pace"], full_reset[1]["seeded_pace"])
        for row in rows:
            self.assertFalse(row["pace_used_to_choose_possessions"])
            self.assertGreater(abs(row["pace_gap"]), 20)
            self.assertEqual(row["quarter_endings_mean"], 4)
            self.assertEqual(row["shot_clock"], 24)
        sample = play_era_regulation(7, ERAS[0], home, away)
        self.assertEqual(sample["period_breaks"], 3)
        self.assertIsNone(sample["overtime"]["schedule"])
        self.assertEqual(sample["seeded_pace"], ERAS[0].pace)
        self.assertNotEqual(sample["possessions_per_team"][0], round(sample["seeded_pace"]))
        for minutes in sample["player_minutes"]:
            self.assertAlmostEqual(sum(minutes.values()), 240.0, places=6)
        for possession in sample["possession_log"]:
            for lineup in possession["lineups"]:
                self.assertEqual(len(set(lineup)), 5)


if __name__ == "__main__":
    unittest.main()
