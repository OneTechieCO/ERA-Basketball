"""Invariants for the isolated shoot-or-pass experiment."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from era_engine.core import clamp, simulate_game as simulate_live
from era_engine.decision_experiments import intelligent_pass_chance, simulate_intelligent
from era_engine.receiver_experiments import simulate_variant_a
from era_engine.seed import initialize_demo
from era_engine.storage import connect, load


def _live_pass_chance(player, style_bonus, pass_count):
    return clamp(
        0.42
        + (player.passing - 70) * 0.004
        + (player.iq - 70) * 0.002
        + style_bonus
        - pass_count * 0.045,
        0.12,
        0.82,
    )


class DecisionExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = connect(str(Path(self.tmp.name) / "test.db"))
        initialize_demo(self.con)
        self.players, self.eras, self.teams = load(self.con)
        self.team_a = self.teams["Bay City Legends"]
        self.team_b = self.teams["Chicago Stampede"]
        self.star = self.players["ERA_0001"]
        self.neutral = replace(self.players["ERA_0002"], defense=73)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def _chance(self, shooter, defender=None, style=0.0, passes=0, remaining=18.0):
        return intelligent_pass_chance(
            shooter,
            self.neutral if defender is None else defender,
            style,
            passes,
            remaining,
        )

    def test_good_shooter_keeps_a_favorable_look(self):
        star_chance = self._chance(self.star, style=0.08)
        live = _live_pass_chance(self.star, 0.08, 0)
        self.assertLess(star_chance, live)
        self.assertGreater(star_chance, 0.25)
        weak_shot = replace(
            self.star,
            shot_skills={"rim": 60, "paint": 60, "mid": 60, "three": 60},
            tendencies={"rim": 0.25, "paint": 0.25, "mid": 0.25, "three": 0.25},
        )
        self.assertLess(star_chance, self._chance(weak_shot, style=0.08))

    def test_tendencies_change_the_look_without_changing_skills(self):
        skills = {"rim": 70, "paint": 70, "mid": 70, "three": 98}
        prefers_three = replace(
            self.star,
            shot_skills=skills,
            tendencies={"rim": 0.05, "paint": 0.05, "mid": 0.05, "three": 0.85},
        )
        prefers_paint = replace(
            self.star,
            shot_skills=skills,
            tendencies={"rim": 0.05, "paint": 0.85, "mid": 0.05, "three": 0.05},
        )
        self.assertLess(self._chance(prefers_three), self._chance(prefers_paint))

    def test_good_passers_still_create(self):
        creator = replace(self.star, passing=96, iq=92)
        reluctant = replace(self.star, passing=45, iq=60)
        self.assertGreater(self._chance(creator) - self._chance(reluctant), 0.08)

    def test_usage_defense_clock_and_style_move_the_decision(self):
        low_usage = replace(self.star, usage=0.12)
        self.assertLess(self._chance(self.star), self._chance(low_usage))
        tight = replace(self.neutral, defense=98)
        loose = replace(self.neutral, defense=60)
        self.assertGreater(self._chance(self.star, tight), self._chance(self.star, loose))
        self.assertGreater(self._chance(self.star, remaining=20), self._chance(self.star, remaining=6))
        self.assertGreater(self._chance(self.star, style=0.08), self._chance(self.star, style=-0.04))
        self.assertGreater(self._chance(self.star, passes=0), self._chance(self.star, passes=3))
        for defender_rating, style, passes, remaining in (
            (40, -0.2, 6, 1.0),
            (99, 0.2, 0, 24.0),
        ):
            chance = self._chance(
                self.star,
                replace(self.neutral, defense=defender_rating),
                style,
                passes,
                remaining,
            )
            self.assertGreaterEqual(chance, 0.12)
            self.assertLessEqual(chance, 0.82)

    def test_game_invariants_and_preserved_clock_gates(self):
        for era_key in ("1995-96", "2023-24"):
            with self.subTest(era=era_key):
                game = simulate_intelligent(
                    self.team_a, self.team_b, self.players, self.eras[era_key], 7
                )
                again = simulate_intelligent(
                    self.team_a, self.team_b, self.players, self.eras[era_key], 7
                )
                self.assertEqual(game, again)
                self.assertEqual(
                    game,
                    simulate_live(self.team_a, self.team_b, self.players, self.eras[era_key], 7),
                )
                self.assertTrue(all(0 <= event["passes"] <= 6 for event in game["events"]))
                self.assertGreater(max(event["passes"] for event in game["events"]), 0)
                self.assertGreater(
                    sum(stats.get("ast", 0) for stats in game["player_stats"].values()),
                    0,
                )
                for stats in game["player_stats"].values():
                    fgm = stats.get("fgm", 0)
                    fg3m = stats.get("fg3m", 0)
                    self.assertLessEqual(fgm, stats.get("fga", 0))
                    self.assertLessEqual(fg3m, stats.get("fg3a", 0))
                    self.assertLessEqual(stats.get("ftm", 0), stats.get("fta", 0))
                    self.assertEqual(
                        stats.get("pts", 0),
                        2 * (fgm - fg3m) + 3 * fg3m + stats.get("ftm", 0),
                    )
                for team in (self.team_a, self.team_b):
                    idx = 0 if team is self.team_a else 1
                    self.assertEqual(
                        game["score"][idx],
                        sum(game["player_stats"][pid].get("pts", 0) for pid in team.player_ids),
                    )
                    assists = sum(game["player_stats"][pid].get("ast", 0) for pid in team.player_ids)
                    makes = sum(game["player_stats"][pid].get("fgm", 0) for pid in team.player_ids)
                    self.assertLessEqual(assists, makes)

        short = replace(self.eras["1995-96"], shot_clock=1)
        with self.assertRaises(RuntimeError) as caught:
            simulate_intelligent(self.team_a, self.team_b, self.players, short, 19)
        game = caught.exception.result
        self.assertFalse(game["completed"])
        self.assertEqual(game["score"], [0, 0])
        self.assertIn("No tie-break point", str(caught.exception))
        self.assertGreater(sum(event["kind"] == "shot_clock_violation" for event in game["events"]), 0)
        self.assertTrue(all(event["passes"] == 0 for event in game["events"]))
        self.assertEqual(sum(stats.get("fga", 0) for stats in game["player_stats"].values()), 0)

    def test_no_shot_quota_over_a_short_sample(self):
        era = self.eras["2015-16"]
        totals = {pid: 0 for pid in self.team_a.player_ids}
        for idx in range(12):
            game = simulate_intelligent(
                self.team_a, self.team_b, self.players, era, 42 + idx * 997
            )
            for pid in self.team_a.player_ids:
                totals[pid] += game["player_stats"][pid].get("fga", 0)
        team_fga = sum(totals.values())
        shares = {pid: totals[pid] / team_fga for pid in totals}
        self.assertGreater(min(shares.values()), 0.10)
        self.assertLess(max(shares.values()), 0.35)
        self.assertGreater(shares["ERA_0001"], 0.16)
        live = simulate_live(self.team_a, self.team_b, self.players, era, 42)
        variant = simulate_variant_a(self.team_a, self.team_b, self.players, era, 42)
        intelligent = simulate_intelligent(self.team_a, self.team_b, self.players, era, 42)
        self.assertEqual(intelligent, live)
        self.assertNotEqual(intelligent, variant)


if __name__ == "__main__":
    unittest.main()
