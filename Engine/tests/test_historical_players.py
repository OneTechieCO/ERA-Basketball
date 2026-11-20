"""Historical roster construction and live-engine compatibility.

These tests use the stored research database. They do not treat a simulated
score as a historical result, and they do not change the fictional demo.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from era_engine.core import free_throw_make_probability, probability_make, selection_weight, simulate_game
from era_engine.historical_players import (
    ASSUMPTIONS,
    OFFENSIVE_EVENT_FREE_THROW_WEIGHT,
    HistoricalDataError,
    attempt_rate_gap,
    compare_game,
    connect_research,
    load_historical_roster,
    load_researched_rosters,
)
from era_engine.research_report import ensure_external_schema
from era_engine.seed import ERAS, demo_players, initialize_demo
from era_engine.storage import connect, create_research_schema, load


def _line(**overrides):
    row = {
        "games": 10, "minutes": 200, "points": 20, "fgm": 8, "fga": 20,
        "fg3m": 2, "fg3a": 6, "ftm": 2, "fta": 4, "assists": 3,
        "offensive_rebounds": 1, "defensive_rebounds": 2, "steals": 1,
        "blocks": 1, "turnovers": 2,
    }
    row.update(overrides)
    return row


def _insert_player(connection, *, player_id, name, season, team_id, abbreviation, source="nba.com via nba_api", **stats):
    connection.execute(
        '''INSERT INTO nba_player_season_research(
             nba_player_id, player_name, season_id, team_id, team_abbreviation,
             source_name, endpoint, dataset_name, counting_measure,
             games, minutes, points, fgm, fga, fg3m, fg3a, ftm, fta, assists,
             offensive_rebounds, defensive_rebounds, steals, blocks, turnovers,
             usage_rate, usage_rate_measure)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (player_id, name, season, team_id, abbreviation, source, "playercareerstats",
         "SeasonTotalsRegularSeason", "season_total", stats["games"], stats["minutes"], stats["points"],
         stats["fgm"], stats["fga"], stats["fg3m"], stats["fg3a"], stats["ftm"], stats["fta"],
         stats["assists"], stats["offensive_rebounds"], stats["defensive_rebounds"], stats["steals"],
         stats["blocks"], stats["turnovers"], None, None),
    )


def _insert_team(connection, season, team_id, name, players):
    totals = {field: 0.0 for field in _line()}
    for player in players:
        if player["abbreviation"] == "TOT":
            continue
        for field in totals:
            value = player["stats"][field]
            totals[field] += 0 if value is None else value
    connection.execute(
        '''INSERT INTO nba_team_season_research(
             team_id, team_name, season_id, source_name, endpoint, dataset_name, counting_measure,
             games, minutes, points, fgm, fga, fg3m, fg3a, ftm, fta, assists,
             offensive_rebounds, defensive_rebounds, steals, blocks, turnovers,
             pace, pace_measure, possessions, possessions_measure)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (team_id, name, season, "nba.com via nba_api", "teamyearbyyearstats", "TeamStats", "season_total",
         10, totals["minutes"], totals["points"], totals["fgm"], totals["fga"], totals["fg3m"],
         totals["fg3a"], totals["ftm"], totals["fta"], totals["assists"], totals["offensive_rebounds"],
         totals["defensive_rebounds"], totals["steals"], totals["blocks"], totals["turnovers"],
         None, None, None, None),
    )


def _memory_roster(players, *, season="1999-00", team_id=999, name="Example Club"):
    connection = sqlite3.connect(":memory:")
    create_research_schema(connection)
    ensure_external_schema(connection)
    for player in players:
        _insert_player(
            connection, player_id=player["id"], name=player["name"], season=season,
            team_id=player.get("team_id", team_id), abbreviation=player["abbreviation"],
            source=player.get("source", "nba.com via nba_api"), **player["stats"],
        )
    roster_players = [player for player in players if player.get("team_id", team_id) == team_id and player["abbreviation"] != "TOT"]
    _insert_team(connection, season, team_id, name, roster_players)
    connection.commit()
    return connection


def _five(names=("A", "B", "C", "D", "E")):
    minutes = (200, 180, 160, 140, 120)
    return [
        {"id": index + 1, "name": name, "abbreviation": "EX", "stats": _line(minutes=minutes[index])}
        for index, name in enumerate(names)
    ]


class HistoricalIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bulls, cls.warriors = load_researched_rosters()

    def test_stored_counting_totals_match_except_the_known_turnover_gap(self):
        for roster, gap in ((self.bulls, 41.0), (self.warriors, 26.0)):
            with self.subTest(team=roster.team_name):
                self.assertEqual(roster.audit["turnover_gap"], gap)
                self.assertFalse(roster.audit["turnover_gap_reconciled"])
                for field, difference in roster.audit["counting_difference_from_team"].items():
                    if field == "turnovers":
                        self.assertEqual(difference, -gap)
                    else:
                        self.assertEqual(difference, 0.0)
                self.assertEqual(roster.audit["source_endpoint"], "playercareerstats")

    def test_split_seasons_use_only_the_team_stint(self):
        salley = next(profile for profile in self.bulls.profiles if profile.player_name == "John Salley")
        self.assertEqual(salley.counting["games"], 17)
        self.assertEqual(salley.counting["minutes"], 191)
        self.assertEqual(salley.team_abbreviation, "CHI")
        thompson = next(profile for profile in self.warriors.profiles if profile.player_name == "Jason Thompson")
        varejao = next(profile for profile in self.warriors.profiles if profile.player_name == "Anderson Varejao")
        self.assertEqual(thompson.counting["minutes"], 179)
        self.assertEqual(varejao.counting["minutes"], 186)
        self.assertEqual(self.warriors.audit["dash_crosscheck"]["absent_from_dash"], ["Jason Thompson"])
        self.assertIn("Jason Thompson", {check["player_name"] for check in self.warriors.audit["split_checks"]})
        self.assertNotIn("TOT", {profile.team_abbreviation for profile in self.bulls.profiles + self.warriors.profiles})

    def test_points_identity_and_attempt_bounds_hold_for_every_stored_player(self):
        for roster in (self.bulls, self.warriors):
            for profile in roster.profiles:
                counting = profile.counting
                expected = 2 * (counting["fgm"] - counting["fg3m"]) + 3 * counting["fg3m"] + counting["ftm"]
                self.assertAlmostEqual(expected, counting["points"])
                self.assertLessEqual(counting["fgm"], counting["fga"])
                self.assertLessEqual(counting["fg3a"], counting["fga"])
                self.assertLessEqual(counting["ftm"], counting["fta"])

    def test_demo_players_remain_fictional(self):
        demo = demo_players()
        self.assertEqual(demo[0].player_id, "ERA_0001")
        self.assertEqual(demo[0].usage, 0.33)
        self.assertTrue(all(not player.player_id.startswith("NBA_") for player in demo))
        historical_ids = {profile.player.player_id for profile in self.bulls.profiles + self.warriors.profiles}
        self.assertTrue(historical_ids.isdisjoint({player.player_id for player in demo}))


class HistoricalProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bulls, cls.warriors = load_researched_rosters()

    def test_only_jordan_and_curry_carry_verified_usage(self):
        jordan = next(profile for profile in self.bulls.profiles if profile.nba_player_id == 893)
        curry = next(profile for profile in self.warriors.profiles if profile.nba_player_id == 201939)
        pippen = next(profile for profile in self.bulls.profiles if profile.player_name == "Scottie Pippen")
        self.assertEqual(jordan.ratings["usage"].classification, "verified")
        self.assertEqual(curry.ratings["usage"].classification, "verified")
        self.assertAlmostEqual(jordan.player.usage, 0.333)
        self.assertAlmostEqual(curry.player.usage, 0.326)
        self.assertNotAlmostEqual(jordan.player.usage, jordan.calculated["fga_share"], places=2)
        self.assertNotAlmostEqual(curry.player.usage, curry.calculated["fga_share"], places=2)
        self.assertEqual(pippen.ratings["usage"].classification, "calculated_proxy")
        self.assertIn("Not NBA USG%", pippen.ratings["usage"].statement)
        self.assertIsNone(pippen.calculated["verified_usage"])
        self.assertEqual(curry.calculated["verified_usage"], curry.player.usage)
        self.assertIsNone(curry.calculated["fga_per_100_team_possessions"])
        self.assertIn("not combined", curry.calculated["fga_per_100_team_possessions_reason"])
        self.assertAlmostEqual(curry.calculated["fga_per_36"], curry.counting["fga"] / curry.counting["minutes"] * 36)
        self.assertAlmostEqual(curry.calculated["fga_per_game"], curry.counting["fga"] / curry.counting["games"])
        self.assertLess(abs(curry.player.shot_propensity), 1.0)
        klay_propensity = next(profile.player.shot_propensity for profile in self.warriors.profiles if profile.player_name == "Klay Thompson")
        green_propensity = next(profile.player.shot_propensity for profile in self.warriors.profiles if profile.player_name == "Draymond Green")
        kukoc_propensity = next(profile.player.shot_propensity for profile in self.bulls.profiles if profile.player_name == "Toni Kukoc")
        self.assertGreater(klay_propensity, curry.player.shot_propensity)
        self.assertLess(klay_propensity, 3.0)
        self.assertLess(green_propensity, 0.0)
        self.assertLess(abs(kukoc_propensity), 1.0)
        self.assertLess(abs(next(profile.player.shot_propensity for profile in self.bulls.profiles if profile.nba_player_id == 917)), 1.5)
        self.assertNotAlmostEqual(curry.player.shot_propensity, curry.player.usage, places=2)
        self.assertNotAlmostEqual(curry.player.shot_propensity, curry.player.event_intensity, places=2)
        self.assertNotAlmostEqual(curry.calculated["fga_per_minute"], curry.player.event_intensity, places=2)
        events = curry.counting["fga"] + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * curry.counting["fta"] + curry.counting["turnovers"]
        team_events = sum(
            profile.counting["fga"] + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * profile.counting["fta"] + profile.counting["turnovers"]
            for profile in self.warriors.profiles
        )
        gap = attempt_rate_gap(curry.counting["fga"], curry.counting["minutes"], events, sum(profile.counting["fga"] for profile in self.warriors.profiles), team_events)
        self.assertAlmostEqual(curry.player.shot_propensity, gap["shot_propensity"])
        self.assertIsNone(demo_players()[0].shot_propensity)
        self.assertIsNone(demo_players()[0].event_intensity)
        klay = next(profile for profile in self.warriors.profiles if profile.player_name == "Klay Thompson")
        self.assertGreater(curry.player.event_intensity, 0.0)
        self.assertGreater(klay.player.event_intensity, 0.0)
        self.assertLess(
            selection_weight(curry.player) / selection_weight(klay.player),
            curry.player.usage / klay.player.usage,
        )
        self.assertNotAlmostEqual(curry.player.event_intensity, curry.player.usage, places=2)
        events = [
            profile.counting["fga"] + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * profile.counting["fta"] + profile.counting["turnovers"]
            for profile in self.bulls.profiles
        ]
        pippen_events = pippen.counting["fga"] + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * pippen.counting["fta"] + pippen.counting["turnovers"]
        self.assertAlmostEqual(pippen.player.usage, pippen_events / sum(events))

    def test_zone_ratings_do_not_invent_a_shot_chart(self):
        longley = next(profile for profile in self.bulls.profiles if profile.player_name == "Luc Longley")
        self.assertEqual(longley.counting["fg3a"], 0)
        self.assertEqual(longley.player.shot_skills["three"], 75)
        self.assertIn("No three-point attempts", longley.ratings["shot_three"].statement)
        self.assertEqual(longley.player.shot_skills["rim"], longley.player.shot_skills["paint"])
        self.assertEqual(longley.player.shot_skills["paint"], longley.player.shot_skills["mid"])
        self.assertIn("Not a shot chart", longley.ratings["shot_rim"].statement)
        jordan = next(profile for profile in self.bulls.profiles if profile.nba_player_id == 893)
        self.assertEqual(jordan.player.tendencies["rim"], jordan.player.tendencies["paint"])
        self.assertEqual(jordan.player.tendencies["paint"], jordan.player.tendencies["mid"])
        self.assertAlmostEqual(sum(jordan.player.tendencies.values()), 1.0)
        self.assertAlmostEqual(
            jordan.calculated["achieved_three_point_attempt_rate"],
            jordan.calculated["shrunk_three_point_attempt_rate"],
            places=3,
        )
        self.assertNotAlmostEqual(jordan.player.tendencies["three"], jordan.calculated["three_point_attempt_rate"], places=3)
        self.assertIsNotNone(jordan.calculated["free_throw_percentage"])
        self.assertTrue(any("midrange formula" in line for line in ASSUMPTIONS))
        self.assertIn("not zero ability", longley.ratings["shot_three"].statement)

    def test_absent_skills_stay_constant_and_counting_skills_keep_their_order(self):
        profiles = self.bulls.profiles + self.warriors.profiles
        self.assertEqual({profile.player.iq for profile in profiles}, {78})
        self.assertEqual({profile.player.defense for profile in profiles}, {75})
        self.assertEqual({profile.ratings["iq"].classification for profile in profiles}, {"uncalibrated"})
        rodman = next(profile for profile in self.bulls.profiles if profile.player_name == "Dennis Rodman")
        kerr = next(profile for profile in self.bulls.profiles if profile.player_name == "Steve Kerr")
        curry = next(profile for profile in self.warriors.profiles if profile.nba_player_id == 201939)
        looney = next(profile for profile in self.warriors.profiles if profile.player_name == "Kevon Looney")
        self.assertGreater(rodman.player.rebounding, kerr.player.rebounding)
        self.assertGreater(curry.player.passing, looney.player.passing)
        self.assertGreater(curry.player.stamina, looney.player.stamina)
        self.assertEqual(rodman.ratings["rebounding"].classification, "uncalibrated")

    def test_shooting_ratings_use_shrunk_totals_and_report_the_formula_cap(self):
        jordan = next(profile for profile in self.bulls.profiles if profile.nba_player_id == 893)
        curry = next(profile for profile in self.warriors.profiles if profile.nba_player_id == 201939)
        kerr = next(profile for profile in self.bulls.profiles if profile.player_name == "Steve Kerr")
        rodman = next(profile for profile in self.bulls.profiles if profile.player_name == "Dennis Rodman")
        wennington = next(profile for profile in self.bulls.profiles if profile.player_name == "Bill Wennington")
        longley = next(profile for profile in self.bulls.profiles if profile.player_name == "Luc Longley")
        bogut = next(profile for profile in self.warriors.profiles if profile.player_name == "Andrew Bogut")
        self.assertEqual(jordan.player.shot_skills["rim"], jordan.player.shot_skills["paint"])
        self.assertEqual(jordan.player.shot_skills["paint"], jordan.player.shot_skills["mid"])
        self.assertAlmostEqual(
            jordan.calculated["reference_two_point_make"],
            jordan.calculated["shrunk_two_point_percentage"],
            places=3,
        )
        self.assertNotIn("cannot reach", jordan.ratings["shot_rim"].statement)
        self.assertEqual(jordan.player.shot_skills["three"], 99)
        self.assertIn("cannot reach", jordan.ratings["shot_three"].statement)
        self.assertLess(
            jordan.calculated["reference_three_point_make"],
            jordan.calculated["shrunk_three_point_percentage"],
        )
        self.assertEqual(curry.player.shot_skills["three"], 99)
        self.assertIn("cannot reach", curry.ratings["shot_three"].statement)
        self.assertIn("cannot reach", kerr.ratings["shot_three"].statement)
        self.assertEqual(kerr.player.shot_skills["three"], 99)
        self.assertEqual(bogut.player.shot_skills["rim"], 99)
        self.assertIn("cannot reach", bogut.ratings["shot_rim"].statement)
        self.assertEqual(longley.player.shot_skills["three"], 75)
        self.assertIn("cannot reach", longley.ratings["tendency_three"].statement)
        self.assertLess(wennington.counting["fg3a"], 5)
        self.assertGreater(wennington.player.shot_skills["three"], 75)
        self.assertLess(wennington.player.shot_skills["three"], 95)
        self.assertLess(wennington.calculated["shrunk_three_point_percentage"], 0.50)
        self.assertLess(rodman.player.shot_skills["three"], 75)
        self.assertGreater(
            rodman.calculated["shrunk_three_point_percentage"],
            rodman.calculated["three_point_percentage"],
        )
        self.assertAlmostEqual(jordan.player.free_throw_probability, jordan.calculated["shrunk_free_throw_percentage"])
        self.assertGreater(jordan.player.free_throw_probability, 0.80)
        self.assertLess(bogut.player.free_throw_probability, 0.60)
        self.assertGreater(bogut.player.free_throw_probability, bogut.calculated["free_throw_percentage"])
        self.assertEqual(bogut.ratings["free_throw"].classification, "calculated_proxy")
        self.assertIn("not the midrange rating", bogut.ratings["free_throw"].statement)
        self.assertEqual(jordan.ratings["shot_three"].classification, "uncalibrated")


class HistoricalRosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bulls, cls.warriors = load_researched_rosters()

    def test_both_researched_teams_build_representative_targets(self):
        self.assertEqual(len(self.bulls.profiles), 15)
        self.assertEqual(len(self.warriors.profiles), 16)
        self.assertEqual(self.bulls.era_key, "1995-96")
        self.assertEqual(self.warriors.era_key, "2015-16")
        self.assertEqual(self.bulls.team.coach_style, "balanced")
        for roster in (self.bulls, self.warriors):
            self.assertAlmostEqual(sum(roster.rotation_targets.values()), 240.0)
            self.assertTrue(all(0 <= value <= 48 for value in roster.rotation_targets.values()))
            self.assertIn("not the lineup from any single historical game", " ".join(roster.assumptions))
        haley = self.bulls.rotation_targets["NBA_917"]
        self.assertLess(haley, 1.0)
        self.assertGreater(self.bulls.rotation_targets["NBA_893"], 30)

    def test_missing_ambiguous_and_inconsistent_rows_are_rejected(self):
        cases = {
            "four players": _five(("A", "B", "C", "D")),
            "duplicate source": _five() + [{"id": 1, "name": "A", "abbreviation": "EX", "source": "other", "stats": _line()}],
            "too many makes": _five()[:4] + [{"id": 5, "name": "E", "abbreviation": "EX", "stats": _line(fgm=30)}],
            "points do not match": _five()[:4] + [{"id": 5, "name": "E", "abbreviation": "EX", "stats": _line(points=99)}],
            "missing games": _five()[:4] + [{"id": 5, "name": "E", "abbreviation": "EX", "stats": _line(games=None)}],
        }
        for label, players in cases.items():
            with self.subTest(case=label):
                connection = _memory_roster(players)
                with self.assertRaises(HistoricalDataError):
                    load_historical_roster(connection, "1999-00", 999)
                connection.close()

    def test_a_player_with_no_field_goal_attempts_gets_neutral_zones(self):
        blank = _line(points=2, fgm=0, fga=0, fg3m=0, fg3a=0, ftm=2, fta=4)
        players = _five()
        players[4] = {"id": 5, "name": "E", "abbreviation": "EX", "stats": blank}
        connection = _memory_roster(players)
        roster = load_historical_roster(connection, "1999-00", 999)
        profile = next(item for item in roster.profiles if item.nba_player_id == 5)
        self.assertEqual(profile.calculated["fga_share"], 0.0)
        self.assertIsNone(profile.calculated["three_point_attempt_rate"])
        self.assertEqual(set(profile.player.tendencies.values()), {0.25})
        self.assertEqual(set(profile.player.shot_skills.values()), {75})
        self.assertEqual(profile.ratings["tendency_three"].classification, "uncalibrated")
        self.assertIn("not a shot chart", profile.ratings["tendency_three"].statement)
        connection.close()

    def test_usage_for_another_team_is_not_applied(self):
        players = _five()
        connection = _memory_roster(players)
        connection.execute(
            '''INSERT INTO research_external_observation(
                 source_name, url, retrieved_on, season_id, entity_kind, entity_id, entity_name,
                 field_name, field_label, displayed_value, normalized_fraction, measurement_definition,
                 methodology, import_status, status_reason, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            ("basketball-reference.com", "http://example.test", "2020-01-01", "1999-00", "player", 1, "A",
             "usg_pct", "USG%", 40.0, 0.40, "on-floor usage", "external", "verified", None,
             '{"labeled_cells": {"team_name_abbr": "TOR", "usg_pct": "40.0", "year_id": "1999-00"}}'),
        )
        connection.commit()
        roster = load_historical_roster(connection, "1999-00", 999)
        star = next(profile for profile in roster.profiles if profile.nba_player_id == 1)
        self.assertEqual(star.ratings["usage"].classification, "calculated_proxy")
        self.assertNotAlmostEqual(star.player.usage, 0.40)
        connection.close()

    def test_research_database_is_opened_read_only(self):
        connection = connect_research()
        with self.assertRaises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE historical_write_probe(x)")
        connection.close()


class HistoricalSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bulls, cls.warriors = load_researched_rosters()
        cls.eras = {era.key: era for era in ERAS}
        tmp = tempfile.TemporaryDirectory()
        cls.tmp = tmp
        connection = connect(str(Path(tmp.name) / "demo.db"))
        initialize_demo(connection)
        cls.demo_players, _, cls.demo_teams = load(connection)
        connection.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _play(self, roster, opponent_name, seed=7, **kwargs):
        merged = dict(self.demo_players)
        merged.update(roster.players)
        opponent = self.demo_teams[opponent_name]
        return simulate_game(
            roster.team, opponent, merged, self.eras[roster.era_key], seed,
            rotation_targets=(dict(roster.rotation_targets), None), **kwargs,
        )

    def test_historical_profiles_play_a_deterministic_game(self):
        samples = (
            (self.bulls, "Bay City Legends", [162, 164], [153, 152], 1),
            (self.warriors, "Chicago Stampede", [142, 144], [140, 139], 0),
        )
        for roster, opponent, score, possessions, overtime in samples:
            with self.subTest(team=roster.team_name):
                game = self._play(roster, opponent)
                self.assertEqual(game, self._play(roster, opponent))
                self.assertEqual(game["score"], score)
                self.assertEqual(game["possessions_by_team"], possessions)
                self.assertEqual(game["overtime"]["periods"], overtime)
                self.assertEqual(game["game_clock_seconds"], 2880.0 + 300.0 * overtime)
                self.assertEqual(game["rotation_mode"], ["scheduled", "full_game"])
                ids = set(roster.team.player_ids)
                self.assertAlmostEqual(
                    sum(game["player_stats"][pid]["min"] for pid in ids),
                    game["game_clock_seconds"] / 12.0,
                )
                self.assertEqual(
                    game["score"][0],
                    sum(game["player_stats"][pid].get("pts", 0) for pid in ids),
                )
                self.assertGreater(sum(game["player_stats"][pid].get("reb", 0) for pid in ids), 0)
                self.assertGreater(sum(game["player_stats"][pid].get("ast", 0) for pid in ids), 0)
                self.assertGreater(sum(game["player_stats"][pid].get("tov", 0) for pid in ids), 0)
                fatigues = [value for event in game["events"] for value in event["player_fatigue"].values()]
                self.assertGreater(max(fatigues), 0)
                self.assertLessEqual(max(fatigues), 0.42)
                for event in game["events"]:
                    offense = set(event["offense_lineup"])
                    self.assertTrue(offense <= ids or offense.isdisjoint(ids))
                    if event.get("offense") in ids:
                        self.assertIn(event["offense"], event["offense_lineup"])
                report = compare_game(roster, game, 0)
                self.assertIn("not proof of historical calibration", report["label"])
                self.assertAlmostEqual(sum(row["simulated_fga_share"] for row in report["players"]), 1.0)
                self.assertAlmostEqual(report["historical_points_per_game"], roster.audit["team_points_per_game"])

    def test_fictional_demo_free_throws_keep_the_midrange_path(self):
        player = next(iter(self.demo_players.values()))
        self.assertIsNone(player.free_throw_probability)
        mid_formula = 0.65 + (player.shot_skills["mid"] - 70) * 0.003
        self.assertAlmostEqual(free_throw_make_probability(player), mid_formula)
        rated = replace(player, free_throw_probability=0.90)
        self.assertAlmostEqual(free_throw_make_probability(rated), 0.90)
        self.assertAlmostEqual(free_throw_make_probability(replace(player, free_throw_probability=0.10)), 0.40)
        self.assertAlmostEqual(free_throw_make_probability(replace(player, free_throw_probability=0.99)), 0.95)
        era = self.eras["2015-16"]
        defender = next(item for item in self.demo_players.values() if item.player_id != player.player_id)
        self.assertEqual(
            probability_make(player, defender, era, "mid", 0.0, 0.0, 0.0),
            probability_make(rated, defender, era, "mid", 0.0, 0.0, 0.0),
        )
        expected = {
            ("1995-96", 0): ([122, 133], 268),
            ("1995-96", 7): ([141, 146], 269),
            ("1995-96", 19): ([157, 154], 271),
            ("2015-16", 0): ([120, 131], 266),
            ("2015-16", 7): ([139, 123], 265),
            ("2015-16", 19): ([163, 178], 272),
        }
        home = self.demo_teams["Bay City Legends"]
        away = self.demo_teams["Chicago Stampede"]
        for (key, seed), (score, possessions) in expected.items():
            with self.subTest(era=key, seed=seed):
                game = simulate_game(home, away, self.demo_players, self.eras[key], seed)
                self.assertEqual(game["score"], score)
                self.assertEqual(game["possession_count"], possessions)

    def test_shot_share_is_not_forced_to_the_historical_share(self):
        game = self._play(self.bulls, "Bay City Legends")
        report = compare_game(self.bulls, game, 0)
        jordan = next(row for row in report["players"] if row["nba_player_id"] == 893)
        haley = next(row for row in report["players"] if row["nba_player_id"] == 917)
        self.assertNotAlmostEqual(jordan["simulated_fga_share"], jordan["historical_fga_share"], places=3)
        self.assertGreater(jordan["simulated_fga_share"], haley["simulated_fga_share"])
        self.assertEqual(haley["simulated_minutes"], 0.0)
        self.assertEqual(haley["simulated_fga"], 0)
        self.assertGreater(jordan["simulated_minutes"], 30)
        curry_game = self._play(self.warriors, "Chicago Stampede")
        curry_report = compare_game(self.warriors, curry_game, 0)
        curry = next(row for row in curry_report["players"] if row["nba_player_id"] == 201939)
        looney = next(row for row in curry_report["players"] if row["player_name"] == "Kevon Looney")
        self.assertEqual(curry["usage_classification"], "verified")
        self.assertNotAlmostEqual(curry["simulated_fga_share"], curry["historical_fga_share"], places=3)
        self.assertGreater(curry["simulated_minutes"], looney["simulated_minutes"])
        self.assertEqual(looney["simulated_minutes"], 0.0)
        self.assertEqual(looney["simulated_fga"], 0)

    def test_overtime_uses_the_same_historical_roster(self):
        game = self._play(self.bulls, "Bay City Legends", equalize_periods=("regulation",))
        self.assertGreaterEqual(game["overtime"]["periods"], 1)
        self.assertTrue(game["completed"])
        ids = self.bulls.team.player_ids
        periods = game["overtime"]["periods"]
        self.assertAlmostEqual(
            sum(game["player_stats"][pid]["min"] for pid in ids),
            240.0 + 25.0 * periods,
        )
        self.assertEqual(game["game_clock_seconds"], 2880.0 + 300.0 * periods)
        for event in game["events"]:
            if event["period"].startswith("OT"):
                self.assertTrue(set(event["offense_lineup"]) <= set(ids) or set(event["offense_lineup"]).isdisjoint(ids))


if __name__ == "__main__":
    unittest.main()
