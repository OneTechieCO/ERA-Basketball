"""Build production Player objects from stored NBA season research.

Verified counting totals stay labeled as verified. Rates, shrunk shooting
targets, and the offensive-event share are calculated proxies. The integers
the simulator reads, other than a verified usage percentage, are uncalibrated
ratings. Nothing here is presented as NBA USG% unless a verified external
usage observation matches that player, season, and team.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .core import ZONES, Era, Player, Team, clamp, probability_make, shot_weights
from .research_report import KNOWN_TURNOVER_GAPS
from .seed import ERAS

PLAYER_ENDPOINT = "playercareerstats"
PLAYER_DATASET = "SeasonTotalsRegularSeason"
TEAM_ENDPOINT = "teamyearbyyearstats"
DASH_ENDPOINT = "leaguedashplayerstats"
DASH_DATASET = "LeagueDashPlayerStats"

COUNTING_FIELDS = (
    "games", "minutes", "points", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "assists", "offensive_rebounds", "defensive_rebounds", "steals", "blocks", "turnovers",
)
TEAM_MINUTE_BUDGET = 240.0
MINUTE_CAP = 48.0
# Possession-estimate weight used only in the labeled involvement proxy.
OFFENSIVE_EVENT_FREE_THROW_WEIGHT = 0.44
NEUTRAL_RATING = 75
RATING_FLOOR = 35
RATING_CAP = 99
# Attempts of prior weight. A player with this many attempts is halfway
# between his own rate and the reference rate. Zero attempts stay at the
# reference rate, so a missing three-point line is not treated as zero ability.
PRIOR_ATTEMPTS = 10.0
SKILL_STEP = 0.004
NEUTRAL_FREE_THROW = 0.75
REBOUND_ANCHOR = 6.0
REBOUND_SLOPE = 3.0
PASSING_ANCHOR = 4.0
PASSING_SLOPE = 5.0
STEAL_ANCHOR = 1.2
STEAL_SLOPE = 12.0
BLOCK_ANCHOR = 0.8
BLOCK_SLOPE = 12.0
OFFENSE_ANCHOR = 15.0
OFFENSE_SLOPE = 2.0
DEFENSE_RATING = 75
IQ_RATING = 78
INJURY_RATING = 75
# Minutes of prior weight. A one-game sample shrinks its attempt-rate gap
# toward zero instead of becoming a shooting role.
SHOT_RATE_PRIOR_MINUTES = 360.0

RESEARCHED_SEASONS = (
    {
        "season_id": "1995-96",
        "team_id": 1610612741,
        "team_name": "Chicago Bulls",
        "team_abbreviation": "CHI",
        "era_key": "1995-96",
    },
    {
        "season_id": "2015-16",
        "team_id": 1610612744,
        "team_name": "Golden State Warriors",
        "team_abbreviation": "GSW",
        "era_key": "2015-16",
    },
)

ASSUMPTIONS = (
    "The roster is the stored playercareerstats season for this team only. A combined total row and any other club are excluded.",
    "A player who changed teams is represented by this team's stint, not by his combined season.",
    "Regulation targets scale stored minutes, including overtime already inside those totals, to 240 minutes. This is not the lineup from any single historical game.",
    "The integer rotation may round a share below one minute down to zero.",
    "coach_style is balanced. No coaching record is stored, and the three-point attempt rate is already carried on tendencies.",
    "The seeded Era for this season key is the existing illustrative era. It is not a historical import.",
    "Defense, IQ, and injury resilience are constants. Those measurements are absent, and a constant IQ does not change who is selected relative to usage.",
    "Rim, paint, and mid share one rating taken from aggregate two-point shooting. Shot location is not stored.",
    "Verified usage percentage is applied only when a verified external observation matches this player, season, and team. Every other usage value is an offensive-event share, not NBA USG%.",
    "Touch selection uses shrunk offensive events per minute for every historical player, including those with verified usage. Verified USG% stays on the usage field. Fictional players leave event intensity unset and still use usage. The rate is not a shot quota and is not reused as the shooting propensity.",
    "Free-throw attempts use a shrunk season rate. Fictional players leave that rate unset and still use the midrange formula. The live field-goal rating range cannot always reach a historical three-point percentage.",
    "Shot propensity is field goals per 36 minutes above the rate implied by this player's offensive-event volume at the roster's field-goal mix. It replaces the usage term in the pass decision. A matching mix is zero, so event volume is not also a shooting bonus. Fictional players leave it unset. It is not a shot quota.",
    "Shooting ratings are inverted at openness 0, fatigue 0, adaptation 0, and defender defense 75 in the illustrative era. A fictional-opponent game is not that context.",
)


class HistoricalDataError(ValueError):
    """The stored research rows cannot build an unambiguous roster."""


@dataclass(frozen=True)
class RatingRecord:
    field: str
    value: float
    classification: str
    statement: str


@dataclass(frozen=True)
class PlayerProfile:
    player: Player
    nba_player_id: int
    player_name: str
    season_id: str
    team_id: int
    team_abbreviation: str
    counting: Mapping[str, float]
    calculated: Mapping[str, float | None]
    ratings: Mapping[str, RatingRecord]

    def rating(self, field: str) -> RatingRecord:
        try:
            return self.ratings[field]
        except KeyError as exc:
            raise KeyError(field) from exc


@dataclass(frozen=True)
class HistoricalRoster:
    season_id: str
    team_id: int
    team_name: str
    team_abbreviation: str
    era_key: str
    team: Team
    players: dict[str, Player]
    profiles: tuple[PlayerProfile, ...]
    rotation_targets: Mapping[str, float]
    assumptions: tuple[str, ...]
    audit: Mapping


def research_database_path() -> Path:
    return Path(__file__).resolve().parents[1] / "research.db"


def connect_research(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the stored research database without allowing writes."""
    database = Path(path) if path is not None else research_database_path()
    if not database.exists():
        raise HistoricalDataError(f"research database is missing: {database}")
    return sqlite3.connect(f"file:{database}?mode=ro", uri=True)


def load_researched_rosters(path: Path | str | None = None) -> tuple[HistoricalRoster, HistoricalRoster]:
    connection = connect_research(path)
    try:
        return tuple(load_historical_roster(connection, spec["season_id"], spec["team_id"]) for spec in RESEARCHED_SEASONS)
    finally:
        connection.close()


def load_historical_roster(connection: sqlite3.Connection, season_id: str, team_id: int) -> HistoricalRoster:
    """Convert one stored team season into Players, targets, and provenance."""
    connection.row_factory = sqlite3.Row
    team_row = _one_team_row(connection, season_id, team_id)
    expected = _season_spec(season_id, team_id)["team_abbreviation"] or None
    player_rows = _player_rows(connection, season_id, team_id, expected)
    if len(player_rows) < 5:
        raise HistoricalDataError(
            f"{team_row['team_name']} {season_id} has {len(player_rows)} stored players; a roster needs five"
        )
    abbreviations = {row["team_abbreviation"] for row in player_rows}
    if len(abbreviations) != 1:
        raise HistoricalDataError(f"{season_id} team {team_id} rows do not share one team abbreviation")
    _require_identities(player_rows)
    audit = _audit(connection, season_id, team_id, team_row, player_rows)
    abbreviation = player_rows[0]["team_abbreviation"]
    usage_by_player = _verified_usage(connection, season_id, abbreviation)
    event_shares = _offensive_event_shares(player_rows)
    intensities = _event_intensities(player_rows)
    fga_total = sum(_number(row["fga"], row, "fga") for row in player_rows)
    team_free_throw_rate = _quotient(
        sum(float(row["ftm"]) for row in player_rows),
        sum(float(row["fta"]) for row in player_rows),
    )
    era = _illustrative_era(season_id)
    team_events = sum(
        float(row["fga"]) + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * float(row["fta"]) + float(row["turnovers"])
        for row in player_rows
    )
    possession_rate = _fga_per_100_possessions(connection, season_id, team_id)
    profiles = tuple(
        _build_profile(
            row, team_row, event_shares[int(row["nba_player_id"])], fga_total,
            usage_by_player, era, team_free_throw_rate, possession_rate,
            intensities[int(row["nba_player_id"])], team_events,
        )
        for row in player_rows
    )
    targets = _scale_minutes([(profile.player.player_id, profile.counting["minutes"]) for profile in profiles])
    players = {profile.player.player_id: profile.player for profile in profiles}
    order = tuple(profile.player.player_id for profile in profiles)
    spec = _season_spec(season_id, team_id)
    return HistoricalRoster(
        season_id=season_id,
        team_id=team_id,
        team_name=str(team_row["team_name"]),
        team_abbreviation=str(abbreviation),
        era_key=spec["era_key"],
        team=Team(str(team_row["team_name"]), order, "balanced"),
        players=players,
        profiles=profiles,
        rotation_targets=MappingProxyType(targets),
        assumptions=ASSUMPTIONS,
        audit=MappingProxyType(audit),
    )


def compare_game(roster: HistoricalRoster, game: dict, team_index: int) -> dict:
    """Line up one simulated game with historical rates.

    The simulated side is a single game. It is not a season projection and,
    when the opponent is from another era or a fictional roster, it is not an
    observed historical result.
    """
    if team_index not in (0, 1):
        raise HistoricalDataError("team_index must be 0 or 1")
    if game["teams"][team_index] != roster.team.name:
        raise HistoricalDataError("the simulated team name does not match this roster")
    ids = roster.team.player_ids
    simulated_fga = sum(game["player_stats"][pid].get("fga", 0) for pid in ids)
    rows = []
    for profile in roster.profiles:
        stats = game["player_stats"][profile.player.player_id]
        fga = stats.get("fga", 0)
        fgm = stats.get("fgm", 0)
        fg3a = stats.get("fg3a", 0)
        rows.append({
            "player_id": profile.player.player_id,
            "nba_player_id": profile.nba_player_id,
            "player_name": profile.player_name,
            "usage_classification": profile.ratings["usage"].classification,
            "usage_value": profile.player.usage,
            "historical_fga_share": profile.calculated["fga_share"],
            "simulated_fga_share": None if simulated_fga == 0 else fga / simulated_fga,
            "historical_minutes_per_game": profile.calculated["minutes_per_game"],
            "target_minutes": roster.rotation_targets[profile.player.player_id],
            "simulated_minutes": stats["min"],
            "historical_points_per_game": profile.calculated["points_per_game"],
            "simulated_points": stats.get("pts", 0),
            "historical_fga_per_game": profile.calculated["fga_per_game"],
            "simulated_fga": fga,
            "historical_field_goal_percentage": profile.calculated["field_goal_percentage"],
            "simulated_field_goal_percentage": None if fga == 0 else fgm / fga,
            "historical_three_point_attempt_rate": profile.calculated["three_point_attempt_rate"],
            "simulated_three_point_attempt_rate": None if fga == 0 else fg3a / fga,
            "historical_rebounds_per_game": profile.calculated["rebounds_per_game"],
            "simulated_rebounds": stats.get("reb", 0),
            "historical_assists_per_game": profile.calculated["assists_per_game"],
            "simulated_assists": stats.get("ast", 0),
            "historical_turnovers_per_game": profile.calculated["turnovers_per_game"],
            "simulated_turnovers": stats.get("tov", 0),
        })
    return {
        "label": (
            "One simulated game beside historical per-game rates. "
            "Not a season projection and not proof of historical calibration."
        ),
        "season_id": roster.season_id,
        "team_name": roster.team_name,
        "opponent": game["teams"][1 - team_index],
        "era": game["era"],
        "seed": game["seed"],
        "team_points": game["score"][team_index],
        "historical_points_per_game": roster.audit["team_points_per_game"],
        "possessions": game["possessions_by_team"][team_index],
        "overtime_periods": game["overtime"]["periods"],
        "simulated_minutes": sum(row["simulated_minutes"] for row in rows),
        "players": rows,
    }


def _season_spec(season_id: str, team_id: int) -> dict:
    for spec in RESEARCHED_SEASONS:
        if spec["season_id"] == season_id and spec["team_id"] == team_id:
            return spec
    return {
        "season_id": season_id,
        "team_id": team_id,
        "team_name": "",
        "team_abbreviation": "",
        "era_key": season_id,
    }


def _rows(connection, sql: str, params: tuple) -> list[sqlite3.Row]:
    return list(connection.execute(sql, params))


def _one_team_row(connection, season_id: str, team_id: int) -> sqlite3.Row:
    rows = _rows(
        connection,
        '''SELECT * FROM nba_team_season_research
           WHERE season_id=? AND team_id=? AND endpoint=?''',
        (season_id, team_id, TEAM_ENDPOINT),
    )
    if len(rows) != 1:
        raise HistoricalDataError(
            f"{season_id} team {team_id} has {len(rows)} {TEAM_ENDPOINT} rows; expected one"
        )
    if not rows[0]["team_name"]:
        raise HistoricalDataError(f"{season_id} team {team_id} has no team name")
    return rows[0]


def _player_rows(connection, season_id: str, team_id: int, abbreviation: str | None) -> list[sqlite3.Row]:
    rows = _rows(
        connection,
        '''SELECT * FROM nba_player_season_research
           WHERE season_id=? AND team_id=? AND endpoint=? AND dataset_name=?
             AND UPPER(COALESCE(team_abbreviation, '')) != 'TOT'
           ORDER BY minutes DESC, nba_player_id ASC''',
        (season_id, team_id, PLAYER_ENDPOINT, PLAYER_DATASET),
    )
    seen = set()
    for row in rows:
        player_id = int(row["nba_player_id"])
        if player_id in seen:
            raise HistoricalDataError(f"duplicate {PLAYER_ENDPOINT} row for {row['player_name']} in {season_id}")
        seen.add(player_id)
        if abbreviation and row["team_abbreviation"] not in (None, abbreviation):
            raise HistoricalDataError(
                f"{row['player_name']} is stored as {row['team_abbreviation']} on team {team_id}"
            )
        for field in COUNTING_FIELDS:
            _number(row[field], row, field)
        if row["games"] <= 0 or row["minutes"] <= 0:
            raise HistoricalDataError(f"{row['player_name']} needs positive games and minutes")
        if row["fgm"] > row["fga"] or row["fg3m"] > row["fg3a"] or row["fg3a"] > row["fga"]:
            raise HistoricalDataError(f"{row['player_name']} has more makes than attempts")
        if row["fg3m"] > row["fgm"] or row["ftm"] > row["fta"]:
            raise HistoricalDataError(f"{row['player_name']} has more makes than attempts")
        if row["fga"] - row["fg3a"] < 0 or row["fgm"] - row["fg3m"] < 0:
            raise HistoricalDataError(f"{row['player_name']} has a negative two-point line")
    return rows


def _require_identities(rows: list[sqlite3.Row]) -> None:
    for row in rows:
        expected = 2 * (row["fgm"] - row["fg3m"]) + 3 * row["fg3m"] + row["ftm"]
        if abs(expected - row["points"]) > 1e-6:
            raise HistoricalDataError(
                f"{row['player_name']} points {row['points']} do not match the stored shooting line {expected}"
            )


def _number(value, row, field: str) -> float:
    if isinstance(value, bool) or value is None:
        name = row["player_name"] if "player_name" in row.keys() else row["team_name"]
        raise HistoricalDataError(f"{name} is missing {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalDataError(f"stored {field} is not a number") from exc
    if number < 0:
        raise HistoricalDataError(f"stored {field} is negative")
    return number


def _audit(connection, season_id: str, team_id: int, team_row, player_rows) -> dict:
    compared = {}
    for field in COUNTING_FIELDS:
        if field in ("games", "minutes"):
            continue
        team_value = team_row[field]
        if team_value is None:
            compared[field] = None
            continue
        player_total = sum(_number(row[field], row, field) for row in player_rows)
        compared[field] = player_total - float(team_value)
    for field, gap in compared.items():
        if field == "turnovers" or gap is None:
            continue
        if abs(gap) > 1e-6:
            raise HistoricalDataError(f"player {field} differs from the team total by {gap}")
    player_turnovers = sum(row["turnovers"] for row in player_rows)
    team_turnovers = None if team_row["turnovers"] is None else float(team_row["turnovers"])
    gap = None if team_turnovers is None else team_turnovers - player_turnovers
    known = KNOWN_TURNOVER_GAPS.get((team_id, season_id))
    if known is not None and gap != known:
        raise HistoricalDataError(f"turnover gap {gap} does not match the carried-forward gap {known}")
    splits = _split_audit(connection, season_id, player_rows)
    dash = _dash_audit(connection, season_id, team_id, player_rows)
    games = _number(team_row["games"], team_row, "games")
    return {
        "player_count": len(player_rows),
        "counting_difference_from_team": compared,
        "turnover_gap": gap,
        "known_turnover_gap": known,
        "turnover_gap_reconciled": False,
        "split_checks": splits,
        "dash_crosscheck": dash,
        "team_points_per_game": None if games == 0 or team_row["points"] is None else float(team_row["points"]) / games,
        "team_fga": None if team_row["fga"] is None else float(team_row["fga"]),
        "source_endpoint": PLAYER_ENDPOINT,
        "source_dataset": PLAYER_DATASET,
    }


def _split_audit(connection, season_id: str, player_rows) -> list[dict]:
    checks = []
    for row in player_rows:
        parts = _rows(
            connection,
            '''SELECT * FROM nba_player_season_research
               WHERE nba_player_id=? AND season_id=? AND endpoint=? AND dataset_name=?''',
            (row["nba_player_id"], season_id, PLAYER_ENDPOINT, PLAYER_DATASET),
        )
        total = [part for part in parts if str(part["team_abbreviation"]).upper() == "TOT"]
        clubs = [part for part in parts if str(part["team_abbreviation"]).upper() != "TOT"]
        if not total:
            continue
        if len(total) != 1:
            raise HistoricalDataError(f"{row['player_name']} has more than one combined season row")
        for field in COUNTING_FIELDS:
            summed = sum(_number(part[field], part, field) for part in clubs)
            expected = _number(total[0][field], total[0], field)
            if abs(summed - expected) > 1e-6:
                raise HistoricalDataError(
                    f"{row['player_name']} {field} splits sum to {summed}, not the combined {expected}"
                )
        checks.append({
            "nba_player_id": int(row["nba_player_id"]),
            "player_name": row["player_name"],
            "clubs": [part["team_abbreviation"] for part in clubs],
            "uses_this_team_stint_only": True,
        })
    return checks


def _dash_audit(connection, season_id: str, team_id: int, player_rows) -> dict:
    dash_rows = _rows(
        connection,
        '''SELECT * FROM nba_player_season_research
           WHERE season_id=? AND team_id=? AND endpoint=? AND dataset_name=?''',
        (season_id, team_id, DASH_ENDPOINT, DASH_DATASET),
    )
    if not dash_rows:
        return {"compared": False, "reason": "no league dash rows are stored for this team"}
    career = {int(row["nba_player_id"]): row for row in player_rows}
    dash = {int(row["nba_player_id"]): row for row in dash_rows}
    extra = sorted(set(dash) - set(career))
    if extra:
        raise HistoricalDataError(f"dash rows name players who are absent from {PLAYER_ENDPOINT}: {extra}")
    absent = [career[player_id]["player_name"] for player_id in career if player_id not in dash]
    for player_id, dash_row in dash.items():
        career_row = career[player_id]
        for field in COUNTING_FIELDS:
            if field == "minutes":
                if abs(float(dash_row[field]) - float(career_row[field])) >= 1:
                    raise HistoricalDataError(f"{career_row['player_name']} dash minutes differ by a minute or more")
                continue
            if dash_row[field] is None:
                continue
            if abs(float(dash_row[field]) - float(career_row[field])) > 1e-6:
                raise HistoricalDataError(f"{career_row['player_name']} dash {field} does not match the career row")
    return {"compared": True, "matched_players": len(dash), "absent_from_dash": absent}


def _verified_usage(connection, season_id: str, abbreviation: str | None) -> dict[int, sqlite3.Row]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_external_observation'"
    ).fetchone()
    if exists is None:
        return {}
    rows = _rows(
        connection,
        '''SELECT * FROM research_external_observation
           WHERE season_id=? AND entity_kind='player' AND field_name='usg_pct'
             AND import_status='verified' ''',
        (season_id,),
    )
    accepted = {}
    for row in rows:
        raw = json.loads(row["raw_json"])
        cells = raw.get("labeled_cells") if isinstance(raw.get("labeled_cells"), dict) else {}
        if abbreviation and cells.get("team_name_abbr") != abbreviation:
            continue
        if row["normalized_fraction"] is None:
            continue
        accepted[int(row["entity_id"])] = row
    return accepted


def _offensive_event_shares(rows: list[sqlite3.Row]) -> dict[int, float]:
    events = {
        int(row["nba_player_id"]): row["fga"] + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * row["fta"] + row["turnovers"]
        for row in rows
    }
    total = sum(events.values())
    if total <= 0:
        raise HistoricalDataError("the roster has no field-goal attempts, free throws, or turnovers")
    return {player_id: value / total for player_id, value in events.items()}


def shrunk_event_intensity(events: float, minutes: float, team_events: float, team_minutes: float) -> dict[str, float]:
    """Offensive events per minute, shrunk toward this roster's own rate.

    Events are FGA + 0.44*FTA + turnovers from the player's team stint.
    The result is a touch-selection input. It is not NBA USG%, not a counted
    touch frequency, and not a field-goal share. A short minute sample is
    pulled toward the roster rate so one game cannot set the weight.
    """
    if minutes <= 0 or team_minutes <= 0 or team_events <= 0:
        raise HistoricalDataError("event intensity needs positive minutes and team events")
    values = (events, minutes, team_events, team_minutes)
    if min(values) < 0 or not all(math.isfinite(value) for value in values):
        raise HistoricalDataError("event intensity inputs must be finite and non-negative")
    raw = events / minutes
    reference = team_events / team_minutes
    weight = minutes / (minutes + SHOT_RATE_PRIOR_MINUTES)
    return {
        "events_per_minute": raw,
        "event_intensity_reference": reference,
        "event_intensity": weight * raw + (1.0 - weight) * reference,
    }


def _event_intensities(rows: list[sqlite3.Row]) -> dict[int, dict[str, float]]:
    counted = {}
    for row in rows:
        player_id = int(row["nba_player_id"])
        events = float(row["fga"]) + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * float(row["fta"]) + float(row["turnovers"])
        counted[player_id] = (events, float(row["minutes"]))
    team_events = sum(events for events, _ in counted.values())
    team_minutes = sum(minutes for _, minutes in counted.values())
    return {
        player_id: shrunk_event_intensity(events, minutes, team_events, team_minutes)
        for player_id, (events, minutes) in counted.items()
    }


def _quotient(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def attempt_rate_gap(fga: float, minutes: float, events: float, team_fga: float, team_events: float) -> dict[str, float]:
    """Field goals per 36 above the rate implied by offensive-event volume.

    Events are FGA + 0.44*FTA + turnovers. Touch selection already uses that
    volume per minute. The implied rate gives those events the roster's
    field-goal mix. The returned gap is observed attempts per 36 minus that
    rate, shrunk toward zero for short minutes. A player whose mix matches
    the roster gets zero, so the same attempts are not also a shooting bonus.
    The gap is not a shot quota, not NBA USG%, and not a field-goal share.
    """
    if minutes <= 0 or team_events <= 0:
        raise HistoricalDataError("attempt rate needs positive minutes and team events")
    values = (fga, minutes, events, team_fga, team_events)
    if min(values) < 0 or not all(math.isfinite(value) for value in values):
        raise HistoricalDataError("attempt rate inputs must be finite and non-negative")
    if events + 1e-9 < fga:
        raise HistoricalDataError("field-goal attempts cannot exceed offensive events")
    observed = fga / minutes * 36.0
    team_fraction = team_fga / team_events
    implied = 0.0 if events == 0 else (events / minutes) * team_fraction * 36.0
    weight = minutes / (minutes + SHOT_RATE_PRIOR_MINUTES)
    return {
        "fga_per_minute": fga / minutes,
        "fga_per_36": observed,
        "shot_event_fraction": 0.0 if events == 0 else fga / events,
        "team_shot_event_fraction": team_fraction,
        "involvement_implied_fga_per_36": implied,
        "shot_propensity": weight * (observed - implied),
    }


def _fga_per_100_possessions(connection, season_id: str, team_id: int) -> dict[str, float | str | None]:
    """Use possessions only when the same team row also has field-goal attempts."""
    rows = _rows(
        connection,
        '''SELECT fga, possessions, possessions_measure
           FROM nba_team_season_research
           WHERE season_id=? AND team_id=?''',
        (season_id, team_id),
    )
    for row in rows:
        possessions = row["possessions"]
        fga = row["fga"]
        if row["possessions_measure"] == "season_total" and possessions not in (None, 0) and fga not in (None, 0):
            return {
                "fga_per_100_team_possessions": float(fga) / float(possessions) * 100.0,
                "fga_per_100_team_possessions_reason": None,
            }
    return {
        "fga_per_100_team_possessions": None,
        "fga_per_100_team_possessions_reason": (
            "No stored team row has both a season-total possessions value and field-goal attempts. "
            "Those fields are not combined across endpoints, and player rows have no possessions denominator."
        ),
    }


REFERENCE_MIX = 0.25
TENDENCY_FLOOR = 0.001


def _illustrative_era(season_id: str) -> Era:
    """Era used only to invert ratings. Games still use the roster's era key."""
    for era in ERAS:
        if era.key == season_id:
            return era
    return Era(
        season_id, 100.0, 0.25, 0.0, True, 1.0,
        rule_note="Neutral reference. No illustrative era is stored for this season.",
    )


def _context_player(skill: int, tendencies: dict[str, float] | None = None) -> Player:
    skills = {zone: int(skill) for zone in ZONES}
    chosen = tendencies or {zone: 0.25 for zone in ZONES}
    return Player(
        "reference", "reference", "reference",
        NEUTRAL_RATING, DEFENSE_RATING, NEUTRAL_RATING, NEUTRAL_RATING,
        IQ_RATING, NEUTRAL_RATING, 0.2, skills, chosen,
    )


def _reference_make(era: Era, zone: str, skill: int) -> float:
    """Make probability at openness 0, fatigue 0, adaptation 0, defense 75."""
    return probability_make(
        _context_player(skill), _context_player(DEFENSE_RATING),
        era, zone, 0.0, 0.0, 0.0,
    )


def _shrink(makes: float, attempts: float, prior: float) -> float:
    if attempts <= 0:
        return prior
    return (attempts * (makes / attempts) + PRIOR_ATTEMPTS * prior) / (attempts + PRIOR_ATTEMPTS)


def _resting_weights(tendencies: dict[str, float], era: Era) -> dict[str, float]:
    raw = shot_weights(_context_player(NEUTRAL_RATING, tendencies), era, "balanced", 0.0)
    total = sum(raw.values())
    return {zone: raw[zone] / total for zone in ZONES}


def _attempt_tendencies(target: float, era: Era) -> tuple[dict[str, float], float, bool]:
    """Choose tendencies so the resting era mix lands on the shrunk attempt rate.

    shot_weights keeps a 25 percent mix with the era three-point share when
    adaptation is 0. The tendency is the input that produces the target after
    that mix. The era floor and ceiling are reported when the target is outside
    them. This does not edit shot_weights.
    """
    ceiling = 1.0 - 3.0 * TENDENCY_FLOOR
    raw = (target - REFERENCE_MIX * era.three_share) / (1.0 - REFERENCE_MIX)
    bounded = raw < TENDENCY_FLOOR or raw > ceiling
    three = float(clamp(raw, TENDENCY_FLOOR, ceiling))
    interior = (1.0 - three) / 3.0
    tendencies = {"rim": interior, "paint": interior, "mid": interior, "three": three}
    achieved = _resting_weights(tendencies, era)["three"]
    return tendencies, achieved, bounded


def _skill_for_rate(era: Era, weights: dict[str, float], target: float) -> tuple[int, float, bool]:
    """Invert the live 0.004 skill step. The 35-99 range is not raised."""
    neutral = sum(weight * _reference_make(era, zone, NEUTRAL_RATING) for zone, weight in weights.items())
    raw = NEUTRAL_RATING + (target - neutral) / SKILL_STEP
    skill = int(round(clamp(raw, RATING_FLOOR, RATING_CAP)))
    achieved = sum(weight * _reference_make(era, zone, skill) for zone, weight in weights.items())
    # A miss no larger than one rating step is the integer resolution, not a cap.
    bounded = (raw < RATING_FLOOR or raw > RATING_CAP) and abs(achieved - target) > SKILL_STEP + 1e-12
    return skill, achieved, bounded


def _bound_sentence(achieved: float, target: float) -> str:
    return (
        f" The live rating range {RATING_FLOOR}-{RATING_CAP} cannot reach the shrunk percentage "
        f"in the reference context. Reference make rate {achieved:.3f}; shrunk target {target:.3f}."
    )


def _shooting_ratings(counting, two_attempts, two_makes, era: Era, team_free_throw: float) -> dict:
    """Map verified shooting totals onto the live formula without changing it."""
    ratings: dict[str, RatingRecord] = {}
    extra: dict[str, float | None] = {}
    if counting["fga"] <= 0:
        statement = "No field-goal attempts. Equal zone tendencies and neutral skills are uncalibrated, not a shot chart."
        for zone in ("rim", "paint", "mid"):
            ratings[f"shot_{zone}"] = RatingRecord(f"shot_{zone}", NEUTRAL_RATING, "uncalibrated", statement)
            ratings[f"tendency_{zone}"] = RatingRecord(f"tendency_{zone}", 0.25, "uncalibrated", statement)
        ratings["shot_three"] = RatingRecord(
            "shot_three", NEUTRAL_RATING, "uncalibrated",
            "No three-point attempts. Neutral rating, not zero ability and not a shot chart.",
        )
        ratings["tendency_three"] = RatingRecord("tendency_three", 0.25, "uncalibrated", statement)
        extra.update({
            "shrunk_three_point_attempt_rate": None,
            "achieved_three_point_attempt_rate": None,
            "shrunk_two_point_percentage": None,
            "reference_two_point_make": None,
            "shrunk_three_point_percentage": None,
            "reference_three_point_make": _reference_make(era, "three", NEUTRAL_RATING),
        })
    else:
        shrunk_rate = _shrink(counting["fg3a"], counting["fga"], era.three_share)
        tendencies, achieved_rate, rate_bounded = _attempt_tendencies(shrunk_rate, era)
        if rate_bounded:
            three_statement = (
                "The live era mix cannot reach the shrunk three-point attempt rate. "
                f"Resting rate {achieved_rate:.3f}; shrunk target {shrunk_rate:.3f}. Not a shot chart."
            )
        else:
            three_statement = (
                "Three-point tendency pre-compensates the live era mix so the resting shot mix "
                "matches the shrunk attempt rate. Not a shot chart."
            )
        interior_statement = "Equal split of the non-three tendency. Shot location is not stored. Not a shot chart."
        for zone in ("rim", "paint", "mid"):
            ratings[f"tendency_{zone}"] = RatingRecord(
                f"tendency_{zone}", tendencies[zone], "uncalibrated", interior_statement,
            )
        ratings["tendency_three"] = RatingRecord(
            "tendency_three", tendencies["three"], "calculated_proxy", three_statement,
        )
        weights = {zone: value for zone, value in _resting_weights(tendencies, era).items() if zone != "three"}
        interior_total = sum(weights.values())
        weights = {zone: value / interior_total for zone, value in weights.items()}
        if two_attempts <= 0:
            two_skill = NEUTRAL_RATING
            two_target = None
            two_achieved = sum(weight * _reference_make(era, zone, two_skill) for zone, weight in weights.items())
            two_statement = (
                "No two-point attempts. Neutral rating copied to rim, paint, and mid. "
                "Not a shot chart or a historical zone percentage."
            )
        else:
            neutral_two = sum(weight * _reference_make(era, zone, NEUTRAL_RATING) for zone, weight in weights.items())
            two_target = _shrink(two_makes, two_attempts, neutral_two)
            two_skill, two_achieved, two_bounded = _skill_for_rate(era, weights, two_target)
            two_statement = (
                "Shared two-point rating inverted so the reference-context mix of rim, paint, and mid "
                "matches the shrunk aggregate two-point rate. The same skill is copied to all three zones. "
                "Not a shot chart or separate zone percentages."
            )
            if two_bounded:
                two_statement += _bound_sentence(two_achieved, two_target)
        for zone in ("rim", "paint", "mid"):
            ratings[f"shot_{zone}"] = RatingRecord(f"shot_{zone}", two_skill, "uncalibrated", two_statement)
        if counting["fg3a"] <= 0:
            three_skill = NEUTRAL_RATING
            three_target = None
            three_achieved = _reference_make(era, "three", three_skill)
            three_statement = (
                "No three-point attempts. Neutral rating from the reference make rate, not zero ability "
                "and not a measured three-point percentage."
            )
        else:
            neutral_three = _reference_make(era, "three", NEUTRAL_RATING)
            three_target = _shrink(counting["fg3m"], counting["fg3a"], neutral_three)
            three_skill, three_achieved, three_bounded = _skill_for_rate(era, {"three": 1.0}, three_target)
            three_statement = (
                "Three-point rating inverted from the shrunk three-point rate in the reference context. "
                "The live make formula is unchanged, and one game is not forced to the season rate."
            )
            if three_bounded:
                three_statement += _bound_sentence(three_achieved, three_target)
        ratings["shot_three"] = RatingRecord("shot_three", three_skill, "uncalibrated", three_statement)
        extra.update({
            "shrunk_three_point_attempt_rate": shrunk_rate,
            "achieved_three_point_attempt_rate": achieved_rate,
            "shrunk_two_point_percentage": two_target,
            "reference_two_point_make": two_achieved,
            "shrunk_three_point_percentage": three_target,
            "reference_three_point_make": three_achieved,
        })
    if counting["fta"] <= 0:
        free_throw = team_free_throw
        ft_statement = "No free-throw attempts. The team free-throw rate is the prior, not this player's measured ability."
    else:
        free_throw = _shrink(counting["ftm"], counting["fta"], team_free_throw)
        ft_statement = (
            "Shrunk season free-throw rate. The live draw uses this probability directly. "
            "It is not the midrange rating, and one game is not forced to the season rate."
        )
    stored = float(clamp(free_throw, 0.40, 0.95))
    if abs(stored - free_throw) > 1e-12:
        ft_statement += f" The live 0.40-0.95 bounds change the rate from {free_throw:.3f} to {stored:.3f}."
    ratings["free_throw"] = RatingRecord("free_throw", stored, "calculated_proxy", ft_statement)
    extra["shrunk_free_throw_percentage"] = stored
    return {"ratings": ratings, "calculated": extra, "free_throw_probability": stored}


def _per36_rating(total: float, minutes: float, anchor: float, slope: float, statement: str) -> RatingRecord:
    per36 = total / minutes * 36
    value = int(round(clamp(NEUTRAL_RATING + (per36 - anchor) * slope, RATING_FLOOR, RATING_CAP)))
    return RatingRecord("pending", value, "uncalibrated", statement)


def _build_profile(row, team_row, event_share: float, fga_total: float, usage_rows, era: Era, team_free_throw_rate: float | None, possession_rate: dict, intensity: dict[str, float], team_events: float) -> PlayerProfile:
    counting = {field: float(row[field]) for field in COUNTING_FIELDS}
    two_attempts = counting["fga"] - counting["fg3a"]
    two_makes = counting["fgm"] - counting["fg3m"]
    calculated = {
        "minutes_per_game": _quotient(counting["minutes"], counting["games"]),
        "points_per_game": _quotient(counting["points"], counting["games"]),
        "fga_per_game": _quotient(counting["fga"], counting["games"]),
        "rebounds_per_game": _quotient(counting["offensive_rebounds"] + counting["defensive_rebounds"], counting["games"]),
        "assists_per_game": _quotient(counting["assists"], counting["games"]),
        "turnovers_per_game": _quotient(counting["turnovers"], counting["games"]),
        "field_goal_percentage": _quotient(counting["fgm"], counting["fga"]),
        "three_point_percentage": _quotient(counting["fg3m"], counting["fg3a"]),
        "free_throw_percentage": _quotient(counting["ftm"], counting["fta"]),
        "two_point_percentage": _quotient(two_makes, two_attempts),
        "three_point_attempt_rate": _quotient(counting["fg3a"], counting["fga"]),
        "fga_share": _quotient(counting["fga"], fga_total),
        "offensive_event_share": event_share,
        "fga_per_100_team_possessions": possession_rate["fga_per_100_team_possessions"],
        "fga_per_100_team_possessions_reason": possession_rate["fga_per_100_team_possessions_reason"],
    }
    verified = usage_rows.get(int(row["nba_player_id"]))
    calculated["verified_usage"] = None if verified is None else float(verified["normalized_fraction"])
    player_events = counting["fga"] + OFFENSIVE_EVENT_FREE_THROW_WEIGHT * counting["fta"] + counting["turnovers"]
    calculated.update(attempt_rate_gap(counting["fga"], counting["minutes"], player_events, fga_total, team_events))
    calculated.update(intensity)
    ratings: dict[str, RatingRecord] = {}
    if verified is not None:
        ratings["usage"] = RatingRecord(
            "usage",
            float(verified["normalized_fraction"]),
            "verified",
            "Verified external on-floor usage percentage for this player, season, and team. Not a field-goal share.",
        )
    else:
        ratings["usage"] = RatingRecord(
            "usage",
            event_share,
            "calculated_proxy",
            (
                "Share of this roster's stored FGA + 0.44*FTA + turnovers. "
                "Not NBA USG%. The 0.44 weight is a proxy coefficient, and the known team turnover gap is not assigned to a player."
            ),
        )
    ratings["event_intensity"] = RatingRecord(
        "event_intensity",
        float(calculated["event_intensity"]),
        "calculated_proxy",
        (
            "Shrunk (FGA + 0.44*FTA + turnovers) per minute from this team stint, pulled toward the roster rate. "
            "Ballhandler and receiver selection use it. It is not NBA USG%, not a touch count, and not a field-goal share. "
            "Shooting propensity is a separate input."
        ),
    )
    ratings["shot_propensity"] = RatingRecord(
        "shot_propensity",
        float(calculated["shot_propensity"]),
        "calculated_proxy",
        (
            "Field goals per 36 minutes above the rate implied by this player's offensive-event volume at the roster's field-goal mix. "
            "A matching mix is zero. It replaces the usage term in the pass decision and does not change touch selection. "
            "It is not a shot quota and not NBA USG%."
        ),
    )
    team_free_throw = NEUTRAL_FREE_THROW if team_free_throw_rate is None else team_free_throw_rate
    shooting = _shooting_ratings(counting, two_attempts, two_makes, era, team_free_throw)
    ratings.update(shooting["ratings"])
    calculated.update(shooting["calculated"])
    ratings["rebounding"] = _per36_rating(
        counting["offensive_rebounds"] + counting["defensive_rebounds"], counting["minutes"],
        REBOUND_ANCHOR, REBOUND_SLOPE,
        "Uncalibrated per-36 rebound rating. Not a rebound percentage.",
    )
    ratings["passing"] = _per36_rating(
        counting["assists"], counting["minutes"], PASSING_ANCHOR, PASSING_SLOPE,
        "Uncalibrated per-36 assist rating. Not a measured passing skill.",
    )
    ratings["steals"] = _per36_rating(
        counting["steals"], counting["minutes"], STEAL_ANCHOR, STEAL_SLOPE,
        "Uncalibrated per-36 steal rating.",
    )
    ratings["blocks"] = _per36_rating(
        counting["blocks"], counting["minutes"], BLOCK_ANCHOR, BLOCK_SLOPE,
        "Uncalibrated per-36 block rating.",
    )
    ratings["offense"] = _per36_rating(
        counting["points"], counting["minutes"], OFFENSE_ANCHOR, OFFENSE_SLOPE,
        "Uncalibrated per-36 points rating. The live possession loop does not read this field.",
    )
    ratings["defense"] = RatingRecord(
        "defense", DEFENSE_RATING, "uncalibrated",
        "Constant. No opponent or defensive-possession data is stored.",
    )
    ratings["iq"] = RatingRecord(
        "iq", IQ_RATING, "uncalibrated",
        "Constant. IQ is not stored, and keeping it constant stops it from changing relative selection.",
    )
    ratings["injury_resilience"] = RatingRecord(
        "injury_resilience", INJURY_RATING, "uncalibrated",
        "Constant. No injury record is stored, and the live game does not simulate injuries.",
    )
    stamina_value = int(round(clamp(counting["minutes"] / counting["games"] / MINUTE_CAP * 100, 25, 99)))
    ratings["stamina"] = RatingRecord(
        "stamina", stamina_value, "uncalibrated",
        "Uncalibrated stint hint from minutes per game. Minute targets, not this rating, set the playing time.",
    )
    for field, record in list(ratings.items()):
        if record.field == "pending":
            ratings[field] = RatingRecord(field, record.value, record.classification, record.statement)
    shot_skills = {zone: int(ratings[f"shot_{zone}"].value) for zone in ZONES}
    tendencies = {zone: float(ratings[f"tendency_{zone}"].value) for zone in ZONES}
    player = Player(
        f"NBA_{int(row['nba_player_id'])}",
        str(row["player_name"]),
        str(row["season_id"]),
        int(ratings["offense"].value),
        int(ratings["defense"].value),
        int(ratings["passing"].value),
        int(ratings["rebounding"].value),
        int(ratings["iq"].value),
        int(ratings["stamina"].value),
        float(ratings["usage"].value),
        shot_skills,
        tendencies,
        steals=int(ratings["steals"].value),
        blocks=int(ratings["blocks"].value),
        injury_resilience=int(ratings["injury_resilience"].value),
        free_throw_probability=shooting["free_throw_probability"],
        shot_propensity=float(calculated["shot_propensity"]),
        event_intensity=float(calculated["event_intensity"]),
    )
    return PlayerProfile(
        player=player,
        nba_player_id=int(row["nba_player_id"]),
        player_name=str(row["player_name"]),
        season_id=str(row["season_id"]),
        team_id=int(team_row["team_id"]),
        team_abbreviation=str(row["team_abbreviation"]),
        counting=MappingProxyType(counting),
        calculated=MappingProxyType(calculated),
        ratings=MappingProxyType(ratings),
    )


def _scale_minutes(pairs: list[tuple[str, float]]) -> dict[str, float]:
    total = sum(minutes for _, minutes in pairs)
    if total <= 0:
        raise HistoricalDataError("roster minutes must be positive")
    targets = {player_id: TEAM_MINUTE_BUDGET * minutes / total for player_id, minutes in pairs}
    for _ in range(len(targets) + 1):
        over = {player_id: value - MINUTE_CAP for player_id, value in targets.items() if value > MINUTE_CAP + 1e-9}
        if not over:
            break
        room = [player_id for player_id, value in targets.items() if player_id not in over and value < MINUTE_CAP]
        if not room:
            raise HistoricalDataError("regulation minutes cannot fit without a player above 48")
        excess = sum(over.values())
        room_total = sum(targets[player_id] for player_id in room)
        for player_id in over:
            targets[player_id] = MINUTE_CAP
        for player_id in room:
            share = targets[player_id] / room_total if room_total else 1 / len(room)
            targets[player_id] += excess * share
    else:
        raise HistoricalDataError("regulation minutes cannot fit without a player above 48")
    drift = TEAM_MINUTE_BUDGET - sum(targets.values())
    adjust = max(targets, key=lambda player_id: (MINUTE_CAP - targets[player_id], targets[player_id], player_id))
    targets[adjust] += drift
    if targets[adjust] < -1e-6 or targets[adjust] > MINUTE_CAP + 1e-6:
        raise HistoricalDataError("regulation minute targets do not sum to 240 inside the 48-minute cap")
    if targets[adjust] < 0:
        targets[adjust] = 0.0
    return targets
