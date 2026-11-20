"""App-facing batch interface for the frozen V1 engine.

Call ``simulate_batch`` with a JSON object. The function plays each game
through ``simulate_game`` and returns a compact summary. It does not call
``simulate_series``, and that series entry point stays limited to 82 games.

This module does not change shooting, passing, rebounds, fouls, touch
selection, shot propensity, the clock, rotations, fatigue, research rows,
or timing distributions. Variant B3 is not used.

CPython is not assumed to run inside an iPhone app. A later connection can
use one of these hosts, and this step does not choose among them:

- a machine or service that runs this module and returns the JSON summary
- a local runner with its own Python runtime, outside the iOS process
- a shipped file of summaries computed ahead of time

No network server and no Swift UI are implemented here.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .core import (
    PROVISIONAL_1995_96_DEVELOPMENT_MEAN,
    PROVISIONAL_1995_96_PACE_REFERENCE,
    Team,
    TimingAssumptions,
    UnresolvedGameError,
    simulate_game,
)
from .historical_players import RESEARCHED_SEASONS, HistoricalDataError, load_researched_rosters
from .playbyplay_timing import shot_trip_duration_sample
from .seed import ERAS, demo_players
from .validation import SERIES_GAME_LIMIT, Sample, check_game

SCHEMA = "era-basketball-batch-v1"
STATISTIC_FIELDS = {
    "pts": "points",
    "fgm": "field goals made",
    "fga": "field goals attempted",
    "fg3m": "three-pointers made",
    "fg3a": "three-pointers attempted",
    "ftm": "free throws made",
    "fta": "free throws attempted",
    "reb": "rebounds",
    "oreb": "offensive rebounds",
    "dreb": "defensive rebounds",
    "ast": "assists",
    "stl": "steals",
    "blk": "blocks",
    "tov": "turnovers",
    "min": "minutes",
    "fga_share": "share of the team's field-goal attempts",
}
MODEL_VERSION = "0.1.0-prototype"
ALLOWED_GAMES = (1, 10, 100, 1000)
TIMING_ORIGINAL = "original"
TIMING_FIRST_TRIP = "observed_2015_16_first_trip"
TIMING_PROVISIONAL_1995_96 = "provisional_1995_96_aggregate_pace"
REQUEST_FIELDS = (
    "team_a", "season_a", "team_b", "season_b", "era", "games", "seed", "timing_mode", "include_events",
)
COUNTING = (
    "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "min",
)
PACE_1995_96 = (
    "Original 1995-96 timing is assumed, not observed. "
    "Reliability batches using this timing finish near 136 possessions per 48 minutes. "
    "An external historical pace reference is about 91. "
    "The seeded era pace is illustrative and is not a possession quota. "
    "No 1995-96 play-by-play timestamps are stored."
)

CATALOG = (
    {
        "id": "fictional_demo",
        "teams": "Bay City Legends and Chicago Stampede",
        "seasons": None,
        "era": "any seeded era key",
        "timing_mode": TIMING_ORIGINAL,
        "output": "fictional_demo",
        "historical_calibration": "not_applicable",
        "note": "Fictional ratings. Functionally repeatable. Not a historical season.",
    },
    {
        "id": "warriors_2015_16_versus_fictional",
        "teams": "Golden State Warriors versus a fictional opponent",
        "seasons": "2015-16 for the Warriors",
        "era": "2015-16",
        "timing_mode": TIMING_FIRST_TRIP,
        "output": "historical_roster_versus_fictional_opponent",
        "historical_calibration": "incomplete",
        "note": (
            "Functionally validated. The first-trip sample is an observed 2015-16 "
            "duration list, not a pass count. Assists and shooting are not calibrated "
            "to the stored season. Not a replay."
        ),
    },
    {
        "id": "bulls_1995_96_versus_fictional",
        "teams": "Chicago Bulls versus a fictional opponent",
        "seasons": "1995-96 for the Bulls",
        "era": "1995-96",
        "timing_mode": TIMING_ORIGINAL,
        "output": "historical_roster_versus_fictional_opponent",
        "historical_calibration": "incomplete",
        "note": PACE_1995_96 + " Not a replay.",
    },
    {
        "id": "bulls_1995_96_provisional_pace",
        "teams": "Chicago Bulls versus a fictional opponent",
        "seasons": "1995-96 for the Bulls",
        "era": "1995-96",
        "timing_mode": TIMING_PROVISIONAL_1995_96,
        "output": "historical_roster_versus_fictional_opponent",
        "historical_calibration": "provisional_in_sample",
        "note": (
            "Optional timing mode. An exponential hold is calibrated to the "
            "stored 91.1 pace factor through the original-timing audit. "
            "That comparison is in-sample. No 1995-96 timestamps are stored. "
            "Not a replay, and not the default."
        ),
    },
    {
        "id": "cross_era",
        "teams": "rosters whose stored seasons differ, or a roster played outside its season",
        "seasons": "1995-96 and 2015-16 are the only stored historical seasons",
        "era": "the selected era environment",
        "timing_mode": "original; the 2015-16 sample only for era 2015-16; the provisional pace mode only for era 1995-96",
        "output": "hypothetical_model_projection",
        "historical_calibration": "unvalidated",
        "note": "Hypothetical model projection. Not evidence about which real team would win.",
    },
)


class BatchRequestError(ValueError):
    """The request cannot be simulated from the stored engine inputs."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Side:
    identifier: str
    team_id: int | None
    season: str | None
    kind: str
    team: Any
    players: dict[str, Any]
    rotation: dict[str, float] | None
    roster: Any
    native_era: str | None
    source: str
    ratings: str


def configuration_catalog() -> list[dict[str, Any]]:
    """Configurations the V1 batch can label. Historical calibration is incomplete."""
    return [dict(row) for row in CATALOG]


def _reject_bool(value: Any, code: str, message: str) -> None:
    if isinstance(value, bool):
        raise BatchRequestError(message, code)


def _mapping(request: Mapping[str, Any] | None, fields: dict[str, Any]) -> dict[str, Any]:
    if request is not None and fields:
        raise BatchRequestError("Pass a request mapping or fields, not both.", "malformed_request")
    if request is None:
        payload = fields
    else:
        if not isinstance(request, Mapping):
            raise BatchRequestError("The request must be a mapping.", "malformed_request")
        payload = dict(request)
    unknown = sorted(set(payload) - set(REQUEST_FIELDS))
    if unknown:
        raise BatchRequestError(f"Unknown request fields: {', '.join(unknown)}.", "malformed_request")
    return payload


def _games(value: Any) -> int:
    _reject_bool(value, "unsupported_game_count", "Games must be 1, 10, 100, or 1000.")
    if value not in ALLOWED_GAMES:
        raise BatchRequestError(
            "Games must be 1, 10, 100, or 1000. simulate_series remains limited to 1 through 82 and is not used here.",
            "unsupported_game_count",
        )
    return int(value)


def _seed(value: Any) -> tuple[int, bool]:
    if value is None:
        return 0, True
    _reject_bool(value, "malformed_seed", "Seed must be an integer of 0 or greater.")
    if isinstance(value, str) or not isinstance(value, int) or value < 0:
        raise BatchRequestError("Seed must be an integer of 0 or greater.", "malformed_seed")
    return value, False


def _flag(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise BatchRequestError("include_events must be true or false.", "malformed_request")
    return value


def _token(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _season(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BatchRequestError("Season must be a string.", "unavailable_season")
    text = value.strip()
    if text == "" or text.casefold() == "fictional":
        return None
    return text


def _load_historical() -> dict[tuple[str, str], Any]:
    global _HISTORICAL
    if _HISTORICAL is not None:
        return _HISTORICAL
    try:
        rosters = load_researched_rosters()
    except HistoricalDataError as exc:
        raise BatchRequestError(str(exc), "historical_data_unavailable") from exc
    found = {}
    for roster in rosters:
        keys = {
            _token(roster.team_name),
            _token(roster.team_abbreviation),
            str(roster.team_id),
        }
        for key in keys:
            found[(key, roster.season_id)] = roster
    _HISTORICAL = found
    return found


def _build_fictional() -> dict[str, _Side]:
    players = {player.player_id: player for player in demo_players()}
    specs = (
        ("Bay City Legends", tuple(f"ERA_{index:04d}" for index in range(1, 6)), "perimeter"),
        ("Chicago Stampede", tuple(f"ERA_{index:04d}" for index in range(6, 11)), "balanced"),
    )
    sides = {}
    for name, ids, style in specs:
        team_players = {pid: players[pid] for pid in ids}
        side = _Side(
            identifier=name,
            team_id=None,
            season=None,
            kind="fictional",
            team=Team(name, ids, style),
            players=team_players,
            rotation=None,
            roster=None,
            native_era=None,
            source="fictional demo fixtures in era_engine.seed",
            ratings="fictional_not_historical",
        )
        sides[_token(name)] = side
    return sides


_FICTIONAL: dict[str, _Side] | None = None
_HISTORICAL: dict[tuple[str, str], Any] | None = None


def _fictional_sides() -> dict[str, _Side]:
    global _FICTIONAL
    if _FICTIONAL is None:
        _FICTIONAL = _build_fictional()
    return _FICTIONAL


def _resolve_side(identifier: Any, season: Any, historical: dict[tuple[str, str], Any], fictional: dict[str, _Side]) -> _Side:
    if identifier is None or (isinstance(identifier, str) and not identifier.strip()):
        raise BatchRequestError("A team identifier is required.", "unknown_team")
    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
        raise BatchRequestError("A team identifier must be a name, abbreviation, or team id.", "unknown_team")
    key = str(identifier) if isinstance(identifier, int) else _token(identifier)
    if isinstance(identifier, int):
        key = str(identifier)
    text_season = _season(season)
    if key in fictional:
        if text_season is not None:
            raise BatchRequestError(
                f"{fictional[key].identifier} is a fictional team and has no historical season.",
                "unavailable_season",
            )
        return fictional[key]
    if text_season is None:
        stored = [item for item in RESEARCHED_SEASONS if _token(item["team_name"]) == key or _token(item["team_abbreviation"]) == key or str(item["team_id"]) == key]
        if stored:
            raise BatchRequestError(
                f"A season is required. Stored seasons for this team: {', '.join(item['season_id'] for item in stored)}.",
                "unavailable_season",
            )
        raise BatchRequestError(f"Unknown team: {identifier}.", "unknown_team")
    roster = historical.get((key, text_season))
    if roster is None:
        known = sorted({item["season_id"] + " " + item["team_name"] for item in RESEARCHED_SEASONS})
        matched_team = any(key in {_token(item["team_name"]), _token(item["team_abbreviation"]), str(item["team_id"])} for item in RESEARCHED_SEASONS)
        if matched_team:
            raise BatchRequestError(
                f"{text_season} is not a stored season for this team. Stored historical seasons: {', '.join(known)}.",
                "unavailable_season",
            )
        raise BatchRequestError(
            f"Unknown team: {identifier}. Stored historical seasons: {', '.join(known)}.",
            "unknown_team",
        )
    return _Side(
        identifier=roster.team_name,
        team_id=roster.team_id,
        season=roster.season_id,
        kind="historical",
        team=roster.team,
        players=roster.players,
        rotation=dict(roster.rotation_targets),
        roster=roster,
        native_era=roster.era_key,
        source="verified team and player counting totals in the research database",
        ratings="derived_provisional",
    )


def _era(value: Any):
    if not isinstance(value, str) or value not in {era.key for era in ERAS}:
        keys = ", ".join(era.key for era in ERAS)
        raise BatchRequestError(f"Era must be one of: {keys}.", "invalid_era")
    return next(era for era in ERAS if era.key == value)


def _timing(mode: Any, era_key: str) -> tuple[TimingAssumptions | None, dict[str, Any]]:
    if mode == TIMING_ORIGINAL:
        return None, {
            "mode": TIMING_ORIGINAL,
            "classification": "assumed_timing",
            "sample_season": None,
            "applied_to_era": era_key,
            "observed_for_requested_era": False,
            "full_timing_model_calibrated": False,
            "possessions_chosen_from_pace": False,
        }
    if mode == TIMING_PROVISIONAL_1995_96:
        if era_key != "1995-96":
            raise BatchRequestError(
                "The provisional 1995-96 aggregate-pace timing applies only when the selected era is 1995-96. "
                f"It will not be applied to {era_key}.",
                "timing_model_not_for_era",
            )
        return TimingAssumptions(development_mean_seconds=PROVISIONAL_1995_96_DEVELOPMENT_MEAN), {
            "mode": TIMING_PROVISIONAL_1995_96,
            "classification": "provisional_aggregate_pace",
            "sample_season": None,
            "applied_to_era": era_key,
            "observed_for_requested_era": False,
            "event_level_timestamps": False,
            "distribution": "exponential",
            "distribution_status": "assumed_not_observed",
            "development_mean_seconds": PROVISIONAL_1995_96_DEVELOPMENT_MEAN,
            "pace_reference": PROVISIONAL_1995_96_PACE_REFERENCE,
            "pace_reference_source": "Basketball-Reference pace factor, Chicago Bulls 1995-96",
            "pace_reference_role": "in_sample_mean_benchmark",
            "full_timing_model_calibrated": False,
            "possessions_chosen_from_pace": False,
        }
    if mode != TIMING_FIRST_TRIP:
        raise BatchRequestError(
            f"timing_mode must be {TIMING_ORIGINAL!r}, {TIMING_FIRST_TRIP!r}, or {TIMING_PROVISIONAL_1995_96!r}.",
            "invalid_timing_mode",
        )
    if era_key != "2015-16":
        raise BatchRequestError(
            "The 2015-16 first-trip sample is not calibrated for "
            f"{era_key} and will not be applied. Use timing_mode {TIMING_ORIGINAL!r} for that era.",
            "timing_sample_not_for_era",
        )
    try:
        sample = shot_trip_duration_sample()
    except (OSError, FileNotFoundError, KeyError, ValueError) as exc:
        raise BatchRequestError(
            f"The 2015-16 first-trip timing sample is unavailable: {exc}",
            "timing_sample_missing",
        ) from exc
    durations = tuple(sample.get("durations") or ())
    if not durations:
        raise BatchRequestError("The 2015-16 first-trip timing sample contains no durations.", "timing_sample_missing")
    return TimingAssumptions(first_trip_targets=durations), {
        "mode": TIMING_FIRST_TRIP,
        "classification": "observed_timing_sample",
        "sample_season": "2015-16",
        "sample_count": len(durations),
        "applied_to_era": era_key,
        "observed_for_requested_era": True,
        "full_timing_model_calibrated": False,
        "possessions_chosen_from_pace": False,
        "passes_in_sample": False,
        "sample_measures": "live game-clock length of a first shot trip",
    }


def _warnings(side_a: _Side, side_b: _Side, era_key: str, timing_meta: dict[str, Any], output: str, include_events: bool, games: int) -> list[dict[str, str]]:
    notes = [{
        "code": "calibration_incomplete",
        "message": "Historical calibration is not complete. These results are model output.",
    }]
    if output == "hypothetical_model_projection":
        notes.append({
            "code": "hypothetical_cross_era",
            "message": "Hypothetical model projection. Not evidence about which real team would win.",
        })
    elif output == "historical_roster_versus_fictional_opponent":
        notes.append({
            "code": "fictional_opponent",
            "message": "One opponent is fictional. This is not a historical replay.",
        })
    else:
        notes.append({
            "code": "fictional_demo",
            "message": "Both teams are fictional demo rosters. This is not a historical season.",
        })
    if any(side.ratings == "derived_provisional" for side in (side_a, side_b)):
        notes.append({
            "code": "ratings_provisional",
            "message": (
                "Counting totals are verified stored statistics. "
                "Passing, defense, IQ, shot skill, and shot propensity are derived or provisional. "
                "The shoot-or-pass decision is frozen for V1."
            ),
        })
    if timing_meta["mode"] == TIMING_ORIGINAL and era_key == "1995-96":
        notes.append({"code": "pace_1995_96_uncalibrated", "message": PACE_1995_96})
    elif timing_meta["mode"] == TIMING_ORIGINAL and era_key == "2015-16":
        notes.append({
            "code": "timing_sample_not_selected",
            "message": "Original timing was selected. The stored 2015-16 first-trip sample was not applied.",
        })
    elif timing_meta["mode"] == TIMING_ORIGINAL:
        notes.append({
            "code": "timing_assumed",
            "message": f"{era_key} has no stored timing sample. Original timing is an assumption, and the seeded pace is not a possession quota.",
        })
    elif timing_meta["mode"] == TIMING_PROVISIONAL_1995_96:
        notes.append({
            "code": "provisional_aggregate_pace",
            "message": (
                "Provisional 1995-96 timing adds an assumed exponential hold before the first shot. "
                f"Its mean is translated from the stored {PROVISIONAL_1995_96_PACE_REFERENCE} pace factor "
                "and the original-timing audit. A comparison with that same pace factor is in-sample, "
                "not independent confirmation. No 1995-96 play-by-play timestamps are stored. "
                "The 2015-16 first-trip sample is not used. Possessions are not fixed at the pace factor. "
                "This timing environment is the selected 1995-96 mode."
            ),
        })
    else:
        notes.append({
            "code": "timing_sample_observed",
            "message": (
                "Timing uses the stored 2015-16 first-trip duration sample. "
                "That sample does not contain pass counts. "
                "It is not applied to any other era."
            ),
        })
        if output != "historical_roster_versus_fictional_opponent" or not (
            {side_a.season, side_b.season} <= {None, "2015-16"} and "2015-16" in {side_a.season, side_b.season}
        ):
            notes.append({
                "code": "timing_sample_outside_matchup",
                "message": "The 2015-16 sample is an observation from that season's play-by-play, not a measurement of this matchup.",
            })
    if include_events:
        notes.append({
            "code": "event_logs_requested",
            "message": f"Play-by-play was requested and is included for {games} games.",
        })
    return notes


def _output_kind(side_a: _Side, side_b: _Side, era_key: str) -> str:
    historical = [side for side in (side_a, side_b) if side.kind == "historical"]
    if not historical:
        return "fictional_demo"
    seasons = {side.season for side in historical}
    if len(historical) == 2 or len(seasons) > 1 or any(side.native_era != era_key for side in historical):
        return "hypothetical_model_projection"
    return "historical_roster_versus_fictional_opponent"


def _rate(made: float, attempts: float) -> float | None:
    if attempts <= 0:
        return None
    return made / attempts


def _empty_samples() -> dict[str, Sample]:
    return {stat: Sample() for stat in (*COUNTING, "fga_share")}


def _average(totals: dict[str, float], games: int) -> dict[str, Any]:
    if games == 0:
        return {stat: None for stat in COUNTING} | {
            "fg_pct": None, "fg3_pct": None, "ft_pct": None, "fga_share": None,
        }
    return {
        **{stat: totals[stat] / games for stat in COUNTING},
        "fg_pct": _rate(totals["fgm"], totals["fga"]),
        "fg3_pct": _rate(totals["fg3m"], totals["fg3a"]),
        "ft_pct": _rate(totals["ftm"], totals["fta"]),
    }


def _distributions(samples: dict[str, Sample]) -> dict[str, Any]:
    return {name: sample.summary() for name, sample in samples.items()}


def _player_row(pid: str, side: _Side, samples: dict[str, Sample], totals: dict[str, float], team_fga: float, games: int) -> dict[str, Any]:
    player = side.players[pid]
    profile = None if side.roster is None else next((item for item in side.roster.profiles if item.player.player_id == pid), None)
    average = _average(totals, games)
    average["fga_share"] = None if team_fga <= 0 else totals["fga"] / team_fga
    row = {
        "player_id": pid,
        "name": player.name,
        "nba_player_id": None if profile is None else profile.nba_player_id,
        "average": average,
        "distribution": _distributions(samples),
    }
    if profile is not None:
        row["stored_season"] = {
            "classification": "verified_counting_totals",
            "source": "playercareerstats team stint",
            "points_per_game": profile.calculated.get("points_per_game"),
            "fga_per_game": profile.calculated.get("fga_per_game"),
            "fga_share": profile.calculated.get("fga_share"),
            "assists_per_game": profile.calculated.get("assists_per_game"),
            "minutes_per_game": profile.calculated.get("minutes_per_game"),
        }
        row["ratings"] = "derived_provisional"
    else:
        row["ratings"] = "fictional_not_historical"
    return row


def _team_block(side: _Side, samples: dict[str, Sample], totals: dict[str, float], games: int, player_rows: list[dict[str, Any]]) -> dict[str, Any]:
    average = _average(totals, games)
    average["fga_share"] = None
    return {
        "identifier": side.identifier,
        "team_id": side.team_id,
        "season": side.season,
        "kind": side.kind,
        "native_era": side.native_era,
        "source": side.source,
        "ratings": side.ratings,
        "average": average,
        "distribution": _distributions({name: sample for name, sample in samples.items() if name != "fga_share"}),
        "players": player_rows,
    }


def _box(game: dict[str, Any], ids: tuple[str, ...]) -> dict[str, float]:
    box = {stat: 0.0 for stat in COUNTING}
    for pid in ids:
        stats = game["player_stats"][pid]
        for stat in COUNTING:
            box[stat] += stats.get(stat, 0)
    return box


def simulate_batch(request: Mapping[str, Any] | None = None, /, **fields: Any) -> dict[str, Any]:
    """Simulate 1, 10, 100, or 1000 games and return a JSON-ready summary.

    Required fields are ``team_a``, ``team_b``, ``era``, ``games``, and
    ``timing_mode``. ``season_a`` and ``season_b`` are required for a stored
    historical team and must be omitted or ``"fictional"`` for a demo team.
    ``seed`` defaults to 0. Seeds then increase by one. ``include_events``
    defaults to false and is the only way to keep play-by-play.
    """
    payload = _mapping(request, fields)
    for name in ("team_a", "team_b", "era", "games", "timing_mode"):
        if name not in payload:
            raise BatchRequestError(f"{name} is required.", "malformed_request")
    games = _games(payload["games"])
    seed, seed_defaulted = _seed(payload.get("seed"))
    include_events = _flag(payload.get("include_events"))
    era = _era(payload["era"])
    timing, timing_meta = _timing(payload["timing_mode"], era.key)
    fictional = _fictional_sides()
    keys = []
    for name in ("team_a", "team_b"):
        identifier = payload[name]
        if isinstance(identifier, int) and not isinstance(identifier, bool):
            keys.append(str(identifier))
        elif isinstance(identifier, str):
            keys.append(_token(identifier))
        else:
            keys.append(None)
    historical = _load_historical() if any(key not in fictional for key in keys) else {}
    side_a = _resolve_side(payload["team_a"], payload.get("season_a"), historical, fictional)
    side_b = _resolve_side(payload["team_b"], payload.get("season_b"), historical, fictional)
    if set(side_a.team.player_ids) & set(side_b.team.player_ids):
        raise BatchRequestError("The two teams share a player and cannot play each other.", "invalid_matchup")
    output = _output_kind(side_a, side_b, era.key)
    players = {**side_a.players, **side_b.players}
    rotations = None
    if side_a.rotation is not None or side_b.rotation is not None:
        rotations = (
            None if side_a.rotation is None else dict(side_a.rotation),
            None if side_b.rotation is None else dict(side_b.rotation),
        )
    warnings = _warnings(side_a, side_b, era.key, timing_meta, output, include_events, games)

    ids = (side_a.team.player_ids, side_b.team.player_ids)
    team_samples = [_empty_samples() for _ in ids]
    team_totals = [{stat: 0.0 for stat in COUNTING} for _ in ids]
    player_samples = {pid: _empty_samples() for group in ids for pid in group}
    player_totals = {pid: {stat: 0.0 for stat in COUNTING} for group in ids for pid in group}
    game_results = []
    incomplete = []
    failed = []
    completed = 0
    team_a_wins = 0
    overtime_games = 0
    overtime_periods = 0
    max_overtime = 0
    possessions = [0.0, 0.0]
    clock_minutes = 0.0
    started = time.perf_counter()
    for offset in range(games):
        game_seed = seed + offset
        try:
            game = simulate_game(
                side_a.team, side_b.team, players, era, game_seed,
                track_events=True, timing=timing, rotation_targets=rotations,
            )
        except UnresolvedGameError as exc:
            incomplete.append({
                "seed": game_seed,
                "status": "incomplete",
                "message": str(exc),
                "returned_as_completed": False,
            })
            continue
        except Exception as exc:
            failed.append({
                "seed": game_seed,
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
            })
            continue
        issues = check_game(game, era, side_a.team.player_ids, side_b.team.player_ids)
        if issues:
            failed.append({"seed": game_seed, "status": "check", "message": issues[:8]})
            continue
        if game["score"][0] == game["score"][1] or game.get("status") != "final":
            incomplete.append({
                "seed": game_seed,
                "status": "incomplete",
                "message": "The game did not produce a winner.",
                "returned_as_completed": False,
            })
            continue
        completed += 1
        periods = int(game["overtime"]["periods"])
        overtime_periods += periods
        max_overtime = max(max_overtime, periods)
        if periods:
            overtime_games += 1
        if game["score"][0] > game["score"][1]:
            team_a_wins += 1
        clock_minutes += game["game_clock_seconds"] / 60.0
        possessions[0] += game["possessions_by_team"][0]
        possessions[1] += game["possessions_by_team"][1]
        boxes = (_box(game, ids[0]), _box(game, ids[1]))
        for index, box in enumerate(boxes):
            for stat in COUNTING:
                team_samples[index][stat].add(box[stat])
                team_totals[index][stat] += box[stat]
            for pid in ids[index]:
                row = game["player_stats"][pid]
                share = 0.0 if box["fga"] <= 0 else row.get("fga", 0) / box["fga"]
                for stat in COUNTING:
                    player_samples[pid][stat].add(row.get(stat, 0))
                    player_totals[pid][stat] += row.get(stat, 0)
                player_samples[pid]["fga_share"].add(share)
        winner = side_a.identifier if game["score"][0] > game["score"][1] else side_b.identifier
        record = {
            "seed": game_seed,
            "score": list(game["score"]),
            "winner": winner,
            "overtime_periods": periods,
            "regulation_score": list(game["regulation_score"]),
        }
        if include_events:
            record["events"] = game["events"]
        game_results.append(record)
    elapsed = time.perf_counter() - started
    if completed == 0:
        status = "incomplete"
    elif completed == games and not incomplete and not failed:
        status = "completed"
    else:
        status = "partial"
    team_blocks = []
    for index, side in enumerate((side_a, side_b)):
        team_fga = team_totals[index]["fga"]
        rows = [
            _player_row(pid, side, player_samples[pid], player_totals[pid], team_fga, completed)
            for pid in side.team.player_ids
        ]
        team_blocks.append(_team_block(side, team_samples[index], team_totals[index], completed, rows))
    for index, block in enumerate(team_blocks):
        block["possessions_per_48"] = None if clock_minutes == 0 else possessions[index] / clock_minutes * 48.0
    result = {
        "schema": SCHEMA,
        "statistic_fields": dict(STATISTIC_FIELDS),
        "model_version": MODEL_VERSION,
        "engine": "simulate_game",
        "series_limit_preserved": SERIES_GAME_LIMIT,
        "variant_b3": "experimental_not_used",
        "historical_calibration_complete": False,
        "status": status,
        "request": {
            "team_a": side_a.identifier,
            "season_a": side_a.season,
            "team_b": side_b.identifier,
            "season_b": side_b.season,
            "era": era.key,
            "games": games,
            "seed": seed,
            "seed_defaulted": seed_defaulted,
            "timing_mode": timing_meta["mode"],
            "include_events": include_events,
        },
        "games_requested": games,
        "games_completed": completed,
        "games_incomplete": len(incomplete),
        "games_failed_checks": len(failed),
        "seeds": {"start": seed, "step": 1, "count": games},
        "wins": {
            "team_a": team_a_wins,
            "team_b": completed - team_a_wins,
            "team_a_win_pct": None if completed == 0 else team_a_wins / completed,
            "team_b_win_pct": None if completed == 0 else (completed - team_a_wins) / completed,
            "ties_returned_as_completed": 0,
        },
        "overtime": {
            "games": overtime_games,
            "periods": overtime_periods,
            "max_periods": max_overtime,
            "mean_periods": None if completed == 0 else overtime_periods / completed,
        },
        "clock": {
            "regulation_seconds": era.game_minutes * 60,
            "overtime_period_seconds": 300,
            "completed_games_excluded_when_clock_invalid": True,
        },
        "teams": {"team_a": team_blocks[0], "team_b": team_blocks[1]},
        "game_results": game_results,
        "incomplete_games": incomplete,
        "check_failures": failed,
        "calibration": {
            "historical_calibration_complete": False,
            "output": output,
            "timing": timing_meta,
            "warnings": warnings,
            "supported_configurations": configuration_catalog(),
            "frozen": [
                "shoot_or_pass",
                "historical_touch_selection",
                "shooting_propensity",
                "shooting_free_throw_rebound_and_foul_formulas",
                "clock_overtime_rotations_and_fatigue",
                "research_records",
                "timing_distributions",
            ],
        },
        "event_logs_retained": include_events,
        "runtime": {
            "seconds": elapsed,
            "seconds_per_completed_game": None if completed == 0 else elapsed / completed,
        },
    }
    json.dumps(result, allow_nan=False)
    return result


def main(argv: list[str] | None = None) -> int:
    """Read one batch request and write its JSON summary."""
    parser = argparse.ArgumentParser(description="Run one ERA Basketball V1 batch")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--season-a", default=None)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--season-b", default=None)
    parser.add_argument("--era", required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timing", required=True, dest="timing_mode")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--output", default="-")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = simulate_batch(
            team_a=args.team_a,
            season_a=args.season_a,
            team_b=args.team_b,
            season_b=args.season_b,
            era=args.era,
            games=args.games,
            seed=args.seed,
            timing_mode=args.timing_mode,
            include_events=args.include_events,
        )
    except BatchRequestError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}))
        return 2
    payload = result
    if args.summary:
        payload = {
            "status": result["status"],
            "games_completed": result["games_completed"],
            "games_incomplete": result["games_incomplete"],
            "games_failed_checks": result["games_failed_checks"],
            "wins": result["wins"],
            "overtime": result["overtime"],
            "timing": result["calibration"]["timing"]["mode"],
            "output": result["calibration"]["output"],
            "warnings": [item["code"] for item in result["calibration"]["warnings"]],
            "team_a_points": result["teams"]["team_a"]["average"]["pts"],
            "team_b_points": result["teams"]["team_b"]["average"]["pts"],
            "possessions_per_48": result["teams"]["team_a"]["possessions_per_48"],
            "event_logs_retained": result["event_logs_retained"],
            "runtime": result["runtime"],
            "bytes": len(json.dumps(result, allow_nan=False).encode()),
        }
    encoded = json.dumps(payload, indent=2, allow_nan=False).encode()
    if args.output == "-":
        print(encoded.decode())
    else:
        from pathlib import Path
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        print(json.dumps({"output": str(destination), "bytes": destination.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
