"""Experimental minutes and substitution scheduler.

This module does not play a game and does not rate players. Regulation output
is a five-player timeline whose minutes add to 240. It is not part of the
live possession engine.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

REGULATION_GAME_MINUTES = 48
ON_COURT = 5
REGULATION_TEAM_MINUTES = REGULATION_GAME_MINUTES * ON_COURT
PLAYER_MINUTE_CAP = 48
STRATEGIES = ("target_minutes", "equal_time", "short_stints")

INTEGRATION = {
    "promoted": True,
    "live_engine": "era_engine.core.simulate_game",
    "game_clock_required_before_live_integration": True,
    "reason": (
        "simulate_game calls this scheduler for a roster with more than five players. "
        "A five-player roster stays in listed order for the whole regulation game. "
        "Substitutions are applied at period breaks, shooting fouls, and "
        "shot-clock violations. The scheduler's fatigue_at_end value is not "
        "an input to the live shooting formula. Overtime minutes are played "
        "by simulate_game and are not part of this 48-minute plan."
    ),
}


@dataclass(frozen=True)
class RotationPlayer:
    """Experimental availability. Stamina here is not an NBA rating."""

    player_id: str
    target_minutes: float = 0.0
    available: bool = True
    stamina: int = 70


def overtime_requirement() -> dict:
    """Overtime is identified and not scheduled.

    Season totals can exceed 240 player-minutes per team game. That residual
    is not converted into an overtime timeline. The live engine plays its own
    five-minute periods and does not read the empty schedule below.
    """
    return {
        "status": "future_requirement",
        "schedule": None,
        "historical_overtime_schedule": None,
    }


def _max_stint(strategy: str, stamina: int) -> int:
    if strategy == "short_stints":
        return 4
    return max(4, round(stamina / 100 * 16))


def _integer_quotas(targets: dict[str, float]) -> dict[str, int]:
    total = sum(targets.values())
    if abs(total - REGULATION_TEAM_MINUTES) > 1e-4:
        raise ValueError("target minutes must sum to 240 for a regulation game")
    scaled = {player_id: value * REGULATION_TEAM_MINUTES / total for player_id, value in targets.items()}
    quotas = {player_id: int(math.floor(value + 1e-8)) for player_id, value in scaled.items()}
    if any(quota > PLAYER_MINUTE_CAP for quota in quotas.values()):
        raise ValueError("a target exceeds 48 regulation minutes")
    leftover = REGULATION_TEAM_MINUTES - sum(quotas.values())
    order = sorted(scaled, key=lambda player_id: (-(scaled[player_id] - quotas[player_id]), -scaled[player_id], player_id))
    for player_id in order:
        if leftover <= 0:
            break
        if quotas[player_id] >= PLAYER_MINUTE_CAP:
            continue
        quotas[player_id] += 1
        leftover -= 1
    if leftover != 0:
        raise ValueError("targets cannot be placed without exceeding 48 minutes")
    return quotas


def _opening_five(quotas: dict[str, int], supplied: tuple[str, ...] | None) -> tuple[tuple[str, ...], str]:
    positive = [player_id for player_id, quota in quotas.items() if quota > 0]
    if len(positive) < ON_COURT:
        raise ValueError("a regulation game needs at least five players with minutes")
    if supplied is None:
        chosen = tuple(sorted(quotas, key=lambda player_id: (-quotas[player_id], player_id))[:ON_COURT])
        return chosen, "largest_targets"
    if len(supplied) != ON_COURT or len(set(supplied)) != ON_COURT:
        raise ValueError("a starting lineup must contain five distinct players")
    unknown = [player_id for player_id in supplied if player_id not in quotas]
    if unknown:
        raise ValueError("starting lineup includes a player who is not on this roster")
    if any(quotas[player_id] <= 0 for player_id in supplied):
        raise ValueError("a supplied starter needs at least one target minute")
    left_out = [player_id for player_id, quota in quotas.items() if player_id not in supplied and quota > REGULATION_GAME_MINUTES - 1]
    if left_out:
        raise ValueError("a player targeted for 48 minutes must be in the opening lineup")
    return tuple(supplied), "caller_supplied"


def schedule_regulation(
    players: list[RotationPlayer] | tuple[RotationPlayer, ...],
    strategy: str = "target_minutes",
    starting_five: tuple[str, ...] | None = None,
) -> dict:
    """Build one regulation rotation. Does not invent overtime or a historical lineup."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown rotation strategy: {strategy}")
    if len({player.player_id for player in players}) != len(players):
        raise ValueError("duplicate player in the rotation roster")
    for player in players:
        if not 0 <= player.stamina <= 100:
            raise ValueError("experimental stamina must be from 0 through 100")
        if player.target_minutes < 0:
            raise ValueError("target minutes cannot be negative")
        if player.target_minutes > PLAYER_MINUTE_CAP:
            raise ValueError("target minutes cannot exceed 48 in regulation")
        if not player.available and player.target_minutes != 0:
            raise ValueError("an unavailable player cannot have target minutes")

    available = [player for player in players if player.available]
    if len(available) < ON_COURT:
        raise ValueError("fewer than five players are available")
    if strategy == "equal_time":
        share = REGULATION_TEAM_MINUTES / len(available)
        targets = {player.player_id: share for player in available}
    else:
        targets = {player.player_id: player.target_minutes for player in available}
    targets.update({player.player_id: 0.0 for player in players if not player.available})
    quotas = _integer_quotas(targets)
    opening, opening_basis = _opening_five(quotas, starting_five)
    stints = {player.player_id: _max_stint(strategy, player.stamina) for player in players}

    remaining = dict(quotas)
    consecutive = {player.player_id: 0 for player in players}
    minute_lineups: list[tuple[str, ...]] = []
    current = list(opening)
    for minute in range(REGULATION_GAME_MINUTES):
        minutes_left_if_benched = REGULATION_GAME_MINUTES - (minute + 1)
        must_play = {
            player_id for player_id, left in remaining.items()
            if left > minutes_left_if_benched
        }
        if len(must_play) > ON_COURT:
            raise ValueError("minute targets cannot keep a valid five-player lineup")
        bench = [player_id for player_id, left in remaining.items() if left > 0 and player_id not in current]
        fatigued = [
            player_id for player_id in current
            if player_id not in must_play and remaining[player_id] > 0 and consecutive[player_id] >= stints[player_id]
        ]
        fatigued.sort(key=lambda player_id: (-consecutive[player_id], player_id))
        resting = {player_id for player_id in current if remaining[player_id] <= 0}
        for player_id in fatigued:
            if len(resting) >= len(bench):
                break
            resting.add(player_id)
        chosen = [player_id for player_id in current if player_id not in resting]
        for player_id in sorted(must_play):
            if player_id not in chosen and remaining[player_id] > 0:
                chosen.append(player_id)
        while len(chosen) > ON_COURT:
            droppable = [player_id for player_id in chosen if player_id not in must_play]
            if not droppable:
                raise ValueError("minute targets cannot keep a valid five-player lineup")
            drop = min(droppable, key=lambda player_id: (remaining[player_id], player_id))
            chosen.remove(drop)
        fill = sorted(
            (player_id for player_id in bench if player_id not in chosen),
            key=lambda player_id: (-remaining[player_id], consecutive[player_id], player_id),
        )
        for player_id in fill:
            if len(chosen) == ON_COURT:
                break
            chosen.append(player_id)
        if len(chosen) != ON_COURT or len(set(chosen)) != ON_COURT:
            raise ValueError("could not fill five on-court players")
        lineup = tuple(sorted(chosen))
        minute_lineups.append(lineup)
        chosen_ids = set(lineup)
        for player_id in remaining:
            if player_id in chosen_ids:
                remaining[player_id] -= 1
                consecutive[player_id] += 1
            else:
                consecutive[player_id] = 0
        current = list(lineup)

    segments = []
    substitutions = []
    previous = None
    for index, lineup in enumerate(minute_lineups):
        if previous == lineup:
            segments[-1]["end_minute"] = index + 1
        else:
            if previous is not None:
                substitutions.append({
                    "minute": index,
                    "players_out": tuple(sorted(set(previous) - set(lineup))),
                    "players_in": tuple(sorted(set(lineup) - set(previous))),
                })
            segments.append({
                "start_minute": index,
                "end_minute": index + 1,
                "on_court": lineup,
            })
            previous = lineup
        on_court = segments[-1]["on_court"]
        segments[-1]["fatigue_at_end"] = {
            player_id: min(1.0, (index - segments[-1]["start_minute"] + 1) / stints[player_id])
            for player_id in on_court
        }

    minutes = {player.player_id: 0 for player in players}
    for lineup in minute_lineups:
        for player_id in lineup:
            minutes[player_id] += 1
    result = {
        "regulation_minutes": REGULATION_GAME_MINUTES,
        "on_court_count": ON_COURT,
        "team_minutes": REGULATION_TEAM_MINUTES,
        "player_minute_cap": PLAYER_MINUTE_CAP,
        "strategy": strategy,
        "opening_five": minute_lineups[0],
        "opening_five_basis": opening_basis,
        "historical_starting_lineup": False,
        "overtime": overtime_requirement(),
        "minutes": minutes,
        "segments": segments,
        "substitutions": substitutions,
        "fatigue_model": "experimental consecutive-minute fatigue; not the live possession-index fatigue",
        "promoted": False,
    }
    validate_regulation(result, players)
    return result


def validate_regulation(result: dict, players: list[RotationPlayer] | tuple[RotationPlayer, ...]) -> None:
    """Raise when a regulation rotation breaks the five-player minute rules."""
    available = {player.player_id for player in players if player.available}
    if sum(result["minutes"].values()) != REGULATION_TEAM_MINUTES:
        raise ValueError("player minutes do not sum to 240")
    if any(minutes > PLAYER_MINUTE_CAP for minutes in result["minutes"].values()):
        raise ValueError("a player exceeds 48 minutes")
    if any(result["minutes"][player.player_id] != 0 for player in players if not player.available):
        raise ValueError("an unavailable player was given minutes")
    cursor = 0
    for segment in result["segments"]:
        on_court = segment["on_court"]
        if segment["start_minute"] != cursor or segment["end_minute"] <= segment["start_minute"]:
            raise ValueError("segments do not cover the regulation game")
        if len(on_court) != ON_COURT or len(set(on_court)) != ON_COURT:
            raise ValueError("a segment does not have five distinct players")
        if any(player_id not in available for player_id in on_court):
            raise ValueError("a segment includes an unavailable player")
        cursor = segment["end_minute"]
    if cursor != REGULATION_GAME_MINUTES:
        raise ValueError("segments do not cover 48 minutes")
    for substitution in result["substitutions"]:
        if len(substitution["players_out"]) != len(substitution["players_in"]):
            raise ValueError("substitution does not preserve lineup size")
        if set(substitution["players_out"]) & set(substitution["players_in"]):
            raise ValueError("substitution lists the same player on both sides")


def describe_season_minutes(player_rows: list[dict], team_games: int) -> dict:
    """Describe stored season minutes. These are not single-game lineups.

    minutes_per_appearance is minutes divided by games played.
    minutes_per_team_game spreads those minutes across every team game,
    including games the player missed. Neither number is a recovered box score.
    """
    if team_games <= 0:
        raise ValueError("team games must be positive")
    described = []
    total_minutes = 0.0
    for row in player_rows:
        games = row.get("games")
        minutes = row.get("minutes")
        if games in (None, 0) or minutes is None:
            per_appearance = None
        else:
            per_appearance = minutes / games
        per_team_game = None if minutes is None else minutes / team_games
        if minutes is not None:
            total_minutes += minutes
        described.append({
            "player_id": row.get("player_id", row.get("nba_player_id")),
            "player_name": row.get("player_name"),
            "games": games,
            "minutes": minutes,
            "minutes_per_appearance": per_appearance,
            "minutes_per_team_game": per_team_game,
            "single_game_minutes": None,
        })
    regulation_total = REGULATION_TEAM_MINUTES * team_games
    return {
        "team_games": team_games,
        "player_count": len(described),
        "total_player_minutes": total_minutes,
        "regulation_player_minutes": regulation_total,
        "minutes_above_regulation": total_minutes - regulation_total,
        "minutes_above_regulation_meaning": (
            "Season player-minutes minus five players times 48 minutes times team games. "
            "A positive remainder can include overtime. It is not an overtime schedule."
        ),
        "players": described,
        "starting_lineup": None,
        "single_game_lineups": None,
        "overtime": overtime_requirement(),
    }


def project_regulation_targets(description: dict) -> dict:
    """Scale season minutes onto a 240-minute regulation budget.

    The result is a projection for the scheduler. It is not a historical game
    and it does not name a starting lineup.
    """
    raw = {
        player["player_id"]: player["minutes_per_team_game"]
        for player in description["players"]
        if player["minutes_per_team_game"] not in (None, 0)
    }
    if len(raw) < ON_COURT:
        raise ValueError("fewer than five players have season minutes to project")
    total = sum(raw.values())
    targets = {player_id: value * REGULATION_TEAM_MINUTES / total for player_id, value in raw.items()}
    for _ in range(len(targets)):
        overflow = sum(max(0.0, value - PLAYER_MINUTE_CAP) for value in targets.values())
        if overflow <= 1e-8:
            break
        capped = {player_id: min(PLAYER_MINUTE_CAP, value) for player_id, value in targets.items()}
        room = {player_id: PLAYER_MINUTE_CAP - value for player_id, value in capped.items() if value < PLAYER_MINUTE_CAP - 1e-8}
        if sum(room.values()) + 1e-8 < overflow:
            raise ValueError("season minutes cannot fit under the 48-minute regulation cap")
        room_total = sum(room.values())
        for player_id, value in room.items():
            capped[player_id] += overflow * value / room_total
        targets = capped
    if abs(sum(targets.values()) - REGULATION_TEAM_MINUTES) > 1e-4:
        raise ValueError("projected targets do not sum to 240")
    if any(value > PLAYER_MINUTE_CAP + 1e-8 for value in targets.values()):
        raise ValueError("projected target exceeds 48 minutes")
    return {
        "targets": targets,
        "label": "proportional_projection_of_season_minutes",
        "historical_game": False,
        "starting_lineup": None,
        "overtime": overtime_requirement(),
    }


def season_minutes_from_research(con: sqlite3.Connection, team_id: int, season_id: str) -> dict:
    """Read stored season totals. Does not update them."""
    team_games = con.execute(
        '''SELECT games FROM nba_team_season_research
           WHERE team_id=? AND season_id=? AND endpoint='teamyearbyyearstats' ''',
        (team_id, season_id),
    ).fetchone()
    if team_games is None or team_games[0] in (None, 0):
        raise ValueError("team games are not available for this season")
    rows = con.execute(
        '''SELECT nba_player_id, player_name, games, minutes
           FROM nba_player_season_research
           WHERE team_id=? AND season_id=? AND endpoint='playercareerstats'
             AND dataset_name='SeasonTotalsRegularSeason'
             AND UPPER(COALESCE(team_abbreviation, '')) != 'TOT'
           ORDER BY player_name''',
        (team_id, season_id),
    ).fetchall()
    described = describe_season_minutes(
        [
            {"player_id": player_id, "player_name": name, "games": games, "minutes": minutes}
            for player_id, name, games, minutes in rows
        ],
        team_games[0],
    )
    described["team_id"] = team_id
    described["season_id"] = season_id
    described["source_endpoint"] = "playercareerstats"
    return described
