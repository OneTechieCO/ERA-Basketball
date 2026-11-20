"""Possession timing from stored NBA play-by-play.

The files under research_playbyplay are the source. This module does not
fetch data and does not write season totals. A pass, dribble, or screen is
not given a duration, because these events do not record one. The shot clock
is not estimated.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "research_playbyplay"
CLOCK = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
REBOUND_COUNTS = re.compile(r"\(Off:(\d+) Def:(\d+)\)")
FREE_THROW_NUMBER = re.compile(r"(\d+) of (\d+)")

PROPOSED_HALFCOURT_MODEL = {
    "implemented": False,
    "purpose": (
        "A later experiment can let a ballhandler shoot, pass, hold, or start a screen "
        "after the ball is in the half court. This sample does not time those actions separately."
    ),
    "observed": [
        "Regulation game-clock possession durations from the stored 2015-16 play-by-play sample",
        "Separate durations for possessions that end in a shot and possessions that end in a turnover",
        "Game-clock time from an offensive rebound to the end of that same possession",
    ],
    "unavailable": [
        "Shot clock remaining at the release",
        "Transition versus half-court labels",
        "Pass, dribble, hold, and screen durations",
    ],
    "do_not_use_as_a_duration": (
        "The gap between the experimental means and the observed median is not a dribble time "
        "and must not be inserted as a constant."
    ),
}

OBSERVATION_RULES = {
    "clock": "Game-clock time remaining in the period. Stopped time does not move it, so a difference excludes stoppages.",
    "shot_clock": "PlayByPlayV3 has no shot-clock field. Remaining shot clock is not estimated.",
    "offensive_rebound": (
        "A player rebound line carries that player's game rebound counts as Off:N Def:N. "
        "The rebound is offensive when Off increases by one, and defensive when Def increases by one."
    ),
    "team_rebound": "A team rebound without Off/Def counts is ambiguous and is not classified.",
    "free_throws": "Free throws are dead-ball events. A rebound between free throws does not continue a live possession.",
    "transition": "These actions do not label transition or half-court. No split is inferred from duration.",
    "passes": "Pass, dribble, and screen durations are not present and are not inferred.",
}


def clock_seconds(value) -> float | None:
    if not isinstance(value, str):
        return None
    match = CLOCK.fullmatch(value)
    if match is None:
        return None
    return float(match.group(1) or 0) * 60.0 + float(match.group(2) or 0)


def load_manifest(root: Path = ROOT) -> dict:
    return json.loads((root / "manifest.json").read_text())


def load_actions(game_id: str, root: Path = ROOT) -> list[dict]:
    payload = json.loads((root / "source" / "playbyplayv3" / f"{game_id}.json").read_text())
    game = payload.get("game") or {}
    if "clock" not in (game.get("actions") or [{}])[0] and game.get("actions"):
        raise ValueError("play-by-play actions have no clock field")
    return list(game.get("actions") or [])


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _other(team: str | None, teams: set[str]) -> str | None:
    rest = [item for item in teams if item and item != team]
    if len(rest) == 1:
        return rest[0]
    return None


def _rebound_kind(action: dict, counts: dict) -> str:
    match = REBOUND_COUNTS.search(action.get("description") or "")
    if match is None:
        return "ambiguous"
    off, defensive = int(match.group(1)), int(match.group(2))
    key = action.get("personId")
    previous = counts.get(key)
    counts[key] = (off, defensive)
    if previous is None:
        if (off, defensive) == (1, 0):
            return "offensive"
        if (off, defensive) == (0, 1):
            return "defensive"
        return "ambiguous"
    gained_off = off - previous[0]
    gained_def = defensive - previous[1]
    if gained_off == 1 and gained_def == 0:
        return "offensive"
    if gained_def == 1 and gained_off == 0:
        return "defensive"
    return "ambiguous"


def _and_one(actions: list[dict], index: int) -> bool:
    current = actions[index]
    if index + 1 >= len(actions):
        return False
    nxt = actions[index + 1]
    return (
        nxt.get("actionType") == "Foul"
        and nxt.get("subType") in ("Shooting", "Shooting Block")
        and nxt.get("clock") == current.get("clock")
        and nxt.get("period") == current.get("period")
    )


def reconstruct_possessions(actions: list[dict], game_id: str) -> list[dict]:
    """Build possessions from made shots, turnovers, rebounds, fouls, and period ends."""
    teams = {action.get("teamTricode") for action in actions if action.get("teamTricode")}
    counts: dict = {}
    possessions: list[dict] = []
    current = None
    boundary = None
    boundary_period = None
    ft = None
    rebound_after_last_free_throw = False
    tainted = False

    def finish(reason: str, action: dict, remaining: float | None) -> None:
        nonlocal current
        if current is None:
            return
        end_period = action.get("period")
        start = current["start_seconds"]
        reliable = (
            current["exclusion"] is None
            and start is not None
            and remaining is not None
            and end_period == current["period"]
            and remaining <= start
        )
        duration = (start - remaining) if reliable else None
        possessions.append({
            "game_id": game_id,
            "period": current["period"],
            "regulation": current["period"] is not None and current["period"] <= 4,
            "offense": current["offense"],
            "start_seconds": start,
            "end_seconds": remaining,
            "duration": duration,
            "end_reason": reason,
            "offensive_rebounds": current["offensive_rebounds"],
            "continuation_seconds": (
                current["rebound_clock"] - remaining
                if reliable and current["rebound_clock"] is not None and remaining is not None and current["rebound_clock"] >= remaining
                else None
            ),
            "shot_clock_remaining": None,
            "reliable": reliable,
            "exclusion": current["exclusion"],
            "period_ending": reason == "period_end",
        })
        current = None

    def open_possession(team: str | None, period, remaining: float | None) -> None:
        nonlocal current, tainted
        if team is None or remaining is None or period is None:
            current = None
            return
        current = {
            "offense": team,
            "period": period,
            "start_seconds": remaining,
            "offensive_rebounds": 0,
            "rebound_clock": None,
            "exclusion": "after_ambiguous_boundary" if tainted else None,
        }
        tainted = False

    def conflicts(team: str | None) -> bool:
        return current is not None and team is not None and team != current["offense"]

    for index, action in enumerate(actions):
        kind = action.get("actionType") or ""
        subtype = action.get("subType") or ""
        period = action.get("period")
        remaining = clock_seconds(action.get("clock"))
        team = action.get("teamTricode") or None

        if kind == "period" and subtype == "start":
            finish("period_end", action, boundary if boundary_period == period else remaining)
            boundary = remaining
            boundary_period = period
            current = None
            ft = None
            rebound_after_last_free_throw = False
            continue
        if kind == "period" and subtype == "end":
            finish("period_end", action, remaining)
            boundary = None
            continue
        if kind in ("", "Substitution", "Timeout", "Jump Ball", "Instant Replay", "Violation"):
            continue

        if kind == "Foul" and subtype == "Offensive":
            continue
        if kind == "Foul" and subtype in ("Shooting", "Shooting Block"):
            if current is not None:
                finish("shooting_foul", action, remaining)
            ft = {"foul_team": team, "last_seen": False}
            rebound_after_last_free_throw = False
            boundary = remaining
            boundary_period = period
            continue
        if kind == "Foul":
            continue

        if kind == "Free Throw":
            numbers = FREE_THROW_NUMBER.search(subtype)
            missed = "MISS" in (action.get("description") or "")
            is_last = numbers is not None and numbers.group(1) == numbers.group(2)
            if ft is None or not is_last:
                continue
            rebound_after_last_free_throw = missed
            if not missed:
                open_possession(ft.get("foul_team"), period, remaining)
                boundary = remaining
                boundary_period = period
            ft = None
            continue

        if kind == "Rebound":
            rebound_kind = _rebound_kind(action, counts)
            if ft is not None:
                continue
            if rebound_after_last_free_throw:
                rebound_after_last_free_throw = False
                if rebound_kind == "offensive":
                    open_possession(team, period, remaining)
                elif rebound_kind == "defensive":
                    open_possession(team, period, remaining)
                else:
                    boundary = remaining
                    boundary_period = period
                    current = None
                continue
            if current is None:
                continue
            if rebound_kind == "offensive" and team == current["offense"]:
                current["offensive_rebounds"] += 1
                if current["rebound_clock"] is None:
                    current["rebound_clock"] = remaining
                continue
            if rebound_kind == "defensive" and team and team != current["offense"]:
                finish("defensive_rebound", action, remaining)
                open_possession(team, period, remaining)
                boundary = remaining
                boundary_period = period
                continue
            current["exclusion"] = "ambiguous_rebound"
            finish("ambiguous_rebound", action, remaining)
            boundary = remaining
            boundary_period = period
            tainted = True
            continue

        if kind == "Missed Shot":
            if conflicts(team):
                current["exclusion"] = "team_mismatch"
                finish("team_mismatch", action, remaining)
                boundary = remaining
                boundary_period = period
            if current is None:
                open_possession(team, period, boundary if boundary_period == period else remaining)
            continue
        if kind == "Made Shot":
            if conflicts(team):
                current["exclusion"] = "team_mismatch"
                finish("team_mismatch", action, remaining)
                boundary = remaining
                boundary_period = period
            elif current is None:
                open_possession(team, period, boundary if boundary_period == period else remaining)
                finish("made_shot", action, remaining)
            else:
                finish("made_shot", action, remaining)
            if not _and_one(actions, index):
                open_possession(_other(team, teams), period, remaining)
                boundary = remaining
                boundary_period = period
            continue
        if kind == "Turnover":
            if conflicts(team):
                current["exclusion"] = "team_mismatch"
                finish("team_mismatch", action, remaining)
                boundary = remaining
                boundary_period = period
            if current is None:
                open_possession(team, period, boundary if boundary_period == period else remaining)
            reason = "shot_clock_violation" if subtype == "Shot Clock Turnover" else "turnover"
            finish(reason, action, remaining)
            open_possession(_other(team, teams), period, remaining)
            boundary = remaining
            boundary_period = period

    return possessions


def _distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def summarize_games(game_ids: list[str], root: Path = ROOT) -> dict:
    possessions = []
    action_counts = {}
    shot_clock_fields = set()
    for game_id in game_ids:
        actions = load_actions(game_id, root)
        action_counts[game_id] = len(actions)
        if actions:
            shot_clock_fields.update(key for key in actions[0] if "shotclock" in key.lower() or key == "shotClock")
        possessions.extend(reconstruct_possessions(actions, game_id))
    excluded = [row for row in possessions if not row["reliable"] or not row["regulation"]]
    possessions = [row for row in possessions if row["reliable"] and row["regulation"] and row["duration"] is not None]
    durations = [row["duration"] for row in possessions]
    shots = [row["duration"] for row in possessions if row["end_reason"] in ("made_shot", "defensive_rebound", "shooting_foul")]
    turnovers = [row["duration"] for row in possessions if row["end_reason"] in ("turnover", "shot_clock_violation")]
    continued = [row for row in possessions if row["offensive_rebounds"] > 0]
    continuations = [row["continuation_seconds"] for row in continued if row["continuation_seconds"] is not None]
    ends = Counter(row["end_reason"] for row in possessions)
    return {
        "games": len(game_ids),
        "actions_by_game": action_counts,
        "possessions": len(possessions),
        "excluded_possessions": len(excluded),
        "exclusion_reasons": dict(Counter(row["exclusion"] or row["end_reason"] for row in excluded)),
        "duration": _distribution(durations),
        "shot_endings": _distribution(shots),
        "turnover_endings": _distribution(turnovers),
        "offensive_rebound_possessions": len(continued),
        "offensive_rebounds": sum(row["offensive_rebounds"] for row in possessions),
        "continuation": _distribution(continuations),
        "period_endings": ends.get("period_end", 0),
        "shot_clock_violations": ends.get("shot_clock_violation", 0),
        "shot_clock_remaining": None,
        "shot_clock_field_present": bool(shot_clock_fields),
        "transition_halfcourt_supported": False,
        "pass_duration": None,
        "end_reasons": dict(ends),
        "rules": OBSERVATION_RULES,
    }


def sample_summary(root: Path) -> dict:
    """Summarize whatever games a stored manifest lists. It does not choose durations."""
    manifest = load_manifest(root)
    game_ids = [game["game_id"] for game in manifest["games"]]
    summary = summarize_games(game_ids, root)
    summary["source"] = manifest.get("source_name")
    summary["retrieved_on"] = manifest.get("retrieved_on")
    summary["game_ids"] = game_ids
    summary["role"] = manifest.get("role", "exploration")
    return summary


SHOT_TRIP_ENDINGS = ("made_shot", "defensive_rebound", "shooting_foul")
FIRST_TRIP_SHOT_CLOCK_SECONDS = 24.0


def shot_trip_duration_sample(root: Path = ROOT) -> dict:
    """Observed live totals for a first shot trip.

    The default directory is the exploration sample. A caller can point this at
    another stored directory, but the production residual uses the exploration
    games only. The duration is the whole trip. It is not a dribble, hold,
    pass, or screen time. Trips that continue after an offensive rebound, trips
    cut by the horn, and trips longer than 24 seconds are left out. A 24-second
    shot clock cannot contain that last group unless a reset was missed, and a
    running inbound can make the game clock longer than the shot clock.
    """
    manifest = load_manifest(root)
    durations: list[float] = []
    excluded = Counter()
    for game in manifest["games"]:
        if not game.get("actions", 1):
            excluded["no_actions"] += 1
            continue
        for row in reconstruct_possessions(load_actions(game["game_id"], root), game["game_id"]):
            if not row["reliable"] or not row["regulation"] or row["duration"] is None:
                excluded["unreliable_or_overtime"] += 1
                continue
            if row["offensive_rebounds"] > 0:
                excluded["offensive_rebound_continuation"] += 1
                continue
            if row["end_reason"] == "period_end":
                excluded["period_end"] += 1
                continue
            if row["duration"] > FIRST_TRIP_SHOT_CLOCK_SECONDS:
                excluded["longer_than_shot_clock"] += 1
                continue
            if row["end_reason"] not in SHOT_TRIP_ENDINGS:
                excluded["not_a_shot_ending"] += 1
                continue
            durations.append(float(row["duration"]))
    return {
        "durations": tuple(durations),
        "count": len(durations),
        "game_ids": [game["game_id"] for game in manifest["games"]],
        "role": manifest.get("role", "exploration"),
        "endings": list(SHOT_TRIP_ENDINGS),
        "shot_clock_limit_seconds": FIRST_TRIP_SHOT_CLOCK_SECONDS,
        "excluded": dict(excluded),
        "what_the_duration_is": "live game-clock length of one shot trip",
        "what_the_duration_is_not": "a dribble, hold, pass, or screen duration",
    }


def warriors_sample_summary(root: Path = ROOT) -> dict:
    manifest = load_manifest(root)
    game_ids = [game["game_id"] for game in manifest["games"] if game["team"] == "GSW"]
    summary = summarize_games(game_ids, root)
    summary["season"] = "2015-16"
    summary["team"] = "GSW"
    summary["source"] = manifest["source_name"]
    summary["retrieved_on"] = manifest["retrieved_on"]
    summary["game_ids"] = game_ids
    return summary


def compare_with_engine(observed: dict, engine: dict) -> dict:
    """Place observed durations beside experimental means. Pace is not a target."""
    observed_mean = observed["duration"]["mean"]
    rows = []
    for name, mean_seconds in engine.items():
        rows.append({
            "model": name,
            "mean_seconds": mean_seconds,
            "minus_observed": None if observed_mean is None else mean_seconds - observed_mean,
        })
    return {
        "observed_mean_seconds": observed_mean,
        "observed_median_seconds": observed["duration"]["p50"],
        "shot_clock_remaining_observed": observed["shot_clock_remaining"],
        "transition_compared": False,
        "models": rows,
        "pace_fitted": False,
    }
