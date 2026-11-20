"""Half-court actions before the live shoot-or-pass decision.

The advance, the pass, and the release keep their existing durations.
Dribble, hold, and screen times are uncalibrated and are not taken from the
play-by-play sample. They are not a single delay chosen to reach 14 seconds.
"""
from __future__ import annotations

import random
from collections import Counter

from .core import (
    POINTS,
    _expected_shot_value,
    clamp,
    intelligent_pass_chance,
    offensive_rebound_reset,
    probability_make,
    shot_weights,
    weighted_pick,
)
from .game_clock_experiments import REGULATION_SECONDS, GameClock, ShotClock, apply_live_action, apply_stoppage
from .live_timing_diagnostics import SEEDS, fictional_matchup
from .possession_timing_experiments import (
    HALF_COURT_STARTS,
    begin_possession,
    recover_offensive_rebound,
)

# One second each. The play-by-play feed does not record these actions.
# The values are not half of the 4-to-6-second gap, and they are not fitted.
UNCALIBRATED_ACTION_SECONDS = {
    "dribble_evaluate": 1.0,
    "hold": 1.0,
    "screen_action": 1.0,
}
MAX_PREPARATORY_ACTIONS = 3
LATE_CLOCK_NO_PREP_SECONDS = 8.0

OBSERVED_CONTEXT = {
    "sample_possessions": 676,
    "excluded_possessions": 133,
    "games": 4,
    "mean_seconds": 14.2,
    "median_seconds": 14.0,
    "p10_seconds": 6.0,
    "p90_seconds": 23.0,
    "used_as_a_duration": False,
}

PROMOTION = {"promoted": False, "integrated_with_live_engine": False}


def preparatory_action_chance(shooter, defender, shot_remaining: float) -> float:
    """Chance of one more half-court action before the live shoot-or-pass call.

    This is not intelligent_pass_chance. A good look, read by a higher-IQ
    player, makes an early decision more likely. Passing skill and defensive
    pressure make another action more likely. At 8 seconds or under, the
    live decision is used immediately.
    """
    if shot_remaining <= LATE_CLOCK_NO_PREP_SECONDS:
        return 0.0
    look = _expected_shot_value(shooter)
    recognition = clamp((shooter.iq - 70) / 40.0, 0.0, 1.0)
    early_shot = clamp((look - 0.85) * (0.5 + recognition), 0.0, 0.6)
    search = clamp((shooter.passing - 70) / 80.0, 0.0, 0.35)
    pressure = clamp((defender.defense - 70) / 80.0, 0.0, 0.35)
    return clamp(0.25 + search + pressure - early_shot, 0.0, 0.8)


def _style_bonus(coach_style: str) -> float:
    return {"perimeter": 0.08, "balanced": 0.0, "post": -0.04}.get(coach_style, 0.0)


def _fatigue(elapsed: float) -> float:
    return clamp(elapsed / REGULATION_SECONDS * 0.42, 0.0, 0.42)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _next_start(end_reason: str, steal: bool) -> str:
    if end_reason == "make":
        return "made_basket_inbound"
    if end_reason in ("shooting_foul", "shot_clock_violation"):
        return "dead_ball_inbound"
    if end_reason == "defensive_rebound":
        return "transition"
    if end_reason == "turnover":
        return "transition" if steal else "dead_ball_inbound"
    return "quarter_throw_in"


def _play_possession(game, rng, era, offense, defense, coach_style, fatigue, start_mode, action_scale: float) -> dict:
    style_bonus = _style_bonus(coach_style)
    shot = ShotClock(era.shot_clock, era.shot_clock)
    categories = Counter()
    counts = Counter()
    passes = 0
    passes_since_rebound = 0
    violations = 0
    offensive_rebounds = 0
    misses = 0
    continuation = 0.0
    in_continuation = False
    prep_used = start_mode not in HALF_COURT_STARTS
    steal = False
    end_reason = "live_ball"
    release_left = []
    shots = Counter()
    makes = Counter()
    reads = {"prep_checks": 0, "prep_chance": 0.0, "finish_checks": 0, "pass_chance": 0.0, "finish_shots": 0, "finish_passes": 0}

    def charge(seconds: float, kind: str) -> dict:
        nonlocal continuation
        action = apply_live_action(game, shot, seconds, kind)
        categories[kind] += action["game_clock_elapsed"]
        if in_continuation:
            continuation += action["game_clock_elapsed"]
        return action

    opening = begin_possession(game, shot, start_mode)
    if opening["stopped"] is not None:
        counts["stopped_throw_in"] += 1
    categories[opening["running"]["kind"]] += opening["running"]["game_clock_elapsed"]
    if categories["half_court_advance"] and categories["transition_push"]:
        raise ValueError("the bring-up was charged twice")
    if not opening["running"]["finished"]:
        end_reason = opening["running"]["reason"] or "period_expired"
        return _record(start_mode, categories, counts, passes, violations, offensive_rebounds, misses, continuation, release_left, end_reason, shot, False, shots, reads, makes)

    while True:
        if game.quarter_remaining == 0.0:
            end_reason = "period_expired"
            break
        ballhandler = rng.choices(
            offense,
            weights=[max(2, player.usage * (70 + player.iq) / 140) for player in offense],
        )[0]
        defender = defense[offense.index(ballhandler)]
        turnover_chance = clamp(
            0.085 + (defender.steals - ballhandler.iq) * 0.00065 + fatigue * 0.03,
            0.025,
            0.16,
        )
        if rng.random() < turnover_chance:
            counts["turnover"] += 1
            steal = rng.random() < 0.48
            apply_stoppage(game, "live_turnover", 0.0)
            end_reason = "turnover"
            break
        if not prep_used and not in_continuation:
            prep_end = _preparatory_actions(game, shot, rng, ballhandler, defender, charge, action_scale, reads)
            prep_used = True
            if prep_end is not None:
                end_reason = prep_end
                if prep_end == "shot_clock_violation":
                    violations += 1
                break

        passer = None
        shooter = ballhandler
        ended = False
        for _ in range(6):
            if game.quarter_remaining == 0.0:
                end_reason = "period_expired"
                ended = True
                break
            pass_chance = intelligent_pass_chance(
                shooter, defense[offense.index(shooter)], style_bonus, passes, shot.remaining
            )
            if shot.remaining <= 2:
                pass_chance = 0.0
            elif shot.remaining <= 4:
                pass_chance *= 0.15
            elif shot.remaining <= 8:
                pass_chance *= 0.50
            reads["finish_checks"] += 1
            reads["pass_chance"] += pass_chance
            if rng.random() >= pass_chance:
                reads["finish_shots"] += 1
                break
            reads["finish_passes"] += 1
            pool = [player for player in offense if player.player_id != shooter.player_id]
            recipient = rng.choices(
                pool,
                weights=[max(1.0, player.usage * (70 + player.iq) / 140) for player in pool],
            )[0]
            passer_defender = defense[offense.index(shooter)]
            recipient_defender = defense[offense.index(recipient)]
            pass_turnover = clamp(
                0.020
                + (70 - shooter.passing) * 0.001
                + (70 - shooter.iq) * 0.0005
                + (passer_defender.steals - 70) * 0.0006
                + (recipient_defender.defense - 70) * 0.0004,
                0.01,
                0.15,
            )
            if rng.random() < pass_turnover:
                counts["turnover"] += 1
                steal = rng.random() < 0.65
                apply_stoppage(game, "pass_turnover", 0.0)
                end_reason = "turnover"
                ended = True
                break
            passer = shooter
            shooter = recipient
            passes += 1
            passes_since_rebound += 1
            passed = charge(2.0, "completed_pass")
            if passed["reason"] == "period_expired":
                end_reason = "period_expired"
                ended = True
                break
        if ended:
            break
        if shot.remaining <= 0:
            violations += 1
            counts["turnover"] += 1
            end_reason = "shot_clock_violation"
            break

        shot_defender = defense[offense.index(shooter)]
        preparation = rng.uniform(1.5, 3.5)
        release_left.append(shot.remaining)
        if preparation > shot.remaining and shot.remaining <= game.quarter_remaining:
            charge(shot.remaining, "shot_clock_violation")
            violations += 1
            counts["turnover"] += 1
            end_reason = "shot_clock_violation"
            break
        if preparation > game.quarter_remaining:
            charge(preparation, "release")
            counts["period_ending_shot"] += 1
            end_reason = "period_expired"
            break
        charge(preparation, "release")
        zone = weighted_pick(rng, shot_weights(shooter, era, coach_style, 0.0))
        openness = clamp(
            (shooter.iq - shot_defender.iq) / 55 + ((passer.passing - 65) / 100 if passer else 0)
            if passer
            else (shooter.iq - shot_defender.iq) / 65,
            -0.8,
            1,
        )
        foul_chance = clamp(
            0.065 + era.physicality * 0.045 + (shot_defender.defense - 70) * -0.0003,
            0.035,
            0.16,
        )
        if rng.random() < foul_chance:
            counts["shooting_foul"] += 1
            free_throws = 3 if zone == "three" else 2
            free_throw_make = clamp(0.65 + (shooter.shot_skills["mid"] - 70) * 0.003, 0.40, 0.95)
            for _ in range(free_throws):
                apply_stoppage(game, "free_throw", 0.0)
                if rng.random() < free_throw_make:
                    counts["points"] += 1
            end_reason = "shooting_foul"
            break
        made = rng.random() < probability_make(shooter, shot_defender, era, zone, fatigue, 0.0, openness)
        counts["fga"] += 1
        shots[shooter.player_id] += 1
        if made:
            counts["make"] += 1
            makes[shooter.player_id] += 1
            counts["points"] += POINTS[zone]
            if passer and rng.random() < clamp(0.55 + (passer.passing - 70) * 0.004, 0.2, 0.88):
                counts["assist"] += 1
            end_reason = "make"
            break
        misses += 1
        if rng.random() < clamp(0.075 + (shot_defender.blocks - 65) * 0.0015, 0.015, 0.24):
            counts["block"] += 1
        if rng.random() >= 0.24 or game.quarter_remaining == 0.0:
            end_reason = "defensive_rebound"
            break
        recovery = recover_offensive_rebound(game, shot, offensive_rebound_reset(era))
        categories["oreb_recovery"] += recovery["game_clock_elapsed"]
        continuation += recovery["game_clock_elapsed"]
        if recovery["game_clock_moved_backward"]:
            raise ValueError("an offensive rebound reset the game clock")
        offensive_rebounds += 1
        if not recovery["continued"]:
            end_reason = "period_expired"
            break
        in_continuation = True
        passes_since_rebound = 0

    return _record(
        start_mode, categories, counts, passes, violations, offensive_rebounds, misses,
        continuation, release_left, end_reason, shot, steal, shots, reads, makes,
    )


def _preparatory_actions(game, shot, rng, ballhandler, defender, charge, action_scale: float, reads: dict):
    """Spend uncalibrated half-court time, then return to the live decision."""
    taken = 0
    names = tuple(UNCALIBRATED_ACTION_SECONDS)
    while taken < MAX_PREPARATORY_ACTIONS and shot.remaining > LATE_CLOCK_NO_PREP_SECONDS and game.quarter_remaining > 0:
        chance = preparatory_action_chance(ballhandler, defender, shot.remaining)
        reads["prep_checks"] += 1
        reads["prep_chance"] += chance
        if rng.random() >= chance:
            return None
        kind = rng.choice(names)
        seconds = UNCALIBRATED_ACTION_SECONDS[kind] * action_scale
        action = charge(seconds, kind)
        taken += 1
        if not action["finished"]:
            if action["reason"] == "shot_clock_expired":
                return "shot_clock_violation"
            return action["reason"] or "period_expired"
    return None


def _record(start_mode, categories, counts, passes, violations, offensive_rebounds, misses, continuation, release_left, end_reason, shot, steal, shots=None, reads=None, makes=None) -> dict:
    return {
        "start_mode": start_mode,
        "game_clock_elapsed": sum(categories.values()),
        "categories": dict(categories),
        "counts": dict(counts),
        "passes": passes,
        "shot_clock_violations": violations,
        "offensive_rebounds": offensive_rebounds,
        "misses": misses,
        "oreb_continuation_seconds": continuation,
        "release_shot_clock_remaining": release_left,
        "period_ending": end_reason == "period_expired",
        "end_reason": end_reason,
        "next_start": _next_start(end_reason, steal),
        "shot_clock_limit": shot.limit,
        "shot_by_player": dict(shots or {}),
        "make_by_player": dict(makes or {}),
        "reads": dict(reads or {}),
    }


def diagnose_game(era, seed: int, home, away, roster, action_scale: float = 1.0) -> dict:
    rng = random.Random(seed)
    game = GameClock()
    teams = (home, away)
    possessions = []
    side = 0
    start_mode = "opening_jump"
    while not game.regulation_complete:
        if game.quarter_remaining == 0.0:
            if not game.start_next_quarter():
                break
            start_mode = "quarter_throw_in"
        team = teams[side]
        opponent = teams[1 - side]
        started = game.quarter_remaining
        result = _play_possession(
            game, rng, era,
            [roster[player_id] for player_id in team.player_ids],
            [roster[player_id] for player_id in opponent.player_ids],
            team.coach_style, _fatigue(game.elapsed), start_mode, action_scale,
        )
        if game.quarter_remaining == 0.0 and started > 0:
            result["period_ending"] = True
        possessions.append(result)
        side = 1 - side
        start_mode = result["next_start"]
    durations = [possession["game_clock_elapsed"] for possession in possessions]
    return {
        "era": era.key,
        "seed": seed,
        "seeded_pace": era.pace,
        "pace_input": None,
        "action_scale": action_scale,
        "possessions_per_team": [
            sum(1 for offset, _possession in enumerate(possessions) if offset % 2 == index)
            for index in (0, 1)
        ],
        "durations": durations,
        "possession_log": possessions,
        "regulation_seconds": sum(game.quarter_lengths()),
        "promoted": False,
    }


def summarize_era(games: list[dict]) -> dict:
    possessions = [possession for game in games for possession in game["possession_log"]]
    total = len(possessions) or 1
    count = len(games)
    durations = [possession["game_clock_elapsed"] for possession in possessions]
    categories = Counter()
    counts = Counter()
    release_left = []
    for possession in possessions:
        categories.update(possession["categories"])
        counts.update(possession["counts"])
        release_left.extend(possession["release_shot_clock_remaining"])
    per_team = [sum(game["possessions_per_team"][index] for game in games) / count for index in (0, 1)]
    continued = [possession["oreb_continuation_seconds"] for possession in possessions if possession["offensive_rebounds"] > 0]
    return {
        "era": games[0]["era"],
        "seeded_pace": games[0]["seeded_pace"],
        "pace_input": None,
        "action_scale": games[0]["action_scale"],
        "possessions_per_team": per_team,
        "possessions_per_team_mean": sum(per_team) / 2,
        "mean_seconds": sum(durations) / len(durations),
        "p10_seconds": _percentile(durations, 0.10),
        "median_seconds": _percentile(durations, 0.50),
        "p90_seconds": _percentile(durations, 0.90),
        "passes_per_possession": sum(possession["passes"] for possession in possessions) / total,
        "turnovers_per_game": counts["turnover"] / count,
        "fga_per_game": counts["fga"] / count,
        "points_per_game": counts["points"] / count,
        "action_seconds_per_possession": {name: value / total for name, value in categories.items()},
        "preparatory_seconds_per_possession": sum(
            categories[name] for name in UNCALIBRATED_ACTION_SECONDS
        ) / total,
        "oreb_continuation_seconds_mean": (sum(continued) / len(continued)) if continued else None,
        "shot_clock_violations_per_game": sum(possession["shot_clock_violations"] for possession in possessions) / count,
        "quarter_endings_per_game": sum(1 for possession in possessions if possession["period_ending"]) / count,
        "mean_shot_clock_remaining_at_release": (sum(release_left) / len(release_left)) if release_left else None,
        "promoted": False,
    }


def run_halfcourt(eras, seeds=SEEDS, action_scale: float = 1.0) -> list[dict]:
    home, away, roster = fictional_matchup()
    rows = []
    for era in eras:
        games = [diagnose_game(era, seed, home, away, roster, action_scale) for seed in seeds]
        rows.append(summarize_era(games))
    return rows


def action_duration_sensitivity(era, seeds=SEEDS, scales=(0.5, 1.0, 2.0)) -> list[dict]:
    """Show the effect of the uncalibrated seconds. None of these scales is a fitted pace."""
    home, away, roster = fictional_matchup()
    rows = []
    for scale in scales:
        games = [diagnose_game(era, seed, home, away, roster, scale) for seed in seeds]
        summary = summarize_era(games)
        summary["uncalibrated"] = True
        summary["fitted_to_observed_mean"] = False
        rows.append(summary)
    return rows
