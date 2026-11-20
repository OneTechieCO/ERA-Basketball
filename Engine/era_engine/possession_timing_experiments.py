"""Possession phases on the experimental game clock.

The live 6-second setup is one shot-clock charge. The previous diagnostic copied
it onto every new possession. This experiment does not stack a throw-in or a
transition push on top of that charge.

No play-by-play timestamps are stored in this project. Durations that are not
the live shot-clock rules are marked uncalibrated. None of them is chosen to
hit a seeded pace.
"""
from __future__ import annotations

import random
from collections import Counter

from .core import (
    clamp,
    intelligent_pass_chance,
    offensive_rebound_reset,
    probability_make,
    shot_weights,
    weighted_pick,
)
from .game_clock_experiments import (
    REGULATION_SECONDS,
    GameClock,
    ShotClock,
    apply_live_action,
    apply_stoppage,
    setup_seconds,
)
from .live_timing_diagnostics import SEEDS, fictional_matchup, run_live_timing

# The live setup constant, reused as a backcourt advance only.
# min(6, one quarter of the shot clock). Not a measured dribble time.
HALF_COURT_ADVANCE_IS_LIVE_SETUP = True

# One live pass-length, used where a 6-second walk-up does not happen.
# Uncalibrated. Not fitted to pace.
TRANSITION_PUSH_SECONDS = 2.0

# Game clock only, between a release and an offensive-rebound reset.
# The shot clock is already stopped on the release. Uncalibrated.
OREB_RECOVERY_SECONDS = 1.0

SETUP_AUDIT = {
    "live_charge": "min(6, shot_clock * 0.25) on the shot clock at the start of every possession",
    "previous_diagnostic": "That charge was also taken off the game clock for every new possession",
    "double_count_if_stacked": (
        "A throw-in, a transition push, and the 6-second charge are the same bring-up "
        "if they are all applied to one trip"
    ),
    "this_experiment": (
        "Backcourt throw-ins and the opening tap use the 6-second charge once. "
        "A steal or defensive rebound uses a 2-second push instead. "
        "An offensive rebound uses neither."
    ),
}

TIMING_PARAMETERS = {
    "half_court_advance_seconds": {
        "value": "setup_seconds(shot_clock)",
        "calibrated": False,
        "provenance": "Live shot-clock setup. Relabeled as the backcourt advance. Not a play-by-play measurement.",
    },
    "transition_push_seconds": {
        "value": TRANSITION_PUSH_SECONDS,
        "calibrated": False,
        "provenance": "Uncalibrated. Equal to one live completed-pass charge. Replaces the 6-second advance.",
    },
    "oreb_recovery_seconds": {
        "value": OREB_RECOVERY_SECONDS,
        "calibrated": False,
        "runs_shot_clock": False,
        "provenance": "Uncalibrated. No play-by-play timestamps are available in this project.",
    },
    "throw_in_running_seconds": {
        "value": 0.0,
        "calibrated": False,
        "provenance": (
            "NBA rule mechanic: a throw-in does not run the game clock. "
            "The clock starts when a player on the court legally touches the ball. "
            "No stored timestamp measures the inbounder's hold."
        ),
    },
    "completed_pass_seconds": {
        "value": 2.0,
        "calibrated": False,
        "provenance": "Live shot-clock rule.",
    },
    "release_seconds": {
        "value": "uniform(1.5, 3.5)",
        "calibrated": False,
        "provenance": "Live shot-clock rule.",
    },
}

NO_PLAY_BY_PLAY = (
    "Season totals and pace rates are stored. Play-by-play timestamps are not. "
    "No duration here was read from a game log."
)

PROMOTION = {"promoted": False, "integrated_with_live_engine": False}

HALF_COURT_STARTS = ("opening_jump", "made_basket_inbound", "dead_ball_inbound", "quarter_throw_in")
STOPPED_THROW_IN_STARTS = ("made_basket_inbound", "dead_ball_inbound", "quarter_throw_in")


def _fatigue(elapsed: float) -> float:
    return clamp(elapsed / REGULATION_SECONDS * 0.42, 0.0, 0.42)


def _style_bonus(coach_style: str) -> float:
    return {"perimeter": 0.08, "balanced": 0.0, "post": -0.04}.get(coach_style, 0.0)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def begin_possession(game: GameClock, shot: ShotClock, start_mode: str) -> dict:
    """Start a trip. A stopped throw-in adds no running time, and only one advance is charged."""
    if start_mode not in HALF_COURT_STARTS and start_mode != "transition":
        raise ValueError(f"unknown possession start: {start_mode}")
    stopped = None
    if start_mode in STOPPED_THROW_IN_STARTS:
        stopped = apply_stoppage(game, "throw_in", 0.0)
    if start_mode == "transition":
        running = apply_live_action(game, shot, TRANSITION_PUSH_SECONDS, "transition_push")
    else:
        running = apply_live_action(game, shot, setup_seconds(shot.limit), "half_court_advance")
    if stopped is not None and running["kind"] == "transition_push":
        raise ValueError("a throw-in and a transition push must not be charged together")
    if running["kind"] == "half_court_advance" and start_mode == "transition":
        raise ValueError("transition must not charge the half-court advance")
    return {"stopped": stopped, "running": running, "start_mode": start_mode}


def recover_offensive_rebound(game: GameClock, shot: ShotClock, reset_seconds: float) -> dict:
    """Secure the miss. The game clock moves. The shot clock stays put, then resets."""
    before_game = game.quarter_remaining
    before_shot = shot.remaining
    elapsed = game.advance_live(min(OREB_RECOVERY_SECONDS, game.quarter_remaining))
    if shot.remaining != before_shot:
        raise ValueError("offensive-rebound recovery must not run the shot clock")
    continued = elapsed + 1e-9 >= OREB_RECOVERY_SECONDS and game.quarter_remaining > 0.0
    if continued:
        shot.reset(reset_seconds)
    return {
        "kind": "oreb_recovery",
        "game_clock_elapsed": elapsed,
        "shot_clock_elapsed": 0.0,
        "stopped_clock": False,
        "continued": continued,
        "game_clock_moved_backward": game.quarter_remaining > before_game,
    }


def _play_possession(game, rng, era, offense, defense, coach_style, fatigue, start_mode) -> dict:
    style_bonus = _style_bonus(coach_style)
    shot = ShotClock(era.shot_clock, era.shot_clock)
    categories = Counter()
    counts = Counter()
    passes = 0
    passes_since_rebound = 0
    violations = 0
    offensive_rebounds = 0
    misses = 0
    putbacks = 0
    release_left = []
    continuation = 0.0
    in_continuation = False
    steal = False
    end_reason = "live_ball"
    release_started = False

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
        raise ValueError("half-court advance and transition were both charged")
    if not opening["running"]["finished"]:
        end_reason = opening["running"]["reason"] or "period_expired"
        return _record(
            start_mode, categories, counts, passes, violations, offensive_rebounds, misses,
            putbacks, release_left, continuation, end_reason, shot, False,
        )

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
            counts["live_turnover"] += 1
            steal = rng.random() < 0.48
            if steal:
                counts["steal"] += 1
            apply_stoppage(game, "live_turnover", 0.0)
            end_reason = "turnover"
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
            if rng.random() >= pass_chance:
                break
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
                counts["pass_turnover"] += 1
                steal = rng.random() < 0.65
                if steal:
                    counts["steal"] += 1
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
            end_reason = "shot_clock_violation"
            break

        shot_defender = defense[offense.index(shooter)]
        preparation = rng.uniform(1.5, 3.5)
        release_left.append(shot.remaining)
        release_started = True
        if preparation > shot.remaining and shot.remaining <= game.quarter_remaining:
            # The whistle comes when the shot clock reaches zero, so the game clock runs to that horn.
            charge(shot.remaining, "shot_clock_violation")
            violations += 1
            end_reason = "shot_clock_violation"
            break
        if preparation > game.quarter_remaining:
            charge(preparation, "release")
            counts["period_ending_shot"] += 1
            end_reason = "period_expired"
            break
        charge(preparation, "release")
        if in_continuation and passes_since_rebound == 0:
            putbacks += 1
            counts["putback"] += 1
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
            counts["free_throw"] += free_throws
            for _ in range(free_throws):
                apply_stoppage(game, "free_throw", 0.0)
                rng.random()
            end_reason = "shooting_foul"
            break
        if rng.random() < probability_make(shooter, shot_defender, era, zone, fatigue, 0.0, openness):
            counts["make"] += 1
            if passer and rng.random() < clamp(0.55 + (passer.passing - 70) * 0.004, 0.2, 0.88):
                counts["assist"] += 1
            end_reason = "make"
            break
        misses += 1
        counts["miss"] += 1
        if rng.random() < clamp(0.075 + (shot_defender.blocks - 65) * 0.0015, 0.015, 0.24):
            counts["block"] += 1
        if rng.random() >= 0.24 or game.quarter_remaining == 0.0:
            counts["defensive_rebound"] += 1
            end_reason = "defensive_rebound"
            break
        recovery = recover_offensive_rebound(game, shot, offensive_rebound_reset(era))
        categories["oreb_recovery"] += recovery["game_clock_elapsed"]
        continuation += recovery["game_clock_elapsed"]
        if recovery["game_clock_moved_backward"]:
            raise ValueError("an offensive rebound reset the game clock")
        offensive_rebounds += 1
        counts["offensive_rebound"] += 1
        if not recovery["continued"]:
            end_reason = "period_expired"
            break
        in_continuation = True
        passes_since_rebound = 0

    return _record(
        start_mode, categories, counts, passes, violations, offensive_rebounds, misses,
        putbacks, release_left, continuation, end_reason, shot, steal, release_started,
    )


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


def _record(
    start_mode, categories, counts, passes, violations, offensive_rebounds, misses,
    putbacks, release_left, continuation, end_reason, shot, steal, release_started=False,
) -> dict:
    game_elapsed = sum(categories.values())
    return {
        "start_mode": start_mode,
        "game_clock_elapsed": game_elapsed,
        "categories": dict(categories),
        "counts": dict(counts),
        "passes": passes,
        "shot_clock_violations": violations,
        "offensive_rebounds": offensive_rebounds,
        "misses": misses,
        "putbacks": putbacks,
        "release_shot_clock_remaining": release_left,
        "oreb_continuation_seconds": continuation,
        "period_ending": end_reason == "period_expired",
        "period_ending_shot": bool(counts.get("period_ending_shot")),
        "end_reason": end_reason,
        "next_start": _next_start(end_reason, steal),
        "shot_clock_remaining": shot.remaining,
        "shot_clock_limit": shot.limit,
        "release_started": release_started,
        "stopped_running_seconds": 0.0,
    }


def diagnose_game(era, seed: int, home, away, roster) -> dict:
    """Play 48 minutes. Pace does not choose the number of possessions."""
    rng = random.Random(seed)
    game = GameClock()
    teams = (home, away)
    possessions = []
    side = 0
    start_mode = "opening_jump"
    period_breaks = 0
    while not game.regulation_complete:
        if game.quarter_remaining == 0.0:
            if not game.start_next_quarter():
                break
            period_breaks += 1
            start_mode = "quarter_throw_in"
        team = teams[side]
        opponent = teams[1 - side]
        started_remaining = game.quarter_remaining
        result = _play_possession(
            game,
            rng,
            era,
            [roster[player_id] for player_id in team.player_ids],
            [roster[player_id] for player_id in opponent.player_ids],
            team.coach_style,
            _fatigue(game.elapsed),
            start_mode,
        )
        result["quarter"] = game.quarter if game.quarter_remaining > 0 else game.quarter
        if game.quarter_remaining == 0.0 and started_remaining > 0:
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
        "possessions": len(possessions),
        "possessions_per_team": [
            sum(1 for offset, _possession in enumerate(possessions) if offset % 2 == index)
            for index in (0, 1)
        ],
        "durations": durations,
        "mean_seconds": (sum(durations) / len(durations)) if durations else 0.0,
        "passes": sum(possession["passes"] for possession in possessions),
        "shot_clock_violations": sum(possession["shot_clock_violations"] for possession in possessions),
        "period_ending_possessions": sum(1 for possession in possessions if possession["period_ending"]),
        "period_breaks": period_breaks,
        "quarter_lengths": game.quarter_lengths(),
        "regulation_seconds": sum(game.quarter_lengths()),
        "possession_log": possessions,
        "promoted": False,
    }


def _duration_distribution(durations: list[float]) -> dict:
    return {
        "min": min(durations) if durations else 0.0,
        "p50": _percentile(durations, 0.50),
        "p90": _percentile(durations, 0.90),
        "max": max(durations) if durations else 0.0,
    }


def summarize_era(games: list[dict]) -> dict:
    count = len(games)
    possessions = [possession for game in games for possession in game["possession_log"]]
    total = len(possessions) or 1
    durations = [possession["game_clock_elapsed"] for possession in possessions]
    categories = Counter()
    by_mode = {}
    release_left = []
    for possession in possessions:
        categories.update(possession["categories"])
        release_left.extend(possession["release_shot_clock_remaining"])
        by_mode.setdefault(possession["start_mode"], []).append(possession["game_clock_elapsed"])
    misses = sum(possession["misses"] for possession in possessions)
    offensive_rebounds = sum(possession["offensive_rebounds"] for possession in possessions)
    transition_possessions = [possession for possession in possessions if possession["start_mode"] == "transition"]
    half_court_possessions = [possession for possession in possessions if possession["start_mode"] in HALF_COURT_STARTS]
    per_team = [
        sum(game["possessions_per_team"][index] for game in games) / count for index in (0, 1)
    ]
    mean_seconds = sum(durations) / len(durations)
    advance_gap = setup_seconds(24) - TRANSITION_PUSH_SECONDS
    return {
        "era": games[0]["era"],
        "games": count,
        "seeded_pace": games[0]["seeded_pace"],
        "pace_input": None,
        "possessions_per_team": per_team,
        "possessions_per_team_mean": sum(per_team) / 2,
        "mean_seconds": mean_seconds,
        "median_seconds": _percentile(durations, 0.50),
        "distribution": _duration_distribution(durations),
        "passes_per_possession": sum(possession["passes"] for possession in possessions) / total,
        "shot_clock_violations_per_game": sum(game["shot_clock_violations"] for game in games) / count,
        "quarter_ending_possessions_per_game": sum(game["period_ending_possessions"] for game in games) / count,
        "mean_shot_clock_remaining_at_release": (sum(release_left) / len(release_left)) if release_left else None,
        "transition_seconds_per_possession": categories["transition_push"] / total,
        "half_court_seconds_per_possession": categories["half_court_advance"] / total,
        "transition_possession_share": len(transition_possessions) / total,
        "half_court_possession_share": len(half_court_possessions) / total,
        "mean_seconds_by_start": {
            mode: (sum(values) / len(values)) for mode, values in sorted(by_mode.items())
        },
        "oreb_recovery_seconds_per_possession": categories["oreb_recovery"] / total,
        "oreb_continuation_seconds_per_possession": (
            sum(possession["oreb_continuation_seconds"] for possession in possessions) / total
        ),
        "oreb_continuation_seconds_per_rebound": (
            sum(possession["oreb_continuation_seconds"] for possession in possessions) / offensive_rebounds
            if offensive_rebounds else None
        ),
        "offensive_rebound_rate": (offensive_rebounds / misses) if misses else None,
        "stopped_throw_ins_per_game": sum(
            possession["counts"].get("stopped_throw_in", 0) for possession in possessions
        ) / count,
        "category_seconds_per_possession": {name: value / total for name, value in categories.items()},
        "category_seconds_per_game": {name: value / count for name, value in categories.items()},
        "seconds_if_seeded_pace_were_the_result": REGULATION_SECONDS / (2 * games[0]["seeded_pace"]),
        "transition_shortening_vs_blanket_setup": (
            len(transition_possessions) / total * advance_gap
        ),
        "promoted": False,
    }


def run_possession_timing(eras, seeds=SEEDS) -> list[dict]:
    home, away, roster = fictional_matchup()
    rows = []
    for era in eras:
        games = [diagnose_game(era, seed, home, away, roster) for seed in seeds]
        rows.append(summarize_era(games))
    return rows


def compare_with_live_diagnostic(eras, seeds=SEEDS) -> list[dict]:
    """Measure this split against the diagnostic that charged 6 seconds on every trip."""
    current = {row["era"]: row for row in run_possession_timing(eras, seeds)}
    previous = {row["era"]: row for row in run_live_timing(eras, seeds)}
    compared = []
    for era in eras:
        now = current[era.key]
        before = previous[era.key]
        mechanical = (
            -now["transition_shortening_vs_blanket_setup"]
            + now["oreb_recovery_seconds_per_possession"]
        )
        delta = now["mean_seconds"] - before["mean_seconds"]
        compared.append({
            "era": era.key,
            "seeded_pace": era.pace,
            "previous_possessions_per_team": before["possessions_per_team_mean"],
            "possessions_per_team": now["possessions_per_team"],
            "possessions_per_team_mean": now["possessions_per_team_mean"],
            "previous_mean_seconds": before["mean_seconds"],
            "mean_seconds": now["mean_seconds"],
            "median_seconds": now["median_seconds"],
            "delta_seconds": delta,
            "transition_shortening_seconds": now["transition_shortening_vs_blanket_setup"],
            "oreb_recovery_seconds_per_possession": now["oreb_recovery_seconds_per_possession"],
            "oreb_continuation_seconds_per_possession": now["oreb_continuation_seconds_per_possession"],
            "mechanical_seconds": mechanical,
            "residual_seconds": delta - mechanical,
            "shot_clock_left_at_release": now["mean_shot_clock_remaining_at_release"],
            "passes_per_possession": now["passes_per_possession"],
            "shot_clock_violations_per_game": now["shot_clock_violations_per_game"],
            "quarter_endings_per_game": now["quarter_ending_possessions_per_game"],
            "pace_gap": now["possessions_per_team_mean"] - era.pace,
            "seconds_short_of_seeded_pace": now["seconds_if_seeded_pace_were_the_result"] - now["mean_seconds"],
            "mean_seconds_by_start": now["mean_seconds_by_start"],
            "transition_possession_share": now["transition_possession_share"],
            "half_court_seconds_per_possession": now["half_court_seconds_per_possession"],
            "transition_seconds_per_possession": now["transition_seconds_per_possession"],
            "promoted": False,
            "pace_input": None,
        })
    return compared
