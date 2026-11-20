"""Variant B2. An offensive action can change the next look.

The live make formula is still probability_make. A screen, dribble, or hold
may change the openness input before that formula clamps it. A pass changes
the shooter. None of these assign a made shot, an assist, or a turnover.
The three-action cap from Variant B is not used. A safeguard stops a chain
that never resolves, and the shot clock still ends a possession.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from .action_decision_experiments import (
    EARLY_RELEASE_SECONDS,
    diagnose_game as variant_game,
    observed_duration_sample,
    summarize_games,
    validation_report,
)
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
from .halfcourt_action_experiments import (
    HALF_COURT_STARTS,
    UNCALIBRATED_ACTION_SECONDS,
    begin_possession,
    diagnose_game as baseline_game,
    recover_offensive_rebound,
)
from .live_timing_diagnostics import SEEDS, fictional_matchup
from .playbyplay_timing import ROOT

DEVELOPMENT_SECONDS = dict(UNCALIBRATED_ACTION_SECONDS)
URGENT_SECONDS = 8.0
# Loop guard. It is not a possession length and it is not Variant B's cap of 3.
SAFEGUARD_ACTIONS = 8

# Added to the live openness input. Not measured from play-by-play.
# probability_make then applies its own 0.035 coefficient and [-0.8, 1] clamp.
OPENNESS_ASSUMPTIONS = {
    "screen_success": 0.22,
    "dribble_separation": 0.12,
    "dribble_contest": -0.10,
    "hold": 0.05,
    "source": "unverified",
    "changes_make_formula": False,
}

PROMOTION = {"promoted": False, "integrated_with_live_engine": False}
FITTED_TO_HELD_OUT = False


def live_openness(shooter, defender, passer) -> float:
    """The openness expression copied from the live possession, before its clamp."""
    if passer:
        return (shooter.iq - defender.iq) / 55 + (passer.passing - 65) / 100
    return (shooter.iq - defender.iq) / 65


def shot_openness(shooter, defender, passer, created_openness: float) -> float:
    """Live openness plus the experimental action input, then the live clamp."""
    return clamp(live_openness(shooter, defender, passer) + created_openness, -0.8, 1.0)


def screen_success_chance(shooter, defender) -> float:
    """Whether the screen frees the ballhandler. Uncalibrated. It does not make the shot."""
    return clamp(
        0.45 + (shooter.iq - 70) * 0.004 + (shooter.passing - 70) * 0.003 - (defender.defense - 70) * 0.004,
        0.15,
        0.80,
    )


def dribble_separation_chance(shooter, defender) -> float:
    """Whether the dribble creates space. Otherwise the defender stays attached."""
    return clamp(
        0.40 + (shooter.iq - 70) * 0.005 - (defender.defense - 70) * 0.004,
        0.15,
        0.75,
    )


def b2_continue_chance(shooter, defender, shot_remaining: float, created_openness: float, pressure_extra: float, actions: int) -> float:
    """Whether to develop again. Quality, pressure, clock, IQ, and role decide.

    At 8 seconds, or after the safeguard, the chance is zero. A created opening
    lowers the chance for a player who can recognize it. Pressure raises it.
    """
    if shot_remaining <= URGENT_SECONDS or actions >= SAFEGUARD_ACTIONS:
        return 0.0
    recognition = clamp((shooter.iq - 70) / 40.0, 0.0, 1.0)
    pressure = clamp((defender.defense - 70) / 25.0 + pressure_extra, 0.0, 1.5)
    saw_opening = clamp(created_openness / 0.25, 0.0, 1.0) * (0.25 + 0.55 * recognition)
    look_gap = max(0.0, 1.02 - _expected_shot_value(shooter))
    search = (0.12 + 0.20 * pressure + 0.15 * look_gap) * (0.55 + 0.45 * recognition)
    role_finish = clamp((shooter.usage - 0.19) * 0.4, 0.0, 0.10) * clamp(created_openness / 0.20, 0.0, 1.0)
    time_room = clamp((shot_remaining - 10.0) / 28.0, 0.0, 0.22)
    repetition = 0.07 * max(0, actions - 1) * (1.0 - min(1.0, max(created_openness, 0.0) / 0.20))
    chance = 0.28 + time_room + search - saw_opening - role_finish - repetition
    return clamp(chance, 0.0, 0.82)


def possession_state(shooter, defender, shot_remaining: float, created_openness: float, pressure_extra: float) -> str:
    if shot_remaining <= URGENT_SECONDS:
        return "urgent"
    pressure = clamp((defender.defense - 70) / 25.0 + pressure_extra, 0.0, 1.5)
    if created_openness >= 0.15:
        return "opportunity_created"
    if pressure >= 0.70:
        return "defensive_pressure"
    return "developing"


def b2_action_weights(shooter, defender, created_openness: float, pressure_extra: float) -> dict[str, float]:
    pressure = max(0.0, (defender.defense - 70) / 25.0 + pressure_extra)
    dribble_iq = max(0.0, (shooter.iq - 70) / 30.0)
    return {
        "dribble_evaluate": 0.70 + dribble_iq * (0.40 if pressure >= 0.80 else 1.0),
        "hold": 0.30,
        "screen_action": 0.35 + max(0.0, (shooter.passing - 70) / 22.0) + 0.45 * pressure,
        "pass": 0.40 + max(0.0, (shooter.passing - 70) / 16.0) + 0.35 * pressure,
    }


def apply_action_effect(kind: str, shooter, defender, created_openness: float, pressure_extra: float, succeeded: bool) -> tuple[float, float]:
    """Update the experimental inputs. The live make formula is not called here."""
    if kind == "screen_action":
        if succeeded:
            created_openness += OPENNESS_ASSUMPTIONS["screen_success"]
            pressure_extra = max(0.0, pressure_extra - 0.20)
        else:
            pressure_extra += 0.35
    elif kind == "dribble_evaluate":
        if succeeded:
            created_openness += OPENNESS_ASSUMPTIONS["dribble_separation"]
            pressure_extra = max(0.0, pressure_extra - 0.10)
        else:
            created_openness += OPENNESS_ASSUMPTIONS["dribble_contest"]
            pressure_extra += 0.25
    elif kind == "hold":
        recognition = clamp((shooter.iq - 70) / 40.0, 0.0, 1.0)
        created_openness += OPENNESS_ASSUMPTIONS["hold"] * (0.4 + 0.6 * recognition)
    else:
        pressure_extra = max(0.0, pressure_extra - 0.15)
    return clamp(created_openness, -0.30, 0.45), clamp(pressure_extra, 0.0, 1.5)


def _style_bonus(coach_style: str) -> float:
    return {"perimeter": 0.08, "balanced": 0.0, "post": -0.04}.get(coach_style, 0.0)


def _fatigue(elapsed: float) -> float:
    return clamp(elapsed / REGULATION_SECONDS * 0.42, 0.0, 0.42)


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


def _stop_choice(created_openness: float, pressure_extra: float, defender, recognition: float) -> str:
    pressure = clamp((defender.defense - 70) / 25.0 + pressure_extra, 0.0, 1.5)
    if created_openness >= 0.12 and recognition >= 0.40:
        return "recognized_opening"
    if pressure >= 0.70:
        return "search_ended"
    return "finish_choice"


def _play_possession(game, rng, era, offense, defense, coach_style, fatigue, start_mode) -> dict:
    style_bonus = _style_bonus(coach_style)
    shot = ShotClock(era.shot_clock, era.shot_clock)
    categories = Counter()
    counts = Counter()
    shots = Counter()
    makes = Counter()
    reads = Counter()
    passes = 0
    passes_since_rebound = 0
    violations = 0
    offensive_rebounds = 0
    misses = 0
    continuation = 0.0
    in_continuation = False
    steal = False
    end_reason = "live_ball"
    release_left = []
    created_openness = 0.0
    pressure_extra = 0.0
    developed = 0
    opener_look = None
    finish_openness = []

    def charge(seconds: float, kind: str) -> dict:
        nonlocal continuation
        action = apply_live_action(game, shot, seconds, kind)
        categories[kind] += action["game_clock_elapsed"]
        if in_continuation:
            continuation += action["game_clock_elapsed"]
        return action

    def live_pass(passer):
        pool = [player for player in offense if player.player_id != passer.player_id]
        recipient = rng.choices(
            pool,
            weights=[max(1.0, player.usage * (70 + player.iq) / 140) for player in pool],
        )[0]
        passer_defender = defense[offense.index(passer)]
        recipient_defender = defense[offense.index(recipient)]
        pass_turnover = clamp(
            0.020
            + (70 - passer.passing) * 0.001
            + (70 - passer.iq) * 0.0005
            + (passer_defender.steals - 70) * 0.0006
            + (recipient_defender.defense - 70) * 0.0004,
            0.01,
            0.15,
        )
        if rng.random() < pass_turnover:
            return {"status": "turnover", "steal": rng.random() < 0.65}
        passed = charge(2.0, "completed_pass")
        if passed["reason"] == "period_expired":
            return {"status": "period"}
        return {"status": "complete", "recipient": recipient}

    opening = begin_possession(game, shot, start_mode)
    if opening["stopped"] is not None:
        counts["stopped_throw_in"] += 1
    categories[opening["running"]["kind"]] += opening["running"]["game_clock_elapsed"]
    if not opening["running"]["finished"]:
        end_reason = opening["running"]["reason"] or "period_expired"
        return _record(locals())

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

        shooter = ballhandler
        passer = None
        if start_mode in HALF_COURT_STARTS and not in_continuation:
            if opener_look is None:
                opener_look = _expected_shot_value(shooter)
            step = _develop_b2(
                game, shot, rng, offense, defense, shooter, charge, live_pass, reads,
                created_openness, pressure_extra,
            )
            shooter = step["shooter"]
            passer = step["passer"]
            passes += step["passes"]
            passes_since_rebound += step["passes"]
            developed += step["actions"]
            created_openness = step["created_openness"]
            pressure_extra = step["pressure_extra"]
            if step["end"] == "turnover":
                counts["turnover"] += 1
                steal = step["steal"]
                apply_stoppage(game, "pass_turnover", 0.0)
                end_reason = "turnover"
                break
            if step["end"] is not None:
                end_reason = step["end"]
                if step["end"] == "shot_clock_violation":
                    violations += 1
                    counts["turnover"] += 1
                break
        elif start_mode == "transition" and not in_continuation:
            reads["early_opportunity"] += 1

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
            if rng.random() >= pass_chance:
                reads["finish_shots"] += 1
                break
            reads["finish_passes"] += 1
            outcome = live_pass(shooter)
            if outcome["status"] == "turnover":
                counts["turnover"] += 1
                steal = outcome["steal"]
                apply_stoppage(game, "pass_turnover", 0.0)
                end_reason = "turnover"
                ended = True
                break
            if outcome["status"] == "period":
                end_reason = "period_expired"
                ended = True
                break
            passer = shooter
            shooter = outcome["recipient"]
            passes += 1
            passes_since_rebound += 1
            created_openness, pressure_extra = apply_action_effect(
                "pass", shooter, defense[offense.index(shooter)], created_openness, pressure_extra, True
            )
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
            end_reason = "period_expired"
            break
        charge(preparation, "release")
        zone = weighted_pick(rng, shot_weights(shooter, era, coach_style, 0.0))
        openness = shot_openness(shooter, shot_defender, passer, created_openness)
        finish_openness.append(openness)
        reads["created_openness_sum"] += created_openness
        reads["created_openness_n"] += 1
        if created_openness >= 0.10:
            reads["created_shots"] += 1
            reads["created_open_sum"] += created_openness
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
        if created_openness >= 0.10:
            reads["created_fga"] += 1
            if made:
                reads["created_makes"] += 1
        else:
            reads["plain_fga"] += 1
            if made:
                reads["plain_makes"] += 1
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
        offensive_rebounds += 1
        if not recovery["continued"]:
            end_reason = "period_expired"
            break
        in_continuation = True
        passes_since_rebound = 0
        created_openness = 0.0
        pressure_extra = 0.0

    if opener_look is not None and release_left:
        final_look = _expected_shot_value(shooter)
        reads["look_start"] += opener_look
        reads["look_end"] += final_look
        reads["look_n"] += 1
        if final_look > opener_look + 0.02:
            reads["look_up"] += 1
    return _record(locals())


def _develop_b2(game, shot, rng, offense, defense, shooter, charge, live_pass, reads, created_openness, pressure_extra):
    taken = 0
    added_passes = 0
    passer = None
    while game.quarter_remaining > 0 and shot.remaining > 0:
        defender = defense[offense.index(shooter)]
        if taken == 0:
            reads[f"state_{possession_state(shooter, defender, shot.remaining, created_openness, pressure_extra)}"] += 1
        if shot.remaining <= URGENT_SECONDS:
            reads["stop_urgent"] += 1
            reads[f"chain_{taken}"] += 1
            break
        if taken >= SAFEGUARD_ACTIONS:
            reads["stop_safeguard"] += 1
            reads[f"chain_{taken}"] += 1
            break
        chance = b2_continue_chance(shooter, defender, shot.remaining, created_openness, pressure_extra, taken)
        reads["continue_checks"] += 1
        reads["continue_chance"] += chance
        if rng.random() >= chance:
            recognition = clamp((shooter.iq - 70) / 40.0, 0.0, 1.0)
            reads["stop_" + _stop_choice(created_openness, pressure_extra, defender, recognition)] += 1
            reads[f"chain_{taken}"] += 1
            break
        kind = weighted_pick(rng, b2_action_weights(shooter, defender, created_openness, pressure_extra))
        reads[kind] += 1
        if kind == "pass":
            outcome = live_pass(shooter)
            if outcome["status"] == "turnover":
                reads["stop_turnover"] += 1
                reads[f"chain_{taken}"] += 1
                return _b2_step(shooter, passer, added_passes, taken, "turnover", outcome["steal"], created_openness, pressure_extra)
            if outcome["status"] == "period":
                reads["stop_clock"] += 1
                reads[f"chain_{taken}"] += 1
                return _b2_step(shooter, passer, added_passes, taken, "period_expired", False, created_openness, pressure_extra)
            passer = shooter
            shooter = outcome["recipient"]
            added_passes += 1
            created_openness, pressure_extra = apply_action_effect(
                "pass", shooter, defense[offense.index(shooter)], created_openness, pressure_extra, True
            )
        else:
            succeeded = True
            if kind == "screen_action":
                succeeded = rng.random() < screen_success_chance(shooter, defender)
                reads["screen_success" if succeeded else "screen_fail"] += 1
            elif kind == "dribble_evaluate":
                succeeded = rng.random() < dribble_separation_chance(shooter, defender)
                reads["dribble_separation" if succeeded else "dribble_contest"] += 1
            action = charge(DEVELOPMENT_SECONDS[kind], kind)
            created_openness, pressure_extra = apply_action_effect(
                kind, shooter, defender, created_openness, pressure_extra, succeeded
            )
            if not action["finished"]:
                reason = "shot_clock_violation" if action["reason"] == "shot_clock_expired" else (action["reason"] or "period_expired")
                reads["stop_clock"] += 1
                reads[f"chain_{taken + 1}"] += 1
                return _b2_step(shooter, passer, added_passes, taken + 1, reason, False, created_openness, pressure_extra)
        taken += 1
    return _b2_step(shooter, passer, added_passes, taken, None, False, created_openness, pressure_extra)


def _b2_step(shooter, passer, passes, actions, end, steal, created_openness, pressure_extra) -> dict:
    return {
        "shooter": shooter,
        "passer": passer,
        "passes": passes,
        "actions": actions,
        "end": end,
        "steal": steal,
        "created_openness": created_openness,
        "pressure_extra": pressure_extra,
    }


def _record(scope: dict) -> dict:
    return {
        "start_mode": scope["start_mode"],
        "game_clock_elapsed": sum(scope["categories"].values()),
        "categories": dict(scope["categories"]),
        "counts": dict(scope["counts"]),
        "shot_by_player": dict(scope["shots"]),
        "make_by_player": dict(scope["makes"]),
        "reads": {key: value for key, value in scope["reads"].items()},
        "passes": scope["passes"],
        "shot_clock_violations": scope["violations"],
        "offensive_rebounds": scope["offensive_rebounds"],
        "misses": scope["misses"],
        "oreb_continuation_seconds": scope["continuation"],
        "release_shot_clock_remaining": scope["release_left"],
        "development_actions": scope["developed"],
        "period_ending": scope["end_reason"] == "period_expired",
        "end_reason": scope["end_reason"],
        "next_start": _next_start(scope["end_reason"], scope["steal"]),
        "shot_clock_limit": scope["shot"].limit,
        "created_openness": scope["created_openness"],
    }


def diagnose_game(era, seed: int, home, away, roster) -> dict:
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
            team.coach_style, _fatigue(game.elapsed), start_mode,
        )
        if game.quarter_remaining == 0.0 and started > 0:
            result["period_ending"] = True
        possessions.append(result)
        side = 1 - side
        start_mode = result["next_start"]
    return {
        "era": era.key,
        "seed": seed,
        "variant": "B2",
        "seeded_pace": era.pace,
        "pace_input": None,
        "possessions_per_team": [
            sum(1 for offset, _possession in enumerate(possessions) if offset % 2 == index)
            for index in (0, 1)
        ],
        "durations": [possession["game_clock_elapsed"] for possession in possessions],
        "possession_log": possessions,
        "regulation_seconds": sum(game.quarter_lengths()),
        "promoted": False,
        "fitted_to_held_out": False,
    }


def _baseline_summary(games: list[dict]) -> dict:
    for game in games:
        game["variant"] = "baseline"
        for possession in game["possession_log"]:
            possession.setdefault("make_by_player", {})
            possession.setdefault("development_actions", 0)
    return summarize_games(games)


def compare_models(eras, seeds=SEEDS) -> list[dict]:
    home, away, roster = fictional_matchup()
    rows = []
    for era in eras:
        baseline = [baseline_game(era, seed, home, away, roster) for seed in seeds]
        variant_b = [variant_game(era, seed, home, away, roster, "B") for seed in seeds]
        variant_b2 = [diagnose_game(era, seed, home, away, roster) for seed in seeds]
        for game in variant_b:
            game["variant"] = "B"
        rows.append({
            "era": era.key,
            "baseline": _baseline_summary(baseline),
            "B": summarize_games(variant_b),
            "B2": summarize_games(variant_b2),
            "observed_2015_16_applied": False,
        })
    return rows


def variant_b_endings(summary: dict) -> dict:
    """Why Variant B stopped, from the counters recorded on its half-court trips."""
    reads = summary["reads"]
    stops = {key[5:]: value for key, value in reads.items() if key.startswith("stop_")}
    chains = {int(key[6:]): value for key, value in reads.items() if key.startswith("chain_")}
    total = sum(stops.values()) or 1
    look_n = reads.get("look_n", 0) or 1
    return {
        "stop_share": {name: value / total for name, value in sorted(stops.items())},
        "chain_share": {length: chains.get(length, 0) / total for length in range(0, 7)},
        "cap_share": stops.get("cap", 0) / total,
        "favorable_share": sum(value for name, value in stops.items() if name.startswith("favorable")) / total,
        "urgent_share": stops.get("urgent", 0) / total,
        "look_improved_share": reads.get("look_up", 0) / look_n,
        "mean_look_change": (reads.get("look_end", 0) - reads.get("look_start", 0)) / look_n,
        "actions_change_shot_input": False,
    }


def b2_opportunity_report(summary: dict) -> dict:
    reads = summary["reads"]
    created_n = reads.get("created_openness_n", 0) or 1
    created_fga = reads.get("created_fga", 0)
    plain_fga = reads.get("plain_fga", 0)
    stops = {key[5:]: value for key, value in reads.items() if key.startswith("stop_")}
    chains = {int(key[6:]): value for key, value in reads.items() if key.startswith("chain_")}
    total = sum(stops.values()) or 1
    return {
        "mean_created_openness": reads.get("created_openness_sum", 0) / created_n,
        "created_shot_share": reads.get("created_shots", 0) / created_n,
        "mean_openness_on_created_shots": (
            reads.get("created_open_sum", 0) / reads["created_shots"]
        ) if reads.get("created_shots") else None,
        "created_fg_pct": (reads["created_makes"] / created_fga) if created_fga else None,
        "plain_fg_pct": (reads["plain_makes"] / plain_fga) if plain_fga else None,
        "screen_successes": reads.get("screen_success", 0),
        "screen_failures": reads.get("screen_fail", 0),
        "dribble_separations": reads.get("dribble_separation", 0),
        "dribble_contests": reads.get("dribble_contest", 0),
        "stop_share": {name: value / total for name, value in sorted(stops.items())},
        "chain_share": {length: chains.get(length, 0) / total for length in range(0, SAFEGUARD_ACTIONS + 1)},
        "safeguard_share": stops.get("safeguard", 0) / total,
        "look_improved_share": reads.get("look_up", 0) / (reads.get("look_n", 0) or 1),
        "make_formula": "probability_make",
        "openness_assumptions": OPENNESS_ASSUMPTIONS,
    }


def historical_comparison(model_row: dict) -> dict:
    exploration = observed_duration_sample(ROOT)
    validation = observed_duration_sample(Path(__file__).resolve().parents[1] / "research_playbyplay" / "validation")
    report = validation_report(model_row, exploration, validation)
    report["parameters_fit_to_validation"] = False
    report["fictional_roster_calibrated"] = False
    report["excluded_exploration"] = exploration["excluded"]
    report["excluded_validation"] = validation["excluded"]
    return report
