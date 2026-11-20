"""Variant B3. Each action updates a possession state the next decision reads.

The live make, pass-turnover, rebound, and foul formulas are unchanged.
Openness is an input to probability_make. A hold does not change that input.
New probabilities and the one-second action times are unverified.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from .action_decision_experiments import summarize_games
from .action_state_experiments import diagnose_game as diagnose_b2
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
from .halfcourt_action_experiments import HALF_COURT_STARTS, begin_possession, recover_offensive_rebound
from .live_timing_diagnostics import SEEDS, fictional_matchup

URGENT_SECONDS = 8.0
SAFEGUARD_ACTIONS = 8
ACTION_SECONDS = {"screen": 1.0, "dribble": 1.0, "hold": 1.0}

# Unverified inputs. probability_make still multiplies openness by 0.035.
OPENNESS_FROM_SEPARATION = 0.20
OPENNESS_FROM_PRESSURE = -0.08

UNVERIFIED = {
    "action_seconds": dict(ACTION_SECONDS),
    "screen_outcomes": ("opening", "switch", "failed"),
    "dribble_outcomes": ("separation", "pressure"),
    "openness_from_separation": OPENNESS_FROM_SEPARATION,
    "openness_from_pressure": OPENNESS_FROM_PRESSURE,
    "decision_weights": "experimental",
    "hold_changes_shot_input": False,
    "automatic_scoring_bonus": False,
    "urgent_finish": "intelligent_pass_chance",
}

PROMOTION = {"promoted": False, "integrated_with_live_engine": False}
FITTED_TO_SAMPLE = False
RESERVED_GAMES_USED_FOR_ASSESSMENT = False


@dataclass
class PossessionState:
    """What the offense knows while this live ball is being played."""

    ballhandler: object
    defender: object
    separation: float = 0.0
    pressure: float = 0.0
    elapsed: float = 0.0
    last_passer: object = None
    development_actions: int = 0
    putback: bool = False
    history: list = field(default_factory=list)

    def quality(self) -> float:
        return opportunity_quality(self.ballhandler, self.defender, self.separation, self.pressure)


def action_openness(separation: float, pressure: float) -> float:
    """Shot-model input created by actions. A hold leaves both parts unchanged."""
    return clamp(separation * OPENNESS_FROM_SEPARATION + pressure * OPENNESS_FROM_PRESSURE, -0.30, 0.45)


def opportunity_quality(player, defender, separation: float, pressure: float) -> float:
    """Decision read of the look. The shot itself still uses probability_make."""
    openness = action_openness(separation, pressure)
    mismatch = max(0.0, defender.defense - 70) * 0.004
    return _expected_shot_value(player) + 0.07 * openness - mismatch


def must_finish(development_actions: int, shot_remaining: float, putback: bool = False) -> bool:
    """A putback does not start another half-court action chain."""
    return putback or shot_remaining <= URGENT_SECONDS or development_actions >= SAFEGUARD_ACTIONS


def screen_outcome_weights(ballhandler, defender) -> dict[str, float]:
    """Unverified. IQ and passing help the screen. Defense makes failure more likely."""
    skill = (ballhandler.iq - 70) * 0.008 + (ballhandler.passing - 70) * 0.006
    contest = (defender.defense - 70) * 0.008
    return {
        "opening": max(0.08, 0.30 + skill - contest),
        "switch": max(0.08, 0.24 + skill * 0.4),
        "failed": max(0.12, 0.42 + contest - skill * 0.4),
    }


def dribble_outcome_weights(ballhandler, defender) -> dict[str, float]:
    """Unverified. A dribble either creates space or draws pressure."""
    read = (ballhandler.iq - 70) * 0.010
    contest = (defender.defense - 70) * 0.008
    return {
        "separation": max(0.08, 0.46 + read - contest),
        "pressure": max(0.08, 0.42 + contest - read * 0.5),
    }


def _teammate_quality(state: PossessionState, offense, defense) -> float:
    others = [player for player in offense if player.player_id != state.ballhandler.player_id]
    if not others:
        return state.quality()
    total = 0.0
    weight = 0.0
    for player in others:
        live_weight = max(1.0, player.usage * (70 + player.iq) / 140)
        defender = defense[offense.index(player)]
        total += live_weight * opportunity_quality(player, defender, 0.0, 0.0)
        weight += live_weight
    return total / weight


def action_weights(state: PossessionState, offense, defense, shot_remaining: float) -> dict[str, float]:
    """Unverified choice among shoot, pass, screen, dribble, and hold."""
    quality = state.quality()
    recognition = clamp((state.ballhandler.iq - 70) / 40.0, 0.0, 1.0)
    pressure_level = clamp((state.defender.defense - 70) / 25.0 + state.pressure, 0.0, 1.5)
    gain = _teammate_quality(state, offense, defense) - quality
    shoot = 0.08 + max(0.0, quality - 1.00) * (3.2 + recognition) - 0.50 * pressure_level
    if state.separation >= 0.70 and pressure_level < 0.75:
        shoot += 0.30 * (0.35 + recognition)
    pass_weight = 0.12 + max(0.0, gain) * (1.6 + recognition) + max(0.0, state.ballhandler.passing - 70) / 90.0
    screen = 0.18 + 0.28 * pressure_level + max(0.0, 1.02 - quality) * 0.80
    dribble = 0.12 + max(0.0, 1.00 - quality) * 0.35 + max(0.0, state.ballhandler.iq - 70) / 80.0
    if quality >= 1.04 and pressure_level < 0.50:
        dribble *= 0.25
        screen *= 0.35
    if state.separation >= 0.70:
        dribble *= 0.30
        screen *= 0.55
    hold = 0.06
    if state.history:
        last = state.history[-1]
        if last["action"] == "screen" and last["outcome"] == "failed":
            screen *= 0.30
            pass_weight += 0.25
        if last["action"] == "dribble" and last["outcome"] == "pressure":
            dribble *= 0.30
            pass_weight += 0.20
    if shot_remaining <= URGENT_SECONDS:
        shoot += 0.40
        pass_weight += 0.20
        screen = 0.0
        dribble = 0.0
        hold = 0.0
    return {
        "shoot": max(0.05, shoot),
        "pass": max(0.05, pass_weight),
        "screen": max(0.0, screen),
        "dribble": max(0.0, dribble),
        "hold": max(0.0, hold),
    }


def leading_action(weights: dict[str, float]) -> str:
    return max(weights, key=weights.get)


def apply_screen(state: PossessionState, outcome: str, switched_defender=None) -> None:
    if outcome == "opening":
        state.separation = min(1.0, state.separation + 1.0)
        state.pressure = max(0.0, state.pressure - 0.25)
    elif outcome == "switch":
        if switched_defender is None:
            raise ValueError("a switch needs the new defender")
        state.defender = switched_defender
    elif outcome == "failed":
        state.pressure = min(1.5, state.pressure + 0.45)
    else:
        raise ValueError(f"unknown screen outcome: {outcome}")


def apply_dribble(state: PossessionState, outcome: str) -> None:
    if outcome == "separation":
        state.separation = min(1.0, state.separation + 0.70)
        state.pressure = max(0.0, state.pressure - 0.15)
    elif outcome == "pressure":
        state.pressure = min(1.5, state.pressure + 0.40)
    else:
        raise ValueError(f"unknown dribble outcome: {outcome}")


def apply_hold(state: PossessionState) -> None:
    """Time passes. Separation, pressure, defender, and ballhandler stay as they are."""
    return None


def apply_pass(state: PossessionState, recipient, recipient_defender) -> None:
    state.last_passer = state.ballhandler
    state.ballhandler = recipient
    state.defender = recipient_defender
    state.separation = 0.0


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


def _record_event(state: PossessionState, action: str, outcome: str, before: str, after: str, shot_remaining: float, offense, defense) -> None:
    if shot_remaining < 0:
        raise ValueError("shot clock became negative")
    state.history.append({
        "action": action,
        "outcome": outcome,
        "ballhandler": state.ballhandler.player_id,
        "defender": state.defender.player_id,
        "separation": round(state.separation, 3),
        "pressure": round(state.pressure, 3),
        "quality": round(state.quality(), 3),
        "decision_before": before,
        "decision_after": after,
        "shot_clock": round(shot_remaining, 3),
        "elapsed": round(state.elapsed, 3),
        "development_actions": state.development_actions,
        "material_change": before != after,
    })


def _play_possession(game, rng, era, offense, defense, coach_style, fatigue, start_mode) -> dict:
    style_bonus = _style_bonus(coach_style)
    shot = ShotClock(era.shot_clock, era.shot_clock)
    categories = Counter()
    counts = Counter()
    shots = Counter()
    makes = Counter()
    reads = Counter()
    passes = 0
    violations = 0
    offensive_rebounds = 0
    misses = 0
    continuation = 0.0
    in_continuation = False
    steal = False
    end_reason = "live_ball"
    release_left = []
    end_detail = "opening"
    state = None

    def charge(seconds: float, kind: str) -> dict:
        nonlocal continuation
        action = apply_live_action(game, shot, seconds, kind)
        if shot.remaining < 0 or game.quarter_remaining < 0:
            raise ValueError("a clock went negative")
        categories[kind] += action["game_clock_elapsed"]
        if state is not None:
            state.elapsed += action["game_clock_elapsed"]
        if in_continuation:
            continuation += action["game_clock_elapsed"]
        return action

    def live_pass(passer):
        pool = [player for player in offense if player.player_id != passer.player_id]
        recipient = rng.choices(
            pool,
            weights=[max(1.0, player.usage * (70 + player.iq) / 140) for player in pool],
        )[0]
        passer_defender = state.defender
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
        return {"status": "complete", "recipient": recipient, "defender": recipient_defender}

    opening = begin_possession(game, shot, start_mode)
    if opening["stopped"] is not None:
        counts["stopped_throw_in"] += 1
    categories[opening["running"]["kind"]] += opening["running"]["game_clock_elapsed"]
    if not opening["running"]["finished"]:
        end_reason = opening["running"]["reason"] or "period_expired"
        end_detail = "period_during_advance"
        return _record(locals())

    need_touch = True
    while True:
        if game.quarter_remaining == 0.0:
            end_reason = "period_expired"
            end_detail = "period"
            break
        if need_touch or state is None:
            ballhandler = rng.choices(
                offense,
                weights=[max(2, player.usage * (70 + player.iq) / 140) for player in offense],
            )[0]
            defender = defense[offense.index(ballhandler)]
            if state is None:
                state = PossessionState(ballhandler, defender)
                state.elapsed = categories[opening["running"]["kind"]]
            else:
                state.ballhandler = ballhandler
                state.defender = defender
                state.separation = 0.0
                state.pressure = 0.0
                state.last_passer = None
                state.development_actions = 0
            need_touch = False
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
                end_detail = "live_ball_turnover"
                break
        if must_finish(state.development_actions, shot.remaining, state.putback):
            finish = _urgent_finish(
                state, game, shot, rng, era, offense, defense, coach_style, style_bonus, fatigue,
                passes, charge, live_pass, counts, shots, makes, reads, release_left, in_continuation,
            )
            passes = finish["passes"]
            violations += finish["violations"]
            misses += finish["misses"]
            offensive_rebounds += finish["offensive_rebounds"]
            continuation += finish["continuation"]
            steal = finish["steal"] or steal
            if finish["status"] == "oreb":
                categories["oreb_recovery"] += finish["recovery"]
                state.putback = True
                in_continuation = True
                need_touch = True
                continue
            end_reason = finish["end_reason"]
            if state.putback:
                end_detail = "putback"
            elif shot.remaining <= URGENT_SECONDS:
                end_detail = "urgent"
            elif state.development_actions >= SAFEGUARD_ACTIONS:
                end_detail = "safeguard"
            else:
                end_detail = "finish"
            break

        weights = action_weights(state, offense, defense, shot.remaining)
        before = leading_action(weights)
        choice = weighted_pick(rng, weights)
        if choice == "shoot":
            finish = _resolve_shot(
                state, game, shot, rng, era, offense, coach_style, fatigue, charge, counts, shots, makes, reads, release_left,
            )
            misses += finish["misses"]
            if finish.get("violation"):
                violations += 1
            if finish["status"] == "oreb":
                offensive_rebounds += 1
                recovery = finish["recovery"]
                categories["oreb_recovery"] += recovery
                continuation += recovery
                state.putback = True
                in_continuation = True
                need_touch = True
                _record_event(state, "shoot", finish["end_reason"], before, "rebound", shot.remaining, offense, defense)
                continue
            end_reason = finish["end_reason"]
            end_detail = "shot_choice"
            _record_event(state, "shoot", end_reason, before, "finish", shot.remaining, offense, defense)
            break
        if choice == "pass":
            outcome = live_pass(state.ballhandler)
            state.development_actions += 1
            if outcome["status"] == "turnover":
                counts["turnover"] += 1
                steal = outcome["steal"]
                apply_stoppage(game, "pass_turnover", 0.0)
                end_reason = "turnover"
                end_detail = "pass_turnover"
                reads["pass_turnover"] += 1
                break
            if outcome["status"] == "period":
                end_reason = "period_expired"
                end_detail = "period"
                break
            apply_pass(state, outcome["recipient"], outcome["defender"])
            passes += 1
            reads["pass"] += 1
            after = leading_action(action_weights(state, offense, defense, shot.remaining)) if not must_finish(state.development_actions, shot.remaining) else "urgent"
            if before != after:
                reads["material_changes"] += 1
            reads["decisions"] += 1
            _record_event(state, "pass", "complete", before, after, shot.remaining, offense, defense)
            continue
        if choice == "hold":
            action = charge(ACTION_SECONDS["hold"], "hold")
            state.development_actions += 1
            apply_hold(state)
            reads["hold"] += 1
            if not action["finished"]:
                end_reason = "period_expired" if action["reason"] == "period_expired" else "shot_clock_violation"
                end_detail = "clock"
                if end_reason == "shot_clock_violation":
                    violations += 1
                    counts["turnover"] += 1
                break
            after = leading_action(action_weights(state, offense, defense, shot.remaining)) if not must_finish(state.development_actions, shot.remaining) else "urgent"
            if before != after:
                reads["material_changes"] += 1
            reads["decisions"] += 1
            _record_event(state, "hold", "time", before, after, shot.remaining, offense, defense)
            continue

        kind = "screen" if choice == "screen" else "dribble"
        if kind == "screen":
            outcome_name = weighted_pick(rng, screen_outcome_weights(state.ballhandler, state.defender))
            switched = None
            if outcome_name == "switch":
                mates = [player for player in offense if player.player_id != state.ballhandler.player_id]
                screener = rng.choice(mates)
                switched = defense[offense.index(screener)]
            apply_screen(state, outcome_name, switched)
        else:
            outcome_name = weighted_pick(rng, dribble_outcome_weights(state.ballhandler, state.defender))
            apply_dribble(state, outcome_name)
        action = charge(ACTION_SECONDS[kind], kind)
        state.development_actions += 1
        reads[f"{kind}_{outcome_name}"] += 1
        if not action["finished"]:
            end_reason = "shot_clock_violation" if action["reason"] == "shot_clock_expired" else (action["reason"] or "period_expired")
            end_detail = "clock"
            if end_reason == "shot_clock_violation":
                violations += 1
                counts["turnover"] += 1
            break
        after = leading_action(action_weights(state, offense, defense, shot.remaining)) if not must_finish(state.development_actions, shot.remaining) else "urgent"
        if before != after:
            reads["material_changes"] += 1
            reads[f"material_{kind}_{outcome_name}"] += 1
        reads["decisions"] += 1
        reads[f"taken_{kind}_{outcome_name}"] += 1
        _record_event(state, kind, outcome_name, before, after, shot.remaining, offense, defense)

    return _record(locals())


def _urgent_finish(state, game, shot, rng, era, offense, defense, coach_style, style_bonus, fatigue, passes, charge, live_pass, counts, shots, makes, reads, release_left, in_continuation):
    """Late clock or the safeguard. The live pass chance decides pass or shoot."""
    result = {"passes": passes, "violations": 0, "misses": 0, "offensive_rebounds": 0, "continuation": 0.0, "steal": False, "status": "end", "end_reason": "live_ball", "recovery": 0.0}
    for _ in range(6):
        if game.quarter_remaining == 0.0 or shot.remaining < 0:
            result["end_reason"] = "period_expired" if game.quarter_remaining == 0.0 else "shot_clock_violation"
            return result
        pass_chance = intelligent_pass_chance(state.ballhandler, state.defender, style_bonus, result["passes"], shot.remaining)
        if shot.remaining <= 2:
            pass_chance = 0.0
        elif shot.remaining <= 4:
            pass_chance *= 0.15
        elif shot.remaining <= 8:
            pass_chance *= 0.50
        if rng.random() >= pass_chance:
            break
        outcome = live_pass(state.ballhandler)
        if outcome["status"] == "turnover":
            counts["turnover"] += 1
            result["steal"] = outcome["steal"]
            apply_stoppage(game, "pass_turnover", 0.0)
            result["end_reason"] = "turnover"
            result["status"] = "end"
            return result
        if outcome["status"] == "period":
            result["end_reason"] = "period_expired"
            return result
        apply_pass(state, outcome["recipient"], outcome["defender"])
        result["passes"] += 1
    shot_result = _resolve_shot(state, game, shot, rng, era, offense, coach_style, fatigue, charge, counts, shots, makes, reads, release_left)
    result["misses"] += shot_result["misses"]
    if shot_result.get("violation"):
        result["violations"] += 1
    result["end_reason"] = shot_result["end_reason"]
    result["status"] = shot_result["status"]
    if shot_result["status"] == "oreb":
        result["offensive_rebounds"] = 1
        result["continuation"] = shot_result["recovery"]
        result["recovery"] = shot_result["recovery"]
    return result


def _resolve_shot(state, game, shot, rng, era, offense, coach_style, fatigue, charge, counts, shots, makes, reads, release_left):
    """Live shot, foul, and rebound resolution. Openness includes the action input."""
    result = {"status": "end", "end_reason": "live_ball", "misses": 0, "recovery": 0.0}
    if shot.remaining <= 0:
        result["end_reason"] = "shot_clock_violation"
        result["violation"] = True
        counts["turnover"] += 1
        return result
    shooter = state.ballhandler
    defender = state.defender
    passer = state.last_passer
    preparation = rng.uniform(1.5, 3.5)
    release_left.append(shot.remaining)
    if preparation > shot.remaining and shot.remaining <= game.quarter_remaining:
        charge(shot.remaining, "shot_clock_violation")
        counts["turnover"] += 1
        result["violation"] = True
        result["end_reason"] = "shot_clock_violation"
        return result
    if preparation > game.quarter_remaining:
        charge(preparation, "release")
        result["end_reason"] = "period_expired"
        return result
    charge(preparation, "release")
    zone = weighted_pick(rng, shot_weights(shooter, era, coach_style, 0.0))
    created = action_openness(state.separation, state.pressure)
    if passer:
        base = (shooter.iq - defender.iq) / 55 + (passer.passing - 65) / 100
    else:
        base = (shooter.iq - defender.iq) / 65
    openness = clamp(base + created, -0.8, 1.0)
    reads["created_openness_sum"] += created
    reads["created_openness_n"] += 1
    foul_chance = clamp(0.065 + era.physicality * 0.045 + (defender.defense - 70) * -0.0003, 0.035, 0.16)
    if rng.random() < foul_chance:
        counts["shooting_foul"] += 1
        free_throws = 3 if zone == "three" else 2
        free_throw_make = clamp(0.65 + (shooter.shot_skills["mid"] - 70) * 0.003, 0.40, 0.95)
        for _ in range(free_throws):
            apply_stoppage(game, "free_throw", 0.0)
            if rng.random() < free_throw_make:
                counts["points"] += 1
        result["end_reason"] = "shooting_foul"
        return result
    made = rng.random() < probability_make(shooter, defender, era, zone, fatigue, 0.0, openness)
    counts["fga"] += 1
    shots[shooter.player_id] += 1
    if made:
        counts["make"] += 1
        makes[shooter.player_id] += 1
        counts["points"] += POINTS[zone]
        if passer and rng.random() < clamp(0.55 + (passer.passing - 70) * 0.004, 0.2, 0.88):
            counts["assist"] += 1
        result["end_reason"] = "make"
        return result
    result["misses"] = 1
    if rng.random() < clamp(0.075 + (defender.blocks - 65) * 0.0015, 0.015, 0.24):
        counts["block"] += 1
    if rng.random() >= 0.24 or game.quarter_remaining == 0.0:
        result["end_reason"] = "defensive_rebound"
        return result
    recovery = recover_offensive_rebound(game, shot, offensive_rebound_reset(era))
    if shot.remaining < 0 or game.quarter_remaining < 0:
        raise ValueError("a clock went negative")
    result["recovery"] = recovery["game_clock_elapsed"]
    if not recovery["continued"]:
        result["end_reason"] = "period_expired"
        return result
    result["status"] = "oreb"
    result["end_reason"] = "offensive_rebound"
    return result


def _record(scope: dict) -> dict:
    state = scope["state"]
    history = list(state.history) if state is not None else []
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
        "development_actions": state.development_actions if state is not None else 0,
        "period_ending": scope["end_reason"] == "period_expired",
        "end_reason": scope["end_reason"],
        "end_detail": scope["end_detail"],
        "next_start": _next_start(scope["end_reason"], scope["steal"]),
        "shot_clock_limit": scope["shot"].limit,
        "history": history,
        "minimum_shot_clock": min((event["shot_clock"] for event in history), default=scope["shot"].remaining),
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
        "variant": "B3",
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
        "fitted_to_sample": False,
    }


def behavior_report(summary: dict) -> dict:
    reads = summary["reads"]
    decisions = reads.get("decisions", 0) or 1
    taken = {key[6:]: value for key, value in reads.items() if key.startswith("taken_")}
    return {
        "material_change_share": reads.get("material_changes", 0) / decisions,
        "decisions": reads.get("decisions", 0),
        "outcomes": {
            "screen_opening": reads.get("screen_opening", 0),
            "screen_switch": reads.get("screen_switch", 0),
            "screen_failed": reads.get("screen_failed", 0),
            "dribble_separation": reads.get("dribble_separation", 0),
            "dribble_pressure": reads.get("dribble_pressure", 0),
            "hold": reads.get("hold", 0),
            "pass": reads.get("pass", 0),
        },
        "material_by_outcome": {
            key[9:]: value for key, value in reads.items() if key.startswith("material_") and key != "material_changes"
        },
        "taken": taken,
        "mean_action_openness": (
            reads.get("created_openness_sum", 0) / reads["created_openness_n"]
        ) if reads.get("created_openness_n") else None,
        "hold_changes_shot_input": False,
        "automatic_scoring_bonus": False,
    }


def compare_b2_b3(eras, seeds=SEEDS) -> list[dict]:
    home, away, roster = fictional_matchup()
    rows = []
    for era in eras:
        b2_games = [diagnose_b2(era, seed, home, away, roster) for seed in seeds]
        b3_games = [diagnose_game(era, seed, home, away, roster) for seed in seeds]
        for game in b2_games:
            game["variant"] = "B2"
        rows.append({
            "era": era.key,
            "B2": summarize_games(b2_games),
            "B3": summarize_games(b3_games),
            "B3_behavior": behavior_report(summarize_games(b3_games)),
            "observed_2015_16_applied_to_other_eras": False,
            "reserved_games_used": False,
        })
    return rows


def example_possession(era, seed: int = 7) -> dict:
    """One possession whose trace shows why it ended. Not a fitted example."""
    home, away, roster = fictional_matchup()
    game = diagnose_game(era, seed, home, away, roster)
    chosen = game["possession_log"][0]
    for possession in game["possession_log"]:
        actions = {event["action"] for event in possession["history"]}
        if "screen" in actions and any(event["material_change"] for event in possession["history"]):
            chosen = possession
            break
    return {
        "era": era.key,
        "seed": seed,
        "start_mode": chosen["start_mode"],
        "end_reason": chosen["end_reason"],
        "end_detail": chosen["end_detail"],
        "seconds": chosen["game_clock_elapsed"],
        "history": chosen["history"],
    }
