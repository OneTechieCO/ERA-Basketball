"""When a half-court offense develops the play, and when it finishes.

The baseline preparatory chance is left unchanged. Variant A and Variant B
are separate decision layers. Once a shot or a pass is taken, the live
shooting, passing, turnover, rebounding, and foul formulas run as they do
in the half-court experiment. Nothing here is fitted to the held-out games,
and the 2015-16 durations are not applied to other eras.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

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
    preparatory_action_chance,
    recover_offensive_rebound,
)
from .live_timing_diagnostics import SEEDS, fictional_matchup
from .playbyplay_timing import ROOT, load_actions, load_manifest, reconstruct_possessions

# The same one-second actions as the baseline. Not refit, and not a stall.
DEVELOPMENT_SECONDS = dict(UNCALIBRATED_ACTION_SECONDS)
MAX_DEVELOPMENT_ACTIONS = 3
URGENT_SECONDS = 8.0
EARLY_RELEASE_SECONDS = 16.0
# Expected points of a neutral look. A higher bar is not a possession-length target.
FAVORABLE_LOOK = 1.05

PROMOTION = {"promoted": False, "integrated_with_live_engine": False}
FITTED_TO_HELD_OUT = False
APPLIES_2015_16_TO_OTHER_ERAS = False

WHY_PREP_IS_RARE = (
    "The baseline rolls one continue chance, and only on a half-court start. "
    "Transition trips never enter it. The chance starts at 0.25 and then subtracts "
    "when the live expected shot value is above 0.85, which is normal for this roster, "
    "so most checks decline. A declined check goes straight to the live shoot-or-pass "
    "call. Dribble, hold, and screen are equally likely only if that first check accepts, "
    "and the result is not fed back into the next decision."
)


def variant_a_continue_chance(shooter, defender, shot_remaining: float, actions_already: int) -> float:
    """Chance of one more development action before the live shoot-or-pass call.

    A good look lowers it, and a higher-IQ player recognizes that look. Pressure
    raises it, more so for a higher-IQ player. Usage tilts a primary option toward
    finishing, without a shot quota. At 8 seconds the live decision is used.
    """
    if shot_remaining <= URGENT_SECONDS:
        return 0.0
    quality = _expected_shot_value(shooter) - 1.0
    recognition = clamp((shooter.iq - 70) / 40.0, 0.0, 1.0)
    pressure = clamp((defender.defense - 70) / 25.0, 0.0, 1.5)
    take_it = quality * (0.8 + recognition)
    pressure_read = pressure * (0.15 + 0.20 * recognition)
    role_finish = clamp((shooter.usage - 0.19) * 0.6, 0.0, 0.15)
    time_to_develop = clamp((shot_remaining - 14.0) / 24.0, 0.0, 0.30)
    chance = 0.08 + time_to_develop + pressure_read - take_it - role_finish - 0.10 * actions_already
    return clamp(chance, 0.0, 0.70)


def variant_a_action_weights(shooter, defender) -> dict[str, float]:
    """Which development action, once the player has decided to continue."""
    pressure = max(0.0, (defender.defense - 70) / 25.0)
    return {
        "dribble_evaluate": 1.0 + max(0.0, (shooter.iq - 70) / 30.0) + 0.4 * pressure,
        "hold": 0.55,
        "screen_action": 0.35 + max(0.0, (shooter.passing - 70) / 20.0) + 0.5 * pressure,
    }


def offensive_state(shooter, defender, shot_remaining: float, development_actions: int, last_action: str | None) -> str:
    """Developing, favorable, or urgent. Transition and putbacks do not use this."""
    if shot_remaining <= URGENT_SECONDS or development_actions >= MAX_DEVELOPMENT_ACTIONS:
        return "urgent" if shot_remaining <= URGENT_SECONDS else "favorable"
    look = _expected_shot_value(shooter)
    pressure = clamp((defender.defense - 70) / 25.0, 0.0, 1.5)
    adjusted = look - 0.10 * max(0.0, pressure)
    iq_bar = FAVORABLE_LOOK - clamp((shooter.iq - 70) / 40.0, 0.0, 1.0) * 0.08
    if last_action in ("screen_action", "pass"):
        if adjusted < iq_bar - 0.08 and pressure > 0.8:
            return "developing"
        return "favorable"
    if last_action == "dribble_evaluate" and shooter.iq >= 85 and adjusted >= iq_bar - 0.04:
        return "favorable"
    if adjusted >= iq_bar and pressure < 1.1:
        return "favorable"
    return "developing"


def variant_b_action_weights(shooter, defender) -> dict[str, float]:
    """Actions available while the possession is still developing. A shot is not one of them."""
    pressure = max(0.0, (defender.defense - 70) / 25.0)
    return {
        "dribble_evaluate": 0.8 + max(0.0, (shooter.iq - 70) / 30.0) + 0.3 * pressure,
        "hold": 0.35,
        "screen_action": 0.30 + max(0.0, (shooter.passing - 70) / 25.0) + 0.4 * pressure,
        "pass": 0.45 + max(0.0, (shooter.passing - 70) / 18.0),
    }


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


def baseline_decision_profile(roster, home) -> list[dict]:
    """The probabilities the baseline actually rolls, before any random draw."""
    offense = [roster[player_id] for player_id in home.player_ids]
    rows = []
    for shooter in offense:
        for defender in offense:
            prep = preparatory_action_chance(shooter, defender, 18.0)
            finish_pass = intelligent_pass_chance(shooter, defender, _style_bonus(home.coach_style), 0, 18.0)
            rows.append({
                "shooter": shooter.player_id,
                "defender": defender.player_id,
                "preparatory": prep,
                "dribble_given_preparatory": prep / 3.0,
                "hold_given_preparatory": prep / 3.0,
                "screen_given_preparatory": prep / 3.0,
                "pass_given_finish": finish_pass,
                "shoot_given_finish": 1.0 - finish_pass,
                "offered_on_transition": False,
                "offered_once_per_halfcourt_trip": True,
            })
    return rows


def _play_possession(game, rng, era, offense, defense, coach_style, fatigue, start_mode, variant: str) -> dict:
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
    developed = 0

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
    if categories["half_court_advance"] and categories["transition_push"]:
        raise ValueError("the bring-up was charged twice")
    if not opening["running"]["finished"]:
        end_reason = opening["running"]["reason"] or "period_expired"
        return _record(
            start_mode, categories, counts, shots, makes, reads, passes, violations,
            offensive_rebounds, misses, continuation, release_left, end_reason, shot, False, developed,
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
            counts["turnover"] += 1
            steal = rng.random() < 0.48
            apply_stoppage(game, "live_turnover", 0.0)
            end_reason = "turnover"
            break

        shooter = ballhandler
        passer = None
        if start_mode in HALF_COURT_STARTS and not in_continuation:
            developed_step = _develop(
                variant, game, shot, rng, offense, defense, shooter, charge, live_pass, reads,
            )
            shooter = developed_step["shooter"]
            passer = developed_step["passer"]
            passes += developed_step["passes"]
            passes_since_rebound += developed_step["passes"]
            developed += developed_step["actions"]
            if developed_step["end"] == "turnover":
                counts["turnover"] += 1
                steal = developed_step["steal"]
                apply_stoppage(game, "pass_turnover", 0.0)
                end_reason = "turnover"
                break
            if developed_step["end"] is not None:
                end_reason = developed_step["end"]
                if developed_step["end"] == "shot_clock_violation":
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
            reads["pass_chance"] += pass_chance
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
        start_mode, categories, counts, shots, makes, reads, passes, violations,
        offensive_rebounds, misses, continuation, release_left, end_reason, shot, steal, developed,
    )


def _step(shooter, passer, passes: int, actions: int, end, steal: bool = False) -> dict:
    return {"shooter": shooter, "passer": passer, "passes": passes, "actions": actions, "end": end, "steal": steal}


def _develop(variant, game, shot, rng, offense, defense, shooter, charge, live_pass, reads):
    """Move the ball through development, then hand the live shoot-or-pass call a shooter."""
    if variant == "A":
        taken = 0
        while taken < MAX_DEVELOPMENT_ACTIONS and shot.remaining > URGENT_SECONDS and game.quarter_remaining > 0:
            defender = defense[offense.index(shooter)]
            chance = variant_a_continue_chance(shooter, defender, shot.remaining, taken)
            reads["continue_checks"] += 1
            reads["continue_chance"] += chance
            if rng.random() >= chance:
                reads["continue_stops"] += 1
                return _step(shooter, None, 0, taken, None)
            kind = weighted_pick(rng, variant_a_action_weights(shooter, defender))
            reads[kind] += 1
            action = charge(DEVELOPMENT_SECONDS[kind], kind)
            taken += 1
            if not action["finished"]:
                reason = "shot_clock_violation" if action["reason"] == "shot_clock_expired" else (action["reason"] or "period_expired")
                return _step(shooter, None, 0, taken, reason)
        return _step(shooter, None, 0, taken, None)

    taken = 0
    added_passes = 0
    last_action = None
    passer = None
    initial_look = _expected_shot_value(shooter)
    while game.quarter_remaining > 0 and shot.remaining > 0:
        defender = defense[offense.index(shooter)]
        state = offensive_state(shooter, defender, shot.remaining, taken, last_action)
        if taken == 0:
            reads[f"state_{state}"] += 1
        if state != "developing":
            _note_variant_b_exit(reads, shooter, initial_look, taken, last_action, shot.remaining)
            return _step(shooter, passer, added_passes, taken, None)
        kind = weighted_pick(rng, variant_b_action_weights(shooter, defender))
        reads[kind] += 1
        if kind == "pass":
            outcome = live_pass(shooter)
            if outcome["status"] == "turnover":
                reads["development_turnovers"] += 1
                _note_variant_b_exit(reads, shooter, initial_look, taken, last_action, shot.remaining, "turnover")
                return _step(shooter, passer, added_passes, taken, "turnover", outcome["steal"])
            if outcome["status"] == "period":
                _note_variant_b_exit(reads, shooter, initial_look, taken, last_action, shot.remaining, "period")
                return _step(shooter, passer, added_passes, taken, "period_expired")
            passer = shooter
            shooter = outcome["recipient"]
            added_passes += 1
            last_action = "pass"
        else:
            action = charge(DEVELOPMENT_SECONDS[kind], kind)
            last_action = kind
            if not action["finished"]:
                reason = "shot_clock_violation" if action["reason"] == "shot_clock_expired" else (action["reason"] or "period_expired")
                _note_variant_b_exit(reads, shooter, initial_look, taken, last_action, shot.remaining, "clock")
                return _step(shooter, passer, added_passes, taken, reason)
        taken += 1
    end = "period_expired" if game.quarter_remaining == 0.0 else None
    _note_variant_b_exit(reads, shooter, initial_look, taken, last_action, shot.remaining, "clock")
    return _step(shooter, passer, added_passes, taken, end)


def _note_variant_b_exit(reads, shooter, initial_look: float, taken: int, last_action, shot_remaining: float, forced: str | None = None) -> None:
    """Record why development ended. This does not draw a random number."""
    if forced == "turnover":
        reason = "turnover"
    elif forced in ("clock", "period"):
        reason = "clock"
    elif shot_remaining <= URGENT_SECONDS:
        reason = "urgent"
    elif taken >= MAX_DEVELOPMENT_ACTIONS:
        reason = "cap"
    elif taken == 0:
        reason = "favorable_initial"
    elif last_action == "screen_action":
        reason = "favorable_screen"
    elif last_action == "pass":
        reason = "favorable_pass"
    elif last_action == "dribble_evaluate" and shooter.iq >= 85:
        reason = "favorable_dribble"
    else:
        reason = "favorable_look"
    reads[f"stop_{reason}"] += 1
    reads[f"chain_{taken}"] += 1
    final_look = _expected_shot_value(shooter)
    reads["look_start"] += initial_look
    reads["look_end"] += final_look
    reads["look_n"] += 1
    if final_look > initial_look + 0.02:
        reads["look_up"] += 1


def _record(start_mode, categories, counts, shots, makes, reads, passes, violations, offensive_rebounds, misses, continuation, release_left, end_reason, shot, steal, developed) -> dict:
    return {
        "start_mode": start_mode,
        "game_clock_elapsed": sum(categories.values()),
        "categories": dict(categories),
        "counts": dict(counts),
        "shot_by_player": dict(shots),
        "make_by_player": dict(makes),
        "reads": {key: value for key, value in reads.items()},
        "passes": passes,
        "shot_clock_violations": violations,
        "offensive_rebounds": offensive_rebounds,
        "misses": misses,
        "oreb_continuation_seconds": continuation,
        "release_shot_clock_remaining": release_left,
        "development_actions": developed,
        "period_ending": end_reason == "period_expired",
        "end_reason": end_reason,
        "next_start": _next_start(end_reason, steal),
        "shot_clock_limit": shot.limit,
    }


def diagnose_game(era, seed: int, home, away, roster, variant: str) -> dict:
    if variant not in ("A", "B"):
        raise ValueError(f"unknown decision variant: {variant}")
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
            team.coach_style, _fatigue(game.elapsed), start_mode, variant,
        )
        if game.quarter_remaining == 0.0 and started > 0:
            result["period_ending"] = True
        possessions.append(result)
        side = 1 - side
        start_mode = result["next_start"]
    return {
        "era": era.key,
        "seed": seed,
        "variant": variant,
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


def _duration_shares(durations: list[float]) -> dict:
    edges = ((0, 6), (6, 10), (10, 14), (14, 18), (18, 23), (23, None))
    total = len(durations) or 1
    shares = {}
    for low, high in edges:
        label = f"{low}_to_{high}" if high is not None else f"{low}_plus"
        if high is None:
            count = sum(value >= low for value in durations)
        else:
            count = sum(low <= value < high for value in durations)
        shares[label] = count / total
    return shares


def summarize_games(games: list[dict]) -> dict:
    possessions = [possession for game in games for possession in game["possession_log"]]
    total = len(possessions) or 1
    count = len(games)
    durations = [possession["game_clock_elapsed"] for possession in possessions]
    categories = Counter()
    counts = Counter()
    shots = Counter()
    makes = Counter()
    reads = Counter()
    release_left = []
    for possession in possessions:
        categories.update(possession["categories"])
        counts.update(possession["counts"])
        shots.update(possession.get("shot_by_player", {}))
        makes.update(possession.get("make_by_player", {}))
        reads.update(possession.get("reads", {}))
        release_left.extend(possession["release_shot_clock_remaining"][:1])
    per_team = [sum(game["possessions_per_team"][index] for game in games) / count for index in (0, 1)]
    continued = [possession["oreb_continuation_seconds"] for possession in possessions if possession["offensive_rebounds"] > 0]
    attempts = sum(shots.values()) or 1
    shot_distribution = {
        player_id: {
            "fga_per_game": shots[player_id] / count,
            "attempt_share": shots[player_id] / attempts,
            "fg_pct": (makes[player_id] / shots[player_id]) if shots[player_id] else None,
        }
        for player_id in sorted(shots)
    }
    development_names = tuple(DEVELOPMENT_SECONDS)
    transitions = [possession for possession in possessions if possession["start_mode"] == "transition"]
    early_transitions = [
        possession for possession in transitions
        if possession["release_shot_clock_remaining"]
        and not any(possession["categories"].get(name) for name in development_names)
    ]
    def _sequence(possession: dict) -> bool:
        from_reads = sum(possession.get("reads", {}).get(name, 0) for name in ("dribble_evaluate", "hold", "screen_action", "pass"))
        from_clock = sum(possession["categories"].get(name, 0.0) / DEVELOPMENT_SECONDS[name] for name in development_names)
        actions = from_reads if from_reads else from_clock
        passed = possession["passes"] > 0 or possession["categories"].get("completed_pass", 0.0) > 0
        return actions >= 2 or (actions >= 1 and passed)
    return {
        "era": games[0]["era"],
        "variant": games[0].get("variant", "baseline"),
        "seeded_pace": games[0]["seeded_pace"],
        "pace_input": None,
        "possessions_per_team": per_team,
        "possessions_per_team_mean": sum(per_team) / 2,
        "mean_seconds": sum(durations) / len(durations),
        "p10_seconds": _percentile(durations, 0.10),
        "median_seconds": _percentile(durations, 0.50),
        "p90_seconds": _percentile(durations, 0.90),
        "duration_shares": _duration_shares(durations),
        "passes_per_possession": sum(possession["passes"] for possession in possessions) / total,
        "assists_per_game": counts["assist"] / count,
        "turnovers_per_game": counts["turnover"] / count,
        "fga_per_game": counts["fga"] / count,
        "fg_pct": (counts["make"] / counts["fga"]) if counts["fga"] else None,
        "points_per_game": counts["points"] / count,
        "action_counts_per_possession": {
            **{name: categories[name] / DEVELOPMENT_SECONDS[name] / total for name in development_names},
            "completed_pass": categories["completed_pass"] / 2.0 / total,
            "release": counts["fga"] / total + counts["shooting_foul"] / total,
        },
        "action_seconds_per_possession": {name: categories[name] / total for name in list(development_names) + ["half_court_advance", "transition_push", "completed_pass", "release", "oreb_recovery"]},
        "shot_distribution": shot_distribution,
        "early_release_share": (
            sum(value >= EARLY_RELEASE_SECONDS for value in release_left) / len(release_left)
        ) if release_left else None,
        "transition_early_finish_share": (len(early_transitions) / len(transitions)) if transitions else None,
        "sequence_share": sum(1 for possession in possessions if _sequence(possession)) / total,
        "oreb_continuation_seconds_mean": (sum(continued) / len(continued)) if continued else None,
        "shot_clock_violations_per_game": sum(possession["shot_clock_violations"] for possession in possessions) / count,
        "quarter_endings_per_game": sum(1 for possession in possessions if possession["period_ending"]) / count,
        "mean_shot_clock_remaining_at_first_release": (sum(release_left) / len(release_left)) if release_left else None,
        "reads": dict(reads),
        "promoted": False,
        "fitted_to_held_out": False,
        "uses_2015_16_as_other_era_fact": False,
    }


def _baseline_summary(games: list[dict]) -> dict:
    """Baseline logs use a slightly different development count, so normalize before summarizing."""
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
        variant_a = [diagnose_game(era, seed, home, away, roster, "A") for seed in seeds]
        variant_b = [diagnose_game(era, seed, home, away, roster, "B") for seed in seeds]
        rows.append({
            "era": era.key,
            "baseline": _baseline_summary(baseline),
            "A": summarize_games(variant_a),
            "B": summarize_games(variant_b),
            "observed_2015_16_applied": False,
        })
    return rows


def observed_duration_sample(root: Path) -> dict:
    """Reliable regulation durations. Excluded possessions stay excluded."""
    manifest = load_manifest(root)
    game_ids = [game["game_id"] for game in manifest["games"]]
    kept = []
    excluded = 0
    for game_id in game_ids:
        for row in reconstruct_possessions(load_actions(game_id, root), game_id):
            if row["reliable"] and row["regulation"] and row["duration"] is not None:
                kept.append(row["duration"])
            else:
                excluded += 1
    return {
        "game_ids": game_ids,
        "role": manifest.get("role", "exploration"),
        "durations": kept,
        "excluded": excluded,
        "excluded_share": excluded / (excluded + len(kept) or 1),
        "shares": _duration_shares(kept),
        "mean": sum(kept) / len(kept),
        "p10": _percentile(kept, 0.10),
        "p50": _percentile(kept, 0.50),
        "p90": _percentile(kept, 0.90),
        "tuned_against": False,
    }


def validation_report(model_row: dict, exploration: dict, validation: dict) -> dict:
    """Place one model beside both samples. The validation sample does not choose parameters."""
    return {
        "variant": model_row["variant"],
        "model_shares": model_row["duration_shares"],
        "exploration_shares": exploration["shares"],
        "validation_shares": validation["shares"],
        "exploration_excluded_share": exploration["excluded_share"],
        "validation_excluded_share": validation["excluded_share"],
        "parameters_fit_to_validation": False,
        "era_scope": "2015-16 Warriors sample only",
    }
