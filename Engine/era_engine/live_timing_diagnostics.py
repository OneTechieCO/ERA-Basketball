"""Time the live possession rules on the experimental game clock.

This does not change the live engine and it does not choose a possession
count from era pace. Probability functions are the live functions. A few
chances exist only inside simulate_game, so those expressions are copied
here and named in COPIED_EXPRESSIONS.
"""
from __future__ import annotations

import random
from collections import Counter

from .core import (
    Team,
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
)
from .seed import demo_players

SEEDS = (7, 19, 42, 101, 256, 997, 2026, 4242)

# Chances that live only inside simulate_game. The numbers are that code.
COPIED_EXPRESSIONS = {
    "live_turnover": "clamp(0.085 + (steals - iq) * 0.00065 + fatigue * 0.03, 0.025, 0.16)",
    "pass_turnover": "clamp(0.020 + passing and iq and steal and defense terms, 0.01, 0.15)",
    "shooting_foul": "clamp(0.065 + physicality * 0.045 + defense term, 0.035, 0.16)",
    "offensive_rebound": "random() < 0.24",
    "shot_preparation": "uniform(1.5, 3.5)",
    "setup": "min(6, shot_clock * 0.25)",
    "completed_pass": "2 seconds, charged by the live clock",
}

APPROXIMATIONS = {
    "fatigue": (
        "Live fatigue is the possession index divided by the pace-drawn game length, "
        "capped at 0.42. This diagnostic ramps that same cap across the 48-minute clock "
        "so pace does not choose the possession count."
    ),
    "free_throws": (
        "The live engine assigns no duration to free throws. They are dead-ball events "
        "with 0 game-clock seconds."
    ),
    "between_possessions": (
        "The live engine charges no time between the end of one possession and the next "
        "setup. That gap is recorded as 0."
    ),
    "rebound_gather": "The live engine charges no rebound-gather time. None is added.",
    "shot_clock_violation": (
        "A release that does not fit ends the possession without charging the leftover "
        "shot-clock time. That is the live shot-clock behavior."
    ),
    "period_boundary": (
        "The live engine has no quarters. An action that would run past a quarter is cut "
        "at the experimental game clock and does not continue."
    ),
    "lineup": (
        "The fictional demo fives stay on the court. The live possession loop does not "
        "substitute, and these are not NBA ratings."
    ),
    "adaptation": "Adaptation is 0, the live default.",
    "matchup": "The defender is the same list index as the offensive player, as in the live game.",
}

PROMOTION = {"promoted": False, "integrated_with_live_engine": False}


def fictional_matchup() -> tuple[Team, Team, dict]:
    """Two fictional demo teams. No historical player is rated here."""
    players = demo_players()
    roster = {player.player_id: player for player in players}
    home = Team("Bay City Legends", tuple(player.player_id for player in players[:5]), "perimeter")
    away = Team("Chicago Stampede", tuple(player.player_id for player in players[5:]), "balanced")
    return home, away, roster


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


def _play_live_possession(game, rng, era, offense, defense, coach_style, fatigue) -> dict:
    """One live possession, stopped at the quarter if the action cannot finish."""
    style_bonus = _style_bonus(coach_style)
    shot = ShotClock(era.shot_clock, era.shot_clock)
    categories = Counter()
    counts = Counter()
    passes = 0
    violations = 0
    offensive_rebounds = 0
    misses = 0
    release_shot_clock = []
    end_reason = "live_ball"

    def charge(seconds: float, kind: str) -> dict:
        action = apply_live_action(game, shot, seconds, kind)
        categories[kind] += action["game_clock_elapsed"]
        return action

    setup = min(6.0, era.shot_clock * 0.25)
    setup_action = charge(setup, "setup")
    if not setup_action["finished"]:
        end_reason = "period_expired"
        return _record(game, shot, categories, counts, passes, violations, offensive_rebounds, misses, release_shot_clock, end_reason)

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
            if rng.random() < 0.48:
                counts["steal"] += 1
            apply_stoppage(game, "live_turnover", 0.0)
            end_reason = "turnover"
            break

        passer = None
        shooter = ballhandler
        pass_failed = False
        for _ in range(6):
            if game.quarter_remaining == 0.0:
                end_reason = "period_expired"
                pass_failed = True
                break
            pass_chance = intelligent_pass_chance(shooter, defense[offense.index(shooter)], style_bonus, passes, shot.remaining)
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
                if rng.random() < 0.65:
                    counts["steal"] += 1
                apply_stoppage(game, "pass_turnover", 0.0)
                end_reason = "turnover"
                pass_failed = True
                break
            passer = shooter
            shooter = recipient
            passes += 1
            # The live clock charges a pass down to zero and then checks for a violation.
            passed = charge(2.0, "completed_pass")
            if passed["reason"] == "period_expired":
                end_reason = "period_expired"
                pass_failed = True
                break
        if pass_failed:
            break
        if shot.remaining <= 0:
            violations += 1
            end_reason = "shot_clock_violation"
            break

        shot_defender = defense[offense.index(shooter)]
        preparation = rng.uniform(1.5, 3.5)
        if preparation > shot.remaining and shot.remaining <= game.quarter_remaining:
            violations += 1
            end_reason = "shot_clock_violation"
            break
        release_shot_clock.append(shot.remaining)
        if preparation > game.quarter_remaining:
            charge(preparation, "release")
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
        foul_chance = clamp(0.065 + era.physicality * 0.045 + (shot_defender.defense - 70) * -0.0003, 0.035, 0.16)
        if rng.random() < foul_chance:
            counts["shooting_foul"] += 1
            free_throws = 3 if zone == "three" else 2
            counts["free_throw"] += free_throws
            for _ in range(free_throws):
                apply_stoppage(game, "free_throw", 0.0)
                rng.random()
            end_reason = "shooting_foul"
            break
        make_probability = probability_make(shooter, shot_defender, era, zone, fatigue, 0.0, openness)
        if rng.random() < make_probability:
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
        offensive_rebounds += 1
        counts["offensive_rebound"] += 1
        shot.reset(offensive_rebound_reset(era))

    return _record(
        game, shot, categories, counts, passes, violations, offensive_rebounds, misses, release_shot_clock, end_reason
    )


def _record(game, shot, categories, counts, passes, violations, offensive_rebounds, misses, release_shot_clock, end_reason):
    game_elapsed = sum(categories.values())
    return {
        "game_clock_elapsed": game_elapsed,
        "shot_clock_time_consumed": game_elapsed,
        "shot_clock_remaining": shot.remaining,
        "shot_clock_limit": shot.limit,
        "dead_ball_seconds": 0.0,
        "between_possession_seconds": 0.0,
        "categories": dict(categories),
        "counts": dict(counts),
        "passes": passes,
        "shot_clock_violations": violations,
        "offensive_rebounds": offensive_rebounds,
        "misses": misses,
        "release_shot_clock_remaining": release_shot_clock,
        "continued_after_offensive_rebound": offensive_rebounds > 0,
        "period_ending": end_reason == "period_expired" or game.quarter_remaining == 0.0,
        "end_reason": end_reason,
    }


def diagnose_game(era, seed: int, home: Team, away: Team, roster: dict) -> dict:
    """Play regulation with live decisions. Era pace is not an input."""
    rng = random.Random(seed)
    game = GameClock()
    teams = (home, away)
    possessions = []
    side = 0
    period_breaks = 0
    while not game.regulation_complete:
        if game.quarter_remaining == 0.0:
            if not game.start_next_quarter():
                break
            period_breaks += 1
        team = teams[side]
        opponent = teams[1 - side]
        offense = [roster[player_id] for player_id in team.player_ids]
        defense = [roster[player_id] for player_id in opponent.player_ids]
        started_quarter = game.quarter
        started_remaining = game.quarter_remaining
        result = _play_live_possession(
            game,
            rng,
            era,
            offense,
            defense,
            team.coach_style,
            _fatigue(game.elapsed),
        )
        result["offense"] = side
        result["quarter"] = started_quarter
        if game.quarter_remaining == 0.0 and started_remaining > 0:
            result["period_ending"] = True
        possessions.append(result)
        side = 1 - side

    durations = [possession["game_clock_elapsed"] for possession in possessions]
    per_team = [sum(1 for possession in possessions if possession["offense"] == index) for index in (0, 1)]
    categories = Counter()
    counts = Counter()
    release_left = []
    for possession in possessions:
        categories.update(possession["categories"])
        counts.update(possession["counts"])
        release_left.extend(possession["release_shot_clock_remaining"])
    misses = sum(possession["misses"] for possession in possessions)
    offensive_rebounds = sum(possession["offensive_rebounds"] for possession in possessions)
    return {
        "era": era.key,
        "seed": seed,
        "seeded_pace": era.pace,
        "pace_input": None,
        "possessions": len(possessions),
        "possessions_per_team": per_team,
        "durations": durations,
        "mean_seconds": (sum(durations) / len(durations)) if durations else 0.0,
        "passes": sum(possession["passes"] for possession in possessions),
        "shot_clock_violations": sum(possession["shot_clock_violations"] for possession in possessions),
        "period_ending_possessions": sum(1 for possession in possessions if possession["period_ending"]),
        "period_breaks": period_breaks,
        "quarter_lengths": game.quarter_lengths(),
        "regulation_seconds": sum(game.quarter_lengths()),
        "categories": dict(categories),
        "counts": dict(counts),
        "misses": misses,
        "offensive_rebounds": offensive_rebounds,
        "release_shot_clock_remaining": release_left,
        "offensive_rebound_possessions": sum(1 for possession in possessions if possession["continued_after_offensive_rebound"]),
        "possession_log": possessions,
        "promoted": False,
        "roster": "fictional_demo",
    }


def _duration_distribution(durations: list[float]) -> dict:
    buckets = {"under_8": 0, "8_to_12": 0, "12_to_16": 0, "16_to_24": 0, "over_24": 0}
    for value in durations:
        if value < 8:
            buckets["under_8"] += 1
        elif value < 12:
            buckets["8_to_12"] += 1
        elif value < 16:
            buckets["12_to_16"] += 1
        elif value <= 24:
            buckets["16_to_24"] += 1
        else:
            buckets["over_24"] += 1
    total = len(durations) or 1
    return {
        "min": min(durations) if durations else 0.0,
        "p10": _percentile(durations, 0.10),
        "p50": _percentile(durations, 0.50),
        "p90": _percentile(durations, 0.90),
        "max": max(durations) if durations else 0.0,
        "shares": {name: count / total for name, count in buckets.items()},
    }


def summarize_era(games: list[dict]) -> dict:
    count = len(games)
    durations = [value for game in games for value in game["durations"]]
    possessions = sum(game["possessions"] for game in games)
    categories = Counter()
    for game in games:
        categories.update(game["categories"])
    misses = sum(game["misses"] for game in games)
    offensive_rebounds = sum(game["offensive_rebounds"] for game in games)
    release_left = [value for game in games for value in game["release_shot_clock_remaining"]]
    per_team = [sum(game["possessions_per_team"][index] for game in games) / count for index in (0, 1)]
    mean_seconds = (sum(durations) / len(durations)) if durations else 0.0
    pace = games[0]["seeded_pace"]
    return {
        "era": games[0]["era"],
        "games": count,
        "seeded_pace": pace,
        "possessions_per_team": per_team,
        "possessions_per_team_mean": sum(per_team) / 2,
        "mean_seconds": mean_seconds,
        "distribution": _duration_distribution(durations),
        "passes_per_possession": sum(game["passes"] for game in games) / possessions,
        "offensive_rebound_rate": (offensive_rebounds / misses) if misses else None,
        "shot_clock_violations_per_game": sum(game["shot_clock_violations"] for game in games) / count,
        "quarter_ending_possessions_per_game": sum(game["period_ending_possessions"] for game in games) / count,
        "category_seconds_per_game": {name: value / count for name, value in categories.items()},
        "category_seconds_per_possession": {name: value / possessions for name, value in categories.items()},
        "mean_shot_clock_remaining_at_release": (sum(release_left) / len(release_left)) if release_left else None,
        "shooting_fouls_per_game": sum(game["counts"].get("shooting_foul", 0) for game in games) / count,
        "free_throws_per_game": sum(game["counts"].get("free_throw", 0) for game in games) / count,
        "turnovers_per_game": sum(
            game["counts"].get("live_turnover", 0) + game["counts"].get("pass_turnover", 0) for game in games
        ) / count,
        "seconds_if_seeded_pace_were_the_result": REGULATION_SECONDS / (2 * pace),
        "pace_input": None,
        "dead_ball_seconds_per_possession": 0.0,
        "between_possession_seconds_per_possession": 0.0,
        "promoted": False,
    }


def run_live_timing(eras, seeds=SEEDS) -> list[dict]:
    home, away, roster = fictional_matchup()
    rows = []
    for era in eras:
        games = [diagnose_game(era, seed, home, away, roster) for seed in seeds]
        rows.append(summarize_era(games))
    return rows


def timing_gaps(row: dict) -> dict:
    """Describe the measured gap. This does not add time to close it."""
    mean_seconds = row["mean_seconds"]
    needed = row["seconds_if_seeded_pace_were_the_result"]
    parts = row["category_seconds_per_possession"]
    return {
        "mean_seconds": mean_seconds,
        "seconds_at_seeded_pace": needed,
        "shortfall_seconds": needed - mean_seconds,
        "setup_seconds": parts.get("setup", 0.0),
        "pass_seconds": parts.get("completed_pass", 0.0),
        "release_seconds": parts.get("release", 0.0),
        "between_possession_seconds": 0.0,
        "dead_ball_seconds": 0.0,
        "shot_clock_left_at_release": row["mean_shot_clock_remaining_at_release"],
        "passes_per_possession": row["passes_per_possession"],
        "clock_accounts_for_the_period": abs(
            sum(row["category_seconds_per_game"].values()) - REGULATION_SECONDS
        ) < 1e-4,
    }
