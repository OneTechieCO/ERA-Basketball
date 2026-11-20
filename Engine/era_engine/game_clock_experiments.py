"""Experimental game clock. It does not play the live engine.

The shot clock limits one live ball. The game clock measures the quarter.
Those durations are tracked separately. A possession is not given the length
of the shot clock, and its length is not chosen to hit an era pace.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .core import offensive_rebound_reset
from .rotation_experiments import REGULATION_GAME_MINUTES, overtime_requirement

QUARTER_SECONDS = 12 * 60
REGULATION_QUARTERS = 4
REGULATION_SECONDS = QUARTER_SECONDS * REGULATION_QUARTERS
SHOT_CLOCK_SECONDS = 24
# Current NBA-era test configuration resets the shot clock to 14 after an
# offensive rebound. That reset is not a game-clock reset.
OFFENSIVE_REBOUND_RESET_SECONDS = 14

# Durations copied from the live shot-clock rules. They charge the shot clock
# during live ball. They are not a target possession length.
SETUP_SECONDS = 6.0
COMPLETED_PASS_SECONDS = 2.0
RELEASE_LOW_SECONDS = 1.5
RELEASE_HIGH_SECONDS = 3.5
MAX_PASSES = 6
LATE_CLOCK_NO_PASS_SECONDS = 2.0

# Frequencies copied only as timing drivers, not as changes to live formulas.
PASS_TURNOVER_PROBABILITY = 0.02
OFFENSIVE_REBOUND_PROBABILITY = 0.24

# Experimental timing drivers. These are not the live pass, foul, or shooting formulas.
EXPERIMENTAL_PASS_ATTEMPT_PROBABILITY = 0.35
EXPERIMENTAL_LIVE_TURNOVER_PROBABILITY = 0.08
EXPERIMENTAL_SHOOTING_FOUL_PROBABILITY = 0.10
EXPERIMENTAL_MAKE_PROBABILITY = 0.45

# Game clock only. The live shot clock has no rebound-gather duration.
REBOUND_GATHER_SECONDS = 1.0

SUBSTITUTION_ASSUMPTION = (
    "Substitutions are read from the experimental rotation at the start of a "
    "possession. The lineup that starts a live action finishes it, even if a "
    "scheduled substitution minute is crossed. This is a simplified experimental "
    "assumption, not a historical substitution rule."
)

PROMOTION = {
    "promoted": False,
    "integrated_with_live_engine": False,
}


def _snap(value: float) -> float:
    return 0.0 if value <= 1e-9 else value


@dataclass
class GameClock:
    """Regulation clock: four 12-minute quarters. Overtime is not started."""

    quarter: int = 1
    quarter_remaining: float = QUARTER_SECONDS
    quarter_length: float = QUARTER_SECONDS
    quarters: int = REGULATION_QUARTERS
    regulation_complete: bool = False

    @property
    def game_remaining(self) -> float:
        if self.regulation_complete:
            return 0.0
        later = (self.quarters - self.quarter) * self.quarter_length
        return later + self.quarter_remaining

    @property
    def elapsed(self) -> float:
        completed = (self.quarter - 1) * self.quarter_length
        if self.regulation_complete and self.quarter_remaining == 0.0:
            return float(self.quarters * self.quarter_length)
        return completed + (self.quarter_length - self.quarter_remaining)

    def advance_live(self, seconds: float) -> float:
        """Advance live playing time. Returns the time that actually elapsed."""
        if seconds < 0:
            raise ValueError("game clock cannot advance by a negative time")
        if self.regulation_complete:
            return 0.0
        elapsed = min(float(seconds), self.quarter_remaining)
        self.quarter_remaining = _snap(self.quarter_remaining - elapsed)
        if self.quarter_remaining == 0.0 and self.quarter == self.quarters:
            self.regulation_complete = True
        return elapsed

    def start_next_quarter(self) -> bool:
        """Start the next regulation quarter. Returns False when regulation is over."""
        if self.quarter_remaining != 0.0:
            raise ValueError("a quarter cannot change while time remains")
        if self.quarter >= self.quarters:
            self.regulation_complete = True
            return False
        self.quarter += 1
        self.quarter_remaining = self.quarter_length
        return True

    def quarter_lengths(self) -> list[float]:
        """Completed quarters are exactly 12 minutes. The current quarter reports time used."""
        lengths = []
        for quarter in range(1, self.quarters + 1):
            completed = quarter < self.quarter or self.regulation_complete
            if completed:
                lengths.append(self.quarter_length)
            elif quarter == self.quarter:
                lengths.append(self.quarter_length - self.quarter_remaining)
            else:
                lengths.append(0.0)
        return lengths


@dataclass
class ShotClock:
    """Possession limit. Resetting it does not move the game clock."""

    remaining: float = SHOT_CLOCK_SECONDS
    limit: float = SHOT_CLOCK_SECONDS

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("shot clock cannot advance by a negative time")
        elapsed = min(float(seconds), self.remaining)
        self.remaining = _snap(self.remaining - elapsed)
        return elapsed

    def reset(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("shot-clock reset must be positive")
        self.limit = float(seconds)
        self.remaining = float(seconds)


def setup_seconds(shot_clock_limit: float) -> float:
    """Live setup charge: min(6, one quarter of the shot clock)."""
    return min(SETUP_SECONDS, shot_clock_limit * 0.25)


def apply_live_action(game: GameClock, shot: ShotClock, intended_seconds: float, kind: str) -> dict:
    """Run one live action on both clocks, stopping at whichever expires first.

    The action does not finish when the period or the shot clock cannot hold
    the full duration. Unused intended time is not carried forward.
    """
    if intended_seconds < 0:
        raise ValueError("an action cannot take negative time")
    allowed = min(intended_seconds, game.quarter_remaining, shot.remaining)
    game_elapsed = game.advance_live(allowed)
    shot_elapsed = shot.advance(allowed)
    finished = allowed + 1e-9 >= intended_seconds
    reason = None
    if not finished:
        if game.quarter_remaining == 0.0:
            reason = "period_expired"
        elif shot.remaining == 0.0:
            reason = "shot_clock_expired"
        else:
            reason = "incomplete"
    return {
        "kind": kind,
        "intended_seconds": intended_seconds,
        "game_clock_elapsed": game_elapsed,
        "shot_clock_elapsed": shot_elapsed,
        "finished": finished,
        "reason": reason,
        "stopped_clock": False,
    }


def apply_stoppage(game: GameClock, kind: str, dead_seconds: float) -> dict:
    """Record a dead ball. The game clock and the shot clock stay where they are."""
    if dead_seconds < 0:
        raise ValueError("a stoppage cannot take negative time")
    before_game = game.quarter_remaining
    return {
        "kind": kind,
        "intended_seconds": dead_seconds,
        "game_clock_elapsed": 0.0,
        "shot_clock_elapsed": 0.0,
        "finished": True,
        "reason": None,
        "stopped_clock": True,
        "quarter_remaining_unchanged": game.quarter_remaining == before_game,
    }


def lineup_at(rotation: dict, elapsed_seconds: float) -> tuple[str, ...]:
    """Read the experimental rotation. Does not change the schedule."""
    if elapsed_seconds < 0:
        raise ValueError("elapsed time cannot be negative")
    minute = min(elapsed_seconds / 60.0, float(REGULATION_GAME_MINUTES) - 1e-9)
    for segment in rotation["segments"]:
        if segment["start_minute"] <= minute < segment["end_minute"]:
            return tuple(segment["on_court"])
    raise ValueError("elapsed time is outside the regulation rotation")


def _release_seconds(rng: random.Random) -> float:
    return rng.uniform(RELEASE_LOW_SECONDS, RELEASE_HIGH_SECONDS)


def _game_only(game: GameClock, intended_seconds: float, kind: str) -> dict:
    """Live game-clock time that the shot clock does not charge. A rebound gather is one."""
    if intended_seconds < 0:
        raise ValueError("an action cannot take negative time")
    elapsed = game.advance_live(min(intended_seconds, game.quarter_remaining))
    finished = elapsed + 1e-9 >= intended_seconds
    return {
        "kind": kind,
        "intended_seconds": intended_seconds,
        "game_clock_elapsed": elapsed,
        "shot_clock_elapsed": 0.0,
        "finished": finished,
        "reason": None if finished else "period_expired",
        "stopped_clock": False,
    }


def _play_possession(
    game: GameClock,
    rng: random.Random,
    rebound_reset: float,
    shot_clock: float = SHOT_CLOCK_SECONDS,
) -> dict:
    """One possession. The shot clock may reset; the game clock does not."""
    shot = ShotClock(shot_clock, shot_clock)
    actions = []
    violations = 0
    stopped = 0.0

    def live(intended: float, kind: str) -> dict:
        action = apply_live_action(game, shot, intended, kind)
        actions.append(action)
        return action

    setup = live(setup_seconds(shot.limit), "setup")
    if not setup["finished"]:
        return _possession_record(game, shot, actions, violations, stopped, setup["reason"])
    if rng.random() < EXPERIMENTAL_LIVE_TURNOVER_PROBABILITY:
        actions.append(apply_stoppage(game, "live_ball_turnover", 0.0))
        return _possession_record(game, shot, actions, violations, stopped, "turnover")

    while True:
        passes = 0
        while passes < MAX_PASSES and shot.remaining > LATE_CLOCK_NO_PASS_SECONDS and game.quarter_remaining > 0:
            if rng.random() >= EXPERIMENTAL_PASS_ATTEMPT_PROBABILITY:
                break
            if rng.random() < PASS_TURNOVER_PROBABILITY:
                actions.append(apply_stoppage(game, "pass_turnover", 0.0))
                return _possession_record(game, shot, actions, violations, stopped, "turnover")
            passed = live(COMPLETED_PASS_SECONDS, "completed_pass")
            if not passed["finished"]:
                reason = "shot_clock_violation" if passed["reason"] == "shot_clock_expired" else passed["reason"]
                if reason == "shot_clock_violation":
                    violations += 1
                return _possession_record(game, shot, actions, violations, stopped, reason)
            passes += 1
        if game.quarter_remaining == 0.0:
            return _possession_record(game, shot, actions, violations, stopped, "period_expired")

        release = live(_release_seconds(rng), "release")
        if not release["finished"]:
            reason = "shot_clock_violation" if release["reason"] == "shot_clock_expired" else release["reason"]
            if reason == "shot_clock_violation":
                violations += 1
            return _possession_record(game, shot, actions, violations, stopped, reason)

        roll = rng.random()
        if roll < EXPERIMENTAL_SHOOTING_FOUL_PROBABILITY:
            for _ in range(2):
                stoppage = apply_stoppage(game, "free_throw", 1.0)
                actions.append(stoppage)
                stopped += stoppage["intended_seconds"]
            return _possession_record(game, shot, actions, violations, stopped, "shooting_foul")
        if roll < EXPERIMENTAL_SHOOTING_FOUL_PROBABILITY + EXPERIMENTAL_MAKE_PROBABILITY:
            return _possession_record(game, shot, actions, violations, stopped, "make")

        gather = _game_only(game, REBOUND_GATHER_SECONDS, "rebound_gather")
        actions.append(gather)
        if not gather["finished"]:
            return _possession_record(game, shot, actions, violations, stopped, "period_expired")
        if rng.random() >= OFFENSIVE_REBOUND_PROBABILITY or game.quarter_remaining == 0.0:
            return _possession_record(game, shot, actions, violations, stopped, "defensive_rebound")
        actions.append({
            "kind": "offensive_rebound",
            "game_clock_elapsed": 0.0,
            "shot_clock_elapsed": 0.0,
            "finished": True,
            "reason": None,
            "stopped_clock": False,
            "shot_clock_reset": rebound_reset,
        })
        shot.reset(rebound_reset)


def _possession_record(game, shot, actions, violations, stopped, end_reason) -> dict:
    game_elapsed = sum(action.get("game_clock_elapsed", 0.0) for action in actions)
    shot_elapsed = sum(action.get("shot_clock_elapsed", 0.0) for action in actions)
    return {
        "quarter": game.quarter if game.quarter_remaining > 0 or game.quarter == REGULATION_QUARTERS else game.quarter,
        "actions": actions,
        "game_clock_elapsed": game_elapsed,
        "shot_clock_time_consumed": shot_elapsed,
        "stopped_clock_seconds": stopped,
        "shot_clock_violations": violations,
        "period_ending": end_reason == "period_expired" or game.quarter_remaining == 0.0,
        "end_reason": end_reason,
        "shot_clock_remaining": shot.remaining,
    }


def play_regulation_game(
    seed: int,
    home_rotation: dict,
    away_rotation: dict,
    rebound_reset: float = OFFENSIVE_REBOUND_RESET_SECONDS,
    shot_clock: float = SHOT_CLOCK_SECONDS,
) -> dict:
    """Play regulation from the clocks and the rotation schedules.

    Possession counts are whatever fits in 48 minutes. Era pace is not an input.
    """
    rng = random.Random(seed)
    game = GameClock()
    period_breaks = 0
    rotations = (home_rotation, away_rotation)
    names = (home_rotation.get("team_name", "home"), away_rotation.get("team_name", "away"))
    minute_maps = tuple(_minute_map(rotation) for rotation in rotations)
    credited = tuple({player_id: 0.0 for player_id in minute_map} for minute_map in minute_maps)
    previous = [None, None]
    substitutions = [0, 0]
    possessions = []
    side = 0
    while not game.regulation_complete:
        if game.quarter_remaining == 0.0:
            if not game.start_next_quarter():
                break
            period_breaks += 1
        elapsed = game.elapsed
        lineups = []
        for index, rotation in enumerate(rotations):
            lineup = lineup_at(rotation, elapsed)
            if len(lineup) != 5 or len(set(lineup)) != 5:
                raise ValueError("rotation did not supply five players")
            if previous[index] is not None and lineup != previous[index]:
                substitutions[index] += 1
            previous[index] = lineup
            lineups.append(lineup)
        started_quarter = game.quarter
        started_remaining = game.quarter_remaining
        result = _play_possession(game, rng, rebound_reset, shot_clock)
        result["offense"] = side
        result["quarter"] = started_quarter
        result["lineups"] = lineups
        result["elapsed_at_start"] = elapsed
        possessions.append(result)
        for index, lineup in enumerate(lineups):
            for player_id in lineup:
                credited[index][player_id] += result["game_clock_elapsed"] / 60.0
        if game.quarter_remaining == 0.0 and started_remaining > 0:
            result["period_ending"] = True
        side = 1 - side

    per_team = [sum(1 for possession in possessions if possession["offense"] == index) for index in (0, 1)]
    elapsed_times = [possession["game_clock_elapsed"] for possession in possessions]
    return {
        "seed": seed,
        "team_names": names,
        "possessions": len(possessions),
        "possessions_per_team": per_team,
        "average_game_clock_per_possession": (sum(elapsed_times) / len(elapsed_times)) if elapsed_times else 0.0,
        "quarter_lengths": game.quarter_lengths(),
        "regulation_seconds": sum(game.quarter_lengths()),
        "substitutions": substitutions,
        "player_minutes": credited,
        "shot_clock_violations": sum(possession["shot_clock_violations"] for possession in possessions),
        "period_ending_possessions": sum(1 for possession in possessions if possession["period_ending"]),
        "possession_log": possessions,
        "period_breaks": period_breaks,
        "shot_clock": shot_clock,
        "rebound_reset": rebound_reset,
        "overtime": overtime_requirement(),
        "substitution_assumption": SUBSTITUTION_ASSUMPTION,
        "promoted": False,
        "pace_input": None,
    }


def play_era_regulation(seed: int, era, home_rotation: dict, away_rotation: dict) -> dict:
    """Play one seeded era. Pace is stored for comparison and does not set the possession count."""
    reset = offensive_rebound_reset(era)
    game = play_regulation_game(
        seed,
        home_rotation,
        away_rotation,
        rebound_reset=reset,
        shot_clock=era.shot_clock,
    )
    game["era"] = era.key
    game["seeded_pace"] = era.pace
    return game


def simulate_seeded_eras(eras, home_rotation: dict, away_rotation: dict, seeds) -> list[dict]:
    """Replay the same seeds in each era. Differences come from the shot-clock rules only."""
    rows = []
    for era in eras:
        games = [play_era_regulation(seed, era, home_rotation, away_rotation) for seed in seeds]
        summary = summarize_games(games)
        summary["era"] = era.key
        summary["seeded_pace"] = era.pace
        summary["shot_clock"] = era.shot_clock
        summary["rebound_reset"] = offensive_rebound_reset(era)
        summary["possessions_per_team"] = summary["possessions_per_team_mean"]
        summary["quarter_endings_mean"] = summary["period_ending_possessions_mean"]
        summary["pace_gap"] = summary["possessions_per_team_combined_mean"] - era.pace
        summary["pace_used_to_choose_possessions"] = False
        rows.append(summary)
    return rows


def _minute_map(rotation: dict) -> dict[str, float]:
    return {player_id: float(minutes) for player_id, minutes in rotation["minutes"].items()}


def summarize_games(games: list[dict]) -> dict:
    """Average experimental games. Pace is compared later and is not fitted here."""
    count = len(games)
    per_team = [sum(game["possessions_per_team"][index] for game in games) / count for index in (0, 1)]
    return {
        "games": count,
        "possessions_per_team_mean": per_team,
        "possessions_per_team_combined_mean": sum(per_team) / 2,
        "average_game_clock_per_possession": sum(game["average_game_clock_per_possession"] for game in games) / count,
        "shot_clock_violations_mean": sum(game["shot_clock_violations"] for game in games) / count,
        "period_ending_possessions_mean": sum(game["period_ending_possessions"] for game in games) / count,
        "substitutions_mean": [
            sum(game["substitutions"][index] for game in games) / count for index in (0, 1)
        ],
        "promoted": False,
    }
