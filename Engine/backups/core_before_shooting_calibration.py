"""Deterministic-seed 5v5 statistical prototype with a regulation game clock.

The live game is four quarters that add up to era.game_minutes. A tied
regulation game continues in five-minute overtime periods until the score is
not tied. Possession length is whatever the existing shot-clock charges
consume on that clock. Those durations are configurable and are not calibrated
NBA timings. Era pace does not choose the possession count. A roster of five
stays on the court. A larger roster uses the regulation rotation scheduler.
Substitutions happen at a period break or a modeled whistle, not on a live ball
or a made-basket inbound. A stopped clock restarts on the first live action.

IMPORTANT: parameters in seed.py are illustrative, not calibrated NBA estimates.
No real-player likenesses, NBA data, or rights-controlled datasets are bundled.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import math
import random
from typing import Any

ZONES = ('rim', 'paint', 'mid', 'three')
POINTS = {'rim': 2, 'paint': 2, 'mid': 2, 'three': 3}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class Era:
    key: str
    pace: float  # approximate team possessions / 48 min
    three_share: float
    physicality: float  # 0..1 contest intensity, not a direct skill penalty
    zone_allowed: bool
    three_distance_factor: float = 1.0
    shot_clock: int = 24
    game_minutes: int = 48
    rule_note: str = ''


    def __post_init__(self):
        if not 0 < self.pace < float('inf'):
            raise ValueError('Era pace must be positive and finite')
        if not 0 <= self.three_share <= 1:
            raise ValueError('Three-point share must be between 0 and 1')
        if not 0 <= self.physicality <= 1:
            raise ValueError('Physicality must be between 0 and 1')
        if not 0 < self.three_distance_factor < float('inf'):
            raise ValueError('Three-point distance factor must be positive and finite')
        if not isinstance(self.shot_clock, int) or self.shot_clock <= 0:
            raise ValueError('Shot clock must be a positive integer')
        if not isinstance(self.game_minutes, int) or self.game_minutes <= 0:
            raise ValueError('Game minutes must be a positive integer')


def offensive_rebound_reset(era: Era) -> int:
    """Shot-clock reset after an offensive rebound.

    Seasons from 2018 onward reset to 14 seconds. Earlier seasons reset to
    the full possession clock. A shorter configured clock is never raised.
    """
    season_start = int(era.key[:4])
    if season_start >= 2018:
        return min(14, era.shot_clock)
    return era.shot_clock


@dataclass
class PossessionClock:
    remaining: float

    def advance(self, seconds: float) -> bool:
        """Consume time. Return True if the clock has expired."""
        if seconds < 0:
            raise ValueError('Elapsed time cannot be negative')
        self.remaining = max(0.0, self.remaining - seconds)
        return self.remaining == 0.0

    def reset(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError('Reset time must be positive')
        self.remaining = float(seconds)


def _snap_clock(value: float) -> float:
    return 0.0 if value <= 1e-9 else value


OVERTIME_PERIOD_SECONDS = 300.0
OVERTIME_RUNAWAY_LIMIT = 15
LATE_MAKE_STOP_QUARTERS_1_TO_3 = 60.0
LATE_MAKE_STOP_FINAL_PERIOD = 120.0
JUMP_BALL_MODEL = (
    'Simplified center jump. No historical tip rate is stored. Each team '
    'jumps with its on-court player who has the highest rebounding rating. '
    'A tie between teammates uses the lower player id. The higher rating '
    'wins the tip. If the two ratings are equal, the lower player id wins, '
    'and a remaining tie goes to team A. The choice does not draw a random '
    'number. This is not a measured jump-ball percentage. Regulation opens as a '
    'period start with team A, so the tip is not drawn before quarter 1.'
)


class UnresolvedGameError(RuntimeError):
    """The overtime safety limit ended with the score still tied.

    result is the partial game. It is not a completed game, and no point
    was added to produce a winner.
    """

    def __init__(self, result: dict[str, Any]):
        self.result = result
        periods = result.get('overtime', {}).get('periods', OVERTIME_RUNAWAY_LIMIT)
        super().__init__(
            f'{periods} overtime periods ended tied. '
            'No tie-break point is awarded. This game is incomplete.'
        )


def made_basket_stops_clock(period_number: int, remaining: float) -> bool:
    """Whether a made basket stops the game clock inside a live period.

    Quarters 1-3 stop during the final minute. Quarter 4 and every overtime
    period stop during the final two minutes. Zero remaining is the period
    horn, so that make is not given this stoppage as well.
    """
    if remaining <= 0.0:
        return False
    if period_number >= 4:
        return remaining <= LATE_MAKE_STOP_FINAL_PERIOD
    return remaining <= LATE_MAKE_STOP_QUARTERS_1_TO_3


def game_clock_stopped_for_start(start: str, make_stopped_clock: bool) -> bool:
    """The game clock is stopped through an inbound, a tip, or a period start.

    A live-ball change keeps it running. A made-basket inbound stops it only
    in the late-period window. Neither clock runs during that dead ball. Both
    start together when the ball is legally touched.
    """
    if start == 'live_ball':
        return False
    if start == 'made_basket_inbound':
        return make_stopped_clock
    if start in ('period_start', 'jump_ball', 'dead_ball_inbound'):
        return True
    raise ValueError(f'unknown possession start {start}')


@dataclass
class RegulationClock:
    """Four equal quarters, then optional five-minute overtime periods.

    The shot clock does not move this clock backward. Overtime starts only
    when start_overtime is called after regulation. Advancing a finished
    regulation clock does not enter overtime by itself.
    """

    quarter_length: float
    quarters: int = 4
    quarter: int = 1
    quarter_remaining: float = 0.0
    regulation_complete: bool = False
    overtime_length: float = OVERTIME_PERIOD_SECONDS
    overtime_remaining: float = 0.0
    in_overtime: bool = False
    completed_overtime: int = 0
    overtime_period: int = 0

    def __post_init__(self) -> None:
        if self.quarter_length <= 0 or self.quarters <= 0:
            raise ValueError('Regulation clock must have a positive length')
        if self.overtime_length <= 0:
            raise ValueError('Overtime period must have a positive length')
        self.quarter_remaining = float(self.quarter_length)

    def period_remaining(self) -> float:
        """Seconds left in the quarter or, during overtime, in that period."""
        if self.in_overtime:
            return self.overtime_remaining
        return self.quarter_remaining

    @property
    def elapsed(self) -> float:
        if self.regulation_complete:
            regulation = self.quarter_length * self.quarters
        else:
            completed = (self.quarter - 1) * self.quarter_length
            regulation = completed + (self.quarter_length - self.quarter_remaining)
        overtime = self.completed_overtime * self.overtime_length
        if self.in_overtime:
            overtime += self.overtime_length - self.overtime_remaining
        return regulation + overtime

    def advance(self, seconds: float) -> float:
        """Advance live playing time. Dead-ball time must not call this."""
        if seconds < 0:
            raise ValueError('Elapsed time cannot be negative')
        if self.in_overtime:
            elapsed = min(float(seconds), self.overtime_remaining)
            self.overtime_remaining = _snap_clock(self.overtime_remaining - elapsed)
            if self.overtime_remaining == 0.0:
                self.in_overtime = False
                self.completed_overtime += 1
            return elapsed
        if self.regulation_complete:
            return 0.0
        elapsed = min(float(seconds), self.quarter_remaining)
        self.quarter_remaining = _snap_clock(self.quarter_remaining - elapsed)
        if self.quarter_remaining == 0.0 and self.quarter == self.quarters:
            self.regulation_complete = True
        return elapsed

    def start_next_quarter(self) -> bool:
        """Start the next quarter. Returns False when regulation is over."""
        if self.quarter_remaining != 0.0:
            raise ValueError('A quarter cannot change while time remains')
        if self.quarter >= self.quarters:
            self.regulation_complete = True
            return False
        self.quarter += 1
        self.quarter_remaining = self.quarter_length
        return True

    def start_overtime(self) -> int:
        """Start the next five-minute period. Regulation is left unchanged."""
        if self.in_overtime:
            raise ValueError('overtime is already running')
        if not self.regulation_complete or self.quarter_remaining != 0.0:
            raise ValueError('overtime starts after regulation')
        self.overtime_period = self.completed_overtime + 1
        self.in_overtime = True
        self.overtime_remaining = float(self.overtime_length)
        return self.overtime_period

    def quarter_lengths(self) -> list[float]:
        if self.regulation_complete:
            return [self.quarter_length] * self.quarters
        lengths = []
        for quarter in range(1, self.quarters + 1):
            if quarter < self.quarter:
                lengths.append(self.quarter_length)
            elif quarter == self.quarter:
                lengths.append(self.quarter_length - self.quarter_remaining)
            else:
                lengths.append(0.0)
        return lengths


def _charge_live(game: RegulationClock, shot: PossessionClock, seconds: float) -> tuple[float, str | None]:
    """Move the game clock and the shot clock by the same live-ball time.

    The action stops at whichever clock runs out first. Neither clock goes
    below zero. A finished action returns None even if it lands on a horn.
    """
    if seconds < 0:
        raise ValueError('Elapsed time cannot be negative')
    quarter_left = game.period_remaining()
    shot_left = shot.remaining
    allowed = min(float(seconds), quarter_left, shot_left)
    elapsed = game.advance(allowed)
    shot.advance(allowed)
    if allowed + 1e-9 >= float(seconds):
        return elapsed, None
    if quarter_left < shot_left:
        return elapsed, 'period'
    return elapsed, 'shot_clock'


@dataclass(frozen=True)
class TimingAssumptions:
    """Durations already used by the live shot clock, plus one game-clock gather.

    These are not measurements of NBA games. setup_cap_seconds repeats the
    existing rule min(cap, one quarter of the shot clock) and now also runs
    the game clock. A completed pass is 2 seconds and a release is uniform
    between the two release bounds, matching the shot-clock rules. The
    rebound gather runs only on the game clock after a missed field goal. It
    does not change the shot clock, and free throws do not use it: free throws
    are dead-ball time and add no game-clock seconds. None of these values
    has been calibrated to historical possession length.
    """

    setup_cap_seconds: float = 6.0
    completed_pass_seconds: float = 2.0
    release_low_seconds: float = 1.5
    release_high_seconds: float = 3.5
    rebound_gather_seconds: float = 1.0
    calibrated: bool = False

    def setup_seconds(self, shot_clock_limit: float) -> float:
        return min(self.setup_cap_seconds, float(shot_clock_limit) * 0.25)


@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    origin_era: str
    offense: int
    defense: int
    passing: int
    rebounding: int
    iq: int
    stamina: int
    usage: float
    # make probabilities per zone *conditional on taking shot* at neutral context
    shot_skills: dict[str, int]
    # relative preferred zone weights (sum need not be 1)
    tendencies: dict[str, float]
    # specializations
    steals: int = 65
    blocks: int = 60
    injury_resilience: int = 75


@dataclass(frozen=True)
class Team:
    name: str
    player_ids: tuple[str, ...]
    coach_style: str = 'balanced'


def _player_fatigue(seconds_played: float, regulation_seconds: float) -> float:
    """Court-time input for the existing fatigue term.

    The coefficient is the live 0.42 scale. A player who has been on the
    court for the whole game matches the previous shared game-clock value.
    A substitute starts from the seconds that player has actually played.
    """
    if regulation_seconds <= 0:
        return 0.0
    return clamp(seconds_played / regulation_seconds * .42, 0, .42)


def _lineup_at(rotation: dict, elapsed_seconds: float) -> tuple[str, ...]:
    """Read one scheduler segment. The lookup does not change the plan."""
    minute = min(max(elapsed_seconds, 0.0) / 60.0, 48.0 - 1e-9)
    for segment in rotation['segments']:
        if segment['start_minute'] <= minute < segment['end_minute']:
            return tuple(segment['on_court'])
    raise ValueError('elapsed time is outside the regulation rotation')


def _rotation_plan(team: Team, roster: dict[str, Player], targets: dict[str, float] | None) -> dict:
    """Five listed players stay in that order. Larger rosters use the scheduler."""
    ids = tuple(team.player_ids)
    if targets is None and len(ids) == 5:
        return {
            'mode': 'full_game',
            'order': ids,
            'planned_minutes': {pid: 48 for pid in ids},
            'planned_substitutions': [],
        }
    from era_engine.rotation_experiments import RotationPlayer, schedule_regulation
    if targets is None:
        strategy = 'equal_time'
        group = [RotationPlayer(pid, 0.0, True, roster[pid].stamina) for pid in ids]
        starting: tuple[str, ...] | None = ids[:5]
    else:
        strategy = 'target_minutes'
        unknown = [pid for pid in targets if pid not in ids]
        if unknown:
            raise ValueError('rotation target names a player who is not on the team')
        group = [
            RotationPlayer(pid, float(targets.get(pid, 0.0)), True, roster[pid].stamina)
            for pid in ids
        ]
        starting = None
    scheduled = schedule_regulation(group, strategy, starting_five=starting)
    return {
        'mode': 'scheduled',
        'rotation': scheduled,
        'planned_minutes': dict(scheduled['minutes']),
        'planned_substitutions': scheduled['substitutions'],
    }


def _active_five(plan: dict, elapsed_seconds: float) -> tuple[str, ...]:
    if plan['mode'] == 'full_game':
        return plan['order']
    return _lineup_at(plan['rotation'], elapsed_seconds)


def _stint_limit_seconds(stamina: int) -> float:
    """Same stint length the regulation scheduler uses, in seconds."""
    return max(4, round(stamina / 100 * 16)) * 60.0


def _preferred_five(plan: dict) -> tuple[str, ...]:
    """Players with the largest regulation minute targets. A zero target stays off."""
    if plan['mode'] == 'full_game':
        return plan['order']
    ranked = sorted(
        plan['planned_minutes'],
        key=lambda pid: (-float(plan['planned_minutes'][pid]), pid),
    )
    chosen = [pid for pid in ranked if float(plan['planned_minutes'][pid]) > 0][:5]
    if len(chosen) != 5:
        raise RuntimeError('overtime needs five players with regulation minutes')
    return tuple(chosen)


def _overtime_five(plan: dict, roster: dict[str, Player], current: tuple[str, ...] | None,
                   stints: dict[str, float], period_break: bool) -> tuple[str, ...]:
    """Period breaks use the preferred five. A whistle only replaces a finished stint.

    The regulation scheduler has no overtime minute plan. This does not invent
    a bench player or add a substitution just to hit a target.
    """
    preferred = _preferred_five(plan)
    if plan['mode'] == 'full_game':
        return preferred
    ranked = [
        pid for pid in sorted(plan['planned_minutes'], key=lambda pid: (-float(plan['planned_minutes'][pid]), pid))
        if float(plan['planned_minutes'][pid]) > 0
    ]
    base = list(preferred if period_break or current is None else current)
    for pid in list(base):
        if stints.get(pid, 0.0) + 1e-9 < _stint_limit_seconds(roster[pid].stamina):
            continue
        for candidate in ranked:
            if candidate not in base:
                base[base.index(pid)] = candidate
                break
    if len(base) != 5 or len(set(base)) != 5:
        return preferred
    return tuple(base)


def _lineup_for_stoppage(plan: dict, roster: dict[str, Player], current: tuple[str, ...] | None,
                         stints: dict[str, float], game: RegulationClock, reason: str) -> tuple[str, ...]:
    """Choose the five allowed on the court at a legal stoppage."""
    if plan['mode'] == 'full_game':
        return plan['order']
    if game.in_overtime:
        return _overtime_five(plan, roster, current, stints, reason == 'period_break' or current is None)
    if current is not None and reason not in ('period_break', 'shooting_foul', 'shot_clock_violation'):
        return current
    return _lineup_at(plan['rotation'], game.elapsed)


def jump_ball_winner(ids_a: tuple[str, ...], ids_b: tuple[str, ...], roster: dict[str, Player]) -> int:
    """Return 0 or 1. See JUMP_BALL_MODEL. This function does not use a random draw."""
    def jumper(ids: tuple[str, ...]) -> str:
        return min(ids, key=lambda pid: (-roster[pid].rebounding, pid))

    player_a = jumper(ids_a)
    player_b = jumper(ids_b)
    rebound_a = roster[player_a].rebounding
    rebound_b = roster[player_b].rebounding
    if rebound_a != rebound_b:
        return 0 if rebound_a > rebound_b else 1
    if player_a != player_b:
        return 0 if player_a < player_b else 1
    return 0


def selection_weight(player: Player) -> float:
    """Weight for the initial ballhandler and for a pass receiver.

    Proportional to usage and IQ. Usage is an involvement weight, not a
    field-goal share, and no shot quota is assigned. The retired max() floors
    sat above every seeded raw weight and made both choices uniform. A tiny
    floor keeps a zero-usage player eligible.
    """
    return max(float(player.usage), 1e-9) * (70 + player.iq) / 140


def weighted_pick(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(max(0.0, x) for x in weights.values())
    if total <= 0:
        raise ValueError('Expected positive weights')
    needle = rng.random() * total
    for k, w in weights.items():
        needle -= max(0.0, w)
        if needle <= 0:
            return k
    return next(reversed(weights))


def shot_weights(p: Player, era: Era, coach_style: str, adaptation: float) -> dict[str, float]:
    """Player preference persists while opportunities shift toward the target era."""
    baseline = {'rim': .25, 'paint': .20, 'mid': .30, 'three': .25}
    baseline['three'] = era.three_share
    remaining = 1.0 - era.three_share
    for z, share in [('rim', .34), ('paint', .26), ('mid', .40)]:
        baseline[z] = remaining * share
    mix = clamp(.25 + .35 * adaptation, .25, .60)
    out = {z: (1 - mix) * max(.001, p.tendencies.get(z, .01)) + mix * baseline[z] for z in ZONES}
    if coach_style == 'perimeter':
        out['three'] *= 1.25
    elif coach_style == 'post':
        out['rim'] *= 1.15
        out['paint'] *= 1.18
    return out


def probability_make(p: Player, defender: Player, era: Era, zone: str, fatigue: float, adaptation: float, openness: float) -> float:
    # Baselines are development placeholders, NOT inferred from NBA history.
    base = {'rim': .61, 'paint': .43, 'mid': .40, 'three': .34}[zone]
    shooting = (p.shot_skills[zone] - 75) * .004
    contest = (defender.defense - 70) * .0018 + era.physicality * .024
    spacing = .018 * (1 if era.zone_allowed else -1) if zone == 'rim' else 0
    distance = -.018 * (era.three_distance_factor - 1) if zone == 'three' else 0
    return clamp(base + shooting - contest + spacing + distance + .035 * openness - .055 * fatigue + .023 * adaptation, .05, .88)


def risk_injury(p: Player, workload: float, fatigue: float) -> float:
    # rare per-game event; can be replaced by calibrated injury model
    return clamp(.003 + .006 * fatigue + .003 * workload + (75 - p.injury_resilience) * .00008, .0003, .035)


def _tendency_shares(player: Player) -> dict[str, float]:
    raw = {zone: max(0.001, player.tendencies.get(zone, 0.01)) for zone in ZONES}
    total = sum(raw.values())
    return {zone: raw[zone] / total for zone in ZONES}


def _expected_shot_value(player: Player) -> float:
    """Neutral expected points of the shots this player already prefers.

    Uses the live zone baselines and the live skill term. It does not change
    a shot once it is taken.
    """
    baseline = {'rim': .61, 'paint': .43, 'mid': .40, 'three': .34}
    shares = _tendency_shares(player)
    value = 0.0
    for zone in ZONES:
        make = clamp(baseline[zone] + (player.shot_skills[zone] - 75) * .004, .05, .88)
        value += shares[zone] * make * POINTS[zone]
    return value


def intelligent_pass_chance(
    shooter: Player,
    defender: Player,
    style_bonus: float,
    pass_count: int,
    clock_remaining: float,
) -> float:
    """Chance the ballhandler passes instead of using the current look.

    A favorable look is the expected points of the shots this player already
    tends to take. Passing skill and IQ still move the ball, and a high
    passing rating does not by itself override that look. No shot share is
    assigned. The late-clock gates still run after this value is returned.
    """
    shot_finish = (1.0 - _expected_shot_value(shooter)) * 0.70
    creation = (shooter.passing - 70) * 0.0035 + (shooter.iq - 70) * 0.0010
    usage_finish = (0.19 - shooter.usage) * 0.35
    pressure = (defender.defense - 73) * 0.0016
    time_room = clamp((clock_remaining - 12.0) / 80.0, -0.05, 0.05)
    read = (shooter.iq - 78) / 80.0
    recognition = read * (shot_finish * 0.40 + pressure * 0.50)
    chance = (
        0.42
        + shot_finish
        + creation
        + usage_finish
        + pressure
        + time_room
        + recognition
        + style_bonus
        - pass_count * 0.045
    )
    return clamp(chance, 0.12, 0.82)


def simulate_game(team_a: Team, team_b: Team, players: dict[str, Player], era: Era, seed: int = 7,
                  adaptation_a: float = .0, adaptation_b: float = .0,
                  track_events: bool = True, timing: TimingAssumptions | None = None,
                  rotation_targets: tuple[dict[str, float] | None, dict[str, float] | None] | None = None,
                  *, equalize_periods: tuple[str, ...] = ()) -> dict[str, Any]:
    """Simulate regulation and, when the score is tied, five-minute overtime periods.

    possessions_per_team is team A's possession count. Team B is
    possessions_by_team[1] and can differ by one. Era pace is stored and is
    not used to choose either count. Timing durations are not calibrated.
    A team with five players keeps that lineup. A larger roster uses the
    regulation scheduler. Substitutions are applied only at a period break,
    a shooting foul, or a shot-clock violation. A made-basket inbound, a
    steal, and a defensive rebound do not open a substitution window.
    Timeouts are not modeled. A stopped throw-in, period start, or jump ball
    does not run either clock until the legal touch. That touch is one live
    setup charge on both clocks, and a clock that is already running is not
    charged a second time. The overtime safety limit raises UnresolvedGameError
    instead of returning a tied game. Optional rotation_targets give each team a regulation minute
    budget that sums to 240. equalize_periods is a test fixture and is empty
    in production. AI coaching is simplified, not fully historical, and there
    is no 3D renderer.
    """
    rng = random.Random(seed)
    teams = (team_a, team_b)
    if any(len(team.player_ids) < 5 for team in teams):
        raise ValueError('A team needs at least five players')
    if any(pid not in players for team in teams for pid in team.player_ids):
        raise ValueError('Unknown player')
    if len(set(team_a.player_ids)) != len(team_a.player_ids):
        raise ValueError('Duplicate player on team A')
    if len(set(team_b.player_ids)) != len(team_b.player_ids):
        raise ValueError('Duplicate player on team B')
    if set(team_a.player_ids) & set(team_b.player_ids):
        raise ValueError('Player cannot appear on both teams')
    if timing is None:
        timing = TimingAssumptions()
    for key in equalize_periods:
        if key != 'regulation' and not (key.startswith('OT') and key[2:].isdigit() and int(key[2:]) > 0):
            raise ValueError('equalize_periods accepts regulation or an overtime period such as OT1')
    equalize_periods = tuple(equalize_periods)
    target_pair = rotation_targets or (None, None)
    if len(target_pair) != 2:
        raise ValueError('rotation targets must provide one budget for each team')
    plans = tuple(_rotation_plan(team, players, target_pair[index]) for index, team in enumerate(teams))
    if any(plan['mode'] == 'scheduled' for plan in plans) and era.game_minutes != 48:
        raise ValueError('the rotation scheduler covers a 48-minute regulation game')
    stats: dict[str, Counter] = {
        pid: Counter()
        for team in teams for pid in team.player_ids
    }
    score = [0, 0]
    events: list[dict[str, Any]] = []
    live_seconds: dict[str, float] = {}
    quarter_length = era.game_minutes * 60 / 4
    game_clock = RegulationClock(quarter_length, 4)
    poss_index = 0
    side = 0
    possessions_by_team = [0, 0]
    period_endings = 0
    played_seconds = {pid: 0.0 for team in teams for pid in team.player_ids}
    previous_lineup: list[tuple[str, ...] | None] = [None, None]
    on_court: list[tuple[str, ...] | None] = [None, None]
    stint_seconds = [{pid: 0.0 for pid in team.player_ids} for team in teams]
    applied_subs: list[list[dict[str, Any]]] = [[], []]
    substitution_window = True
    window_reason = 'period_break'
    period_scores: list[dict[str, Any]] = []
    period_label = 'Q1'
    period_points_start = [0, 0]
    regulation_score: list[int] | None = None
    fixture_points: list[dict[str, Any]] = []
    jump_balls: list[dict[str, Any]] = []
    possession_start = 'period_start'
    make_stopped_clock = False

    def grant_equalizer(key: str) -> None:
        trailer = 0 if score[0] < score[1] else 1
        amount = abs(score[0] - score[1])
        pid = on_court[trailer][0]
        stats[pid]['fta'] += amount
        stats[pid]['ftm'] += amount
        stats[pid]['pts'] += amount
        score[trailer] += amount
        fixture_points.append({
            'period': key,
            'player_id': pid,
            'team_index': trailer,
            'free_throws': amount,
            'points': amount,
            'label': 'test fixture free throws; not a production tie-break',
        })

    def close_period(equalize_key: str | None = None) -> None:
        if equalize_key and equalize_key in equalize_periods and score[0] != score[1]:
            grant_equalizer(equalize_key)
        period_scores.append({
            'period': period_label,
            'points': [score[0] - period_points_start[0], score[1] - period_points_start[1]],
            'running_score': [score[0], score[1]],
            'seconds': OVERTIME_PERIOD_SECONDS if period_label.startswith('OT') else quarter_length,
        })

    def apply_lineups(reason: str, event_period: int) -> None:
        nonlocal substitution_window
        for index in (0, 1):
            current = _lineup_for_stoppage(
                plans[index], players, on_court[index], stint_seconds[index], game_clock, reason,
            )
            if len(current) != 5 or len(set(current)) != 5:
                raise RuntimeError('a live lineup must contain five distinct players')
            previous = previous_lineup[index]
            if previous is not None and set(current) != set(previous):
                applied_subs[index].append({
                    'possession': poss_index,
                    'quarter': event_period,
                    'elapsed_seconds': game_clock.elapsed,
                    'players_out': sorted(set(previous) - set(current)),
                    'players_in': sorted(set(current) - set(previous)),
                    'reason': reason,
                })
            previous_lineup[index] = current
            on_court[index] = current
        substitution_window = False

    phase = 'regulation'

    def begin_overtime() -> bool:
        nonlocal period_label, period_points_start, substitution_window, window_reason, side, phase
        nonlocal possession_start, make_stopped_clock
        if game_clock.completed_overtime >= OVERTIME_RUNAWAY_LIMIT:
            return False
        number = game_clock.start_overtime()
        period_label = f'OT{number}'
        period_points_start = [score[0], score[1]]
        substitution_window = True
        window_reason = 'period_break'
        apply_lineups('period_break', 4 + number)
        side = jump_ball_winner(on_court[0], on_court[1], players)
        jump_balls.append({
            'period': period_label,
            'winner_index': side,
            'team': teams[side].name,
            'lineups': [list(on_court[0]), list(on_court[1])],
        })
        phase = 'overtime'
        possession_start = 'jump_ball'
        make_stopped_clock = False
        return True

    while True:
        if phase == 'overtime':
            if not game_clock.in_overtime:
                close_period(period_label)
                if score[0] != score[1] or not begin_overtime():
                    break
                continue
        elif game_clock.quarter_remaining == 0.0:
            if game_clock.regulation_complete:
                close_period('regulation')
                regulation_score = [score[0], score[1]]
                if score[0] != score[1] or not begin_overtime():
                    break
                continue
            close_period()
            if not game_clock.start_next_quarter():
                break
            period_label = f'Q{game_clock.quarter}'
            period_points_start = [score[0], score[1]]
            substitution_window = True
            window_reason = 'period_break'
            possession_start = 'period_start'
            make_stopped_clock = False
            continue
        team, opp = teams[side], teams[1 - side]
        adapt = adaptation_a if side == 0 else adaptation_b
        regulation_seconds = quarter_length * 4
        event_period = 4 + game_clock.overtime_period if game_clock.in_overtime else game_clock.quarter
        if substitution_window or on_court[0] is None:
            apply_lineups(window_reason, event_period)
        offense_ids = on_court[side]
        defense_ids = on_court[1 - side]
        fatigue_of = {
            pid: _player_fatigue(played_seconds[pid], regulation_seconds)
            for pid in offense_ids + defense_ids
        }
        offense = [players[pid] for pid in offense_ids]
        defense = [players[pid] for pid in defense_ids]
        # New possessions use the era clock, 24 seconds for the seeded historical eras.
        clock = PossessionClock(era.shot_clock)
        started_elapsed = game_clock.elapsed
        pass_count = 0
        start_state = possession_start
        clock_stopped = game_clock_stopped_for_start(start_state, make_stopped_clock)
        setup_game_clock = 0.0
        pending_make_stop = False

        def charge_both(seconds: float, kind: str) -> tuple[float, str | None]:
            elapsed, reason = _charge_live(game_clock, clock, seconds)
            live_seconds[kind] = live_seconds.get(kind, 0.0) + elapsed
            return elapsed, reason

        def charge_game(seconds: float, kind: str) -> float:
            if seconds < 0:
                raise ValueError('Elapsed time cannot be negative')
            elapsed = game_clock.advance(min(float(seconds), game_clock.period_remaining()))
            live_seconds[kind] = live_seconds.get(kind, 0.0) + elapsed
            return elapsed

        def stamp(payload: dict[str, Any]) -> None:
            payload['possession'] = poss_index
            payload['quarter'] = event_period
            payload['period'] = period_label
            payload['possession_start'] = start_state
            payload['game_clock_stopped_at_start'] = clock_stopped
            payload['setup_game_clock_seconds'] = setup_game_clock
            payload['setup_shot_clock_seconds'] = setup_shot_clock
            payload['elapsed_seconds'] = game_clock.elapsed
            payload['shot_clock_remaining'] = clock.remaining
            payload['quarter_remaining'] = game_clock.period_remaining()
            payload['period_ending'] = game_clock.period_remaining() == 0.0
            payload['offense_lineup'] = list(offense_ids)
            payload['defense_lineup'] = list(defense_ids)
            payload['player_fatigue'] = {pid: fatigue_of[pid] for pid in offense_ids + defense_ids}
            payload.setdefault('passes', pass_count)
            if track_events:
                events.append(payload)

        setup_seconds = timing.setup_seconds(era.shot_clock)
        if setup_seconds < 0:
            raise ValueError('Elapsed time cannot be negative')
        # A stopped throw-in, period start, or jump ball does not run either
        # clock before the ball is legally touched. This charge is that touch:
        # both clocks start together. A running clock, including a made basket
        # outside the stop window, is already live, so the charge is not added
        # again on top of the inbound.
        shot_before_setup = clock.remaining
        setup_game_clock, setup_reason = charge_both(setup_seconds, 'setup')
        setup_shot_clock = shot_before_setup - clock.remaining
        possession_end = None
        if game_clock.period_remaining() == 0.0 or setup_reason == 'period':
            possession_end = 'period_expired'
            stamp({'kind': 'period_expired', 'offense': None, 'passes': 0})
        elif setup_reason == 'shot_clock' or clock.remaining <= 0:
            possession_end = 'shot_clock_violation'
            stamp({'kind': 'shot_clock_violation', 'offense': None, 'passes': 0})
        else:
            while True:
                ballhandler = rng.choices(offense, weights=[selection_weight(p) for p in offense])[0]
                # Defensive assignment uses same position index, with IQ-based help.
                defender = defense[offense.index(ballhandler)]
                if rng.random() < clamp(.085 + (defender.steals - ballhandler.iq) * .00065 + fatigue_of[ballhandler.player_id] * .03, .025, .16):
                    stats[ballhandler.player_id]['tov'] += 1
                    if rng.random() < .48:
                        stats[defender.player_id]['stl'] += 1
                    stamp({'kind': 'turnover', 'offense': ballhandler.player_id, 'defender': defender.player_id, 'passes': 0})
                    possession_end = 'turnover'
                    break
                # Up to six passes. The pass rate falls as the ball moves, and the
                # live late-clock gates still force a shot when time is short.
                passer = None
                shooter = ballhandler
                pass_count = 0
                max_passes = 6
                pass_failed = False
                period_cut = False
                style_bonus = {'perimeter': .08, 'balanced': .0, 'post': -.04}.get(team.coach_style, 0.0)
                for _ in range(max_passes):
                    pass_chance = intelligent_pass_chance(
                        shooter,
                        defense[offense.index(shooter)],
                        style_bonus,
                        pass_count,
                        clock.remaining,
                    )
                    if clock.remaining <= 2:
                        pass_chance = 0.0
                    elif clock.remaining <= 4:
                        pass_chance *= 0.15
                    elif clock.remaining <= 8:
                        pass_chance *= 0.50
                    if rng.random() >= pass_chance:
                        break
                    pool = [p for p in offense if p.player_id != shooter.player_id]
                    recipient = rng.choices(pool, weights=[selection_weight(p) for p in pool])[0]
                    passer_defender = defense[offense.index(shooter)]
                    recipient_defender = defense[offense.index(recipient)]
                    pass_tov_chance = clamp(
                        .020
                        + (70 - shooter.passing) * .001
                        + (70 - shooter.iq) * .0005
                        + (passer_defender.steals - 70) * .0006
                        + (recipient_defender.defense - 70) * .0004,
                        .01,
                        .15,
                    )
                    if rng.random() < pass_tov_chance:
                        stats[shooter.player_id]['tov'] += 1
                        if rng.random() < .65:
                            stats[recipient_defender.player_id]['stl'] += 1
                        stamp({'kind': 'pass_turnover', 'offense': shooter.player_id, 'defender': recipient_defender.player_id, 'passes': pass_count})
                        pass_failed = True
                        possession_end = 'pass_turnover'
                        break
                    quarter_left = game_clock.period_remaining()
                    shot_left = clock.remaining
                    _, reason = charge_both(timing.completed_pass_seconds, 'completed_pass')
                    # A pass that only runs out the shot clock still counts, so the
                    # next decision rolls once more and then becomes a violation.
                    # A period horn stops the possession before that shot.
                    if reason == 'period' or (game_clock.period_remaining() == 0.0 and quarter_left < shot_left):
                        period_cut = True
                        break
                    passer = shooter
                    shooter = recipient
                    pass_count += 1
                    if game_clock.period_remaining() == 0.0:
                        period_cut = True
                        break
                if period_cut:
                    stamp({'kind': 'period_expired', 'offense': shooter.player_id, 'passes': pass_count})
                    possession_end = 'period_expired'
                    break
                if pass_failed:
                    break
                if clock.remaining <= 0:
                    stats[shooter.player_id]['tov'] += 1
                    stamp({'kind': 'shot_clock_violation', 'offense': shooter.player_id, 'passes': pass_count})
                    possession_end = 'shot_clock_violation'
                    break
                defender = defense[offense.index(shooter)]
                shot_preparation = rng.uniform(timing.release_low_seconds, timing.release_high_seconds)
                quarter_left = game_clock.period_remaining()
                shot_left = clock.remaining
                if quarter_left < shot_preparation and quarter_left <= shot_left:
                    charge_both(quarter_left, 'period')
                    stamp({'kind': 'period_expired', 'offense': shooter.player_id, 'passes': pass_count})
                    possession_end = 'period_expired'
                    break
                if shot_left < shot_preparation:
                    charge_both(shot_left, 'shot_clock_expiration')
                    stats[shooter.player_id]['tov'] += 1
                    stamp({'kind': 'shot_clock_violation', 'offense': shooter.player_id, 'passes': pass_count})
                    possession_end = 'shot_clock_violation'
                    break
                charge_both(shot_preparation, 'release')
                zone = weighted_pick(rng, shot_weights(shooter, era, team.coach_style, adapt))
                openness = clamp((shooter.iq - defender.iq) / 55 + (passer.passing - 65) / 100 if passer else (shooter.iq - defender.iq) / 65, -.8, 1)
                # Shooting foul simplified; no bonus, team foul counts or detailed referee decisions.
                # Free throws are dead-ball time and do not advance the game clock.
                fouled = rng.random() < clamp(.065 + era.physicality * .045 + (defender.defense - 70) * -.0003, .035, .16)
                if fouled:
                    stats[defender.player_id]['pf'] += 1
                    fta = 3 if zone == 'three' else 2
                    stats[shooter.player_id]['fta'] += fta
                    ft_p = clamp(.65 + (shooter.shot_skills['mid'] - 70) * .003, .40, .95)
                    made = sum(rng.random() < ft_p for _ in range(fta))
                    stats[shooter.player_id]['ftm'] += made
                    stats[shooter.player_id]['pts'] += made
                    score[side] += made
                    stamp({'kind': 'shooting_foul', 'offense': shooter.player_id, 'defender': defender.player_id, 'free_throws': fta, 'ft_made': made, 'passes': pass_count})
                    possession_end = 'shooting_foul'
                    break
                prob = probability_make(shooter, defender, era, zone, fatigue_of[shooter.player_id], adapt, openness)
                made = rng.random() < prob
                stats[shooter.player_id]['fga'] += 1
                if zone == 'three':
                    stats[shooter.player_id]['fg3a'] += 1
                if made:
                    stats[shooter.player_id]['fgm'] += 1
                    if zone == 'three':
                        stats[shooter.player_id]['fg3m'] += 1
                    pts = POINTS[zone]
                    score[side] += pts
                    stats[shooter.player_id]['pts'] += pts
                    assister_id = None
                    if passer and rng.random() < clamp(.55 + (passer.passing - 70) * .004, .2, .88):
                        stats[passer.player_id]['ast'] += 1
                        assister_id = passer.player_id
                    pending_make_stop = made_basket_stops_clock(event_period, game_clock.period_remaining())
                    stamp({'kind': 'made_shot', 'offense': shooter.player_id, 'defender': defender.player_id, 'zone': zone, 'points': pts, 'make_probability': round(prob, 3), 'passes': pass_count, 'assister': assister_id, 'clock_stopped': pending_make_stop})
                    possession_end = 'made_shot'
                    break
                # A miss creates exactly one credited rebound (simplification); occasional block credited.
                if rng.random() < clamp(.075 + (defender.blocks - 65) * .0015, .015, .24):
                    stats[defender.player_id]['blk'] += 1
                offensive_reb = rng.random() < .24
                pool = offense if offensive_reb else defense
                rebounder = rng.choices(pool, weights=[max(10, p.rebounding) for p in pool])[0]
                stats[rebounder.player_id]['oreb' if offensive_reb else 'dreb'] += 1
                stats[rebounder.player_id]['reb'] += 1
                reset_seconds = offensive_rebound_reset(era) if offensive_reb else None
                if timing.rebound_gather_seconds > 0 and game_clock.period_remaining() > 0:
                    charge_game(timing.rebound_gather_seconds, 'rebound_gather')
                miss_event = {'kind': 'miss', 'offense': shooter.player_id, 'defender': defender.player_id, 'zone': zone, 'rebounder': rebounder.player_id, 'offensive_rebound': offensive_reb, 'passes': pass_count}
                if offensive_reb:
                    miss_event['shot_clock_reset'] = reset_seconds
                stamp(miss_event)
                if not offensive_reb or game_clock.period_remaining() == 0.0:
                    if offensive_reb and game_clock.period_remaining() == 0.0:
                        stamp({'kind': 'period_expired', 'offense': shooter.player_id, 'passes': pass_count})
                        possession_end = 'period_expired'
                    else:
                        possession_end = 'defensive_rebound'
                    break
                clock.reset(reset_seconds)
        if possession_end is None:
            raise RuntimeError('a possession ended without a recorded result')
        # A running clock must consume time. A stopped clock may end before
        # any timed action: the inbound itself adds none, and an immediate
        # turnover or pre-touch violation does not invent a duration.
        if not clock_stopped and game_clock.elapsed <= started_elapsed and game_clock.period_remaining() > 0:
            raise RuntimeError('a possession ended without using game-clock time')
        played = game_clock.elapsed - started_elapsed
        for index, ids in enumerate(on_court):
            active = set(ids)
            for pid in stint_seconds[index]:
                if pid in active:
                    stint_seconds[index][pid] += played
                else:
                    stint_seconds[index][pid] = 0.0
        for pid in offense_ids:
            played_seconds[pid] += played
        for pid in defense_ids:
            played_seconds[pid] += played
        if game_clock.period_remaining() == 0.0:
            period_endings += 1
            substitution_window = True
            window_reason = 'period_break'
            possession_start = 'period_start'
            make_stopped_clock = False
        elif possession_end == 'made_shot':
            # A made-basket inbound is not a substitution window, including
            # when the late-period rule has stopped the clock.
            substitution_window = False
            window_reason = None
            possession_start = 'made_basket_inbound'
            make_stopped_clock = pending_make_stop
        elif possession_end in ('shooting_foul', 'shot_clock_violation'):
            substitution_window = True
            window_reason = possession_end
            possession_start = 'dead_ball_inbound'
            make_stopped_clock = False
        elif possession_end in ('turnover', 'pass_turnover', 'defensive_rebound'):
            substitution_window = False
            window_reason = None
            possession_start = 'live_ball'
            make_stopped_clock = False
        else:
            raise RuntimeError(f'unclassified possession end {possession_end}')
        possessions_by_team[side] += 1
        poss_index += 1
        side = 1 - side

    if regulation_score is None:
        raise RuntimeError('regulation ended without a recorded score')
    for team in teams:
        covered = sum(played_seconds[pid] for pid in team.player_ids)
        if abs(covered - 5 * game_clock.elapsed) > 1e-3:
            raise RuntimeError('player time does not cover the game clock')
    for pid, stat in stats.items():
        seconds = played_seconds[pid]
        stat['gp'] = 1 if seconds > 1e-9 else 0
        stat['min'] = seconds / 60.0
    timing_report = {
        'setup_cap_seconds': timing.setup_cap_seconds,
        'setup_rule': (
            'min(setup_cap_seconds, shot_clock * 0.25) is one live-ball charge. '
            'A stopped throw-in, period start, or jump ball does not run either '
            'clock before the legal touch. That charge then starts both clocks. '
            'A running clock, including a made basket outside the stop window, '
            'is already live, so the charge is not applied twice.'
        ),
        'completed_pass_seconds': timing.completed_pass_seconds,
        'release_seconds': [timing.release_low_seconds, timing.release_high_seconds],
        'rebound_gather_seconds': timing.rebound_gather_seconds,
        'rebound_gather_scope': 'game clock only, after a missed field goal, unverified',
        'free_throws': 'dead ball; the game clock does not advance',
        'calibrated': False,
    }
    result = {
        'era': era.key, 'seed': seed, 'teams': [team_a.name, team_b.name], 'score': score,
        'possessions_per_team': possessions_by_team[0],
        'possessions_by_team': possessions_by_team,
        'possession_count': poss_index,
        'player_stats': {pid: dict(v) for pid, v in stats.items()}, 'events': events,
        'regulation_score': list(regulation_score),
        'period_scores': period_scores,
        'regulation_seconds': quarter_length * 4,
        'game_clock_seconds': game_clock.elapsed,
        'quarter_lengths': game_clock.quarter_lengths(),
        'quarters_completed': game_clock.quarters if game_clock.regulation_complete else game_clock.quarter - 1,
        'period_ending_possessions': period_endings,
        'live_seconds': live_seconds,
        'pace_parameter': era.pace,
        'pace_used_to_choose_possessions': False,
        'timing_calibrated': False,
        'timing_assumptions': timing_report,
        'overtime': {
            'periods': game_clock.completed_overtime,
            'period_seconds': OVERTIME_PERIOD_SECONDS,
            'jump_balls': jump_balls,
            'jump_ball_model': 'highest_rebounding_on_court',
            'jump_ball_documentation': JUMP_BALL_MODEL,
            'timeouts_modeled': False,
            'timeout_limitation': (
                'Timeouts are not modeled, so a timeout is never treated as a substitution window.'
            ),
            'made_field_goal_limitation': (
                'A made field goal opens a made-basket inbound. That inbound does not '
                'authorize a substitution, including when the late-period rule stops the clock. '
                'Timeouts are not modeled.'
            ),
            'legal_stoppages': ['period_break', 'shooting_foul', 'shot_clock_violation'],
            'not_a_stoppage': [
                'turnover', 'pass_turnover', 'made_shot', 'defensive_rebound', 'offensive_rebound',
            ],
            'tie_broken_by_rule': False,
            'runaway_limit': OVERTIME_RUNAWAY_LIMIT,
            'unresolved_tie': score[0] == score[1],
            'fixture_points': fixture_points,
            'equalize_periods': list(equalize_periods),
        },
        'game_clock': 'overtime' if game_clock.completed_overtime else 'regulation',
        'rotations_live': True,
        'rotation_mode': [plan['mode'] for plan in plans],
        'substitutions': applied_subs,
        'substitution_counts': [len(applied_subs[0]), len(applied_subs[1])],
        'player_seconds': dict(played_seconds),
        'completed': score[0] != score[1],
        'status': 'final' if score[0] != score[1] else 'incomplete',
        'possession_starts': {
            'states': [
                'period_start', 'jump_ball', 'made_basket_inbound', 'dead_ball_inbound', 'live_ball',
            ],
            'regulation_opening': 'period_start',
            'overtime_opening': 'jump_ball',
            'late_made_basket_stop_seconds': {
                'quarters_1_to_3': LATE_MAKE_STOP_QUARTERS_1_TO_3,
                'quarter_4_and_overtime': LATE_MAKE_STOP_FINAL_PERIOD,
            },
            'made_basket_authorizes_substitution': False,
            'timeouts_modeled': False,
            'stopped_inbound_shot_clock_seconds': 0.0,
            'simplification': (
                'A stopped throw-in does not run the game clock or the shot clock '
                'before the ball is legally touched. Period starts and jump balls '
                'follow the same rule: regulation opens on a period-start inbound, '
                'and overtime opens on the jump-ball touch. The tip awards possession '
                'immediately, with no separate tap-to-control gap. The one setup '
                'charge is that touch, so both clocks start together. A made basket '
                'outside the late window leaves the clock running, and that running '
                'possession takes the same charge once. Timeouts are not modeled.'
            ),
        },
    }
    if score[0] == score[1]:
        result['overtime']['unresolved_tie'] = True
        result['overtime']['incomplete_reason'] = (
            f'{OVERTIME_RUNAWAY_LIMIT} overtime periods ended tied. '
            'No tie-break point is awarded. This game is incomplete.'
        )
        raise UnresolvedGameError(result)
    return result


def simulate_series(team_a: Team, team_b: Team, players: dict[str, Player], era: Era,
                    games: int = 10, seed: int = 7, adapt_over_season: bool = True) -> dict[str, Any]:
    if not 1 <= games <= 82:
        raise ValueError('games must be between 1 and 82')
    totals = {pid: Counter() for pid in team_a.player_ids + team_b.player_ids}
    per_game = []
    highlights = []
    for idx in range(games):
        a = clamp(idx / 65, 0, 1) if adapt_over_season else 0
        result = simulate_game(team_a, team_b, players, era, seed + idx * 997, a, a)
        per_game.append({'game': idx + 1, 'score': result['score'], 'player_stats': result['player_stats']})
        for pid, s in result['player_stats'].items():
            totals[pid].update(s)
        highlights.extend([{'game': idx + 1, **ev} for ev in result['events'] if ev['kind'] in ('made_shot', 'shooting_foul')][:4])
    averages = {}
    for pid, s in totals.items():
        averages[pid] = {
            **{stat: round(s[stat] / games, 3) for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'pf']},
            'fg_pct': round(s['fgm'] / s['fga'], 4) if s['fga'] else None,
            'fg3_pct': round(s['fg3m'] / s['fg3a'], 4) if s['fg3a'] else None,
            'ft_pct': round(s['ftm'] / s['fta'], 4) if s['fta'] else None,
        }
    return {'model_version': '0.1.0-prototype', 'era': era.key, 'games': games, 'seed': seed,
            'team_names': [team_a.name, team_b.name], 'averages': averages,
            'game_results': per_game, 'highlights': highlights}


def estimate_season(team_a: Team, team_b: Team, players: dict[str, Player], era: Era,
                    runs: int = 60, seed: int = 7) -> dict[str, Any]:
    """Repeat season-length simulations to estimate a numerical uncertainty interval.
    Not a statistically validated historical confidence interval.
    """
    if not 2 <= runs <= 5000:
        raise ValueError('runs must be between 2 and 5000')
    targets = {pid: [] for pid in team_a.player_ids + team_b.player_ids}
    for index in range(runs):
        result = simulate_series(team_a, team_b, players, era, games=82,
                                 seed=seed + index * 100003)
        for pid in targets:
            targets[pid].append(result['averages'][pid]['pts'])
    def quantile(values: list[float], q: float) -> float:
        xs = sorted(values)
        pos = q * (len(xs) - 1)
        lower = int(pos)
        frac = pos - lower
        return xs[lower] * (1 - frac) + xs[min(lower + 1, len(xs) - 1)] * frac
    return {'model_version': '0.1.0-prototype', 'runs': runs,
            'note': 'Intervals describe Monte Carlo variability under assumed parameters, NOT real-world calibrated confidence.',
            'players': {pid: {'points_per_game_mean': round(sum(xs) / len(xs), 2),
                              'simulation_p05': round(quantile(xs, .05), 2),
                              'simulation_p95': round(quantile(xs, .95), 2)} for pid, xs in targets.items()}}
