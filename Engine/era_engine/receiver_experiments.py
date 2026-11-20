"""Isolated receiver-selection experiments. The live engine is not modified.

Both variants replace only who receives a pass. The decision to pass, the
pass-turnover formula, the shot clock, shooting, rebounding, and assists stay
on the live code path.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

from .core import POINTS, ZONES, Player, clamp

_RECIPIENT_LINE = (
    "recipient = rng.choices(pool, weights=[max(1.0, p.usage * (70 + p.iq) / 140) for p in pool])[0]"
)
# Receiver variants stay on the pre-decision engine. Loading the promoted
# core.py would stack this experiment on the intelligent pass decision.
_LIVE_SOURCE = Path(__file__).with_name("core_before_intelligent_decision.py.bak").read_text()
if _LIVE_SOURCE.count(_RECIPIENT_LINE) != 1:
    raise RuntimeError("Live receiver line was not found exactly once")


def tendency_shares(player: Player) -> dict[str, float]:
    raw = {zone: max(0.001, player.tendencies.get(zone, 0.01)) for zone in ZONES}
    total = sum(raw.values())
    return {zone: raw[zone] / total for zone in ZONES}


def shooting_profile(player: Player) -> float:
    """Shot skill at the shots this player already prefers."""
    shares = tendency_shares(player)
    return sum(shares[zone] * player.shot_skills[zone] for zone in ZONES)


def expected_shot_value(player: Player) -> float:
    """Neutral expected points of the player's own shot mix.

    Uses the live zone baselines and the live skill term. It does not change
    a shot once it is taken.
    """
    baseline = {'rim': .61, 'paint': .43, 'mid': .40, 'three': .34}
    shares = tendency_shares(player)
    value = 0.0
    for zone in ZONES:
        make = clamp(baseline[zone] + (player.shot_skills[zone] - 75) * .004, .05, .88)
        value += shares[zone] * make * POINTS[zone]
    return value


def receiver_weight_a(player: Player) -> float:
    """Variant A: raise the chance a pass goes to a high-usage, high-skill player.

    The floor keeps every teammate reachable. No shot share is assigned in advance.
    """
    skill_scale = clamp(shooting_profile(player) / 78.0, 0.70, 1.35)
    usage_tilt = (max(player.usage, 0.05) / 0.19) ** 1.35
    return max(0.25, usage_tilt * skill_scale)


def receiver_weight_b(player: Player) -> float:
    """Variant B: weight a receiver by usage times the value of the shots he tends to take."""
    involvement = (70 + player.iq) / 140
    return max(0.20, player.usage * expected_shot_value(player) * involvement)


def _load_variant(name: str, weight):
    module = types.ModuleType(name)
    module._receiver_weight = weight
    sys.modules[name] = module
    source = _LIVE_SOURCE.replace(
        _RECIPIENT_LINE,
        "recipient = rng.choices(pool, weights=[_receiver_weight(p) for p in pool])[0]",
        1,
    )
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module.simulate_game


simulate_variant_a = _load_variant("era_receiver_variant_a", receiver_weight_a)
simulate_variant_b = _load_variant("era_receiver_variant_b", receiver_weight_b)
VARIANTS = {"A": simulate_variant_a, "B": simulate_variant_b}
