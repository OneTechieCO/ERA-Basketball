"""Isolated ballhandler and receiver weight experiments.

Production uses the proportional usage-and-IQ weight for both selections.
Variant C is that same weight. Variants A and B put back one of the retired
floors so the contrast stays measurable. Each copy keeps the intelligent
shoot-or-pass decision, the 0.020 pass-turnover base, the shot clock,
shooting, rebounding, fouls, and era settings. No shot share is assigned.
"""
from __future__ import annotations

import sys
import types
from collections import Counter
from pathlib import Path

from .core import Player

_BALLHANDLER_LINE = (
    "ballhandler = rng.choices(offense, weights=[selection_weight(p) for p in offense])[0]"
)
_RECEIVER_LINE = (
    "recipient = rng.choices(pool, weights=[selection_weight(p) for p in pool])[0]"
)
_CORE_SOURCE = Path(__file__).with_name("core.py").read_text()
if _CORE_SOURCE.count(_BALLHANDLER_LINE) != 1 or _CORE_SOURCE.count(_RECEIVER_LINE) != 1:
    raise RuntimeError("Live selection lines were not found exactly once")


def touch_weight(player: Player) -> float:
    """Positive weight proportional to offensive usage and basketball IQ.

    The live max() floors are omitted. Those floors sit above every seeded
    raw weight and make selection uniform.
    """
    return max(float(player.usage), 1e-9) * (70 + player.iq) / 140


def retired_ballhandler_weight(player: Player) -> float:
    """The old floor. It is above every seeded raw weight, so the choice was uniform."""
    return max(2, player.usage * (70 + player.iq) / 140)


def retired_receiver_weight(player: Player) -> float:
    """The old floor. It is above every seeded raw weight, so the choice was uniform."""
    return max(1.0, player.usage * (70 + player.iq) / 140)


def live_ballhandler_weight(player: Player) -> float:
    return retired_ballhandler_weight(player)


def live_receiver_weight(player: Player) -> float:
    return retired_receiver_weight(player)


def _picker(weight, counter: Counter):
    def pick(rng, pool):
        weights = [weight(player) for player in pool]
        chosen = rng.choices(pool, weights=weights)[0]
        counter[chosen.player_id] += 1
        return chosen

    return pick


def _counters():
    return {"ballhandler": Counter(), "receiver": Counter()}


COUNTS = {
    "live": _counters(),
    "A": _counters(),
    "B": _counters(),
    "C": _counters(),
}


def reset_counts() -> None:
    for buckets in COUNTS.values():
        for counter in buckets.values():
            counter.clear()


def _load(name: str, label: str, ball_weight, receiver_weight):
    module = types.ModuleType(name)
    module._pick_ballhandler = _picker(ball_weight, COUNTS[label]["ballhandler"])
    module._pick_receiver = _picker(receiver_weight, COUNTS[label]["receiver"])
    sys.modules[name] = module
    source = _CORE_SOURCE.replace(
        _BALLHANDLER_LINE, "ballhandler = _pick_ballhandler(rng, offense)", 1
    ).replace(
        _RECEIVER_LINE, "recipient = _pick_receiver(rng, pool)", 1
    )
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module.simulate_game


# Counted live uses the production weight so its box scores match core.py.
simulate_counted_live = _load(
    "era_selection_counted_live", "live", touch_weight, touch_weight
)
# A: proportional ballhandler weight, retired receiver floor.
simulate_ballhandler = _load(
    "era_selection_variant_a", "A", touch_weight, retired_receiver_weight
)
# B: retired ballhandler floor, proportional receiver weight.
simulate_receiver = _load(
    "era_selection_variant_b", "B", retired_ballhandler_weight, touch_weight
)
# C: proportional weight for both selections.
simulate_both = _load(
    "era_selection_variant_c", "C", touch_weight, touch_weight
)
VARIANTS = {
    "A": simulate_ballhandler,
    "B": simulate_receiver,
    "C": simulate_both,
}
