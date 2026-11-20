"""Shoot-or-pass decision promoted into the live engine.

era_engine.core.simulate_game now calls intelligent_pass_chance. This module
re-exports that decision so the experiment checks keep one definition.
Receiver Variant A and Variant B are not combined with it.
"""
from __future__ import annotations

from .core import intelligent_pass_chance, simulate_game as simulate_intelligent

__all__ = ["intelligent_pass_chance", "simulate_intelligent"]
