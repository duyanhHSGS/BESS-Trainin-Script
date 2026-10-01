"""Typed contracts for the deterministic controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSettings:
    """Tunable rule constants, all independent from measured future data."""

    history_days: int = 30
    peak_quantile: float = 0.90
    robust_mad_multiplier: float = 3.0
    sustainable_peak_hours: float = 5.0
    reserve_usable_fraction: float = 0.20
    daytime_start_slot: int = 24  # 06:00
    daytime_end_slot: int = 70  # 17:30, exclusive
    overnight_end_slot: int = 24  # 06:00, exclusive
    numeric_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.history_days < 1:
            raise ValueError("history_days must be >= 1")
        if not 0.0 <= self.peak_quantile <= 1.0:
            raise ValueError("peak_quantile must be in [0, 1]")
        if self.robust_mad_multiplier < 0.0:
            raise ValueError("robust_mad_multiplier must be >= 0")
        if self.sustainable_peak_hours <= 0.0:
            raise ValueError("sustainable_peak_hours must be > 0")
        if not 0.0 <= self.reserve_usable_fraction <= 1.0:
            raise ValueError("reserve_usable_fraction must be in [0, 1]")
        if not 0 <= self.overnight_end_slot <= 96:
            raise ValueError("overnight_end_slot must be in [0, 96]")
        if not 0 <= self.daytime_start_slot < self.daytime_end_slot <= 96:
            raise ValueError("daytime charging slots must form a valid window")
        if self.numeric_tolerance <= 0.0:
            raise ValueError("numeric_tolerance must be > 0")


@dataclass(frozen=True)
class Decision:
    """One executable action. Positive battery power means discharge."""

    battery_power_kw: float
    grid_power_kw: float
    charge_pv_kw: float
    charge_grid_kw: float
    discharge_kw: float
    curtailed_pv_kw: float
    peak_target_kw: float | None
    allowed_grid_kw: float | None
    reason: str
    deadline_impossible: bool = False

