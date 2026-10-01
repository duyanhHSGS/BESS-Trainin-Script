"""Causal, deterministic battery dispatch rules.

The controller knows current load/PV/SOC, completed history, and the published
tariff calendar.  It never reads future load/PV and has no learned component.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rule_based import mama
from rule_based.models import Decision, RuleSettings


@dataclass(frozen=True)
class PeakTarget:
    historical_peak_kw: float
    sustainable_shave_kw: float
    target_kw: float


def robust_high_quantile(
    values: Sequence[float],
    quantile: float,
    mad_multiplier: float,
) -> float:
    """Nearest-rank high quantile after one-sided median/MAD winsorisation."""
    if not values:
        raise ValueError("values must not be empty")
    samples = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(samples)) or np.any(samples < 0.0):
        raise ValueError("peak history must contain finite non-negative values")
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    if mad > 0.0 and mad_multiplier > 0.0:
        robust_sigma = 1.4826 * mad
        samples = np.minimum(samples, median + mad_multiplier * robust_sigma)
    ordered = np.sort(samples)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[rank])


class RuleBasedController:
    """Stateful rule controller for sequential 15-minute measurements."""

    def __init__(
        self,
        cfg: mama.DrlConfig,
        settings: RuleSettings | None = None,
    ) -> None:
        self.cfg = cfg
        self.settings = settings or RuleSettings()
        self._daily_peak_history: deque[float] = deque(
            maxlen=self.settings.history_days
        )
        self._tariffs = np.zeros(mama.STEPS_PER_DAY, dtype=np.float64)
        self._cheapest_price = 0.0
        self._max_future_tariff = np.zeros(mama.STEPS_PER_DAY, dtype=np.float64)
        self._same_price_remaining = np.ones(mama.STEPS_PER_DAY, dtype=np.int64)
        self._daytime_charge_slots_remaining = np.zeros(
            mama.STEPS_PER_DAY, dtype=np.int64
        )
        self._daily_nobess_grid: list[float] = []
        self._block_grid_sum_kw = 0.0
        self._running_month_peak_kw = 0.0
        self._expected_slot = 0
        self._day_started = False
        self._target: PeakTarget | None = None

    @property
    def peak_history(self) -> tuple[float, ...]:
        return tuple(self._daily_peak_history)

    @property
    def peak_target(self) -> PeakTarget | None:
        return self._target

    @property
    def running_month_peak_kw(self) -> float:
        return self._running_month_peak_kw

    @property
    def reserve_soc(self) -> float:
        usable_soc = self.cfg.soc_max - self.cfg.soc_min
        return self.cfg.soc_min + (
            self.settings.reserve_usable_fraction * usable_soc
        )

    def start_month(self) -> None:
        """Reset only current-month meter state; causal daily history survives."""
        if self._day_started:
            raise RuntimeError("cannot start a month during an unfinished day")
        self._running_month_peak_kw = 0.0

    def start_day(self, tariffs: Sequence[float]) -> PeakTarget | None:
        if self._day_started:
            raise RuntimeError("previous day is incomplete")
        if len(tariffs) != mama.STEPS_PER_DAY:
            raise ValueError("tariffs must contain exactly 96 slots")
        tariff_array = np.asarray(tariffs, dtype=np.float64)
        if not np.all(np.isfinite(tariff_array)) or np.any(tariff_array < 0.0):
            raise ValueError("tariffs must be finite and non-negative")
        self._tariffs = tariff_array
        self._cache_tariff_lookups()
        self._daily_nobess_grid = []
        self._block_grid_sum_kw = 0.0
        self._expected_slot = 0
        self._day_started = True
        self._target = self._build_peak_target()
        return self._target

    def _cache_tariff_lookups(self) -> None:
        """Precompute known-tariff queries used by every dispatch tick."""
        tolerance = self.settings.numeric_tolerance
        self._cheapest_price = float(np.min(self._tariffs))
        self._max_future_tariff = np.maximum.accumulate(self._tariffs[::-1])[::-1]
        self._same_price_remaining.fill(1)
        for slot in range(mama.STEPS_PER_DAY - 2, -1, -1):
            if abs(float(self._tariffs[slot + 1] - self._tariffs[slot])) <= tolerance:
                self._same_price_remaining[slot] = (
                    self._same_price_remaining[slot + 1] + 1
                )
        self._daytime_charge_slots_remaining.fill(0)
        count = 0
        for slot in range(self.settings.daytime_end_slot - 1, -1, -1):
            if (
                slot >= self.settings.daytime_start_slot
                and self._tariffs[slot] <= self._cheapest_price + tolerance
            ):
                count += 1
            self._daytime_charge_slots_remaining[slot] = count

    def _build_peak_target(self) -> PeakTarget | None:
        if not self._daily_peak_history:
            return None
        historical = robust_high_quantile(
            self._daily_peak_history,
            self.settings.peak_quantile,
            self.settings.robust_mad_multiplier,
        )
        usable_output_kwh = (
            self.cfg.e_cap_kwh
            * (self.cfg.soc_max - self.cfg.soc_min)
            * self.cfg.eta_dis
        )
        sustainable = min(
            self.cfg.p_rated_kw,
            usable_output_kwh / self.settings.sustainable_peak_hours,
        )
        return PeakTarget(
            historical_peak_kw=historical,
            sustainable_shave_kw=sustainable,
            target_kw=max(0.0, historical - sustainable),
        )

    def _effective_peak_target_kw(self) -> float | None:
        if self._target is None:
            return None
        return max(self._target.target_kw, self._running_month_peak_kw)

    def _remaining_block_slots(self, slot: int) -> int:
        phase = slot % mama.DEMAND_BLOCK_SLOTS
        return mama.DEMAND_BLOCK_SLOTS - phase

    def _allowed_grid_kw(self, slot: int) -> float | None:
        target = self._effective_peak_target_kw()
        if target is None:
            return None
        total_block_budget = target * mama.DEMAND_BLOCK_SLOTS
        remaining_budget = max(0.0, total_block_budget - self._block_grid_sum_kw)
        return remaining_budget / self._remaining_block_slots(slot)

    def _remaining_slots_at_price(self, slot: int, price: float) -> int:
        if abs(float(self._tariffs[slot]) - price) > self.settings.numeric_tolerance:
            return 0
        return int(self._same_price_remaining[slot])

    def _remaining_daytime_charge_slots(self, slot: int) -> int:
        begin = max(slot, self.settings.daytime_start_slot)
        return int(self._daytime_charge_slots_remaining[begin])

    def _scheduled_grid_charge_kw(self, slot: int, soc: float) -> tuple[float, bool]:
        stored_deficit_kwh = max(
            0.0,
            (self.cfg.soc_max - soc) * self.cfg.e_cap_kwh,
        )
        if stored_deficit_kwh <= self.settings.numeric_tolerance:
            return 0.0, False

        if slot < self.settings.overnight_end_slot:
            remaining_slots = self.settings.overnight_end_slot - slot
        elif (
            self.settings.daytime_start_slot
            <= slot
            < self.settings.daytime_end_slot
        ):
            if (
                self._tariffs[slot]
                > self._cheapest_price + self.settings.numeric_tolerance
            ):
                return 0.0, False
            remaining_slots = self._remaining_daytime_charge_slots(slot)
        else:
            return 0.0, False

        if remaining_slots <= 0:
            return 0.0, True
        remaining_hours = remaining_slots * mama.DT_HOURS
        requested_kw = stored_deficit_kwh / (
            self.cfg.eta_ch * remaining_hours
        )
        return requested_kw, requested_kw > self.cfg.p_rated_kw + 1e-9

    def _economic_discharge_kw(self, slot: int, soc: float) -> float:
        current_price = float(self._tariffs[slot])
        cheapest_price = self._cheapest_price
        most_expensive_remaining = float(self._max_future_tariff[slot])
        if current_price + self.settings.numeric_tolerance < most_expensive_remaining:
            return 0.0
        break_even = cheapest_price / (self.cfg.eta_ch * self.cfg.eta_dis)
        break_even += self.cfg.degradation_cost_per_kwh_discharged
        if current_price <= break_even + self.settings.numeric_tolerance:
            return 0.0
        remaining_slots = self._remaining_slots_at_price(slot, current_price)
        if remaining_slots <= 0:
            return 0.0
        energy_above_reserve_kwh = max(
            0.0,
            (soc - self.reserve_soc) * self.cfg.e_cap_kwh * self.cfg.eta_dis,
        )
        return energy_above_reserve_kwh / (remaining_slots * mama.DT_HOURS)

    def decide(
        self,
        *,
        slot: int,
        load_kw: float,
        pv_kw: float,
        soc: float,
    ) -> Decision:
        """Choose and record one action using only causally available values."""
        if not self._day_started:
            raise RuntimeError("start_day must be called before decide")
        if slot != self._expected_slot:
            raise ValueError(f"expected slot {self._expected_slot}, received {slot}")
        if not math.isfinite(load_kw) or load_kw < 0.0:
            raise ValueError("load_kw must be finite and non-negative")
        if not math.isfinite(pv_kw) or pv_kw < 0.0:
            raise ValueError("pv_kw must be finite and non-negative")
        if not math.isfinite(soc) or not self.cfg.soc_min <= soc <= self.cfg.soc_max:
            raise ValueError("soc must be finite and inside configured limits")

        bounds = mama.physical_bounds(
            self.cfg,
            soc=soc,
            load_kw=load_kw,
            pv_kw=pv_kw,
        )
        net_load_kw = max(0.0, load_kw - pv_kw)
        pv_surplus_kw = max(0.0, pv_kw - load_kw)
        allowed_grid_kw = self._allowed_grid_kw(slot)
        peak_discharge_kw = 0.0
        if allowed_grid_kw is not None:
            peak_discharge_kw = max(0.0, net_load_kw - allowed_grid_kw)

        pv_charge_kw = min(pv_surplus_kw, bounds.max_charge_kw)
        deadline_impossible = False
        if peak_discharge_kw > self.settings.numeric_tolerance:
            discharge_kw = min(peak_discharge_kw, bounds.max_discharge_kw)
            decision = self._discharge_decision(
                discharge_kw,
                net_load_kw,
                allowed_grid_kw,
                "peak_police",
            )
        elif pv_surplus_kw > self.settings.numeric_tolerance:
            scheduled_kw, deadline_impossible = self._scheduled_grid_charge_kw(
                slot, soc
            )
            decision = self._charge_decision(
                scheduled_grid_kw=scheduled_kw,
                pv_charge_kw=pv_charge_kw,
                pv_surplus_kw=pv_surplus_kw,
                net_load_kw=net_load_kw,
                max_charge_kw=bounds.max_charge_kw,
                allowed_grid_kw=allowed_grid_kw,
                reason="pv_self_consumption",
                deadline_impossible=deadline_impossible,
            )
        else:
            economic_kw = self._economic_discharge_kw(slot, soc)
            if economic_kw > self.settings.numeric_tolerance:
                discharge_kw = min(economic_kw, bounds.max_discharge_kw)
                decision = self._discharge_decision(
                    discharge_kw,
                    net_load_kw,
                    allowed_grid_kw,
                    "economic_discharge",
                )
            else:
                scheduled_kw, deadline_impossible = self._scheduled_grid_charge_kw(
                    slot, soc
                )
                reason = "scheduled_charge" if scheduled_kw > 0.0 else "no_op"
                decision = self._charge_decision(
                    scheduled_grid_kw=scheduled_kw,
                    pv_charge_kw=0.0,
                    pv_surplus_kw=0.0,
                    net_load_kw=net_load_kw,
                    max_charge_kw=bounds.max_charge_kw,
                    allowed_grid_kw=allowed_grid_kw,
                    reason=reason,
                    deadline_impossible=deadline_impossible,
                )

        self._record_slot(slot, decision.grid_power_kw, net_load_kw)
        return decision

    def _discharge_decision(
        self,
        discharge_kw: float,
        net_load_kw: float,
        allowed_grid_kw: float | None,
        reason: str,
    ) -> Decision:
        grid_kw = max(0.0, net_load_kw - discharge_kw)
        return Decision(
            battery_power_kw=discharge_kw,
            grid_power_kw=grid_kw,
            charge_pv_kw=0.0,
            charge_grid_kw=0.0,
            discharge_kw=discharge_kw,
            curtailed_pv_kw=0.0,
            peak_target_kw=self._effective_peak_target_kw(),
            allowed_grid_kw=allowed_grid_kw,
            reason=reason,
        )

    def _charge_decision(
        self,
        *,
        scheduled_grid_kw: float,
        pv_charge_kw: float,
        pv_surplus_kw: float,
        net_load_kw: float,
        max_charge_kw: float,
        allowed_grid_kw: float | None,
        reason: str,
        deadline_impossible: bool,
    ) -> Decision:
        remaining_power_kw = max(0.0, max_charge_kw - pv_charge_kw)
        if allowed_grid_kw is None:
            peak_safe_grid_charge_kw = remaining_power_kw
        else:
            peak_safe_grid_charge_kw = max(0.0, allowed_grid_kw - net_load_kw)
        grid_charge_kw = min(
            scheduled_grid_kw,
            remaining_power_kw,
            peak_safe_grid_charge_kw,
        )
        total_charge_kw = pv_charge_kw + grid_charge_kw
        grid_kw = net_load_kw + grid_charge_kw
        final_reason = (
            reason if total_charge_kw > self.settings.numeric_tolerance else "no_op"
        )
        return Decision(
            battery_power_kw=-total_charge_kw,
            grid_power_kw=grid_kw,
            charge_pv_kw=pv_charge_kw,
            charge_grid_kw=grid_charge_kw,
            discharge_kw=0.0,
            curtailed_pv_kw=max(0.0, pv_surplus_kw - pv_charge_kw),
            peak_target_kw=self._effective_peak_target_kw(),
            allowed_grid_kw=allowed_grid_kw,
            reason=final_reason,
            deadline_impossible=deadline_impossible,
        )

    def _record_slot(self, slot: int, grid_kw: float, nobess_grid_kw: float) -> None:
        self._block_grid_sum_kw += grid_kw
        self._daily_nobess_grid.append(nobess_grid_kw)
        if (slot + 1) % mama.DEMAND_BLOCK_SLOTS == 0:
            demand_kw = self._block_grid_sum_kw / mama.DEMAND_BLOCK_SLOTS
            self._running_month_peak_kw = max(
                self._running_month_peak_kw,
                demand_kw,
            )
            self._block_grid_sum_kw = 0.0
        self._expected_slot += 1
        if self._expected_slot == mama.STEPS_PER_DAY:
            daily_peak = mama.fixed_pmax_day(
                mama.as_float_array(self._daily_nobess_grid)
            )
            self._daily_peak_history.append(daily_peak)
            self._day_started = False


# TODO(RULE-BASELINE-CALIBRATION): freeze the quantile, MAD cap, reserve, and
# sustainable-duration constants only after multi-site untouched-holdout results.
