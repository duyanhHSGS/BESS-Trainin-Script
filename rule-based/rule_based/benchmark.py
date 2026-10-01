"""Run and score the pure rules over private historical site data."""

from __future__ import annotations

import calendar
from dataclasses import asdict
from typing import Any

import numpy as np

from rule_based import mama
from rule_based.controller import RuleBasedController
from rule_based.data import SiteDay, group_days_by_month
from rule_based.models import RuleSettings

MIN_MONTH_COVERAGE = 0.80
VALIDATION_MONTHS = 2
TEST_MONTHS = 1


def classify_months(days: list[SiteDay]) -> dict[str, tuple[str, float]]:
    """Return DRL-compatible chronological split labels and coverage."""
    by_month: dict[str, set[int]] = {}
    for day in days:
        key = day.date_iso[:7]
        by_month.setdefault(key, set()).add(int(day.date_iso[-2:]))
    coverage: dict[str, float] = {}
    for key, day_numbers in sorted(by_month.items()):
        year, month = (int(value) for value in key.split("-"))
        coverage[key] = len(day_numbers) / calendar.monthrange(year, month)[1]

    eligible = [
        key for key, value in coverage.items() if value >= MIN_MONTH_COVERAGE
    ]
    labels = dict.fromkeys(coverage, "ineligible")
    holdout_count = VALIDATION_MONTHS + TEST_MONTHS
    if len(eligible) >= holdout_count + 1:
        for key in eligible[:-holdout_count]:
            labels[key] = "train"
        for key in eligible[-holdout_count:-TEST_MONTHS]:
            labels[key] = "validation"
        for key in eligible[-TEST_MONTHS:]:
            labels[key] = "test"
    else:
        for key in eligible:
            labels[key] = "unassigned"
    return {key: (labels[key], value) for key, value in coverage.items()}


def _sum_scores(scores: list[dict[str, Any]]) -> dict[str, float | int | None]:
    additive = (
        "energy_cost_vnd",
        "demand_cost_vnd",
        "electricity_bill_vnd",
        "degradation_cost_vnd",
        "terminal_settlement_vnd",
        "total_cost_vnd",
        "throughput_kwh",
        "discharged_kwh",
        "equivalent_full_cycles",
    )
    result: dict[str, float | int | None] = {
        key: sum(float(score[key]) for score in scores) for key in additive
    }
    result["pmax_month_kw"] = max(
        (float(score["pmax_month_kw"]) for score in scores), default=0.0
    )
    result["terminal_soc_fraction"] = (
        scores[-1]["terminal_soc_fraction"] if scores else None
    )
    result["months_scored"] = len(scores)
    return result


def _split_score(monthly: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    selected = [month for month in monthly if month["split"] == label]
    if not selected:
        return None
    rule = _sum_scores([month["rule"] for month in selected])
    nobess = _sum_scores([month["no_bess"] for month in selected])
    saving_pct = 100.0 * (
        float(nobess["total_cost_vnd"]) - float(rule["total_cost_vnd"])
    ) / max(float(nobess["total_cost_vnd"]), 1e-9)
    return {
        "months": [month["month"] for month in selected],
        "saving_pct": saving_pct,
        "rule": rule,
        "no_bess": nobess,
    }


def run_benchmark(
    days: list[SiteDay],
    cfg: mama.DrlConfig,
    settings: RuleSettings | None = None,
) -> dict[str, Any]:
    """Evaluate calendar months while carrying only causal completed-day history."""
    if not days:
        raise ValueError("benchmark requires at least one complete day")
    controller = RuleBasedController(cfg, settings)
    month_classification = classify_months(days)
    rule_month_scores: list[dict[str, Any]] = []
    nobess_month_scores: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    total_curtailment_kwh = 0.0
    total_pv_charge_kwh = 0.0
    total_grid_charge_kwh = 0.0
    total_pv_potential_kwh = 0.0
    total_direct_pv_kwh = 0.0
    deadline_impossible_slots = 0
    deadline_days = 0
    deadline_success_days = 0
    peak_target_violation_blocks = 0
    reason_slots: dict[str, int] = {}
    violation_days = 0

    for month_days in group_days_by_month(days):
        controller.start_month()
        soc = cfg.soc_eod
        grid_days: list[np.ndarray] = []
        bess_days: list[np.ndarray] = []
        soc_days: list[np.ndarray] = []
        nobess_grid_days: list[np.ndarray] = []

        for day in month_days:
            tariffs = mama.tariff_vector(cfg, day=day)
            controller.start_day(tariffs)
            grid = np.zeros(mama.STEPS_PER_DAY, dtype=np.float64)
            battery = np.zeros(mama.STEPS_PER_DAY, dtype=np.float64)
            soc_path = np.zeros(mama.STEPS_PER_DAY + 1, dtype=np.float64)
            target_path: list[float | None] = []
            soc_path[0] = soc
            nobess_grid = np.maximum(0.0, day.load - day.pv_potential)
            total_pv_potential_kwh += float(np.sum(day.pv_potential)) * mama.DT_HOURS
            total_direct_pv_kwh += float(
                np.sum(np.minimum(day.load, day.pv_potential))
            ) * mama.DT_HOURS

            for slot in range(mama.STEPS_PER_DAY):
                decision = controller.decide(
                    slot=slot,
                    load_kw=float(day.load[slot]),
                    pv_kw=float(day.pv_potential[slot]),
                    soc=soc,
                )
                soc = mama.apply_soc(cfg, soc, decision.battery_power_kw)
                grid[slot] = decision.grid_power_kw
                battery[slot] = decision.battery_power_kw
                soc_path[slot + 1] = soc
                total_curtailment_kwh += decision.curtailed_pv_kw * mama.DT_HOURS
                total_pv_charge_kwh += decision.charge_pv_kw * mama.DT_HOURS
                total_grid_charge_kwh += decision.charge_grid_kw * mama.DT_HOURS
                deadline_impossible_slots += int(decision.deadline_impossible)
                reason_slots[decision.reason] = reason_slots.get(decision.reason, 0) + 1
                target_path.append(decision.peak_target_kw)
                if slot == controller.settings.overnight_end_slot - 1:
                    deadline_days += 1
                    deadline_success_days += int(
                        soc >= cfg.soc_max - controller.settings.numeric_tolerance
                    )

            for start in range(0, mama.STEPS_PER_DAY, mama.DEMAND_BLOCK_SLOTS):
                target = target_path[start]
                if target is None:
                    continue
                block_demand = float(
                    np.mean(grid[start : start + mama.DEMAND_BLOCK_SLOTS])
                )
                peak_target_violation_blocks += int(
                    block_demand > target + controller.settings.numeric_tolerance
                )

            grid_days.append(grid)
            bess_days.append(battery)
            soc_days.append(soc_path)
            nobess_grid_days.append(nobess_grid)

        rule_score = mama.score_month(
            grid_days,
            cfg,
            days=month_days,
            p_bess_days=bess_days,
            soc_days=soc_days,
        )
        nobess_score = mama.score_month(nobess_grid_days, cfg, days=month_days)
        hard = mama.check_hard_constraints(grid_days, soc_days, cfg)
        violation_days += int(hard["zero_export_violation_days"])
        violation_days += int(hard["soc_violation_days"])
        rule_month_scores.append(rule_score)
        nobess_month_scores.append(nobess_score)
        monthly.append(
            {
                "month": month_days[0].date_iso[:7],
                "stored_days": len(month_days),
                "coverage": month_classification[month_days[0].date_iso[:7]][1],
                "split": month_classification[month_days[0].date_iso[:7]][0],
                "rule": rule_score,
                "no_bess": nobess_score,
                "saving_pct": 100.0 * (
                    float(nobess_score["total_cost_vnd"])
                    - float(rule_score["total_cost_vnd"])
                ) / max(float(nobess_score["total_cost_vnd"]), 1e-9),
                "hard_constraints": hard,
            }
        )

    rule_total = _sum_scores(rule_month_scores)
    nobess_total = _sum_scores(nobess_month_scores)
    saving_pct = 100.0 * (
        float(nobess_total["total_cost_vnd"])
        - float(rule_total["total_cost_vnd"])
    ) / max(float(nobess_total["total_cost_vnd"]), 1e-9)
    validation = _split_score(monthly, "validation")
    test = _split_score(monthly, "test")
    return {
        "schema_version": "1.0",
        "policy": "pure_causal_rule_based_v1",
        "site_id": cfg.site_id,
        "settings": asdict(settings or RuleSettings()),
        "metrics": {
            "saving_pct": saving_pct,
            "validation_saving_pct": (
                validation["saving_pct"] if validation is not None else None
            ),
            "test_saving_pct": test["saving_pct"] if test is not None else None,
            "rule": rule_total,
            "no_bess": nobess_total,
            "pv_charge_kwh": total_pv_charge_kwh,
            "grid_charge_kwh": total_grid_charge_kwh,
            "curtailed_pv_kwh": total_curtailment_kwh,
            "pv_self_consumption_pct": 100.0 * (
                total_direct_pv_kwh + total_pv_charge_kwh
            ) / max(total_pv_potential_kwh, 1e-9),
            "deadline_impossible_slots": deadline_impossible_slots,
            "deadline_success_days": deadline_success_days,
            "deadline_days": deadline_days,
            "deadline_success_pct": 100.0 * deadline_success_days
            / max(deadline_days, 1),
            "peak_target_violation_blocks": peak_target_violation_blocks,
            "hard_constraint_violation_days": violation_days,
            "decision_slots": reason_slots,
        },
        "splits": {
            "minimum_month_coverage": MIN_MONTH_COVERAGE,
            "validation": validation,
            "test": test,
        },
        "monthly": monthly,
    }
