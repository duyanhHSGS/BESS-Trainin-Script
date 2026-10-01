from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rule_based import mama
from rule_based.benchmark import classify_months, run_benchmark
from rule_based.data import SiteDay, group_days_by_month, load_site_days
from rule_based.models import RuleSettings

PRIVATE_ROOT = Path(__file__).resolve().parents[2]


def config() -> mama.DrlConfig:
    original = mama.load_bess_drl_config(PRIVATE_ROOT / "sites/amy/config.json")
    bess = replace(original.bess, e_cap_kwh=100.0, p_rated_kw=50.0)
    return replace(original, bess=bess)


def site_day(date_iso: str, load_kw: float, pv_kw: float) -> SiteDay:
    return SiteDay(
        date_iso=date_iso,
        day_type="working",
        load=np.full(96, load_kw, dtype=np.float64),
        pv_potential=np.full(96, pv_kw, dtype=np.float64),
    )


def write_csv(path: Path, *, duplicate_last: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("date_iso", "day_type", "step", "P_load_kW", "P_pv_kW"),
        )
        writer.writeheader()
        for slot in range(96):
            writer.writerow(
                {
                    "date_iso": "2026-01-01",
                    "day_type": "working",
                    "step": slot,
                    "P_load_kW": 100,
                    "P_pv_kW": 20,
                }
            )
        if duplicate_last:
            writer.writerow(
                {
                    "date_iso": "2026-01-01",
                    "day_type": "working",
                    "step": 95,
                    "P_load_kW": 100,
                    "P_pv_kW": 20,
                }
            )


def test_loader_accepts_one_complete_ordered_day(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    write_csv(path)
    days = load_site_days(path)
    assert len(days) == 1
    assert days[0].load.shape == (96,)
    assert days[0].pv_potential[0] == pytest.approx(20.0)


def test_loader_rejects_duplicate_slot(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    write_csv(path, duplicate_last=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_site_days(path)


def test_loader_rejects_incomplete_day(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    write_csv(path)
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly steps"):
        load_site_days(path)


def test_grouping_uses_calendar_months() -> None:
    groups = group_days_by_month(
        [
            site_day("2026-01-31", 1.0, 0.0),
            site_day("2026-02-01", 1.0, 0.0),
        ]
    )
    assert [len(group) for group in groups] == [1, 1]


def test_month_classification_matches_drl_holdout_protocol() -> None:
    days: list[SiteDay] = []
    for month, count in ((1, 25), (2, 23), (3, 25), (4, 24), (5, 25)):
        days.extend(
            site_day(f"2026-{month:02d}-{day:02d}", 1.0, 0.0)
            for day in range(1, count + 1)
        )
    classified = classify_months(days)
    assert classified["2026-01"][0] == "train"
    assert classified["2026-02"][0] == "train"
    assert classified["2026-03"][0] == "validation"
    assert classified["2026-04"][0] == "validation"
    assert classified["2026-05"][0] == "test"


def test_benchmark_is_deterministic_and_physically_clean() -> None:
    days = [
        site_day("2026-01-01", 80.0, 0.0),
        site_day("2026-01-02", 80.0, 120.0),
        site_day("2026-02-01", 100.0, 0.0),
    ]
    settings = RuleSettings(sustainable_peak_hours=6.3)
    first = run_benchmark(days, config(), settings)
    second = run_benchmark(days, config(), settings)
    assert first == second
    assert first["policy"] == "pure_causal_rule_based_v1"
    assert first["metrics"]["hard_constraint_violation_days"] == 0
    assert first["metrics"]["pv_charge_kwh"] > 0.0
    assert 0.0 <= first["metrics"]["pv_self_consumption_pct"] <= 100.0
    assert first["metrics"]["deadline_days"] == 3
    assert len(first["monthly"]) == 2


def test_benchmark_never_claims_export_when_pv_exceeds_load() -> None:
    result = run_benchmark(
        [site_day("2026-01-01", 0.0, 200.0)],
        config(),
    )
    assert result["metrics"]["hard_constraint_violation_days"] == 0
    assert result["metrics"]["curtailed_pv_kwh"] > 0.0


def test_benchmark_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_benchmark([], config())
