"""Strict loader for private site CSV measurements."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from rule_based import mama


@dataclass(frozen=True)
class SiteDay:
    date_iso: str
    day_type: str
    load: np.ndarray
    pv_potential: np.ndarray


def load_site_days(path: Path) -> list[SiteDay]:
    """Load complete days and reject malformed/duplicated 15-minute slots."""
    by_day: dict[str, dict[int, tuple[float, float, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date_iso", "day_type", "step", "P_load_kW", "P_pv_kW"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                date_iso = row["date_iso"]
                date.fromisoformat(date_iso)
                slot = int(row["step"])
                load_kw = float(row["P_load_kW"])
                pv_kw = float(row["P_pv_kW"])
                day_type = row["day_type"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed CSV row {line_number}: {exc}") from exc
            if not 0 <= slot < mama.STEPS_PER_DAY:
                raise ValueError(f"row {line_number}: step must be in 0..95")
            if not np.isfinite(load_kw) or load_kw < 0.0:
                raise ValueError(f"row {line_number}: invalid load")
            if not np.isfinite(pv_kw) or pv_kw < 0.0:
                raise ValueError(f"row {line_number}: invalid PV")
            slots = by_day.setdefault(date_iso, {})
            if slot in slots:
                raise ValueError(f"row {line_number}: duplicate {date_iso} step {slot}")
            slots[slot] = (load_kw, pv_kw, day_type)

    days: list[SiteDay] = []
    for date_iso, slots in sorted(by_day.items()):
        if set(slots) != set(range(mama.STEPS_PER_DAY)):
            raise ValueError(f"{date_iso}: day must contain exactly steps 0..95")
        day_types = {entry[2] for entry in slots.values()}
        if len(day_types) != 1:
            raise ValueError(f"{date_iso}: inconsistent day_type values")
        days.append(
            SiteDay(
                date_iso=date_iso,
                day_type=day_types.pop(),
                load=np.asarray(
                    [slots[index][0] for index in range(mama.STEPS_PER_DAY)],
                    dtype=np.float64,
                ),
                pv_potential=np.asarray(
                    [slots[index][1] for index in range(mama.STEPS_PER_DAY)],
                    dtype=np.float64,
                ),
            )
        )
    if not days:
        raise ValueError("CSV contains no site days")
    return days


def group_days_by_month(days: list[SiteDay]) -> list[list[SiteDay]]:
    grouped: dict[str, list[SiteDay]] = {}
    for day in days:
        grouped.setdefault(day.date_iso[:7], []).append(day)
    return [
        sorted(grouped[key], key=lambda day: day.date_iso) for key in sorted(grouped)
    ]
