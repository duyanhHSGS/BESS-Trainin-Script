"""Small bridge to stable benchmark contracts owned by the parent repository.

The private application deliberately imports configuration, tariff, scoring, and
physical-bound helpers from its "mama" repository.  It does not import a PPO agent,
policy observation, reward, checkpoint, or learned action.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BESS_DRL_SRC = REPO_ROOT / "bess-drl" / "src"
DRL_ENGINE = BESS_DRL_SRC / "bess_drl" / "training" / "drl_engine"

for path in (BESS_DRL_SRC, DRL_ENGINE):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

# Compatibility boundary: the DRL training engine is executable-script code and
# imports ``common`` as a top-level module.  These two imports must therefore come
# after its directories are registered.  Keep the E402 suppression this narrow.
from common import (  # noqa: E402
    DEMAND_BLOCK_SLOTS,
    DT_HOURS,
    STEPS_PER_DAY,
    DrlConfig,
    check_hard_constraints,
    fixed_pmax_day,
    load_bess_drl_config,
    score_month,
    tariff_vector,
)
from bess_drl.engine.feasible_action import (  # noqa: E402
    FeasiblePowerBounds,
    feasible_power_bounds,
)


def physical_bounds(
    cfg: DrlConfig,
    *,
    soc: float,
    load_kw: float,
    pv_kw: float,
) -> FeasiblePowerBounds:
    """Return the parent's authoritative one-slot physical power limits."""
    return feasible_power_bounds(
        soc_fraction=soc,
        load_kw=load_kw,
        pv_kw=pv_kw,
        p_rated_kw=cfg.p_rated_kw,
        e_cap_kwh=cfg.e_cap_kwh,
        soc_min=cfg.soc_min,
        soc_max=cfg.soc_max,
        eta_charge=cfg.eta_ch,
        eta_discharge=cfg.eta_dis,
        dt_hours=DT_HOURS,
        allow_export=False,
    )


def apply_soc(cfg: DrlConfig, soc: float, battery_power_kw: float) -> float:
    """Apply one 15-minute AC-side battery setpoint to SOC."""
    discharge_kw = max(0.0, battery_power_kw)
    charge_kw = max(0.0, -battery_power_kw)
    next_soc = soc + (
        charge_kw * DT_HOURS * cfg.eta_ch
        - discharge_kw * DT_HOURS / cfg.eta_dis
    ) / cfg.e_cap_kwh
    return min(cfg.soc_max, max(cfg.soc_min, next_soc))


def as_float_array(values: list[float]) -> np.ndarray:
    """Build the float64 arrays expected by the shared scorer."""
    return np.asarray(values, dtype=np.float64)


__all__ = [
    "DEMAND_BLOCK_SLOTS",
    "DT_HOURS",
    "STEPS_PER_DAY",
    "DrlConfig",
    "apply_soc",
    "as_float_array",
    "check_hard_constraints",
    "fixed_pmax_day",
    "load_bess_drl_config",
    "physical_bounds",
    "score_month",
    "tariff_vector",
]
