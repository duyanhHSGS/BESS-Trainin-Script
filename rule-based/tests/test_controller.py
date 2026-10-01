from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rule_based import mama
from rule_based.controller import RuleBasedController, robust_high_quantile
from rule_based.models import RuleSettings

PRIVATE_ROOT = Path(__file__).resolve().parents[2]


def make_config(
    *,
    energy_kwh: float = 100.0,
    power_kw: float = 50.0,
    eta_ch: float = 0.9,
    eta_dis: float = 0.9,
    degradation: float = 50.0,
) -> mama.DrlConfig:
    base = mama.load_bess_drl_config(PRIVATE_ROOT / "sites/amy/config.json")
    bess = replace(
        base.bess,
        e_cap_kwh=energy_kwh,
        p_rated_kw=power_kw,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
        soc_min=0.2,
        soc_max=0.9,
    )
    tariff = replace(
        base.tariff,
        price_off=100.0,
        price_mid=150.0,
        price_peak=300.0,
        peak_windows="17:30-22:30",
        off_windows="00:00-06:00",
        sunday_no_peak=False,
    )
    economics = replace(
        base.economics,
        degradation_cost_per_kwh_discharged=degradation,
    )
    return replace(base, bess=bess, tariff=tariff, economics=economics)


def flat_tariffs(price: float = 100.0) -> np.ndarray:
    return np.full(mama.STEPS_PER_DAY, price, dtype=np.float64)


def tou_tariffs() -> np.ndarray:
    result = np.full(mama.STEPS_PER_DAY, 150.0, dtype=np.float64)
    result[:24] = 100.0
    result[70:90] = 300.0
    return result


def run_complete_day(
    controller: RuleBasedController,
    *,
    load_kw: float,
    pv_kw: float = 0.0,
    soc: float = 0.9,
    tariffs: np.ndarray | None = None,
) -> float:
    controller.start_day(flat_tariffs() if tariffs is None else tariffs)
    for slot in range(mama.STEPS_PER_DAY):
        decision = controller.decide(
            slot=slot,
            load_kw=load_kw,
            pv_kw=pv_kw,
            soc=soc,
        )
        soc = mama.apply_soc(controller.cfg, soc, decision.battery_power_kw)
    return soc


def warm_target(
    controller: RuleBasedController,
    historical_peak_kw: float = 100.0,
) -> None:
    run_complete_day(controller, load_kw=historical_peak_kw)


class TestSettingsAndHistory:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"history_days": 0}, "history_days"),
            ({"peak_quantile": 1.1}, "peak_quantile"),
            ({"sustainable_peak_hours": 0.0}, "sustainable_peak_hours"),
            ({"reserve_usable_fraction": -0.1}, "reserve_usable_fraction"),
        ],
    )
    def test_invalid_settings_are_rejected(
        self, kwargs: dict[str, float | int], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            RuleSettings(**kwargs)

    def test_robust_quantile_caps_one_sided_outlier(self) -> None:
        result = robust_high_quantile([90.0, 100.0, 110.0, 1000.0], 0.9, 3.0)
        assert result < 200.0
        assert result > 100.0

    def test_quantile_rejects_bad_history(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            robust_high_quantile([1.0, float("nan")], 0.9, 3.0)

    def test_first_day_has_no_future_invented_peak_target(self) -> None:
        controller = RuleBasedController(make_config())
        assert controller.start_day(flat_tariffs()) is None

    def test_only_complete_day_enters_history(self) -> None:
        controller = RuleBasedController(make_config())
        controller.start_day(flat_tariffs())
        for slot in range(95):
            controller.decide(slot=slot, load_kw=80.0, pv_kw=0.0, soc=0.9)
        assert controller.peak_history == ()
        controller.decide(slot=95, load_kw=80.0, pv_kw=0.0, soc=0.9)
        assert controller.peak_history == (80.0,)

    def test_history_is_bounded_to_previous_30_days(self) -> None:
        controller = RuleBasedController(make_config(), RuleSettings(history_days=30))
        for peak in range(35):
            run_complete_day(controller, load_kw=float(peak))
        assert len(controller.peak_history) == 30
        assert controller.peak_history[0] == 5.0

    def test_target_subtracts_sustainable_physical_shave(self) -> None:
        settings = RuleSettings(sustainable_peak_hours=6.3)
        controller = RuleBasedController(make_config(), settings)
        warm_target(controller, 100.0)
        target = controller.start_day(flat_tariffs())
        assert target is not None
        assert target.historical_peak_kw == pytest.approx(100.0)
        assert target.sustainable_shave_kw == pytest.approx(10.0)
        assert target.target_kw == pytest.approx(90.0)


class TestPhysicsAndPv:
    def test_pv_surplus_charges_before_other_rules(self) -> None:
        settings = RuleSettings(
            overnight_end_slot=0,
            daytime_start_slot=70,
            daytime_end_slot=71,
        )
        controller = RuleBasedController(make_config(), settings)
        controller.start_day(flat_tariffs())
        decision = controller.decide(slot=0, load_kw=10.0, pv_kw=30.0, soc=0.5)
        assert decision.reason == "pv_self_consumption"
        assert decision.charge_pv_kw == pytest.approx(20.0)
        assert decision.grid_power_kw == pytest.approx(0.0)
        assert decision.battery_power_kw == pytest.approx(-20.0)

    def test_full_battery_curtailed_pv_never_exports(self) -> None:
        controller = RuleBasedController(make_config())
        controller.start_day(flat_tariffs())
        decision = controller.decide(slot=0, load_kw=0.0, pv_kw=40.0, soc=0.9)
        assert decision.battery_power_kw == pytest.approx(0.0)
        assert decision.grid_power_kw == pytest.approx(0.0)
        assert decision.curtailed_pv_kw == pytest.approx(40.0)

    def test_pv_charge_is_capped_by_rated_power(self) -> None:
        controller = RuleBasedController(make_config(power_kw=25.0))
        controller.start_day(flat_tariffs())
        decision = controller.decide(slot=0, load_kw=0.0, pv_kw=100.0, soc=0.2)
        assert decision.charge_pv_kw == pytest.approx(25.0)
        assert decision.curtailed_pv_kw == pytest.approx(75.0)

    def test_charge_is_capped_by_soc_headroom(self) -> None:
        cfg = make_config(energy_kwh=100.0, power_kw=100.0)
        controller = RuleBasedController(cfg)
        controller.start_day(flat_tariffs())
        decision = controller.decide(slot=0, load_kw=0.0, pv_kw=100.0, soc=0.89)
        expected = (0.9 - 0.89) * 100.0 / (0.9 * 0.25)
        assert decision.charge_pv_kw == pytest.approx(expected)

    def test_discharge_cannot_cross_hard_soc_floor(self) -> None:
        cfg = make_config()
        controller = RuleBasedController(cfg)
        warm_target(controller, 100.0)
        controller.start_day(flat_tariffs())
        decision = controller.decide(slot=0, load_kw=200.0, pv_kw=0.0, soc=cfg.soc_min)
        assert decision.discharge_kw == pytest.approx(0.0)

    def test_inputs_outside_physics_are_rejected(self) -> None:
        cfg = make_config()
        controller = RuleBasedController(cfg)
        controller.start_day(flat_tariffs())
        with pytest.raises(ValueError, match="soc"):
            controller.decide(slot=0, load_kw=1.0, pv_kw=0.0, soc=0.0)


class TestPeakPolice:
    def make_warm_controller(self) -> RuleBasedController:
        controller = RuleBasedController(
            make_config(), RuleSettings(sustainable_peak_hours=6.3)
        )
        warm_target(controller, 100.0)
        controller.start_month()
        controller.start_day(flat_tariffs())
        return controller

    def test_first_slot_shaves_to_target(self) -> None:
        controller = self.make_warm_controller()
        decision = controller.decide(slot=0, load_kw=120.0, pv_kw=0.0, soc=0.9)
        assert decision.allowed_grid_kw == pytest.approx(90.0)
        assert decision.discharge_kw == pytest.approx(30.0)
        assert decision.grid_power_kw == pytest.approx(90.0)

    def test_second_slot_uses_remaining_block_budget(self) -> None:
        controller = self.make_warm_controller()
        first = controller.decide(slot=0, load_kw=80.0, pv_kw=0.0, soc=0.9)
        soc = mama.apply_soc(controller.cfg, 0.9, first.battery_power_kw)
        second = controller.decide(slot=1, load_kw=120.0, pv_kw=0.0, soc=soc)
        assert second.allowed_grid_kw == pytest.approx(100.0)
        assert second.discharge_kw == pytest.approx(20.0)
        assert (first.grid_power_kw + second.grid_power_kw) / 2 == pytest.approx(90.0)

    def test_peak_police_can_use_soft_reserve(self) -> None:
        controller = self.make_warm_controller()
        decision = controller.decide(
            slot=0,
            load_kw=120.0,
            pv_kw=0.0,
            soc=controller.reserve_soc,
        )
        assert decision.reason == "peak_police"
        assert decision.discharge_kw > 0.0

    def test_billed_monthly_peak_becomes_target_floor(self) -> None:
        controller = self.make_warm_controller()
        cfg = controller.cfg
        first = controller.decide(slot=0, load_kw=200.0, pv_kw=0.0, soc=0.21)
        soc = mama.apply_soc(cfg, 0.21, first.battery_power_kw)
        controller.decide(slot=1, load_kw=200.0, pv_kw=0.0, soc=soc)
        assert controller.running_month_peak_kw > 90.0
        third = controller.decide(slot=2, load_kw=150.0, pv_kw=0.0, soc=soc)
        assert third.peak_target_kw == pytest.approx(controller.running_month_peak_kw)

    def test_out_of_order_slot_is_rejected(self) -> None:
        controller = self.make_warm_controller()
        with pytest.raises(ValueError, match="expected slot"):
            controller.decide(slot=1, load_kw=0.0, pv_kw=0.0, soc=0.9)


class TestChargingAndArbitrage:
    def test_overnight_charge_is_evenly_recomputed(self) -> None:
        cfg = make_config()
        controller = RuleBasedController(cfg)
        controller.start_day(tou_tariffs())
        decision = controller.decide(slot=0, load_kw=0.0, pv_kw=0.0, soc=0.3)
        expected = (0.9 - 0.3) * 100.0 / (0.9 * 6.0)
        assert decision.charge_grid_kw == pytest.approx(expected)

    def test_last_overnight_slot_reports_impossible_deadline(self) -> None:
        cfg = make_config(power_kw=10.0)
        controller = RuleBasedController(cfg)
        controller.start_day(tou_tariffs())
        soc = 0.2
        for slot in range(24):
            decision = controller.decide(slot=slot, load_kw=0.0, pv_kw=0.0, soc=soc)
            if slot < 23:
                # Simulate an unavailable charger despite prior requests.
                continue
            assert decision.deadline_impossible
            assert abs(decision.battery_power_kw) <= cfg.p_rated_kw

    def test_charge_is_capped_by_peak_safe_headroom(self) -> None:
        controller = RuleBasedController(
            make_config(), RuleSettings(sustainable_peak_hours=6.3)
        )
        warm_target(controller, 100.0)
        controller.start_month()
        controller.start_day(tou_tariffs())
        decision = controller.decide(slot=0, load_kw=85.0, pv_kw=0.0, soc=0.3)
        assert decision.allowed_grid_kw == pytest.approx(90.0)
        assert decision.charge_grid_kw == pytest.approx(5.0)
        assert decision.grid_power_kw == pytest.approx(90.0)

    def test_daytime_grid_charge_requires_cheapest_tariff(self) -> None:
        cfg = make_config()
        controller = RuleBasedController(cfg)
        tariffs = tou_tariffs()
        controller.start_day(tariffs)
        soc = 0.3
        for slot in range(25):
            decision = controller.decide(slot=slot, load_kw=0.0, pv_kw=0.0, soc=soc)
            if slot < 24:
                soc = 0.3  # emulate unavailable overnight charging
        assert decision.charge_grid_kw == pytest.approx(0.0)

    def test_economic_discharge_preserves_soft_reserve(self) -> None:
        cfg = make_config(degradation=0.0)
        controller = RuleBasedController(cfg)
        tariffs = tou_tariffs()
        controller.start_day(tariffs)
        soc = controller.reserve_soc
        for slot in range(71):
            decision = controller.decide(
                slot=slot,
                load_kw=100.0,
                pv_kw=0.0,
                soc=soc,
            )
        assert decision.reason == "no_op"
        assert decision.discharge_kw == pytest.approx(0.0)

    def test_economic_discharge_spreads_energy_over_peak_period(self) -> None:
        cfg = make_config(degradation=0.0)
        controller = RuleBasedController(cfg)
        controller.start_day(tou_tariffs())
        soc = 0.9
        for slot in range(71):
            decision = controller.decide(
                slot=slot,
                load_kw=100.0,
                pv_kw=0.0,
                soc=soc,
            )
        energy = (0.9 - controller.reserve_soc) * cfg.e_cap_kwh * cfg.eta_dis
        expected = energy / (20 * mama.DT_HOURS)
        assert decision.reason == "economic_discharge"
        assert decision.discharge_kw == pytest.approx(expected)

    def test_degradation_can_make_arbitrage_unprofitable(self) -> None:
        cfg = make_config(degradation=1000.0)
        controller = RuleBasedController(cfg)
        controller.start_day(tou_tariffs())
        for slot in range(71):
            decision = controller.decide(
                slot=slot,
                load_kw=100.0,
                pv_kw=0.0,
                soc=0.9,
            )
        assert decision.reason == "no_op"

    def test_mid_tariff_waits_for_known_more_expensive_period(self) -> None:
        cfg = make_config(degradation=0.0)
        controller = RuleBasedController(cfg)
        controller.start_day(tou_tariffs())
        soc = 0.9
        for slot in range(25):
            decision = controller.decide(
                slot=slot,
                load_kw=100.0,
                pv_kw=0.0,
                soc=soc,
            )
        assert decision.discharge_kw == pytest.approx(0.0)
