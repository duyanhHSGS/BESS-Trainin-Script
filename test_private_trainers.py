from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import date
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("private-trainers.py")
SPEC = importlib.util.spec_from_file_location("private_trainers_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
TRAINERS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINERS
SPEC.loader.exec_module(TRAINERS)


class PrivateTrainerManifestTests(unittest.TestCase):
    def test_manifest_hardware_matches_reported_site_table(self) -> None:
        expected = {
            "amy": (525.0, 1500.0),
            "namduoc": (500.0, 1000.0),
            "newing": (2450.0, 3500.0),
            "youngone": (500.0, 1000.0),
            "songwol": (450.0, 1250.0),
            "minhdanh": (250.0, 500.0),
        }
        for slug, (power, energy) in expected.items():
            with self.subTest(site=slug):
                spec = TRAINERS.SITES[slug]
                self.assertEqual(spec.expected_p_rated_kw, power)
                self.assertEqual(spec.expected_e_cap_kwh, energy)

    def test_trainall_order_is_exactly_the_official_six(self) -> None:
        self.assertEqual(
            TRAINERS.TRAIN_ALL_ORDER,
            ("amy", "namduoc", "newing", "youngone", "songwol", "minhdanh"),
        )

    def test_tande_is_preserved_but_not_in_trainall(self) -> None:
        self.assertIn("tande", TRAINERS.SITES)
        self.assertFalse(TRAINERS.SITES["tande"].include_in_trainall)
        self.assertNotIn("tande", TRAINERS.TRAIN_ALL_ORDER)

    def test_namduoc_is_explicitly_blocked(self) -> None:
        spec = TRAINERS.SITES["namduoc"]
        self.assertFalse(spec.enabled)
        self.assertIn("dataset audit failed", spec.disabled_reason)

    def test_every_site_config_matches_its_manifest_hardware(self) -> None:
        for slug, spec in TRAINERS.SITES.items():
            with self.subTest(site=slug):
                raw = json.loads(spec.config_path.read_text(encoding="utf-8"))
                TRAINERS.validate_config(spec, raw)

    def test_validate_config_rejects_wrong_site(self) -> None:
        spec = TRAINERS.SITES["youngone"]
        raw = json.loads(spec.config_path.read_text(encoding="utf-8"))
        raw["siteId"] = "newing"
        with self.assertRaises(TRAINERS.PreflightError):
            TRAINERS.validate_config(spec, raw)

    def test_validate_config_rejects_wrong_power(self) -> None:
        spec = TRAINERS.SITES["youngone"]
        raw = json.loads(spec.config_path.read_text(encoding="utf-8"))
        raw["bess"]["pRatedKw"] = 999
        with self.assertRaisesRegex(TRAINERS.PreflightError, "BESS power"):
            TRAINERS.validate_config(spec, raw)

    def test_validate_config_rejects_wrong_energy(self) -> None:
        spec = TRAINERS.SITES["youngone"]
        raw = json.loads(spec.config_path.read_text(encoding="utf-8"))
        raw["bess"]["eCapKwh"] = 999
        with self.assertRaisesRegex(TRAINERS.PreflightError, "BESS energy"):
            TRAINERS.validate_config(spec, raw)

    def test_build_command_hardcodes_iq2_experiment_receipt(self) -> None:
        spec = TRAINERS.SITES["newing"]
        command = TRAINERS.build_command(spec)
        self.assertEqual(command[0], sys.executable)
        self.assertIn(str(TRAINERS.TRAINER), command)
        self.assertIn(str(spec.csv_path), command)
        self.assertIn(str(spec.config_path), command)
        self.assertIn("1500000", command)
        self.assertIn("0,1,2", command)
        self.assertIn("2880", command)
        self.assertIn("3e-5", command)
        self.assertIn("3e-4", command)
        self.assertIn("ppo-iq2-coherent-bc-memory-newing", command)

    def test_output_directory_is_scoped_by_site_and_run(self) -> None:
        youngone = TRAINERS.SITES["youngone"].output_dir
        newing = TRAINERS.SITES["newing"].output_dir
        self.assertNotEqual(youngone, newing)
        self.assertEqual(youngone.name, TRAINERS.RUN_NAME)
        self.assertEqual(newing.name, TRAINERS.RUN_NAME)
        self.assertIn("youngone", youngone.parts)
        self.assertIn("newing", newing.parts)

    def test_trainall_skips_disabled_site_and_tande(self) -> None:
        with mock.patch.object(TRAINERS, "run_site") as run_site:
            TRAINERS.run_all(dry_run=True)
        called_slugs = [call.args[0].slug for call in run_site.call_args_list]
        self.assertEqual(called_slugs, ["amy", "newing", "youngone", "songwol", "minhdanh"])
        self.assertNotIn("namduoc", called_slugs)
        self.assertNotIn("tande", called_slugs)

    def test_trainall_uses_bounded_parallel_workers_for_real_training(self) -> None:
        recorded_workers: list[int] = []

        class ImmediateExecutor:
            def __init__(self, max_workers: int) -> None:
                recorded_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def submit(self, fn, *args, **kwargs):
                future = TRAINERS.concurrent.futures.Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:  # pragma: no cover - exercised below
                    future.set_exception(exc)
                return future

        with mock.patch.object(
            TRAINERS.concurrent.futures,
            "ThreadPoolExecutor",
            ImmediateExecutor,
        ), mock.patch.object(TRAINERS, "run_site") as run_site:
            TRAINERS.run_all(dry_run=False)

        self.assertEqual(recorded_workers, [TRAINERS.TRAINALL_MAX_WORKERS])
        called_slugs = sorted(call.args[0].slug for call in run_site.call_args_list)
        self.assertEqual(
            called_slugs,
            sorted(["amy", "newing", "youngone", "songwol", "minhdanh"]),
        )
        self.assertTrue(all(call.kwargs == {"dry_run": False} for call in run_site.call_args_list))

    def test_trainall_parallel_mode_aggregates_site_failures(self) -> None:
        class ImmediateExecutor:
            def __init__(self, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def submit(self, fn, *args, **kwargs):
                future = TRAINERS.concurrent.futures.Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
                return future

        def fake_run_site(spec, *, dry_run: bool = False) -> None:
            if spec.slug in {"newing", "songwol"}:
                raise TRAINERS.PreflightError(f"boom-{spec.slug}")

        with mock.patch.object(
            TRAINERS.concurrent.futures,
            "ThreadPoolExecutor",
            ImmediateExecutor,
        ), mock.patch.object(TRAINERS, "run_site", side_effect=fake_run_site):
            with self.assertRaises(SystemExit) as raised:
                TRAINERS.run_all(dry_run=False)

        message = str(raised.exception)
        self.assertIn("Newing: boom-newing", message)
        self.assertIn("Songwol: boom-songwol", message)

    def test_direct_disabled_site_fails_before_training(self) -> None:
        with self.assertRaises(TRAINERS.PreflightError):
            TRAINERS.run_site(TRAINERS.SITES["namduoc"], dry_run=True)

    def test_real_enabled_sites_pass_launcher_preflight(self) -> None:
        for slug, site in TRAINERS.SITES.items():
            if not site.enabled:
                continue
            with self.subTest(site=slug):
                audit, train, validation, test, _raw = TRAINERS.preflight(site)
                self.assertGreaterEqual(len(audit.eligible_months), 4)
                self.assertGreaterEqual(len(train), 1)
                self.assertEqual(len(validation), 2)
                self.assertEqual(len(test), 1)

    def test_real_namduoc_data_cannot_form_required_split(self) -> None:
        site = TRAINERS.SITES["namduoc"]
        raw = json.loads(site.config_path.read_text(encoding="utf-8"))
        TRAINERS.validate_config(site, raw)
        audit = TRAINERS.audit_csv(site.csv_path)
        with self.assertRaises(TRAINERS.PreflightError):
            TRAINERS.split_eligible_months(audit)


class PrivateTrainerDatasetAuditTests(unittest.TestCase):
    @staticmethod
    def _write_day(path: Path, *, slots: list[int]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "day_index",
                    "date_iso",
                    "day_type",
                    "step",
                    "P_load_kW",
                    "P_pv_kW",
                ),
            )
            writer.writeheader()
            for step in slots:
                writer.writerow(
                    {
                        "day_index": 1,
                        "date_iso": "2026-01-01",
                        "day_type": "working",
                        "step": step,
                        "P_load_kW": 500.0 + step,
                        "P_pv_kW": max(0.0, 100.0 - step),
                    }
                )

    def test_audit_accepts_exactly_96_unique_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            self._write_day(path, slots=list(range(96)))
            audit = TRAINERS.audit_csv(path)
        self.assertEqual(audit.rows, 96)
        self.assertEqual(audit.stored_days, 1)
        self.assertEqual(audit.first_date, date(2026, 1, 1))
        self.assertAlmostEqual(audit.month_coverage["2026-01"], 1 / 31)

    def test_audit_rejects_missing_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            self._write_day(path, slots=list(range(95)))
            with self.assertRaisesRegex(TRAINERS.PreflightError, "steps 0..95"):
                TRAINERS.audit_csv(path)

    def test_audit_rejects_duplicate_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            slots = [*range(95), 94]
            self._write_day(path, slots=slots)
            with self.assertRaisesRegex(TRAINERS.PreflightError, "unique steps"):
                TRAINERS.audit_csv(path)

    def test_split_matches_two_validation_one_test_protocol(self) -> None:
        audit = TRAINERS.DatasetAudit(
            rows=1,
            stored_days=1,
            first_date=date(2026, 1, 1),
            last_date=date(2026, 5, 31),
            month_coverage={},
            eligible_months=(
                "2026-01",
                "2026-02",
                "2026-03",
                "2026-04",
                "2026-05",
            ),
        )
        train, validation, test = TRAINERS.split_eligible_months(audit)
        self.assertEqual(train, ("2026-01", "2026-02"))
        self.assertEqual(validation, ("2026-03", "2026-04"))
        self.assertEqual(test, ("2026-05",))

    def test_split_rejects_fewer_than_four_eligible_months(self) -> None:
        audit = TRAINERS.DatasetAudit(
            rows=1,
            stored_days=1,
            first_date=date(2026, 1, 1),
            last_date=date(2026, 3, 31),
            month_coverage={},
            eligible_months=("2026-01", "2026-02", "2026-03"),
        )
        with self.assertRaisesRegex(TRAINERS.PreflightError, "at least 4"):
            TRAINERS.split_eligible_months(audit)

    def test_prepare_output_creates_scoped_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(TRAINERS, "PRIVATE_ROOT", Path(tmp)):
                spec = deepcopy(TRAINERS.SITES["youngone"])
                TRAINERS._prepare_output(spec)
                self.assertTrue(spec.output_dir.is_dir())

    def test_prepare_output_refuses_nonempty_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(TRAINERS, "PRIVATE_ROOT", Path(tmp)):
                spec = deepcopy(TRAINERS.SITES["youngone"])
                spec.output_dir.mkdir(parents=True)
                (spec.output_dir / "old.pt").write_bytes(b"brain")
                with self.assertRaisesRegex(TRAINERS.PreflightError, "overwrite"):
                    TRAINERS._prepare_output(spec)


if __name__ == "__main__":
    unittest.main()
