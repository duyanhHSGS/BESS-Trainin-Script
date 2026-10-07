from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).with_name("private-trainers.py")
SITE_CUSTOMIZE_PATH = Path(__file__).with_name("sitecustomize.py")
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
            "tande": (450.0, 1250.0),
        }
        for slug, (power, energy) in expected.items():
            with self.subTest(site=slug):
                spec = TRAINERS.SITES[slug]
                self.assertEqual(spec.expected_p_rated_kw, power)
                self.assertEqual(spec.expected_e_cap_kwh, energy)

    def test_trainall_order_includes_tande(self) -> None:
        self.assertEqual(
            TRAINERS.TRAIN_ALL_ORDER,
            (
                "amy",
                "namduoc",
                "newing",
                "youngone",
                "songwol",
                "minhdanh",
                "tande",
            ),
        )

    def test_tande_is_enabled_in_trainall(self) -> None:
        self.assertIn("tande", TRAINERS.SITES)
        self.assertTrue(TRAINERS.SITES["tande"].enabled)
        self.assertTrue(TRAINERS.SITES["tande"].include_in_trainall)
        self.assertIn("tande", TRAINERS.TRAIN_ALL_ORDER)

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

    def test_build_command_preserves_rejected_iq7_receipt_for_audit(self) -> None:
        spec = TRAINERS.SITES["newing"]
        job = TRAINERS.SeedJob(spec=spec, seed=1, cpu_threads=4, gpu_id="0")
        command = TRAINERS.build_command(job)
        self.assertEqual(command[0], sys.executable)
        self.assertIn(str(TRAINERS.TRAINER), command)
        self.assertIn(str(spec.csv_path), command)
        self.assertIn(str(spec.config_path), command)
        self.assertIn("1500000", command)
        self.assertEqual(command[command.index("--seeds") + 1], "1")
        self.assertIn("2880", command)
        self.assertIn("3e-5", command)
        self.assertIn("3e-4", command)
        self.assertIn("iq7_drop_stale_actor_inputs_v1-newing-seed1", command)

    def test_failure_ledger_names_every_noncausal_lineage_member(self) -> None:
        self.assertEqual(
            set(TRAINERS.REJECTED_EXPERIMENTS),
            {
                "iq5_current_slot_actor_v1",
                "iq6_causal_peak_target_actor_v1",
                "iq7_drop_stale_actor_inputs_v1",
            },
        )

    def test_rejected_experiments_fail_for_base_and_scoped_run_names(self) -> None:
        for run_name in TRAINERS.REJECTED_EXPERIMENTS:
            for candidate in (run_name, f"{run_name}-newing-seed1"):
                with self.subTest(run_name=candidate):
                    with self.assertRaisesRegex(
                        TRAINERS.PreflightError,
                        "rejected non-causal experiment",
                    ):
                        TRAINERS.validate_experiment_contract(candidate)

    def test_iq4_causal_baseline_is_not_rejected(self) -> None:
        TRAINERS.validate_experiment_contract(TRAINERS.CAUSAL_BASELINE)

    def test_empty_experiment_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(TRAINERS.PreflightError, "must not be empty"):
            TRAINERS.validate_experiment_contract("  ")

    def test_preflight_blocks_rejected_run_before_reading_private_inputs(self) -> None:
        with mock.patch.object(
            TRAINERS,
            "TRAINER",
            Path("/definitely/missing/trainer.py"),
        ):
            with self.assertRaisesRegex(
                TRAINERS.PreflightError,
                "rejected non-causal experiment",
            ):
                TRAINERS.preflight(TRAINERS.SITES["newing"])

    def test_output_directory_is_scoped_by_site_and_run(self) -> None:
        youngone = TRAINERS.SITES["youngone"].output_dir
        newing = TRAINERS.SITES["newing"].output_dir
        self.assertNotEqual(youngone, newing)
        self.assertEqual(youngone.name, TRAINERS.RUN_NAME)
        self.assertEqual(newing.name, TRAINERS.RUN_NAME)
        self.assertIn("youngone", youngone.parts)
        self.assertIn("newing", newing.parts)

    def test_trainall_skips_only_disabled_site(self) -> None:
        with mock.patch.object(TRAINERS, "run_site") as run_site:
            TRAINERS.run_all(dry_run=True)
        called_slugs = [call.args[0].slug for call in run_site.call_args_list]
        self.assertEqual(
            called_slugs,
            ["amy", "newing", "youngone", "songwol", "minhdanh", "tande"],
        )
        self.assertNotIn("namduoc", called_slugs)

    def test_allocate_cpu_threads_uses_every_cpu(self) -> None:
        self.assertEqual(TRAINERS.allocate_cpu_threads(6, 14), (3, 3, 2, 2, 2, 2))

    def test_allocate_cpu_threads_runs_all_trainers_when_cpus_are_fewer(self) -> None:
        self.assertEqual(TRAINERS.allocate_cpu_threads(6, 2), (1, 1, 1, 1, 1, 1))

    def test_allocate_cpu_threads_rejects_empty_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_count"):
            TRAINERS.allocate_cpu_threads(0, 8)

    def test_allocate_cpu_threads_handles_unknown_cpu_count(self) -> None:
        with mock.patch.object(TRAINERS.os, "cpu_count", return_value=None):
            self.assertEqual(TRAINERS.allocate_cpu_threads(2), (1, 1))

    def test_seed_jobs_run_every_seed_and_consume_the_cpu_budget(self) -> None:
        jobs = TRAINERS._seed_jobs(
            TRAINERS.SITES["tande"],
            cpu_threads=20,
            gpu_ids=("0", "1"),
            gpu_offset=0,
        )
        self.assertEqual([job.seed for job in jobs], [0, 1, 2])
        self.assertEqual([job.cpu_threads for job in jobs], [7, 7, 6])
        self.assertEqual([job.gpu_id for job in jobs], ["0", "1", "0"])

    def test_seed_jobs_rotate_gpu_offset_between_sites(self) -> None:
        jobs = TRAINERS._seed_jobs(
            TRAINERS.SITES["tande"],
            cpu_threads=3,
            gpu_ids=("0", "1", "2", "3"),
            gpu_offset=3,
        )
        self.assertEqual([job.gpu_id for job in jobs], ["3", "0", "1"])

    def test_gpu_override_supports_explicit_cpu_mode(self) -> None:
        self.assertEqual(
            TRAINERS.detect_gpu_ids({TRAINERS.GPU_OVERRIDE_ENV: "cpu"}),
            (),
        )

    def test_gpu_override_parses_unique_ids(self) -> None:
        self.assertEqual(
            TRAINERS.detect_gpu_ids({TRAINERS.GPU_OVERRIDE_ENV: "0, 2"}),
            ("0", "2"),
        )

    def test_gpu_override_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(TRAINERS.PreflightError, "duplicate"):
            TRAINERS.detect_gpu_ids({TRAINERS.GPU_OVERRIDE_ENV: "0,0"})

    def test_gpu_discovery_reads_nvidia_smi_indices(self) -> None:
        completed = SimpleNamespace(stdout="0\n2\n")
        with mock.patch.object(
            TRAINERS.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(TRAINERS.detect_gpu_ids({}), ("0", "2"))
        self.assertIn("--query-gpu=index", run.call_args.args[0])

    def test_gpu_discovery_falls_back_to_cpu_without_nvidia_smi(self) -> None:
        with mock.patch.object(
            TRAINERS.subprocess, "run", side_effect=FileNotFoundError
        ):
            self.assertEqual(TRAINERS.detect_gpu_ids({}), ())

    def test_seed_environment_caps_numeric_threads_and_gpu_visibility(self) -> None:
        job = TRAINERS.SeedJob(
            spec=TRAINERS.SITES["tande"],
            seed=1,
            cpu_threads=3,
            gpu_id="2",
        )
        env = TRAINERS._seed_environment(job)
        for variable in TRAINERS.CPU_THREAD_ENV_VARS:
            with self.subTest(variable=variable):
                self.assertEqual(env[variable], "3")
        self.assertEqual(env[TRAINERS.CPU_COUNT_ENV], "3")
        self.assertEqual(
            env["PYTHONPATH"].split(TRAINERS.os.pathsep)[0],
            str(TRAINERS.PRIVATE_ROOT),
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "2")

    def test_sitecustomize_applies_cpu_quota_before_trainer_imports(self) -> None:
        original_cpu_count = TRAINERS.os.cpu_count
        original_process_cpu_count = getattr(TRAINERS.os, "process_cpu_count", None)
        custom_spec = importlib.util.spec_from_file_location(
            "private_sitecustomize_under_test",
            SITE_CUSTOMIZE_PATH,
        )
        if custom_spec is None or custom_spec.loader is None:
            self.fail("could not load private sitecustomize module")
            return
        custom_module = importlib.util.module_from_spec(custom_spec)
        try:
            with mock.patch.dict(
                TRAINERS.os.environ,
                {TRAINERS.CPU_COUNT_ENV: "5"},
            ):
                custom_spec.loader.exec_module(custom_module)
                self.assertEqual(TRAINERS.os.cpu_count(), 5)
        finally:
            setattr(TRAINERS.os, "cpu_count", original_cpu_count)
            if original_process_cpu_count is not None:
                setattr(
                    TRAINERS.os,
                    "process_cpu_count",
                    original_process_cpu_count,
                )

    def test_cpu_seed_environment_hides_parent_gpu(self) -> None:
        job = TRAINERS.SeedJob(
            spec=TRAINERS.SITES["tande"],
            seed=0,
            cpu_threads=1,
        )
        with mock.patch.dict(TRAINERS.os.environ, {"CUDA_VISIBLE_DEVICES": "9"}):
            env = TRAINERS._seed_environment(job)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "void")

    def test_trainall_runs_every_enabled_site_in_parallel(self) -> None:
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
        ), mock.patch.object(
            TRAINERS,
            "allocate_cpu_threads",
            return_value=(3, 3, 2, 2, 2, 2),
        ), mock.patch.object(
            TRAINERS, "detect_gpu_ids", return_value=("0", "1")
        ), mock.patch.object(TRAINERS, "run_site") as run_site:
            TRAINERS.run_all(dry_run=False)

        self.assertEqual(recorded_workers, [6])
        called_slugs = sorted(call.args[0].slug for call in run_site.call_args_list)
        self.assertEqual(
            called_slugs,
            sorted(
                ["amy", "newing", "youngone", "songwol", "minhdanh", "tande"]
            ),
        )
        self.assertEqual(
            sorted(call.kwargs["cpu_threads"] for call in run_site.call_args_list),
            [2, 2, 2, 2, 3, 3],
        )
        self.assertTrue(
            all(call.kwargs["dry_run"] is False for call in run_site.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["gpu_ids"] == ("0", "1") for call in run_site.call_args_list)
        )
        self.assertEqual(
            sorted(call.kwargs["gpu_offset"] for call in run_site.call_args_list),
            [0, 3, 6, 9, 12, 15],
        )

    def test_run_site_runs_all_seeds_in_parallel_then_aggregates(self) -> None:
        spec = TRAINERS.SITES["tande"]
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
                except BaseException as exc:
                    future.set_exception(exc)
                return future

        with (
            mock.patch.object(TRAINERS, "preflight") as preflight,
            mock.patch.object(TRAINERS, "_print_preflight"),
            mock.patch.object(TRAINERS, "_prepare_output"),
            mock.patch.object(TRAINERS, "_run_seed_job") as run_seed,
            mock.patch.object(TRAINERS, "_aggregate_seed_outputs") as aggregate,
            mock.patch.object(
                TRAINERS.concurrent.futures,
                "ThreadPoolExecutor",
                ImmediateExecutor,
            ),
        ):
            preflight.return_value = (
                mock.sentinel.audit,
                ("train",),
                ("validation",),
                ("test",),
                {},
            )
            TRAINERS.run_site(spec, cpu_threads=8, gpu_ids=("0", "1"))

        self.assertEqual(recorded_workers, [3])
        jobs = [call.args[0] for call in run_seed.call_args_list]
        self.assertEqual([job.seed for job in jobs], [0, 1, 2])
        self.assertEqual([job.cpu_threads for job in jobs], [3, 3, 2])
        self.assertEqual([job.gpu_id for job in jobs], ["0", "1", "0"])
        aggregate.assert_called_once()
        self.assertEqual(list(aggregate.call_args.args[1]), jobs)

    def test_run_site_rejects_invalid_cpu_budget(self) -> None:
        spec = TRAINERS.SITES["tande"]
        with self.assertRaisesRegex(ValueError, "cpu_threads"):
            TRAINERS.run_site(spec, cpu_threads=0)

    def test_run_site_reports_failed_seed_without_aggregating(self) -> None:
        spec = TRAINERS.SITES["tande"]

        def fail_seed(job) -> None:
            if job.seed == 1:
                raise subprocess.CalledProcessError(7, ["trainer"])

        with (
            mock.patch.object(TRAINERS, "preflight") as preflight,
            mock.patch.object(TRAINERS, "_print_preflight"),
            mock.patch.object(TRAINERS, "_prepare_output"),
            mock.patch.object(TRAINERS, "_run_seed_job", side_effect=fail_seed),
            mock.patch.object(TRAINERS, "_aggregate_seed_outputs") as aggregate,
        ):
            preflight.return_value = (
                mock.sentinel.audit,
                ("train",),
                ("validation",),
                ("test",),
                {},
            )
            with self.assertRaisesRegex(TRAINERS.PreflightError, "seed 1"):
                TRAINERS.run_site(spec, cpu_threads=3)
        aggregate.assert_not_called()

    def test_load_seed_result_reads_schema_2_1_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            TRAINERS, "PRIVATE_ROOT", Path(tmp)
        ):
            spec = deepcopy(TRAINERS.SITES["tande"])
            job = TRAINERS._seed_jobs(
                spec, cpu_threads=3, gpu_ids=(), gpu_offset=0
            )[0]
            spec.output_dir.mkdir(parents=True)
            job.checkpoint_path.write_bytes(b"checkpoint")
            job.evaluation_path.write_text(
                json.dumps({
                    "schema_version": "2.1",
                    "policy_tag": job.tag,
                    "metrics": {"test_saving_pct": 12.34},
                }),
                encoding="utf-8",
            )
            checkpoint = {"meta": {"validation_cost_vnd": 123.0}}
            fake_torch = SimpleNamespace(load=mock.Mock(return_value=checkpoint))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                validation_cost, test_saving, loaded = TRAINERS._load_seed_result(job)

            self.assertEqual(validation_cost, 123.0)
            self.assertEqual(test_saving, 12.34)
            self.assertIs(loaded, checkpoint)

    def test_load_seed_result_prefers_metrics_over_legacy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            TRAINERS, "PRIVATE_ROOT", Path(tmp)
        ):
            spec = deepcopy(TRAINERS.SITES["tande"])
            job = TRAINERS._seed_jobs(
                spec, cpu_threads=3, gpu_ids=(), gpu_offset=0
            )[0]
            spec.output_dir.mkdir(parents=True)
            job.checkpoint_path.write_bytes(b"checkpoint")
            job.evaluation_path.write_text(
                json.dumps({
                    "schema_version": "2.1",
                    "metrics": {"test_saving_pct": 8.5},
                    "summary": {"test_saving_pct": 99.0},
                }),
                encoding="utf-8",
            )
            checkpoint = {"meta": {"validation_cost_vnd": 321.0}}
            fake_torch = SimpleNamespace(load=mock.Mock(return_value=checkpoint))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                _validation_cost, test_saving, _loaded = TRAINERS._load_seed_result(job)

            self.assertEqual(test_saving, 8.5)

    def test_load_seed_result_accepts_legacy_summary_during_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            TRAINERS, "PRIVATE_ROOT", Path(tmp)
        ):
            spec = deepcopy(TRAINERS.SITES["tande"])
            job = TRAINERS._seed_jobs(
                spec, cpu_threads=3, gpu_ids=(), gpu_offset=0
            )[0]
            spec.output_dir.mkdir(parents=True)
            job.checkpoint_path.write_bytes(b"checkpoint")
            job.evaluation_path.write_text(
                json.dumps({"summary": {"test_saving_pct": 7.25}}),
                encoding="utf-8",
            )
            checkpoint = {"meta": {"validation_cost_vnd": 456.0}}
            fake_torch = SimpleNamespace(load=mock.Mock(return_value=checkpoint))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                _validation_cost, test_saving, _loaded = TRAINERS._load_seed_result(job)

            self.assertEqual(test_saving, 7.25)

    def test_load_seed_result_rejects_versioned_legacy_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            TRAINERS, "PRIVATE_ROOT", Path(tmp)
        ):
            spec = deepcopy(TRAINERS.SITES["tande"])
            job = TRAINERS._seed_jobs(
                spec, cpu_threads=3, gpu_ids=(), gpu_offset=0
            )[0]
            spec.output_dir.mkdir(parents=True)
            job.checkpoint_path.write_bytes(b"checkpoint")
            job.evaluation_path.write_text(
                json.dumps({
                    "schema_version": "2.1",
                    "summary": {"test_saving_pct": 99.0},
                }),
                encoding="utf-8",
            )
            checkpoint = {"meta": {"validation_cost_vnd": 654.0}}
            fake_torch = SimpleNamespace(load=mock.Mock(return_value=checkpoint))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(
                    TRAINERS.PreflightError, "metrics.test_saving_pct"
                ):
                    TRAINERS._load_seed_result(job)

    def test_load_seed_result_rejects_missing_test_saving_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            TRAINERS, "PRIVATE_ROOT", Path(tmp)
        ):
            spec = deepcopy(TRAINERS.SITES["tande"])
            job = TRAINERS._seed_jobs(
                spec, cpu_threads=3, gpu_ids=(), gpu_offset=0
            )[0]
            spec.output_dir.mkdir(parents=True)
            job.checkpoint_path.write_bytes(b"checkpoint")
            job.evaluation_path.write_text(
                json.dumps({"schema_version": "2.1", "metrics": {}}),
                encoding="utf-8",
            )
            checkpoint = {"meta": {"validation_cost_vnd": 789.0}}
            fake_torch = SimpleNamespace(load=mock.Mock(return_value=checkpoint))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(
                    TRAINERS.PreflightError, "metrics.test_saving_pct"
                ):
                    TRAINERS._load_seed_result(job)

    def test_aggregate_selects_lowest_validation_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            TRAINERS, "PRIVATE_ROOT", Path(tmp)
        ):
            spec = deepcopy(TRAINERS.SITES["tande"])
            spec.output_dir.mkdir(parents=True)
            jobs = TRAINERS._seed_jobs(
                spec,
                cpu_threads=3,
                gpu_ids=(),
                gpu_offset=0,
            )
            for job in jobs:
                job.evaluation_path.write_text(
                    json.dumps({
                        "schema_version": "2.1",
                        "policy_tag": job.tag,
                        "metrics": {"test_saving_pct": 10.0 + job.seed},
                    }),
                    encoding="utf-8",
                )
            costs = {0: 300.0, 1: 100.0, 2: 200.0}

            def fake_result(job):
                return (
                    costs[job.seed],
                    10.0 + job.seed,
                    {"meta": {"seed": job.seed}},
                )

            fake_torch = SimpleNamespace(save=mock.Mock())
            with mock.patch.object(
                TRAINERS, "_load_seed_result", side_effect=fake_result
            ), mock.patch.dict(sys.modules, {"torch": fake_torch}):
                TRAINERS._aggregate_seed_outputs(spec, jobs)

            saved_checkpoint = fake_torch.save.call_args.args[0]
            self.assertEqual(saved_checkpoint["meta"]["selected_seed"], 1)
            self.assertEqual(
                saved_checkpoint["meta"]["validation_cost_vnd_by_seed"],
                {"0": 300.0, "1": 100.0, "2": 200.0},
            )
            canonical_evaluation = json.loads(
                (
                    spec.output_dir
                    / f"evaluation_{TRAINERS.RUN_NAME}-tande.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                canonical_evaluation["seed_selection"]["selected_seed"],
                1,
            )

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

        def fake_run_site(
            spec,
            *,
            dry_run: bool = False,
            cpu_threads: int | None = None,
            gpu_ids=(),
            gpu_offset: int = 0,
        ) -> None:
            if spec.slug in {"newing", "songwol"}:
                raise TRAINERS.PreflightError(f"boom-{spec.slug}")

        with mock.patch.object(
            TRAINERS.concurrent.futures,
            "ThreadPoolExecutor",
            ImmediateExecutor,
        ), mock.patch.object(
            TRAINERS, "detect_gpu_ids", return_value=()
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
        with mock.patch.object(TRAINERS, "RUN_NAME", TRAINERS.CAUSAL_BASELINE):
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
