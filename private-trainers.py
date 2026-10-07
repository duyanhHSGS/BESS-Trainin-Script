"""Private multi-site training launcher and experiment failure ledger.

IQ5, IQ6, and IQ7 are retained only as forensic receipts.  Their offline actor
observations consumed the completed 15-minute average for slot ``t`` before
choosing ``action[t]``.  Production instead had only a boundary-time live
sample, so those runs were not causal deployment comparisons and must never be
launched or promoted again.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

PRIVATE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PRIVATE_ROOT.parent
TRAINER = REPO_ROOT / "bess-drl/src/bess_drl/training/drl_engine/run_train_dataset.py"
RUN_NAME = "iq9_big_brain_256_v1"
# TODO(IQ9-BIG-BRAIN): keep this receipt pinned to the 256-wide IQ8-capacity ablation.
# IQ8 returns to the causal IQ4 observation timing, then adds only IQ6's useful
# completed-history peak-target philosophy. IQ5/IQ6/IQ7 remain rejected below.
REJECTED_EXPERIMENTS: dict[str, str] = {
    "iq5_current_slot_actor_v1": (
        "first actor-observation leak: day.load[t]/day.pv_potential[t] were "
        "completed slot averages presented before action[t]"
    ),
    "iq6_causal_peak_target_actor_v1": (
        "inherited IQ5's non-causal current-slot actor inputs"
    ),
    "iq7_drop_stale_actor_inputs_v1": (
        "inherited IQ5's leak and removed four causal previous-slot inputs"
    ),
}
CAUSAL_BASELINE = "iq4_privileged_critic_v1"
# TODO(IQ8-CAUSAL-PEAK): keep the explicit obs_t causality regression pinned:
# slot-t measured aggregates must not affect obs_t/action_t and may first affect
# obs_(t+1); never relax this contract for a better-looking holdout score.
MIN_MONTH_COVERAGE = 0.80
VAL_MONTHS = 2
TEST_MONTHS = 1
STEPS_PER_DAY = 96
SEEDS: tuple[int, ...] = (0, 1, 2)
GPU_OVERRIDE_ENV = "PRIVATE_TRAINER_GPUS"
CPU_COUNT_ENV = "PRIVATE_TRAINER_CPU_COUNT"
CPU_THREAD_ENV_VARS: tuple[str, ...] = (
    "DRL_TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# TODO(GPU-TRAIN): remove the forward-compatible GPU assignment shim after the
# core PPO trainer exposes a real device contract and moves models/buffers to it.

# Keep the experiment receipt explicit. Do not silently inherit trainer defaults:
# changing a default in run_train_dataset.py must not mutate an old private run.
TRAIN_ARGS: tuple[str, ...] = (
    "--steps", "1500000",
    "--rollout", "2880",
    "--eval-every", "20",
    "--min-month-coverage", "0.8",
    "--actor-lr", "3e-5",
    "--critic-lr", "3e-4",
    "--init-std", "0.15",
    "--clip-penalty", "100.0",
    "--bc-epochs", "10",
    "--lambda-energy", "0.97",
    "--lambda-peak", "0.97",
)

# TODO(PRIVATE-MULTISITE): replace copied reference tariff/economics in each site
# config with authoritative site-specific contracts before production comparison.


class PreflightError(RuntimeError):
    """Raised before expensive training when private experiment inputs are unsafe."""


def validate_experiment_contract(run_name: str) -> None:
    """Reject experiment receipts known to violate the causal actor contract."""
    normalized = run_name.strip()
    if not normalized:
        raise PreflightError("experiment run name must not be empty")
    for rejected_name, failure in REJECTED_EXPERIMENTS.items():
        if normalized == rejected_name or normalized.startswith(f"{rejected_name}-"):
            raise PreflightError(
                f"rejected non-causal experiment {normalized!r}: {failure}; "
                f"return to {CAUSAL_BASELINE} and retrain from scratch"
            )


@dataclass(frozen=True)
class SiteSpec:
    slug: str
    display_name: str
    expected_p_rated_kw: float
    expected_e_cap_kwh: float
    enabled: bool = True
    include_in_trainall: bool = True
    disabled_reason: str = ""

    @property
    def root(self) -> Path:
        return PRIVATE_ROOT / "sites" / self.slug

    @property
    def csv_path(self) -> Path:
        return self.root / "data.csv"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def output_dir(self) -> Path:
        return self.root / "results" / RUN_NAME


SITES: dict[str, SiteSpec] = {
    "amy": SiteSpec("amy", "Á Mỹ", 525.0, 1500.0),
    "namduoc": SiteSpec(
        "namduoc",
        "Nam Dược",
        500.0,
        1000.0,
        enabled=False,
        disabled_reason=(
            "dataset audit failed: August has one stored day and its load sensor "
            "is flat at 0 kW while PV is alive"
        ),
    ),
    "newing": SiteSpec("newing", "Newing", 2450.0, 3500.0),
    "youngone": SiteSpec("youngone", "YoungOne", 500.0, 1000.0),
    "songwol": SiteSpec("songwol", "Songwol", 450.0, 1250.0),
    "minhdanh": SiteSpec("minhdanh", "Minh Danh", 250.0, 500.0),
    "tande": SiteSpec("tande", "Tande", 450.0, 1250.0),
}

TRAIN_ALL_ORDER: tuple[str, ...] = (
    "amy",
    "namduoc",
    "newing",
    "youngone",
    "songwol",
    "minhdanh",
    "tande",
)


@dataclass(frozen=True)
class DatasetAudit:
    rows: int
    stored_days: int
    first_date: date
    last_date: date
    month_coverage: Mapping[str, float]
    eligible_months: tuple[str, ...]


@dataclass(frozen=True)
class SeedJob:
    spec: SiteSpec
    seed: int
    cpu_threads: int
    gpu_id: str | None = None

    @property
    def tag(self) -> str:
        return f"{RUN_NAME}-{self.spec.slug}-seed{self.seed}"

    @property
    def checkpoint_path(self) -> Path:
        return self.spec.output_dir / f"policy_{self.tag}.pt"

    @property
    def evaluation_path(self) -> Path:
        return self.spec.output_dir / f"evaluation_{self.tag}.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreflightError(f"missing config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreflightError(f"invalid JSON config {path}: {exc}") from exc


def validate_config(spec: SiteSpec, raw: Mapping[str, Any]) -> None:
    site_id = raw.get("siteId")
    if site_id != spec.slug:
        raise PreflightError(
            f"{spec.slug}: config siteId={site_id!r}, expected {spec.slug!r}"
        )
    bess = raw.get("bess")
    if not isinstance(bess, Mapping):
        raise PreflightError(f"{spec.slug}: config is missing bess object")
    try:
        power = float(bess["pRatedKw"])
        energy = float(bess["eCapKwh"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError(
            f"{spec.slug}: config requires numeric bess.pRatedKw/eCapKwh"
        ) from exc
    if power != spec.expected_p_rated_kw:
        raise PreflightError(
            f"{spec.slug}: BESS power {power:g} kW != expected "
            f"{spec.expected_p_rated_kw:g} kW"
        )
    if energy != spec.expected_e_cap_kwh:
        raise PreflightError(
            f"{spec.slug}: BESS energy {energy:g} kWh != expected "
            f"{spec.expected_e_cap_kwh:g} kWh"
        )


def audit_csv(path: Path, min_month_coverage: float = MIN_MONTH_COVERAGE) -> DatasetAudit:
    if not path.is_file():
        raise PreflightError(f"missing CSV: {path}")

    required = {"date_iso", "step", "P_load_kW", "P_pv_kW"}
    steps_by_day: dict[date, list[int]] = defaultdict(list)
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise PreflightError(
                f"{path}: missing CSV columns {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            try:
                day = date.fromisoformat(row["date_iso"])
                step = int(row["step"])
                float(row["P_load_kW"])
                float(row["P_pv_kW"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PreflightError(
                    f"{path}: malformed row {row_number}: {exc}"
                ) from exc
            steps_by_day[day].append(step)

    if not steps_by_day:
        raise PreflightError(f"{path}: CSV contains no data rows")

    expected_steps = set(range(STEPS_PER_DAY))
    for day, steps in sorted(steps_by_day.items()):
        if len(steps) != STEPS_PER_DAY or set(steps) != expected_steps:
            raise PreflightError(
                f"{path}: {day.isoformat()} must contain exactly steps 0..95; "
                f"found {len(steps)} rows / {len(set(steps))} unique steps"
            )

    days_by_month: dict[tuple[int, int], set[int]] = defaultdict(set)
    for day in steps_by_day:
        days_by_month[(day.year, day.month)].add(day.day)

    month_coverage: dict[str, float] = {}
    for (year, month), day_numbers in sorted(days_by_month.items()):
        expected_days = calendar.monthrange(year, month)[1]
        key = f"{year:04d}-{month:02d}"
        month_coverage[key] = len(day_numbers) / expected_days

    eligible = tuple(
        key for key, coverage in month_coverage.items()
        if coverage >= min_month_coverage
    )
    ordered_days = sorted(steps_by_day)
    return DatasetAudit(
        rows=rows,
        stored_days=len(ordered_days),
        first_date=ordered_days[0],
        last_date=ordered_days[-1],
        month_coverage=month_coverage,
        eligible_months=eligible,
    )


def split_eligible_months(
    audit: DatasetAudit,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    holdout_count = VAL_MONTHS + TEST_MONTHS
    if len(audit.eligible_months) < holdout_count + 1:
        raise PreflightError(
            "need at least 4 calendar months at >=80% coverage for "
            f"train/validation/test; found {len(audit.eligible_months)}: "
            f"{list(audit.eligible_months)}"
        )
    train = audit.eligible_months[:-holdout_count]
    holdout = audit.eligible_months[-holdout_count:]
    return train, holdout[:VAL_MONTHS], holdout[VAL_MONTHS:]


def build_command(job: SeedJob) -> list[str]:
    return [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(TRAINER),
        "--csv",
        str(job.spec.csv_path),
        "--config-json",
        str(job.spec.config_path),
        *TRAIN_ARGS,
        "--seeds",
        str(job.seed),
        "--tag",
        job.tag,
    ]


def preflight(
    spec: SiteSpec,
) -> tuple[
    DatasetAudit,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
]:
    validate_experiment_contract(RUN_NAME)
    if not TRAINER.is_file():
        raise PreflightError(f"trainer not found: {TRAINER}")
    raw = _load_json(spec.config_path)
    validate_config(spec, raw)
    audit = audit_csv(spec.csv_path)
    train, validation, test = split_eligible_months(audit)
    return audit, train, validation, test, raw


def _print_preflight(
    spec: SiteSpec,
    audit: DatasetAudit,
    train: Sequence[str],
    validation: Sequence[str],
    test: Sequence[str],
    raw_config: Mapping[str, Any],
) -> None:
    print("=" * 72)
    print(f"SITE       : {spec.display_name} ({spec.slug})")
    print(f"CSV        : {spec.csv_path}")
    print(f"CONFIG     : {spec.config_path}")
    print(
        f"BESS       : {spec.expected_p_rated_kw:g} kW / "
        f"{spec.expected_e_cap_kwh:g} kWh"
    )
    print(
        f"DATA       : {audit.first_date} -> {audit.last_date} | "
        f"{audit.stored_days} stored days / {audit.rows} rows"
    )
    print(f"TRAIN      : {', '.join(train)}")
    print(f"VALIDATION : {', '.join(validation)}")
    print(f"TEST       : {', '.join(test)}")
    print(f"OUTPUT     : {spec.output_dir}")
    config_hash = str(raw_config.get("meta", {}).get("configHash", ""))
    if "TODO" in config_hash:
        print(
            "WARNING    : tariff/economics use shared reference values; "
            "TODO confirm site contract"
        )
    newest_calendar_month = max(audit.month_coverage)
    if test and test[-1] != newest_calendar_month:
        print(
            "WARNING    : newest calendar month is not eligible; trainer will test "
            f"{test[-1]}, not {newest_calendar_month}"
        )
    print("=" * 72)


def _prepare_output(spec: SiteSpec) -> None:
    output = spec.output_dir
    if output.exists() and any(output.iterdir()):
        raise PreflightError(
            f"refusing to overwrite non-empty run directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)


def allocate_cpu_threads(
    training_count: int,
    cpu_count: int | None = None,
) -> tuple[int, ...]:
    """Split logical CPUs across simultaneous site trainers."""
    if training_count < 1:
        raise ValueError("training_count must be >= 1")
    detected_cpus = cpu_count if cpu_count is not None else os.cpu_count()
    logical_cpus = max(1, detected_cpus or 1)
    base, remainder = divmod(logical_cpus, training_count)
    if base == 0:
        return (1,) * training_count
    return tuple(
        base + int(index < remainder) for index in range(training_count)
    )


def detect_gpu_ids(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return configured or discoverable NVIDIA GPU identifiers."""
    source = os.environ if environ is None else environ
    override = source.get(GPU_OVERRIDE_ENV)
    if override is not None:
        normalized = override.strip()
        if not normalized or normalized.lower() in {"none", "cpu", "off"}:
            return ()
        gpu_ids = tuple(
            part.strip() for part in normalized.split(",") if part.strip()
        )
        if len(gpu_ids) != len(set(gpu_ids)):
            raise PreflightError(f"{GPU_OVERRIDE_ENV} contains duplicate GPU IDs")
        return gpu_ids

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ()
    return tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )


def _seed_jobs(
    spec: SiteSpec,
    *,
    cpu_threads: int | None,
    gpu_ids: Sequence[str],
    gpu_offset: int,
) -> tuple[SeedJob, ...]:
    total_threads = max(1, cpu_threads or os.cpu_count() or 1)
    budgets = allocate_cpu_threads(len(SEEDS), total_threads)
    return tuple(
        SeedJob(
            spec=spec,
            seed=seed,
            cpu_threads=budget,
            gpu_id=(
                gpu_ids[(gpu_offset + index) % len(gpu_ids)]
                if gpu_ids
                else None
            ),
        )
        for index, (seed, budget) in enumerate(
            zip(SEEDS, budgets, strict=True)
        )
    )


def _seed_environment(job: SeedJob) -> dict[str, str]:
    env = os.environ.copy()
    env["DRL_RESULTS_DIR"] = str(job.spec.output_dir)
    thread_count = str(job.cpu_threads)
    env[CPU_COUNT_ENV] = thread_count
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PRIVATE_ROOT), python_path) if part
    )
    for variable in CPU_THREAD_ENV_VARS:
        env[variable] = thread_count
    if job.gpu_id is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = job.gpu_id
        env["NVIDIA_VISIBLE_DEVICES"] = job.gpu_id
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
    return env


def _run_seed_job(job: SeedJob) -> None:
    command = build_command(job)
    gpu = job.gpu_id if job.gpu_id is not None else "none"
    print(
        f"[seed] START {job.spec.display_name} seed={job.seed} "
        f"threads={job.cpu_threads} visible-gpu={gpu}",
        flush=True,
    )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_seed_environment(job),
        check=True,
    )
    print(
        f"[seed] DONE {job.spec.display_name} seed={job.seed}",
        flush=True,
    )


def _load_seed_result(job: SeedJob) -> tuple[float, float, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise PreflightError("PyTorch is required to merge seed checkpoints") from exc
    if not job.checkpoint_path.is_file():
        raise PreflightError(f"missing seed checkpoint: {job.checkpoint_path}")
    if not job.evaluation_path.is_file():
        raise PreflightError(f"missing seed evaluation: {job.evaluation_path}")
    checkpoint = torch.load(
        job.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("meta"), dict
    ):
        raise PreflightError(f"invalid seed checkpoint: {job.checkpoint_path}")
    meta = checkpoint["meta"]
    try:
        validation_cost = float(meta["validation_cost_vnd"])
        evaluation = json.loads(job.evaluation_path.read_text(encoding="utf-8"))
        metrics = evaluation.get("metrics")
        if isinstance(metrics, dict) and "test_saving_pct" in metrics:
            test_saving = float(metrics["test_saving_pct"])
        else:
            # TODO(PRIVATE-EVAL-SCHEMA): remove this legacy fallback after all
            # retained unversioned evaluation artifacts have migrated to schema >= 2.1.
            summary = evaluation.get("summary")
            if (
                evaluation.get("schema_version") is not None
                or not isinstance(summary, dict)
                or "test_saving_pct" not in summary
            ):
                raise KeyError("metrics.test_saving_pct")
            test_saving = float(summary["test_saving_pct"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid seed result for seed {job.seed}: {exc}") from exc
    return validation_cost, test_saving, checkpoint


def _aggregate_seed_outputs(spec: SiteSpec, jobs: Sequence[SeedJob]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise PreflightError("PyTorch is required to merge seed checkpoints") from exc
    results = [
        (job, *_load_seed_result(job))
        for job in jobs
    ]
    selected_job, selected_cost, selected_saving, selected_checkpoint = min(
        results,
        key=lambda result: result[1],
    )
    validation_by_seed = {
        str(job.seed): validation_cost
        for job, validation_cost, _test_saving, _checkpoint in results
    }
    savings_by_seed = {
        str(job.seed): test_saving
        for job, _validation_cost, test_saving, _checkpoint in results
    }
    meta = selected_checkpoint["meta"]
    meta.update({
        "seeds": list(SEEDS),
        "selected_seed": selected_job.seed,
        "selection_protocol": "mean_val_cost_over_seeds_then_best_seed",
        "parallel_seed_execution": True,
        "validation_cost_vnd_by_seed": validation_by_seed,
        "test_saving_pct_by_seed": [
            savings_by_seed[str(seed)] for seed in SEEDS
        ],
    })
    canonical_tag = f"{RUN_NAME}-{spec.slug}"
    canonical_checkpoint = spec.output_dir / f"policy_{canonical_tag}.pt"
    torch.save(selected_checkpoint, canonical_checkpoint)

    selected_evaluation = json.loads(
        selected_job.evaluation_path.read_text(encoding="utf-8")
    )
    selected_evaluation["policy_tag"] = canonical_tag
    selected_evaluation["seed_selection"] = {
        "protocol": "parallel_seeds_then_min_validation_cost",
        "selected_seed": selected_job.seed,
        "selected_validation_cost_vnd": selected_cost,
        "selected_test_saving_pct": selected_saving,
        "validation_cost_vnd_by_seed": validation_by_seed,
        "test_saving_pct_by_seed": savings_by_seed,
    }
    canonical_evaluation = spec.output_dir / f"evaluation_{canonical_tag}.json"
    canonical_evaluation.write_text(
        json.dumps(selected_evaluation, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"[seed] SELECT {spec.display_name} seed={selected_job.seed} "
        f"validation={selected_cost:.2f} test-saving={selected_saving:.2f}%",
        flush=True,
    )


def run_site(
    spec: SiteSpec,
    *,
    dry_run: bool = False,
    cpu_threads: int | None = None,
    gpu_ids: Sequence[str] = (),
    gpu_offset: int = 0,
) -> None:
    if not spec.enabled:
        raise PreflightError(f"{spec.display_name} disabled: {spec.disabled_reason}")
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError("cpu_threads must be >= 1")
    audit, train, validation, test, raw = preflight(spec)
    _print_preflight(spec, audit, train, validation, test, raw)
    jobs = _seed_jobs(
        spec,
        cpu_threads=cpu_threads,
        gpu_ids=gpu_ids,
        gpu_offset=gpu_offset,
    )
    for job in jobs:
        print("COMMAND    :", " ".join(build_command(job)))
    if dry_run:
        print("DRY RUN    : no training started")
        return

    _prepare_output(spec)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        future_to_job = {
            executor.submit(_run_seed_job, job): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                future.result()
            except subprocess.CalledProcessError as exc:
                failures.append(f"seed {job.seed}: {exc}")
    if failures:
        raise PreflightError(
            f"{spec.display_name} seed training failed: {' | '.join(failures)}"
        )
    _aggregate_seed_outputs(spec, jobs)


def run_all(*, dry_run: bool = False) -> None:
    enabled_specs: list[SiteSpec] = []
    for slug in TRAIN_ALL_ORDER:
        spec = SITES[slug]
        if not spec.enabled:
            print(f"[trainall] SKIP {spec.display_name}: {spec.disabled_reason}")
            continue
        enabled_specs.append(spec)

    # Keep dry-run deterministic/readable: preflight output from concurrent sites
    # would interleave and make the experiment receipt needlessly hard to audit.
    if dry_run:
        failures: list[str] = []
        for spec in enabled_specs:
            try:
                run_site(spec, dry_run=True)
            except (PreflightError, subprocess.CalledProcessError) as exc:
                failures.append(f"{spec.display_name}: {exc}")
                print(f"[trainall] FAILED {spec.display_name}: {exc}", file=sys.stderr)
        if failures:
            raise SystemExit("trainall failed: " + " | ".join(failures))
        return

    if not enabled_specs:
        return

    workers = len(enabled_specs)
    cpu_budgets = allocate_cpu_threads(workers)
    gpu_ids = detect_gpu_ids()
    print(
        f"[trainall] starting {len(enabled_specs)} enabled sites with "
        f"{workers} parallel workers across {sum(cpu_budgets)} logical CPU threads "
        f"(per-site budgets: {', '.join(map(str, cpu_budgets))}); "
        f"visible GPUs: {', '.join(gpu_ids) if gpu_ids else 'none'}",
        flush=True,
    )
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_spec = {
            executor.submit(
                run_site,
                spec,
                dry_run=False,
                cpu_threads=cpu_threads,
                gpu_ids=gpu_ids,
                gpu_offset=site_index * len(SEEDS),
            ): spec
            for site_index, (spec, cpu_threads) in enumerate(
                zip(enabled_specs, cpu_budgets, strict=True)
            )
        }
        for future in concurrent.futures.as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                future.result()
                print(f"[trainall] DONE {spec.display_name}", flush=True)
            except (PreflightError, subprocess.CalledProcessError) as exc:
                failures.append(f"{spec.display_name}: {exc}")
                print(
                    f"[trainall] FAILED {spec.display_name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
    if failures:
        raise SystemExit("trainall failed: " + " | ".join(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hardcoded private multi-site PPO trainer/orchestrator."
    )
    parser.add_argument(
        "target",
        choices=["trainall", *SITES],
        help="One site slug or trainall for the seven-site batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every preflight and print exact commands without training.",
    )
    args = parser.parse_args(argv)

    try:
        if args.target == "trainall":
            run_all(dry_run=args.dry_run)
        else:
            run_site(
                SITES[args.target],
                dry_run=args.dry_run,
                gpu_ids=detect_gpu_ids(),
            )
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
