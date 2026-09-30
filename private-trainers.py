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
RUN_NAME = "ppo-iq2-coherent-bc-memory"
MIN_MONTH_COVERAGE = 0.80
VAL_MONTHS = 2
TEST_MONTHS = 1
STEPS_PER_DAY = 96
CPU_THREAD_ENV_VARS: tuple[str, ...] = (
    "DRL_TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# TODO(PARALLEL-TRAIN): benchmark the CPU split per training host and make the
# launcher GPU-aware once PPO has an explicit device/batching contract.

# Keep the experiment receipt explicit. Do not silently inherit trainer defaults:
# changing a default in run_train_dataset.py must not mutate an old private run.
TRAIN_ARGS: tuple[str, ...] = (
    "--steps", "1500000",
    "--seeds", "0,1,2",
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


def build_command(spec: SiteSpec) -> list[str]:
    tag = f"{RUN_NAME}-{spec.slug}"
    return [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(TRAINER),
        "--csv",
        str(spec.csv_path),
        "--config-json",
        str(spec.config_path),
        *TRAIN_ARGS,
        "--tag",
        tag,
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


def run_site(
    spec: SiteSpec,
    *,
    dry_run: bool = False,
    cpu_threads: int | None = None,
) -> None:
    if not spec.enabled:
        raise PreflightError(f"{spec.display_name} disabled: {spec.disabled_reason}")
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError("cpu_threads must be >= 1")
    audit, train, validation, test, raw = preflight(spec)
    _print_preflight(spec, audit, train, validation, test, raw)
    command = build_command(spec)
    print("COMMAND    :", " ".join(command))
    if dry_run:
        print("DRY RUN    : no training started")
        return

    _prepare_output(spec)
    env = os.environ.copy()
    env["DRL_RESULTS_DIR"] = str(spec.output_dir)
    if cpu_threads is not None:
        thread_count = str(cpu_threads)
        for variable in CPU_THREAD_ENV_VARS:
            env[variable] = thread_count
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


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
    print(
        f"[trainall] starting {len(enabled_specs)} enabled sites with "
        f"{workers} parallel workers across {sum(cpu_budgets)} logical CPU threads "
        f"(per-site budgets: {', '.join(map(str, cpu_budgets))})",
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
            ): spec
            for spec, cpu_threads in zip(enabled_specs, cpu_budgets, strict=True)
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
            run_site(SITES[args.target], dry_run=args.dry_run)
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
