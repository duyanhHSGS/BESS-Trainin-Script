"""Command line entrypoint for private rule-based benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rule_based import mama
from rule_based.benchmark import run_benchmark
from rule_based.data import load_site_days
from rule_based.models import RuleSettings

PRIVATE_ROOT = Path(__file__).resolve().parents[2]
SITES_ROOT = PRIVATE_ROOT / "sites"
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"


def available_sites() -> tuple[str, ...]:
    return tuple(
        path.name
        for path in sorted(SITES_ROOT.iterdir())
        if (path / "config.json").is_file() and (path / "data.csv").is_file()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pure causal rule-based BESS benchmark",
    )
    parser.add_argument("sites", nargs="+", help="site slug(s), or 'all'")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--peak-quantile", type=float, default=0.90)
    parser.add_argument("--sustainable-peak-hours", type=float, default=5.0)
    parser.add_argument("--reserve-fraction", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    known = available_sites()
    requested = list(known) if args.sites == ["all"] else args.sites
    unknown = sorted(set(requested).difference(known))
    if unknown:
        raise SystemExit(f"unknown site(s): {', '.join(unknown)}")
    settings = RuleSettings(
        history_days=args.history_days,
        peak_quantile=args.peak_quantile,
        sustainable_peak_hours=args.sustainable_peak_hours,
        reserve_usable_fraction=args.reserve_fraction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for site in requested:
        site_root = SITES_ROOT / site
        cfg = mama.load_bess_drl_config(site_root / "config.json")
        result = run_benchmark(load_site_days(site_root / "data.csv"), cfg, settings)
        output_path = args.output_dir / f"{site}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics = result["metrics"]
        headline = metrics["test_saving_pct"]
        saving = headline if headline is not None else metrics["saving_pct"]
        scope = "test" if headline is not None else "all-stored"
        print(
            f"{site}: {scope}-saving={saving:.3f}% "
            f"violations={metrics['hard_constraint_violation_days']} -> {output_path}"
        )
    return 0
