"""Apply the private launcher's CPU quota before trainer imports execute."""

from __future__ import annotations

import os

CPU_COUNT_ENV = "PRIVATE_TRAINER_CPU_COUNT"


def _apply_cpu_quota() -> None:
    raw_count = os.environ.get(CPU_COUNT_ENV)
    if raw_count is None:
        return
    try:
        count = int(raw_count)
    except ValueError:
        return
    if count < 1:
        return

    def configured_cpu_count() -> int:
        return count

    # The core trainer sizes its LP-oracle ProcessPoolExecutor with os.cpu_count().
    # TODO(CPU-BUDGET): remove this private startup shim after the core trainer
    # accepts an explicit oracle-worker budget.
    setattr(os, "cpu_count", configured_cpu_count)
    if hasattr(os, "process_cpu_count"):
        setattr(os, "process_cpu_count", configured_cpu_count)


_apply_cpu_quota()
