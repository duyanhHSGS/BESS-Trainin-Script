# Pure causal rule-based BESS benchmark

This private application is the non-learning comparison policy for DRL experiments.
It imports battery configuration, tariffs, physical bounds, fixed 30-minute billing
math, and scoring from the parent repository ("mama"). It imports no PPO policy,
observation builder, reward, checkpoint, or future load/PV information.

Priority order:

1. physical SOC/power/efficiency/zero-export limits;
2. PV self-consumption;
3. fixed-block Peak Police;
4. evenly recomputed midnight-to-06:00 charging;
5. a soft 20% usable-energy reserve;
6. break-even tariff arbitrage and cheap daytime charging;
7. no-op.

Peak targets use only the previous 30 complete days' no-BESS daily fixed-block
maxima. A robust Q90 is reduced by the battery's sustainable five-hour shave.
The already-billed monthly peak is a floor because yesterday's meter cannot be
un-rung 🔔.

The headline `test_saving_pct` uses the same chronological data protocol as the
DRL trainer: calendar months with at least 80% stored-day coverage, the last two
eligible months for validation, and the last eligible month for untouched test.
Earlier and partial months may warm causal history but never leak future values.

Run one site or all private sites:

```bash
.venv-linux/bin/python private-data-and-results/rule-based/run.py amy
.venv-linux/bin/python private-data-and-results/rule-based/run.py all
```

Runtime JSON files go to `rule-based/results/` and are intentionally ignored.

Run tests:

```bash
.venv-linux/bin/python -m pytest private-data-and-results/rule-based/tests
```

TODO(RULE-BASELINE-CALIBRATION): freeze controller constants after the complete
multi-site untouched-holdout comparison against the DRL policies.
