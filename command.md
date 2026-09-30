# Private Multi-Site PPO Training

All private experiment inputs are site-scoped under `private-data-and-results/sites/`.
The core PPO trainer stays generic; all site selection, config selection, run naming,
output paths, and experiment arguments live in `private-data-and-results/private-trainers.py`.

## Layout

```text
private-data-and-results/
├── private-trainers.py
├── test_private_trainers.py
├── command.md
├── archived/
│   └── old-layout/
│       ├── offline_data_Youngone.csv
│       └── offline_tande.csv
└── sites/
    ├── amy/
    │   ├── data.csv
    │   └── config.json
    ├── namduoc/
    │   ├── data.csv
    │   └── config.json
    ├── newing/
    │   ├── data.csv
    │   └── config.json
    ├── youngone/
    │   ├── data.csv
    │   └── config.json
    ├── songwol/
    │   ├── data.csv
    │   └── config.json
    ├── minhdanh/
    │   ├── data.csv
    │   └── config.json
    └── tande/
        ├── data.csv
        ├── config.json
        └── report.md
```

Training outputs are created at runtime under:

```text
private-data-and-results/sites/<site>/results/ppo-iq2-coherent-bc-memory/
```

The launcher refuses to overwrite a non-empty run directory.

## One-button training

Run every currently enabled site at the same time. The launcher divides all available
logical CPUs across the site trainer subprocesses, including PyTorch and common numeric
libraries. Each site keeps its own sequential seed-selection loop and isolated result
directory:

```bash
cd /home/admin/Desktop/CodeProjects/bess-infra
.venv-linux/bin/python -X utf8 private-data-and-results/private-trainers.py trainall
```

Dry-run every preflight and print the exact commands without training. Dry-run stays sequential so the audit output remains readable:

```bash
.venv-linux/bin/python -X utf8 private-data-and-results/private-trainers.py trainall --dry-run
```

Train one site:

```bash
.venv-linux/bin/python -X utf8 private-data-and-results/private-trainers.py youngone
.venv-linux/bin/python -X utf8 private-data-and-results/private-trainers.py newing
.venv-linux/bin/python -X utf8 private-data-and-results/private-trainers.py songwol
```

Tande is included in `trainall` and can also be selected directly.

## Hardcoded IQ2 experiment receipt

`private-trainers.py` explicitly passes these values instead of inheriting mutable
trainer defaults:

- run: `ppo-iq2-coherent-bc-memory`
- steps: `1,500,000`
- seeds: `0,1,2`
- rollout: `2880`
- evaluation cadence: every `20` PPO updates
- minimum month coverage: `0.8`
- actor LR: `3e-5`
- critic LR: `3e-4`
- initial std: `0.15`
- clip penalty: `100.0`
- behaviour-cloning epochs: `10`
- lambda energy: `0.97`
- lambda peak: `0.97`

The launcher sets `DRL_RESULTS_DIR` to the selected site's run directory and adds the
site slug to the trainer tag. For `trainall`, it also assigns each simultaneous trainer
a share of the machine's logical CPUs through `DRL_TORCH_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. If there are fewer
CPUs than enabled sites, every site still starts immediately with a one-thread budget.

## Preflight contract

Before expensive training, the launcher verifies:

- trainer file exists;
- CSV and config exist;
- config `siteId` matches the selected site;
- BESS kW and kWh match the hardcoded site manifest;
- every stored CSV day contains exactly 96 unique steps `0..95`;
- numeric load/PV rows are parseable;
- at least four calendar months meet the 80% coverage threshold;
- the chronological split contains training months, two validation months, and one test month;
- a previous non-empty run directory will not be overwritten.

If the newest calendar month is below the coverage threshold, preflight prints a warning
showing which older month will actually become test.

## Site hardware manifest

| Site | P_BESS kW | E_BESS kWh | trainall |
| --- | ---: | ---: | --- |
| Á Mỹ | 525 | 1500 | yes |
| Nam Dược | 500 | 1000 | **blocked** |
| Newing | 2450 | 3500 | yes |
| YoungOne | 500 | 1000 | yes |
| Songwol | 450 | 1250 | yes |
| Minh Danh | 250 | 500 | yes |
| Tande | 450 | 1250 | yes |

Nam Dược is blocked because the newly pulled dataset cannot form the required
train/validation/test split and its only August day has a dead-looking 0 kW load signal.

## Config warning

The battery sizes above are site-specific. For newly created site configs, tariff and
economics fields currently inherit the existing shared reference experiment values so the
training plumbing is explicit and reproducible.

**TODO:** replace those shared reference tariff/economic values with authoritative
site-specific contracts before treating cross-site savings as production-comparable numbers.

## Tests

Tests intentionally live in the private zone:

```bash
.venv-linux/bin/python -m pytest private-data-and-results/test_private_trainers.py
```

They cover manifest hardware, real configs, real dataset preflight, 96-slot integrity,
coverage/split behavior, command construction, trainall filtering, disabled-site behavior,
and output overwrite protection.

Historical checkpoint/result files referenced by older reports may live outside the tracked
checkout or in ignored runtime storage. This reorganization does not fabricate or delete
those unavailable files; all new outputs use the site-scoped layout above.
