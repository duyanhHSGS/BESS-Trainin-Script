# Private Multi-Site PPO Training

All private experiment inputs are site-scoped under `private-data-and-results/sites/`.
The core PPO trainer stays generic; all site selection, config selection, run naming,
output paths, and experiment arguments live in `private-data-and-results/private-trainers.py`.

## Layout

```text
private-data-and-results/
├── private-trainers.py
├── sitecustomize.py
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
private-data-and-results/sites/<site>/results/iq4_privileged_critic_v1/
```

The launcher refuses to overwrite a non-empty run directory.

Each seed writes a distinct checkpoint and evaluation artifact. After all three finish,
the launcher selects the checkpoint with the lowest validation cost and writes canonical
`policy_<run>-<site>.pt` and `evaluation_<run>-<site>.json` artifacts. The canonical
checkpoint/evaluation records validation cost and test saving for every seed.

## One-button training

Run every currently enabled site at the same time. Every site's seeds `0`, `1`, and `2`
also run as separate subprocesses at the same time, so `trainall` launches the complete
site-by-seed matrix instead of leaving seeds in a sequential queue. The launcher divides
all logical CPUs across sites and then across their seeds. Each site keeps an isolated
result directory:

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

**Tande IQ2 regression guard (2026-09-30):** the run `ppo-iq2-coherent-bc-memory-tande` on checkout `0f6b6c3` was intentionally aborted after seed 0 completed, seed 1 reached 641,376 steps, and seed 2 had not started. Seed 1's best-validation checkpoint reached 13.9119% validation saving and 13.0452% diagnostic test saving, both below the existing Tande BASE (15.5378% validation / 14.23% test) and IQ1 (15.3005% / 14.00%) references. **Do not rerun that exact Tande IQ2 recipe unchanged.** Any future Tande IQ2-derived experiment needs a materially different written hypothesis and a new run name; preserve the failed artifacts as evidence.

**TODO(IQ2-TANDE-GUARD):** before launching Tande in a future batch, confirm the experiment is not merely reproducing the rejected `ppo-iq2-coherent-bc-memory-tande` recipe and record the changed hypothesis in `sites/tande/report.md`.

## Hardcoded IQ4 experiment receipt

`private-trainers.py` explicitly passes these values instead of inheriting mutable
trainer defaults:

- run: `iq4_privileged_critic_v1`
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
site slug and seed to the trainer tag. It assigns each simultaneous seed trainer a share
of the machine's logical CPUs through `DRL_TORCH_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. If there are fewer
CPUs than jobs, every seed still starts immediately with a one-thread budget.

The private `sitecustomize.py` startup shim also exposes only that assigned CPU count to
each child interpreter. This keeps the core trainer's LP-oracle process pool inside the
same quota instead of letting every seed spawn workers for the entire machine.

The launcher discovers NVIDIA GPU IDs with `nvidia-smi` and rotates visible devices
across seed jobs. Override discovery with `PRIVATE_TRAINER_GPUS=0,1` or force CPU-only
visibility with `PRIVATE_TRAINER_GPUS=cpu`. PPO rollout collection and tiny inference
steps remain on CPU, while optimizer-heavy behaviour cloning and PPO updates use the
core trainer's `DRL_TRAIN_DEVICE=auto` contract and select CUDA when available. The
launcher therefore assigns real GPU visibility to seed jobs without changing PPO math.
This private launcher deliberately does not modify root-repository code.

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
coverage/split behavior, command construction, site/seed parallelism, CPU allocation,
GPU discovery/visibility, canonical seed selection, disabled-site behavior, and output
overwrite protection.

Historical checkpoint/result files referenced by older reports may live outside the tracked
checkout or in ignored runtime storage. This reorganization does not fabricate or delete
those unavailable files; all new outputs use the site-scoped layout above.
