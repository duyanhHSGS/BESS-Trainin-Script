# BESS DRL Experiment Report — Tande

## Goal

Target:
- Use `fixed_dataset_v1_tande` as the Tande production PPO baseline.
- Compare every future Tande IQ against the same Tande data/config/split.
- Preserve validation-only model selection; never choose a policy using final-test performance.

This file is the single rolling experiment scoreboard for Tande BASE + future Tande IQ runs.

---

## Baselines Before DRL

These are the two non-learning reference controllers used to judge every PPO experiment. They use the same Tande test data, tariff, BESS physics, and bill scorer as the DRL policies.

Test reference period: `2026-08-01` -> `2026-08-31` (100.0% August coverage).

| Baseline | What it means | Total Cost | Energy Cost | Demand Cost | Degradation / Terminal | Peak | Saving vs No-BESS | Peak Cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **No-BESS** | Battery does nothing: 0 kW charge/discharge. This is the real denominator for savings. | **367,288,396.84 VND** | 123,120,446.41 VND | 244,167,950.43 VND | 0 VND | **856.73 kW** | 0.00% | 0.00 kW |
| **Oracle** | Perfect-foresight whole-month LP. It knows the future load/PV and optimizes the same physical/economic problem. This is the practical lower-bound / god-mode reference, not a deployable controller. | **272,763,419.07 VND** | 118,072,832.48 VND | 150,777,923.56 VND | 3,849,885.25 VND degradation + 62,777.78 VND terminal | **529.05 kW** | **25.7359%** | **327.68 kW** |

Interpretation:
- **No-BESS baseline:** if the battery literally sits there doing nothing, this full August costs **367.288M VND** and reaches a **856.73 kW** monthly peak. Every DRL saving percentage is measured against this bill.
- **Oracle baseline:** with impossible perfect future knowledge, the best whole-month optimization found costs **272.763M VND**, about **94.525M VND less than No-BESS**, or **25.7359% saving**. It cuts the peak to **529.05 kW**.
- The gap between No-BESS and Oracle is the total economic opportunity available on this test month. A DRL policy is judged by how much of that opportunity it captures without seeing the future.
- Tande BASE captures **55.31%** of this Oracle opportunity; IQ1 GRU captures **54.42%**.

Important: unlike the Youngone benchmark, Tande has a complete August billing month: **31 / 31 days = 100% coverage**.

---

## Experiment Leaderboard

| IQ | Run | Commit | Main Change | Validation Saving | Test Saving | Peak Cut | Selected Seed | Result |
|---|---|---|---|---:|---:|---:|---:|---|
| BASE | `fixed_dataset_v1_tande` | `cbdf7d0` | Production feed-forward PPO baseline | **15.5378%** | **14.23%** | 204.43 kW | 1 | **Money-saving reference** |
| IQ1 | `ppo-iq1-gru-memory-uncle-tande` | `87e6e93` | Separate 128-wide actor/critic GRUs with 96-step TBPTT and billing-month memory reset | 15.3005% | 14.00% | **216.83 kW** | 1 | Stronger peak shave; weaker savings |

---

## Current Best

Best validation:
- BASE `fixed_dataset_v1_tande`: **15.5378%**

Best selected-policy test:
- BASE `fixed_dataset_v1_tande`: **14.23%**

Best individual seed test seen inside BASE:
- BASE seed 0: **14.66%**

Best peak shaving:
- IQ1 `ppo-iq1-gru-memory-uncle-tande`: **216.83 kW peak cut** / **639.90 kW peak**

Current money-saving champion:
- BASE feed-forward PPO, selected seed 1
- Commit: `cbdf7d0`
- Champion checkpoint: `private-data-and-results/results/policy_fixed_dataset_v1_tande.pt`

Current peak-shaving champion:
- IQ1 GRU, selected seed 1
- Commit: `87e6e93`
- Checkpoint: `private-data-and-results/results/policy_ppo-iq1-gru-memory-uncle-tande.pt`

---

## BASE — fixed_dataset_v1_tande

### Identity

- Run: `fixed_dataset_v1_tande`
- Commit: `cbdf7d02651b55abb2e264401ee2a5486b2eb3db`
- Branch used after temporary detached checkout: `dev-2-testing`
- Algorithm: production PPO
- Architecture: feed-forward actor + separate energy/peak critics
- Seeds: `0,1,2`
- Selected seed: `1`
- Training steps: `1,500,000` per seed
- Observation dimension: 17
- Observation schema: `causal_block_aware`
- Action interval: 15 minutes
- Action mapping: `physical_feasible_15m`
- Lambda peak: `0.97`
- Lambda energy: `0.97`
- Behavior-cloning epochs: 10

### Hypothesis

None. This is the production feed-forward PPO baseline used to compare future Tande IQ experiments.

The site data changes to Tande while the virtual BESS hardware, tariff, and economics remain the same as the first cross-site Youngone setup. This isolates site behavior from hardware/economic changes.

### Dataset Split

- CSV: `private-data-and-results/sites/tande/data.csv`
- Dataset SHA-256: `2427e1a14b418608f93394894693809286da1b408b96420e3fe9b1c7e1e2141d`
- Rows: 28,800
- Unique days: 300
- Data range: `2025-09-06` -> `2026-08-31`
- Resolution: 15 minutes
- Training months: 7
- Training months: `2025-09`, `2025-12`, `2026-01`, `2026-02`, `2026-03`, `2026-04`, `2026-05`
- Validation months: `2026-06`, `2026-07`
- Test month: `2026-08`
- Test range: `2026-08-01` -> `2026-08-31`
- Test month coverage: **100.0%**
- Minimum eligible month coverage: 80%
- Sparse October/November are rejected by the >=80% rule.
- Largest training load: 1,345.45 kW
- Automatic `p_ref`: **1500 kW**

Tande load / PV character:
- Load minimum: 0.00 kW
- Load mean: 241.42 kW
- Load median: 24.72 kW
- Load p90: 836.05 kW
- Load p95: 961.89 kW
- Load p99: 1,136.74 kW
- Load maximum: **1,345.45 kW**
- PV mean: 144.04 kW
- PV maximum: **1,176.25 kW**
- Mean net load: 97.84 kW
- Maximum net load: 1,097.69 kW
- PV exceeds load in about 0.5% of slots
- Net load is zero in about 2.0% of slots

Tande is highly spiky: ordinary load can be tiny while the upper tail jumps above 1 MW, with very large daytime PV. Peak shaving therefore dominates a lot of the economics.

### BESS / Economics

- Site ID: `tande`
- Config: `private-data-and-results/sites/tande/config.json`
- Capacity: 1250 kWh
- Rated power: 450 kW
- Charge efficiency: 0.90
- Discharge efficiency: 0.90
- SOC min/max: 0.20 / 0.90
- Export: disabled
- Billing mode: 2TC
- Demand charge: 285,000 VND/kW
- Peak window: 17:30-22:30
- Off-peak window: 00:00-06:00
- Degradation: 500 VND/kWh discharged

### Validation

Selected seed 1 checkpoint:
- Best validation cost: **680.786M VND**
- Best checkpoint step: **519,375**
- Validation saving: **15.5378%**
- Validation Oracle gap: 15.3979%
- Validation peak: 786.41 kW

### Final Test

- No-BESS cost: 367.288M VND
- DRL cost: **315.009M VND**
- Oracle cost: 272.763M VND
- DRL saving: **14.23%**
- Oracle saving vs No-BESS: 25.7359%
- Oracle gap: **15.49%**
- Opportunity captured: **55.31%**
- Test coverage: **100.0%**

Cost breakdown:
- Energy cost: 122,392,503.19 VND
- Demand cost: 185,905,803.18 VND
- Degradation cost: 7,321,751.90 VND
- Terminal settlement: -611,218.42 VND

### Peak Shaving

- No-BESS peak: 856.73 kW
- DRL peak: **652.30 kW**
- Oracle peak: 529.05 kW
- DRL peak cut: **204.43 kW**
- Oracle peak cut: 327.68 kW
- Demand at the original No-BESS peak event under DRL: 625.99 kW
- Cut at that original peak event: 230.74 kW

Diagnostics:
- Physical clip rate: **39.11%**
- Requested/executed mismatch: 16,525.57 kWh
- Sign flips: 193
- Peak-window grid charging: **921.24 kWh**
- Terminal SOC: 85.10%

### Seed Comparison

| Seed | Best Validation Saving | Best Validation Step | Test Saving |
|---:|---:|---:|---:|
| 0 | 15.0056% | 461,665 | **14.66%** |
| **1** | **15.5378%** | **519,375** | 14.23% |
| 2 | 10.9157% | 230,835 | 11.17% |

Seed 1 was selected because checkpoint selection is based on validation performance, not test performance. Seed 0 has the strongest test number but is intentionally not selected.

### Training Behavior

The strongest validation checkpoints appeared well before 1.5M steps. Later PPO updates often degraded validation performance.

Examples:
- Seed 0 peaked at step 461,665 with 15.0056% validation saving, then degraded.
- Seed 1 peaked at step 519,375 with **15.5378%**, then drifted downward.
- Seed 2 was much less stable and never matched seeds 0/1.

The checkpoint-selection guard prevented later weaker learners from replacing stronger earlier policies.

### Artifacts

- `private-data-and-results/results/policy_fixed_dataset_v1_tande.pt`
- `private-data-and-results/results/evaluation_fixed_dataset_v1_tande.json`
- `private-data-and-results/results/training_curve_fixed_dataset_v1_tande.csv`
- `private-data-and-results/sites/tande/config.json`
- `private-data-and-results/sites/tande/data.csv`

Per-seed policy/evaluation/curve artifacts are also preserved for seeds 0, 1, and 2.

### Conclusion

BASE establishes the Tande feed-forward PPO reference:
- Validation: **15.5378%**
- Test: **14.23%**
- Peak cut: **204.43 kW**
- Selected seed: 1
- Opportunity captured: **55.31%**

Future Tande IQs should be compared against this row using the same data/config/split whenever possible.

---

## IQ1 — ppo-iq1-gru-memory-uncle-tande

### Identity

- Experiment family: `ppo-iq1-gru-memory-uncle-tande`
- Commit: `87e6e93` (`iq1-gru`)
- Branch: `dev-2-testing`
- Algorithm: production PPO with recurrent GRU memory
- Seeds: `0,1,2`
- Selected seed: `1`
- Training steps: `1,500,000` per seed
- Observation dimension: 17 — unchanged from BASE
- Observation schema: `causal_block_aware` — unchanged from BASE
- Action interval: 15 minutes — unchanged from BASE
- Action mapping: `physical_feasible_15m` — unchanged from BASE

### Main Change

Only the policy/value architecture gained memory:
- 128-wide actor encoder + actor GRU + actor head
- 128-wide critic encoder + critic GRU
- Separate energy and peak value heads preserved
- Dual PopArt critics preserved
- Recurrent sequence length: 96 steps = 1 day
- Memory reset contract: billing-month boundary
- Recurrent PPO optimizer updates happen only after a completed billing-month episode
- Oracle behavior cloning is chronological in recurrent mode
- Production inference carries actor GRU state and resets it at the existing billing-month reset event

The 17-eye observation contract, reward/economics, BESS physics, tariff, feasible-action mapping, Tande CSV, and split were intentionally kept unchanged so IQ1 directly tests whether memory helps.

### Hypothesis

A feed-forward policy sees the current 17-eye snapshot but cannot directly remember what happened earlier in the billing month. A GRU may learn better month-scale peak-shaving behavior by carrying historical context in hidden state.

### Dataset / Economics

Same Tande data split and BESS/tariff/economics as BASE:
- Train: 7 months
- Validation: 2 months
- Test: `2026-08-01` -> `2026-08-31`
- Test coverage: **100.0%**
- BESS: 1250 kWh / 450 kW
- Demand charge: 285,000 VND/kW
- Degradation: 500 VND/kWh discharged
- `p_ref`: 1500 kW

### Seed Results

| Seed | Best Validation Saving | Best Validation Step | Final Test Saving | Notes |
|---:|---:|---:|---:|---|
| 0 | 14.0820% | 159,840 | 13.13% | Early champion, later drift |
| **1** | **15.3005%** | **213,696** | **14.00%** | **Selected by validation** |
| 2 | 12.9025% | 213,696 | 5.82% | Weakest generalization |

All three seeds were trained independently in parallel under the same code/data/config/step budget. Seed 1 is selected because it has the strongest validation checkpoint.

### Validation — Selected Seed 1

- Best validation cost: **682.699M VND**
- Best checkpoint step: **213,696**
- Validation saving: **15.3005%**
- Validation Oracle gap: 15.7221%
- Validation peak: 744.25 kW

Compared with BASE validation:
- BASE: **15.5378%**
- IQ1: 15.3005%
- Delta: **-0.2373 percentage points**

### Final Test — Selected Seed 1

- No-BESS cost: 367,288,396.84 VND
- DRL cost: **315,851,390.71 VND**
- Oracle cost: 272,763,419.07 VND
- DRL saving: **14.00%**
- Oracle gap: **15.80%**
- Opportunity captured: **54.42%**
- Test coverage: **100.0%**

### Peak Shaving

- No-BESS peak: 856.73 kW
- IQ1 DRL peak: **639.90 kW**
- Oracle peak: 529.05 kW
- IQ1 peak cut: **216.83 kW**
- Oracle peak cut: 327.68 kW
- Demand at BASE peak event under IQ1: 605.47 kW
- Cut at BASE peak event under IQ1: 251.26 kW

Diagnostics:
- Physical clip rate: **50.60%**
- Requested/executed mismatch: 27,733.56 kWh
- Sign flips: 298
- Peak-window grid charging: **2,799.61 kWh**
- Terminal SOC: 90.00%

### Compared with BASE

| Metric | BASE | IQ1 GRU | Delta |
|---|---:|---:|---:|
| Validation saving | **15.5378%** | 15.3005% | -0.2373 pp |
| Test saving | **14.23%** | 14.00% | -0.23 pp |
| Test DRL cost | **315.009M VND** | 315.851M VND | +0.843M VND |
| Test DRL peak | 652.30 kW | **639.90 kW** | **-12.40 kW** |
| Peak cut | 204.43 kW | **216.83 kW** | **+12.40 kW** |
| Oracle gap | **15.49%** | 15.80% | +0.31 pp |
| Opportunity captured | **55.31%** | 54.42% | -0.89 pp |
| Physical clip rate | **39.11%** | 50.60% | +11.49 pp |
| Peak-window grid charging | **921.24 kWh** | 2,799.61 kWh | +1,878.37 kWh |

IQ1 cuts the monthly peak harder than BASE, but the extra demand-charge improvement does not compensate for worse energy-side behavior:
- IQ1 energy cost: 128.251M VND vs BASE 122.393M VND
- IQ1 demand cost: 182.370M VND vs BASE 185.906M VND

The much higher physical clipping and peak-window grid charging are important targets for the next experiment rather than reasons to add hard-coded police immediately.

### Training Behavior

GRU did not remove PPO checkpoint drift by itself.

Seed 1:
- Step 213,696: **15.3005% validation saving — best checkpoint**
- Final 1.5M-step learner: about 9.8% validation saving

Seed 0:
- Step 159,840: 14.0820% validation saving — best checkpoint
- Later training never beat that early champion

Seed 2:
- Step 213,696: 12.9025% validation saving — best checkpoint
- Final test generalization was much weaker at 5.82%

Checkpoint selection remains essential even with recurrent memory. Memory Uncle can still wander off and lick the wall later 😂.

### Artifacts

Champion seed 1 canonical artifacts:
- `private-data-and-results/results/policy_ppo-iq1-gru-memory-uncle-tande.pt`
- `private-data-and-results/results/evaluation_ppo-iq1-gru-memory-uncle-tande.json`
- `private-data-and-results/results/training_curve_ppo-iq1-gru-memory-uncle-tande.csv`

Per-seed policy/evaluation/curve artifacts are also preserved for seeds 0, 1, and 2.

### Conclusion

On Tande, IQ1 GRU does **not** beat BASE on the primary money-saving objective in this run:
- Validation: 15.3005% vs BASE **15.5378%**
- Test: 14.00% vs BASE **14.23%**

But IQ1 improves peak shaving:
- Peak: **639.90 kW** vs BASE 652.30 kW
- Peak cut: **216.83 kW** vs BASE 204.43 kW

BASE remains the Tande money-saving reference. IQ1 is currently the stronger peak-shaving reference.

---

## Future IQ Entry Template

### IQX — run-name

- Commit: `...`
- Main change: ...
- Selected seed: ...

Hypothesis:
- ...

Result:
- Validation saving: ...
- Test saving: ...
- No-BESS cost: ...
- DRL cost: ...
- Oracle cost: ...
- Peak cut: ...
- Oracle gap: ...

Compared with BASE:
- Validation: ... percentage points
- Test: ... percentage points
- Peak cut: ... kW
- Cost: ... VND

Compared with current champion:
- Validation: ... percentage points
- Test: ... percentage points
- Peak cut: ... kW
- Cost: ... VND

Conclusion:
- Keep / Reject / Needs more evidence
- Why: ...

Artifacts:
- `...`

---

## TODO

- TODO: append every new Tande IQ to the leaderboard and add one detailed section below it.
- TODO: always record run name, commit, exact main change, validation, test, peak cut, selected seed, and artifact paths.
- TODO: compare every experiment against Tande BASE and the current best references.
- TODO: never call an IQ better only because validation improved; always check untouched test economics.
- TODO: preserve failed experiments too. Bad results are evidence, not trash.
- TODO: investigate the high BASE 39.11% and IQ1 50.60% physical clip rates as policy-learning inefficiency without immediately adding hard-coded rules.
- TODO: investigate why IQ1 increases peak-window grid charging from 921.24 kWh to 2,799.61 kWh.
- TODO: keep Tande and Youngone reports separate because their site distributions and test-month coverage differ.
- TODO: if real Tande-specific BESS hardware or tariff configuration becomes available, create a separate experiment rather than silently replacing this same-hardware cross-site benchmark.
