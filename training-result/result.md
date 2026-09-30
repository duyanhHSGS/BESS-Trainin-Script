# IQ2 Multi-Site Training Report

Date: 2026-09-29  
Run: `ppo-iq2-coherent-bc-memory`  
Training commit: `0f6b6c3` (`iq2-fix-recurrent-bc`)  
Execution: Á Mỹ completed first; Newing, YoungOne, Songwol, and Minh Danh completed in parallel. All four parallel site wrappers exited 0.  
Scope: five enabled `trainall` sites. Nam Dược remains blocked by dataset/split quality rules. Tande is direct-only and was not part of this IQ2 batch.

## Executive result

IQ2 completed successfully on all five enabled sites, but it does **not** improve holdout bill saving broadly versus IQ1.

- IQ2 beats IQ1 on holdout saving at **1/5 sites**: Newing.
- IQ2 beats IQ0 on holdout saving at **1/5 sites**: Á Mỹ.
- The largest IQ2 regression versus IQ1 is **YoungOne: -3.24 percentage points**.
- The largest IQ2 improvement versus IQ1 is **Newing: +2.13 percentage points**, recovering the severe IQ1 regression.
- IQ2 remains physically aggressive on some sites, especially Á Mỹ with a **48.09% physical clip rate**.

## Holdout comparison

| Site | Test month | IQ0 saving | IQ1 saving | IQ2 saving | IQ2 - IQ1 | IQ2 - IQ0 | IQ2 peak cut |
|---|---|---:|---:|---:|---:|---:|---:|
| Á Mỹ | 2026-08 | 0.74% | 1.67% | **1.26%** | -0.41 pp | +0.52 pp | 105.92 kW |
| Newing | 2026-07 | 0.58% | -1.67% | **0.46%** | **+2.13 pp** | -0.12 pp | 31.44 kW |
| YoungOne | 2026-08 | 5.37% | 7.81% | **4.57%** | **-3.24 pp** | -0.80 pp | 98.47 kW |
| Songwol | 2026-08 | 1.58% | 1.76% | **1.31%** | -0.45 pp | -0.27 pp | 148.04 kW |
| Minh Danh | 2026-08 | 8.23% | 7.21% | **6.48%** | -0.73 pp | -1.75 pp | 75.76 kW |

## IQ2 selected checkpoints

| Site | Selected seed | Best validation step | Best validation saving | IQ2 test saving | Clip rate | Day coverage |
|---|---:|---:|---:|---:|---:|---:|
| Á Mỹ | 0 | 372,480 | 0.2695% | 1.26% | 48.09% | 96.77% |
| Newing | 2 | 113,280 | 1.1338% | 0.46% | 11.22% | 100.00% |
| YoungOne | 2 | 1,433,376 | 4.2962% | 4.57% | 34.38% | 96.77% |
| Songwol | 1 | 1,148,928 | 1.6170% | 1.31% | 26.51% | 100.00% |
| Minh Danh | 2 | 606,144 | 8.9261% | 6.48% | 27.93% | 87.10% |

## Checkpoint integrity

| Site | Final checkpoint SHA-256 |
|---|---|
| Á Mỹ | `0892d62e945052293c6e547addb9542af06e38db787d37226d309b3b2009275e` |
| Newing | `6a0f766cbf15ef8d3cb327f659c414ad3a702a1774dbcc27f3de238e8fb3634b` |
| YoungOne | `739fb6b646ad58e52c1051829d3282ba035d4a8a09e63c32ca22a5bd29415387` |
| Songwol | `9e0583b552b9a537290b4a3bfe99e2829b83caa8d9ae192b4a8bc08367e43f58` |
| Minh Danh | `16fca81319ae4a8105086f42d0f1ecd3ba1daa58d5e2d734b9588021eec72faf` |

## Per-site notes

### Á Mỹ
IQ2 lands between IQ0 and IQ1 on holdout saving. Peak cut remains strong, but the 48.09% clip rate is a major warning that the requested policy actions frequently hit physical constraints.

### Newing
IQ2 fixes the catastrophic IQ1 behavior: saving returns from -1.67% to +0.46%, and peak cut returns from -276.52 kW to +31.44 kW. It still trails IQ0 slightly on bill saving.

### YoungOne
IQ2 is materially worse than IQ1 on the holdout: 4.57% versus 7.81%. IQ1 remains the stronger current-hardware result.

### Songwol
IQ2 is stable and positive but trails both IQ0 and IQ1 on holdout saving.

### Minh Danh
IQ2 remains strongly positive, but trails IQ0 and IQ1. Test coverage is only 87.10%, so comparisons should keep the reduced coverage in view.

## Status of non-batch sites

- **Nam Dược:** not trained in IQ2 `trainall`; blocked by the existing dataset/split-quality guard.
- **Tande:** not trained in IQ2 `trainall`; the launcher treats it as direct-only.

## Interpretation

The coherent recurrent-BC fix is real and successfully trains, but the current IQ2 experiment does not show a broad cross-site economic gain. Its clearest benefit is Newing, where it removes IQ1's severe negative generalization. On the other four trained sites, IQ1 or IQ0 remains stronger on holdout bill saving.

The physical clip rates are also high enough on several sites that future work should distinguish policy-learning quality from feasibility projection effects.

## TODO

- TODO(IQ3): investigate why coherent recurrent BC improves Newing but regresses YoungOne, Songwol, and Minh Danh.
- TODO(IQ3): reduce physical clipping, especially Á Mỹ and YoungOne, without sacrificing peak shaving.
- TODO(IQ3): compare selected checkpoints using identical validation/test coverage rules and add an aggregate multi-site selection criterion.
- TODO(TRAIN-RUNTIME): keep site-level parallel training and benchmark the worker cap / GPU-aware execution separately from PPO algorithm changes.
