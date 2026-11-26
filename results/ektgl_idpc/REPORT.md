# EKT-GL-IDPC-DU benchmark

Paths and traces checked: 3360. Feasible paths: 3360.

| Method | Seeds | Feasible task runs | Total task runs | Mean solver seconds per seed |
|---|---:|---:|---:|---:|
| ektgl | 20 | 840 | 840 | 49.52 |
| individual | 20 | 840 | 840 | 54.13 |
| mfea | 20 | 840 | 840 | 268.36 |
| sto | 20 | 840 | 840 | 27.30 |

| Control | EKT-GL wins / ties / losses at final state | At matched evaluation cap | Unavailable at cap |
|---|---:|---:|---:|
| individual | 2 / 824 / 14 | 2 / 824 / 14 | 0 |
| mfea | 128 / 689 / 23 | 250 / 572 / 18 | 0 |
| sto | 11 / 811 / 18 | 11 / 811 / 18 | 0 |

Pairs in which both methods are infeasible are excluded from wins, ties and losses.

Each matched pair uses the last generation snapshot within the smaller final GPA budget.
Evaluation requests and local search checks include memoized repeats. Solver timings exclude graph loading and output verification.
A missing AVG or STD indicates an infeasible seed or insufficient repetitions for sample STD.
See the manifest for the two algorithm configurations and MFEA groups. Seed IDs and results are in the run files.

[Per-run paths](runs.csv) ? [Task statistics](summary.csv) ? [Convergence](convergence.csv) ? [Evaluation caps](matched_budgets.csv) ? [Runtime](timings.csv)
