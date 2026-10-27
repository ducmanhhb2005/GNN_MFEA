# CEC 2020 reproduction: measured results

Configuration: `{"decoder": "algorithm_1", "edge_order": "input", "files": ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x100x1000000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x100x285357.idpc"], "generations": 500, "grouping": "explicit", "groups": [["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x100x1000000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x100x285357.idpc"]], "methods": ["ea", "mfea"], "mutation_rate": 0.05, "population": 100, "rmp": 0.5, "runs": 30, "seed": 100}`.

Pairing `adjacent` is INFERRED from repeated Rs values in Tables III/IV, not specified by the authors.
GPA follows Algorithm 1 line 13: arriving-node sigma and literal modulo. The conflicting prose interpretation is not exposed as a variant.
All original edges are used; there is no graph pruning or extra preprocessing.
A no-outgoing-edge sigma sentinel is stored as numeric 1 so the arriving-node modulo remains defined; the article leaves this case unresolved.
Edge order is an implementation assumption; `input` preserves the dataset order. PMX and TPX draw separate cuts.
Initial chromosomes are evaluated on every task; offspring only on inherited skill. Population size is the TOTAL per solver group.
Uniform parent sampling, randomized rank ties and global elitist survival are framework assumptions; the full loop is not published.
For EA the same encoding/operators/selection are used with one task. No GNN, exact oracle, path repair or benchmark answers enter the search.

| Algorithm | Instance | Runs | BF | AVG | Paper BF | Paper AVG | Valid | BF match | AVG match* |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| EA | idpc_100x100x1000000.idpc | 30 | 103 | 109.1 | 103 | 104.2 | 30 | True | False |
| EA | idpc_50x100x285357.idpc | 30 | 52 | 55.67 | 52 | 52.6 | 30 | True | False |
| MFEA | idpc_100x100x1000000.idpc | 30 | 103 | 103.4 | 102 | 102 | 30 | False | False |
| MFEA | idpc_50x100x285357.idpc | 30 | 52 | 54.77 | 52 | 52 | 30 | True | False |

Both BF and printed-precision AVG match on 0/4 algorithm/instance rows.
A row matching two statistics is not proof of reproducing the author's implementation or the complete paper.
* AVG match is compared after rounding to the one decimal place printed in the tables. Seeds, grouping, edge order and full MFEA loop details are absent from the article; Java and Python random streams differ.

All finite paths were independently checked for connectivity, repeated vertices and domain re-entry.
`manifest.json` records dataset/code SHA256, parameters and seed range. `paths.json` records original edge IDs, nodes and domain sequences (zero-based node/edge IDs).
`runs.csv` and `summary.csv` contain measured costs and gaps; `comparison.csv` contains NIB/NIE/RPD; `convergence.csv` includes generation 0.
Infeasible costs are null in JSON and inf in aggregate CSV. STD uses the sample (n-1) convention.
MFEA seconds are the shared group runtime and are not additive across tasks. Python timings are not comparable directly with the paper's Java/i7 timings.
Wall time for this invocation: 426.9 seconds.
