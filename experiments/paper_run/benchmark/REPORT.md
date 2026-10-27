# CEC 2020 reproduction: measured results

Configuration: `{"decoder": "algorithm_1", "edge_order": "input", "files": ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_10x10x1000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_10x20x2713.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_10x5x425.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_15x15x3375.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_15x30x12111.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_15x7x1504.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_20x10x2492.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_20x20x8000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_20x40x26104.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_25x12x4817.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_25x25x15625.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_25x50x57147.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_30x15x10025.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_30x30x27000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_30x60x89772.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_35x17x13934.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_35x35x42875.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_35x70x123585.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_40x20x18485.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_40x40x64000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_40x80x130681.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_45x22x43769.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_45x45x91125.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_45x90x322081.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x100x1000000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x200x2296097.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x50x461319.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x100x285357.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x25x38961.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x50x125000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_60x120x434337.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_60x30x99470.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_60x60x216000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_70x140x923343.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_70x35x120810.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_70x70x343000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_80x160x1490468.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_80x40x175762.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_80x80x512000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_90x180x1644367.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_90x45x260195.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_90x90x729000.idpc"], "generations": 500, "grouping": "adjacent", "groups": [["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_10x20x2713.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_15x30x12111.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_20x40x26104.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_25x50x57147.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_30x60x89772.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_35x70x123585.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_40x80x130681.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_45x90x322081.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_10x10x1000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_15x15x3375.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_20x20x8000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_25x25x15625.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_30x30x27000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_35x35x42875.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_40x40x64000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_45x45x91125.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_10x5x425.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_15x7x1504.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_20x10x2492.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_25x12x4817.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_30x15x10025.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_35x17x13934.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_40x20x18485.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set1\\idpc_45x22x43769.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x100x285357.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_60x120x434337.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_70x140x923343.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_80x160x1490468.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_90x180x1644367.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x200x2296097.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x50x125000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_60x60x216000.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_70x70x343000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_80x80x512000.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_90x90x729000.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x100x1000000.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_50x25x38961.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_60x30x99470.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_70x35x120810.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_80x40x175762.idpc"], ["D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_90x45x260195.idpc", "D:\\GNN_MFEA\\data\\benchmark\\set2\\idpc_100x50x461319.idpc"]], "methods": ["ea", "mfea"], "mutation_rate": 0.05, "population": 100, "rmp": 0.5, "runs": 30, "seed": 100}`.

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
| EA | idpc_100x200x2296097.idpc | 30 | 101 | 106 | 101 | 105.9 | 30 | True | False |
| EA | idpc_100x50x461319.idpc | 30 | 115 | 117.8 | 115 | 121.2 | 30 | True | False |
| EA | idpc_10x10x1000.idpc | 30 | 13 | 13 | 13 | 13 | 30 | True | True |
| EA | idpc_10x20x2713.idpc | 30 | 12 | 12 | 12 | 12 | 30 | True | True |
| EA | idpc_10x5x425.idpc | 30 | 12 | 12.2 | 12 | 12 | 30 | True | False |
| EA | idpc_15x15x3375.idpc | 30 | 16 | 16 | 16 | 16 | 30 | True | True |
| EA | idpc_15x30x12111.idpc | 30 | 16 | 16 | 16 | 16 | 30 | True | True |
| EA | idpc_15x7x1504.idpc | 30 | 18 | 18 | 18 | 18 | 30 | True | True |
| EA | idpc_20x10x2492.idpc | 30 | 22 | 24.2 | 22 | 23.5 | 30 | True | False |
| EA | idpc_20x20x8000.idpc | 30 | 21 | 21 | 21 | 21 | 30 | True | True |
| EA | idpc_20x40x26104.idpc | 30 | 21 | 21.27 | 21 | 21 | 30 | True | False |
| EA | idpc_25x12x4817.idpc | 30 | 27 | 27.97 | 27 | 27.9 | 30 | True | False |
| EA | idpc_25x25x15625.idpc | 30 | 26 | 26 | 26 | 26 | 30 | True | True |
| EA | idpc_25x50x57147.idpc | 30 | 26 | 26 | 26 | 26 | 30 | True | True |
| EA | idpc_30x15x10025.idpc | 30 | 33 | 33.17 | 33 | 33.1 | 30 | True | False |
| EA | idpc_30x30x27000.idpc | 30 | 33 | 34.47 | 33 | 34 | 30 | True | False |
| EA | idpc_30x60x89772.idpc | 30 | 32 | 32.53 | 32 | 32 | 30 | True | False |
| EA | idpc_35x17x13934.idpc | 30 | 38 | 45.27 | 38 | 44.5 | 30 | True | False |
| EA | idpc_35x35x42875.idpc | 30 | 38 | 39.1 | 39 | 39 | 30 | False | False |
| EA | idpc_35x70x123585.idpc | 30 | 36 | 36.7 | 36 | 36 | 30 | True | False |
| EA | idpc_40x20x18485.idpc | 30 | 42 | 45.2 | 42 | 45.7 | 30 | True | False |
| EA | idpc_40x40x64000.idpc | 30 | 42 | 42.27 | 42 | 42.1 | 30 | True | False |
| EA | idpc_40x80x130681.idpc | 30 | 42 | 42.67 | 42 | 42 | 30 | True | False |
| EA | idpc_45x22x43769.idpc | 30 | 49 | 49.33 | 48 | 49 | 30 | False | False |
| EA | idpc_45x45x91125.idpc | 30 | 46 | 46.37 | 46 | 46 | 30 | True | False |
| EA | idpc_45x90x322081.idpc | 30 | 46 | 47.57 | 46 | 46 | 30 | True | False |
| EA | idpc_50x100x285357.idpc | 30 | 52 | 55.67 | 52 | 52.6 | 30 | True | False |
| EA | idpc_50x25x38961.idpc | 30 | 61 | 61 | 53 | 60.8 | 30 | False | False |
| EA | idpc_50x50x125000.idpc | 30 | 53 | 53.73 | 53 | 53 | 30 | True | False |
| EA | idpc_60x120x434337.idpc | 30 | 61 | 63.13 | 61 | 61.2 | 30 | True | False |
| EA | idpc_60x30x99470.idpc | 30 | 64 | 69.17 | 64 | 68.8 | 30 | True | False |
| EA | idpc_60x60x216000.idpc | 30 | 63 | 63.57 | 63 | 63.3 | 30 | True | False |
| EA | idpc_70x140x923343.idpc | 30 | 73 | 77 | 73 | 73.7 | 30 | True | False |
| EA | idpc_70x35x120810.idpc | 30 | 72 | 74.87 | 72 | 72 | 30 | True | False |
| EA | idpc_70x70x343000.idpc | 30 | 71 | 74.13 | 71 | 72.2 | 30 | True | False |
| EA | idpc_80x160x1490468.idpc | 30 | 81 | 84.17 | 81 | 81.4 | 30 | True | False |
| EA | idpc_80x40x175762.idpc | 30 | 82 | 85.03 | 82 | 85.8 | 30 | True | False |
| EA | idpc_80x80x512000.idpc | 30 | 85 | 89.37 | 85 | 88.4 | 30 | True | False |
| EA | idpc_90x180x1644367.idpc | 30 | 91 | 94.37 | 91 | 92.1 | 30 | True | False |
| EA | idpc_90x45x260195.idpc | 30 | 93 | 102.8 | 95 | 102 | 30 | False | False |
| EA | idpc_90x90x729000.idpc | 30 | 91 | 95.1 | 91 | 91.7 | 30 | True | False |
| MFEA | idpc_100x100x1000000.idpc | 30 | 103 | 107.2 | 102 | 102 | 30 | False | False |
| MFEA | idpc_100x200x2296097.idpc | 30 | 101 | 103 | 101 | 101 | 30 | True | False |
| MFEA | idpc_100x50x461319.idpc | 30 | 104 | 114.7 | 102 | 102.1 | 30 | False | False |
| MFEA | idpc_10x10x1000.idpc | 30 | 13 | 13 | 7 | 12.2 | 30 | False | False |
| MFEA | idpc_10x20x2713.idpc | 30 | 12 | 12 | 12 | 12 | 30 | True | True |
| MFEA | idpc_10x5x425.idpc | 30 | 12 | 12 | 7 | 7 | 30 | False | False |
| MFEA | idpc_15x15x3375.idpc | 30 | 16 | 16 | 10 | 10 | 30 | False | False |
| MFEA | idpc_15x30x12111.idpc | 30 | 16 | 16 | 10 | 10 | 30 | False | False |
| MFEA | idpc_15x7x1504.idpc | 30 | 18 | 18 | 18 | 18 | 30 | True | True |
| MFEA | idpc_20x10x2492.idpc | 30 | 22 | 22.5 | 22 | 22 | 30 | True | False |
| MFEA | idpc_20x20x8000.idpc | 30 | 21 | 21 | 15 | 15.4 | 30 | False | False |
| MFEA | idpc_20x40x26104.idpc | 30 | 21 | 21 | 15 | 15.2 | 30 | False | False |
| MFEA | idpc_25x12x4817.idpc | 30 | 27 | 27.97 | 27 | 27 | 30 | True | False |
| MFEA | idpc_25x25x15625.idpc | 30 | 26 | 26 | 18 | 24.1 | 30 | False | False |
| MFEA | idpc_25x50x57147.idpc | 30 | 26 | 26 | 18 | 24.7 | 30 | False | False |
| MFEA | idpc_30x15x10025.idpc | 30 | 33 | 33.13 | 33 | 33 | 30 | True | False |
| MFEA | idpc_30x30x27000.idpc | 30 | 32 | 33.87 | 32 | 32 | 30 | True | False |
| MFEA | idpc_30x60x89772.idpc | 30 | 32 | 32 | 32 | 32 | 30 | True | True |
| MFEA | idpc_35x17x13934.idpc | 30 | 38 | 45.2 | 38 | 38 | 30 | True | False |
| MFEA | idpc_35x35x42875.idpc | 30 | 39 | 39 | 37 | 37 | 30 | False | False |
| MFEA | idpc_35x70x123585.idpc | 30 | 36 | 36.17 | 28 | 35.5 | 30 | False | False |
| MFEA | idpc_40x20x18485.idpc | 30 | 42 | 45.8 | 42 | 42 | 30 | True | False |
| MFEA | idpc_40x40x64000.idpc | 30 | 42 | 42 | 42 | 42 | 30 | True | True |
| MFEA | idpc_40x80x130681.idpc | 30 | 42 | 42 | 42 | 42 | 30 | True | True |
| MFEA | idpc_45x22x43769.idpc | 30 | 49 | 49 | 47 | 47 | 30 | False | False |
| MFEA | idpc_45x45x91125.idpc | 30 | 46 | 46.03 | 46 | 46 | 30 | True | True |
| MFEA | idpc_45x90x322081.idpc | 30 | 46 | 46 | 46 | 46 | 30 | True | True |
| MFEA | idpc_50x100x285357.idpc | 30 | 52 | 54.97 | 52 | 52 | 30 | True | False |
| MFEA | idpc_50x25x38961.idpc | 30 | 58 | 61.37 | 53 | 53 | 30 | False | False |
| MFEA | idpc_50x50x125000.idpc | 30 | 53 | 53.23 | 52 | 52 | 30 | False | False |
| MFEA | idpc_60x120x434337.idpc | 30 | 61 | 61.23 | 61 | 61 | 30 | True | False |
| MFEA | idpc_60x30x99470.idpc | 30 | 69 | 69 | 62 | 62 | 30 | False | False |
| MFEA | idpc_60x60x216000.idpc | 30 | 63 | 63.1 | 62 | 62 | 30 | False | False |
| MFEA | idpc_70x140x923343.idpc | 30 | 73 | 74.17 | 72 | 72 | 30 | False | False |
| MFEA | idpc_70x35x120810.idpc | 30 | 72 | 72.2 | 72 | 72 | 30 | True | False |
| MFEA | idpc_70x70x343000.idpc | 30 | 71 | 74.4 | 71 | 71 | 30 | True | False |
| MFEA | idpc_80x160x1490468.idpc | 30 | 81 | 84.33 | 81 | 81 | 30 | True | False |
| MFEA | idpc_80x40x175762.idpc | 30 | 82 | 87.5 | 82 | 82 | 30 | True | False |
| MFEA | idpc_80x80x512000.idpc | 30 | 83 | 90.7 | 82 | 82 | 30 | False | False |
| MFEA | idpc_90x180x1644367.idpc | 30 | 91 | 92.23 | 91 | 91 | 30 | True | False |
| MFEA | idpc_90x45x260195.idpc | 30 | 99 | 102 | 93 | 93.2 | 30 | False | False |
| MFEA | idpc_90x90x729000.idpc | 30 | 91 | 96.07 | 91 | 91 | 30 | True | False |

Both BF and printed-precision AVG match on 15/84 algorithm/instance rows.
A row matching two statistics is not proof of reproducing the author's implementation or the complete paper.
* AVG match is compared after rounding to the one decimal place printed in the tables. Seeds, grouping, edge order and full MFEA loop details are absent from the article; Java and Python random streams differ.

All finite paths were independently checked for connectivity, repeated vertices and domain re-entry.
`manifest.json` records dataset/code SHA256, parameters and seed range. `paths.json` records original edge IDs, nodes and domain sequences (zero-based node/edge IDs).
`runs.csv` and `summary.csv` contain measured costs and gaps; `comparison.csv` contains NIB/NIE/RPD; `convergence.csv` includes generation 0.
Infeasible costs are null in JSON and inf in aggregate CSV. STD uses the sample (n-1) convention.
MFEA seconds are the shared group runtime and are not additive across tasks. Python timings are not comparable directly with the paper's Java/i7 timings.
Wall time for this invocation: 2367.5 seconds.
