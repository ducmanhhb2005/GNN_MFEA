# TSPLIB coordinates used by the TSP experiment

The 18 `.tsp` files are EUC_2D TSPLIB instances, downloaded from the
[TSPLIB mirror](https://github.com/mastqe/tsplib) on 2026-11-24.
Original benchmark: [TSPLIB, Heidelberg University](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

Instance names and task order match Table I of the supplied EKT-GL paper.
`run_tsp.py` records a SHA-256 and download URL for each input in every seed
artifact. No random substitutes are used for the paper instances.

TSP costs use one global min and max over the entire coordinate matrix, and continuous
Euclidean distance, rather than the integer-rounded EUC_2D TSPLIB metric.
Do not compare these costs directly to TSPLIB's published integer optima.
This scalar normalization preserves shape. Dividing TSPLIB's rounded optima by
the global coordinate range reproduces all 18 rounded LKH reference values in
Table IX; this is an inference of the paper's normalization, not an explicit
formula quoted from the paper. CVRP normalization is unchanged.
