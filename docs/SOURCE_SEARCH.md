# Search for the authors' public source code - 2026-10-22

**Related public implementations were found. No repository was verified as the official source for the supplied CEC 2020 paper.** Not finding an official link is not evidence that the authors never released code.

## Search and provenance

Checked the exact title and DOI (10.1109/CEC48606.2020.9185701), IDPC-DU/IDPCDU repository names and README contents, the original Mendeley dataset, and author pages/accounts. GitHub's REST API returned relevant repositories that were absent from ordinary web-search results.

[Nguyen Binh Long's publication page](https://longnb.dev/publications/) lists the article and links to his [GitHub account](https://github.com/binhlongmm99), but the article entry only exposes a Paper link. Its currently public repositories did not identify a repository for this article. The [Mendeley v3 dataset](https://data.mendeley.com/datasets/t726xwcjf9/3) supplies instances; the available ZIP contains set1.rar, set2.rar and test_file_structure.txt, with no Java source.

## Related Java implementations

- [longnh176040/MFEA-IDPC-DU](https://github.com/longnh176040/MFEA-IDPC-DU), created September 16, 2020. Download pinned to commit `f7677dcb7a7d40974677633b23cb7ee84a9a05ce`.
- [longnh176040/IDPC-DU-EDGES-UPDATE](https://github.com/longnh176040/IDPC-DU-EDGES-UPDATE), created September 17, 2020. Download pinned to commit `50b05b0fcd65439a7673d1ee12f4bbcdb08c7c1c`.

Both implement Java classes for the same two-segment encoding, USS, evolutionary operators and IDPC-DU costs. There is no README identifying this as the CEC authors' official implementation, and the account metadata does not establish an author's identity.

Concrete issues in the first repository:

1. [Circle.java lines 52-53](https://github.com/longnh176040/MFEA-IDPC-DU/blob/f7677dcb7a7d40974677633b23cb7ee84a9a05ce/IDPC_DU_EDGES/Circle.java#L52-L53) draws a random edge index into `pos` but then assigns `edgeIdx[i] = nodeOut[i]`. Thus it does not perform the random initialization specified in the paper.
2. [Circle.java lines 114-118](https://github.com/longnh176040/MFEA-IDPC-DU/blob/f7677dcb7a7d40974677633b23cb7ee84a9a05ce/IDPC_DU_EDGES/Circle.java#L114-L118) penalizes any domain previously used, adding every traversed domain immediately. This also penalizes consecutive edges of the same domain; the article permits consecutive edges and only forbids re-entry after leaving a domain.
3. [Main.java](https://github.com/longnh176040/MFEA-IDPC-DU/blob/f7677dcb7a7d40974677633b23cb7ee84a9a05ce/IDPC_DU_EDGES/Main.java) uses 10,000 generations and hard-coded file paths, rather than the paper's 500-generation run setup.
4. The mutation's edge-index bound uses the number of nodes rather than the selected position's maximum parallel-edge count.

The updated repository still contains the initialization and domain-tracking issues. Consequently these repositories cannot be treated as a validated reference merely because their names match the problem. They were downloaded for inspection; their algorithms have not been substituted into the Python baseline, and no successful reproduction is claimed from them.

## Separate Python implementation

[wind123mountain/EvolutionComputation](https://github.com/wind123mountain/EvolutionComputation), created November 2021, contains an independent IDPC-DU implementation. Selected reference source files are pinned to commit `8468c57a750f6d3d3e1b58485ace8cd096eee832`.

Its `IDPCDU/BuildGraph.py` uses domain-priority/end-domain-node encoding and within-domain Floyd-Warshall distances. `SR_EncodingV1.py` uses an ability-vector/SREMTO scheme. These differ from the paper's node-priority/edge-index GPA encoding and cannot be relabeled as that paper's original MFEA.

A further repository, [kudo313/Multipoint-IDPC-DU](https://github.com/kudo313/Multipoint-IDPC-DU), concerns a multipoint problem and was excluded from the single source-target MFEA baseline.

## Local copies

`docs/external_sources/` holds the two Java repositories and selected files from the separate Python repository. Each has a `SOURCE.json` with repository URL, pinned commit, date and checksums. This directory is reference material, not a runtime dependency. Only selected source files and provenance are retained; experimental result traces are excluded. Current measured results are in `experiments/paper_run/`. External source copies remain reference material only and are never imported by the optimizer.


## Framework and normalization reference

The CEC 2020 article cites Gupta, Ong and Feng, DOI `10.1109/TEVC.2015.2458037`. The [author-hosted preprint](https://memetic-computing.org/publication/journal/Multifactorial_Evolution_ONLINE_VERSION.pdf) describes the MFEA framework in Algorithms 1–3 and the min–max convergence normalization on page 10. This source supplies the cited framework and reporting formula; its continuous/random-key representations, local learning and operators are not substituted for the IDPC-DU paper's representation/operators.
