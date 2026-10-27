"""Run reconstructed CEC 2020 MFEA/EA on unmodified IDPC-DU data.

Default: all 42 graphs, 30 seeds 100..129, population 100, 500 generations.
GPA follows Algorithm 1. Task pairing and MFEA framework details are assumptions;
this command does not guarantee reproduction of the published table values.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from idpc_du import Graph, validate_path
from mfea.solver import solve_mfea, solve_ea
from experiments.metrics import comparison_metrics

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments" / "paper_reference.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def load_reference():
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def build_groups(files, grouping):
    """Pair adjacent sizes separately for each set and domain/node ratio."""
    files = [Path(f) for f in files]
    if grouping == "explicit":
        return [files]
    if grouping != "adjacent":
        raise ValueError("unknown grouping")
    buckets = {}
    for f in files:
        n, d, _ = map(int, f.stem.removeprefix("idpc_").split("x"))
        # floor(N/2) includes 15x7, 25x12, 35x17 and 45x22.
        ratio = "half" if d == n // 2 else "equal" if d == n else "double" if d == 2*n else None
        if ratio is None:
            raise ValueError(f"No paper ratio for {f.name}; use --grouping explicit")
        buckets.setdefault((f.parent.name, ratio), []).append((n, f))
    groups = []
    for key, bucket in sorted(buckets.items()):
        ordered = [f for _, f in sorted(bucket)]
        if len(ordered) % 2:
            raise ValueError(f"Unpaired task in {key}; supply both sizes or use --grouping explicit")
        groups.extend(ordered[i:i+2] for i in range(0, len(ordered), 2))
    return groups


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def run_group(job):
    """Load a graph group once, checkpoint every seed, and validate every path."""
    algorithm, files, config, output, fingerprint = job
    group_id = hashlib.sha256((algorithm + "|" + "|".join(files)).encode()).hexdigest()[:16]
    results_dir = Path(output) / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs = None
    written = []
    for trial in range(config["runs"]):
        seed = config["seed"] + trial
        destination = results_dir / f"{algorithm}_{group_id}_{seed}.json"
        written.append(str(destination))
        if destination.exists():
            old = json.loads(destination.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fingerprint:
                raise ValueError(f"Checkpoint does not match configuration: {destination}")
            continue
        if graphs is None:
            graphs = [Graph.from_file(f) for f in files]
        started = time.perf_counter()
        diagnostics = {}
        parameters = dict(seed=seed, population=config["population"],
                          generations=config["generations"], mutation_rate=config["mutation_rate"],
                          diagnostics=diagnostics)
        if algorithm == "EA":
            if len(graphs) != 1:
                raise ValueError("EA must optimize exactly one independent task")
            best, history = solve_ea(graphs[0], **parameters)
        elif algorithm == "MFEA":
            best, history = solve_mfea(graphs, rmp=config["rmp"], **parameters)
        else:
            raise ValueError(f"Unknown paper algorithm: {algorithm}")
        elapsed = time.perf_counter() - started
        rows = []
        for task, (graph, filename, (cost, path)) in enumerate(zip(graphs, files, best)):
            valid = math.isfinite(cost) and math.isclose(validate_path(graph, path), cost)
            if math.isfinite(cost) and not valid:
                raise AssertionError(f"Invalid returned path: {filename}, seed {seed}")
            rows.append(dict(
                algorithm=algorithm, instance=Path(filename).name, subset=Path(filename).parent.name,
                task_group=";".join(Path(f).name for f in files), seed=seed,
                cost=cost, valid=valid, seconds=elapsed,
                population=config["population"], generations=config["generations"],
                rmp=config["rmp"], mutation_rate=config["mutation_rate"],
                decoder="algorithm_1", edge_order="input", edges=len(graph.edges),
                # Includes K initial evaluations per chromosome and N children/generation.
                group_evaluations=sum(diagnostics["evaluations"]),
                evaluations=diagnostics["evaluations"][task],
                infeasible_evaluations=diagnostics["infeasible_evaluations"][task],
                objective_min=diagnostics["objective_min"][task],
                objective_max=diagnostics["objective_max"][task],
                edge_ids=path, nodes=[graph.source]+[graph.by_id[i].v for i in path],
                domains=[graph.by_id[i].domain for i in path],
                convergence=[costs[task] for costs in history]))
        content = json_safe(dict(fingerprint=fingerprint, rows=rows))
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(content, separators=(",", ":"))+"\n", encoding="utf-8")
        temporary.replace(destination)
        if (trial+1) % 5 == 0 or trial+1 == config["runs"]:
            print(f"{algorithm} {';'.join(Path(f).stem for f in files)}: "
                  f"{trial+1}/{config['runs']} seeds; costs={[v[0] for v in best]}", flush=True)
    return written


def summarize(rows, reference):
    buckets = {}
    for row in rows:
        buckets.setdefault((row["algorithm"], row["instance"]), []).append(row)
    summaries = []
    for (algorithm, instance), bucket in sorted(buckets.items()):
        costs = [r["cost"] if r["cost"] is not None else math.inf for r in bucket]
        finite = all(math.isfinite(c) for c in costs)
        avg = statistics.mean(costs) if finite else math.inf
        sd = statistics.stdev(costs) if finite and len(costs)>1 else 0.0 if finite else math.inf
        paper = reference["instances"].get(instance, {}).get(algorithm, {})
        pb, pa = paper.get("BF"), paper.get("AVG")
        summaries.append(dict(
            algorithm=algorithm, instance=instance, subset=bucket[0]["subset"],
            runs=len(costs), BF=min(costs), AVG=avg, STD_sample=sd,
            feasible=sum(bool(r["valid"]) for r in bucket),
            paper_BF=pb, paper_AVG=pa, delta_BF=min(costs)-pb if pb is not None else "",
            delta_AVG=avg-pa if pa is not None else "",
            matches_BF=math.isclose(min(costs), pb, abs_tol=1e-9) if pb is not None else "",
            # Tables show one decimal place; comparison is to the printed precision.
            matches_AVG=math.isfinite(avg) and round(avg,1)==pa if pa is not None else "",
            seconds_mean=statistics.mean(r["seconds"] for r in bucket)))
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output, paths, manifest, wall):
    reference = load_reference()
    rows = []
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if result["fingerprint"] != manifest["fingerprint"]:
            raise ValueError(f"Mismatched result: {path}")
        rows.extend(result["rows"])
    rows.sort(key=lambda r:(r["algorithm"],r["instance"],r["seed"]))
    compact = [{k:v for k,v in r.items() if k not in ("convergence","edge_ids","nodes","domains")} for r in rows]
    write_csv(output/"runs.csv", compact)
    # Paths use original zero-based edge IDs; node labels are also zero-based.
    (output/"paths.json").write_text(json.dumps([{k:v for k,v in r.items() if k != "convergence"} for r in rows], indent=2)+"\n",encoding="utf-8")
    summaries = summarize(rows, reference)
    write_csv(output/"summary.csv", summaries)
    convergence = []
    for algorithm, instance in sorted({(r["algorithm"],r["instance"]) for r in rows}):
        bucket=[r for r in rows if (r["algorithm"],r["instance"])==(algorithm,instance)]
        for generation in range(manifest["config"]["generations"]+1):
            values=[r["convergence"][generation] for r in bucket]
            mean=statistics.mean(values) if all(v is not None for v in values) else math.inf
            convergence.append(dict(algorithm=algorithm,instance=instance,generation=generation,AVG=mean))
    write_csv(output/"convergence.csv",convergence)
    compare = comparison_metrics(summaries, reference)
    write_csv(output/"comparison.csv",compare)
    def fmt(v):
        return f"{v:.4g}" if isinstance(v,float) else str(v)
    config=manifest["config"]
    lines=["# CEC 2020 reproduction: measured results", "",
        f"Configuration: `{json.dumps(config,sort_keys=True)}`.", "",
        "Pairing `adjacent` is INFERRED from repeated Rs values in Tables III/IV, not specified by the authors.",
        "GPA follows Algorithm 1 line 13: arriving-node sigma and literal modulo. The conflicting prose interpretation is not exposed as a variant.",
        "All original edges are used; there is no graph pruning or extra preprocessing.",
        "A no-outgoing-edge sigma sentinel is stored as numeric 1 so the arriving-node modulo remains defined; the article leaves this case unresolved.",
        "Edge order is an implementation assumption; `input` preserves the dataset order. PMX and TPX draw separate cuts.",
        "Initial chromosomes are evaluated on every task; offspring only on inherited skill. Population size is the TOTAL per solver group.",
        "Uniform parent sampling, randomized rank ties and global elitist survival are framework assumptions; the full loop is not published.",
        "For EA the same encoding/operators/selection are used with one task. No GNN, exact oracle, path repair or benchmark answers enter the search.", "",
        "| Algorithm | Instance | Runs | BF | AVG | Paper BF | Paper AVG | Valid | BF match | AVG match* |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in summaries:
        values=[r[k] for k in ("algorithm","instance","runs","BF","AVG","paper_BF","paper_AVG","feasible","matches_BF","matches_AVG")]
        lines.append("| "+" | ".join(map(fmt,values))+" |")
    matches=sum(r["matches_BF"] is True and r["matches_AVG"] is True for r in summaries)
    lines += ["",f"Both BF and printed-precision AVG match on {matches}/{len(summaries)} algorithm/instance rows.",
        "A row matching two statistics is not proof of reproducing the author's implementation or the complete paper.",
        "* AVG match is compared after rounding to the one decimal place printed in the tables. Seeds, grouping, edge order and full MFEA loop details are absent from the article; Java and Python random streams differ.",
        "", "All finite paths were independently checked for connectivity, repeated vertices and domain re-entry.",
        "`manifest.json` records dataset/code SHA256, parameters and seed range. `paths.json` records original edge IDs, nodes and domain sequences (zero-based node/edge IDs).",
        "`runs.csv` and `summary.csv` contain measured costs and gaps; `comparison.csv` contains NIB/NIE/RPD; `convergence.csv` includes generation 0.",
        "Infeasible costs are null in JSON and inf in aggregate CSV. STD uses the sample (n-1) convention.",
        "MFEA seconds are the shared group runtime and are not additive across tasks. Python timings are not comparable directly with the paper's Java/i7 timings.",
        f"Wall time for this invocation: {wall:.1f} seconds.", ""]
    (output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Both statistics match: {matches}/{len(summaries)} rows. Report: {output/'REPORT.md'}",flush=True)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set",choices=["set1","set2","all"],default="all")
    parser.add_argument("--instances",nargs="+",help="Explicit joint MFEA task group; overrides --set")
    parser.add_argument("--grouping",choices=["adjacent","explicit"],default=None)
    parser.add_argument("--methods",nargs="+",choices=["ea","mfea"],default=["ea","mfea"])
    parser.add_argument("--runs",type=int,default=30)
    parser.add_argument("--seed",type=int,default=100)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--population",type=int,default=100)
    parser.add_argument("--generations",type=int,default=500)
    parser.add_argument("--rmp",type=float,default=0.5)
    parser.add_argument("--mutation-rate",type=float,default=0.05)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"experiments"/"paper_run"/"benchmark")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    if args.runs<1 or args.workers<1 or args.population<4 or args.generations<1 or not 0<=args.rmp<=1 or not 0<=args.mutation_rate<=1:
        parser.error("invalid run/worker/population/generation/rate parameters")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    reference=load_reference()
    source_pdf=ROOT/reference["source_pdf"]
    if not source_pdf.is_file() or sha256(source_pdf)!=reference["source_sha256"]:
        parser.error("Supplied paper PDF does not match the reference checksum")
    if args.instances:
        files=[Path(f).resolve() for f in args.instances]
    else:
        files=[ROOT/"data"/"benchmark"/r["set"]/name for name,r in reference["instances"].items() if args.set=="all" or r["set"]==args.set]
    if len(files)!=len(set(files)):
        parser.error("duplicate instance paths")
    missing=[str(f) for f in files if not f.is_file()]
    if missing:
        parser.error("Missing dataset files: "+", ".join(missing)+". Unpack the original ZIP with python tools/unpack_idpc.py <archive> --set all")
    grouping=args.grouping or ("explicit" if args.instances else "adjacent")
    if grouping=="explicit" and not args.instances:
        parser.error("--grouping explicit requires --instances")
    try:
        groups=build_groups(files,grouping) if "mfea" in args.methods else []
    except ValueError as exc:
        parser.error(str(exc))
    config={k:getattr(args,k) for k in ("runs","seed","population","generations","rmp","mutation_rate")}
    config.update(decoder="algorithm_1",edge_order="input",grouping=grouping,methods=sorted(args.methods),
        groups=[[str(f) for f in group] for group in groups],files=sorted(map(str,files)))
    code_files=sorted(list((ROOT/"idpc_du").glob("*.py"))+list((ROOT/"mfea").glob("*.py"))
                      +list((ROOT/"experiments").glob("*.py"))+list((ROOT/"experiments"/"_support").glob("*.py")))+[REFERENCE]
    identity=dict(config=config,dataset_sha256={str(f):sha256(f) for f in sorted(files)},
        code_sha256={str(f.relative_to(ROOT)):sha256(f) for f in code_files},python=sys.version,platform=platform.platform())
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=dict(identity,fingerprint=fingerprint,reference_doi=reference["doi"],
        protocol_note="Adjacent-size pairing inferred from equal Rs values; full original grouping, RNG seeds and edge ordering not published.")
    output=args.out_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        if not args.resume:
            parser.error("Output exists. Use --resume with identical settings or a new --out-dir")
        if json.loads(manifest_path.read_text(encoding="utf-8")).get("fingerprint")!=fingerprint:
            parser.error("Cannot resume: code, data, Python environment or experiment configuration changed")
    elif any(output.iterdir()):
        parser.error("Output directory contains files without a manifest; use a new --out-dir")
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    plans=[]
    if "ea" in args.methods:
        plans.extend(("EA",[str(f)]) for f in sorted(files))
    if "mfea" in args.methods:
        plans.extend(("MFEA",list(map(str,group))) for group in groups)
    plans.sort(key=lambda plan: sum(Path(f).stat().st_size for f in plan[1]), reverse=True)
    jobs=[(algorithm,group,config,str(output),fingerprint) for algorithm,group in plans]
    print(f"{len(jobs)} solver groups x {args.runs} seeds; GPA=Algorithm1, grouping={grouping}, workers={args.workers}",flush=True)
    started=time.perf_counter()
    checkpoints=[]
    if args.workers==1:
        for job in jobs:
            checkpoints.extend(run_group(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_group,job) for job in jobs]
            for future in as_completed(futures):
                checkpoints.extend(future.result())
    write_reports(output,checkpoints,manifest,time.perf_counter()-started)


if __name__ == "__main__":
    main()
