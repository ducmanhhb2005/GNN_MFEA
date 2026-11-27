"""Read-only provenance, independent validation and reanalysis of frozen IDPC archives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import mannwhitneyu

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.config import Config
from ekt_gl.idpc.v2.core import EvaluationTracker, initialize, limits_for, project, seed_for
from ekt_gl.idpc.v2.evolution import Member, assign_ranks
from ekt_gl.idpc.v2.gpa_fast import PackedGPA, warmup

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESEARCH = RESULTS / "idpc_v3_research"
LEGACY = RESULTS / "ektgl_idpc"
V2 = {p: RESULTS / "idpc_v2" / d for p, d in
      (("native", "native_all42_v2"), ("matched_ls", "matched_ls_all42_v2"))}
LEGACY_COMMIT = "0a4e11d"
SEEDS = tuple(range(100, 120))
PILOT = (4200, 4201, 4202)
CONFIRMATION = tuple(range(4300, 4320))
OUTCOMES = ("win", "tie", "loss")
PAIRS = (
    ("E1", "legacy:ektgl", "legacy:mfea"), ("E2", "legacy:ektgl", "native:mfea"),
    ("E3", "native:ektgl", "legacy:mfea"), ("E4", "native:ektgl", "native:mfea"),
    ("ekt_versions", "legacy:ektgl", "native:ektgl"), ("mfea_versions", "native:mfea", "legacy:mfea"),
    ("sto_ea", "legacy:sto", "native:ea"), ("ekt_sto", "legacy:ektgl", "legacy:sto"),
    ("ekt_individual", "legacy:ektgl", "legacy:individual"),
    ("matched_reference", "matched_ls:mfea", "matched_ls:ektgl"),
)


def sha(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def structured(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows):
    rows = iter(rows)
    first = next(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=list(first))
        output.writeheader()
        output.writerow(first)
        output.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode("utf-8", errors="replace").strip()


def verify_manifests(out):
    legacy = json_read(LEGACY / "manifest.json")
    v2 = {profile: json_read(folder / "manifest.json") for profile, folder in V2.items()}
    data = {r["task"]: r for r in v2["native"]["tasks"]}
    assert len(data) == 42
    assert v2["native"]["tasks"] == v2["matched_ls"]["tasks"]
    assert {r["name"]: r["sha256"] for r in legacy["tasks"]} == {k: r["sha256"] for k,r in data.items()}
    for row in data.values():
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Dataset fingerprint mismatch: {row['task']}")
    for profile, manifest in v2.items():
        for name, expected in manifest["sources"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Frozen source mismatch {profile}:{name}")
    legacy_tree = Path(tempfile.gettempdir()) / "gnn_mfea_legacy_v3_0a4e11d"
    artifacts = []
    for name, expected in legacy["sources"].items():
        name = Path(name).as_posix()
        source_bytes = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{LEGACY_COMMIT}:{name}"])
        observed = hashlib.sha256(source_bytes).hexdigest()
        if observed != expected or (legacy_tree.exists() and sha(legacy_tree / name) != expected):
            raise ValueError(f"Legacy source mismatch {name}")
        artifacts.append(dict(kind="source",path=name, expected_sha256=expected,
                              observed_sha256=observed, status="verified"))
    for record in sorted((LEGACY / "runs").glob("*.json")):
        run = json_read(record)
        assert run["fingerprint"] == legacy["fingerprint"]
        assert set(run["tasks"]) == set(data)
        artifacts.append(dict(kind="checkpoint",path=record.relative_to(ROOT).as_posix(),
                              expected_sha256=None, observed_sha256=sha(record), status="identity_verified"))
    assert sum(r["kind"] == "checkpoint" for r in artifacts) == 80
    table(out / "00_inventory/legacy_artifacts.csv", artifacts)
    info = dict(branch=git("branch", "--show-current"), commit=git("rev-parse", "HEAD"),
                legacy_commit=git("rev-parse", LEGACY_COMMIT),
                aggregate_data_sha256=hashlib.sha256(json.dumps(
                    {k:r["sha256"] for k,r in data.items()}, sort_keys=True).encode()).hexdigest(),
                profiles={p: m["fingerprint"] for p,m in v2.items()},
                legacy_fingerprint=legacy["fingerprint"], pilot_seeds=PILOT,
                confirmation_seeds=CONFIRMATION, tasks=list(data), v2_source_count=len(v2["native"]["sources"]))
    structured(out / "experiment_manifest.json", info)
    env = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
               v2_hardware=v2["native"]["hardware"], v2_dependencies=v2["native"]["dependencies"],
               legacy_environment=legacy["environment"])
    structured(out / "00_inventory/environment.json", env)
    write(out / "00_inventory/repository_inventory.md",
          f"# Repository inventory\n\nBranch {info['branch']} at {info['commit']}. "
          f"Legacy worktree at {legacy_tree}, commit {info['legacy_commit']}. "
          f"All 26 legacy source hashes, 19 V2 source hashes and 42 dataset hashes passed. "
          f"There are 80 legacy run records and two complete V2 archives. "
          f"Archived records were read in place. Dataset digest {info['aggregate_data_sha256']}.\n\n"
          "Ground truth in the project root includes Ch2/Ch6 lecture PDFs, CEC2020 sections IV-A to IV-F "
          "and Algorithms 1–4, and IEEE TEVC architecture and workflow on PDF pages 4–6. "
          "The V2 specification is the root IDPC_DU_BENCHMARK_AGENT_SPEC_v2.md.\n")
    structured(out / "01_baseline_audit/baseline_freeze_manifest.json",
               dict(profiles=v2, methods=["ea","mfea"], dataset=info["aggregate_data_sha256"]))
    for method in ("ea","mfea"):
        structured(out / f"01_baseline_audit/{method}_effective_config.yaml",
                   {p: dict(m["effective_config"], method_config=m["method_config"][method],
                            evaluation_budget=m["evaluation_budget"])
                    for p,m in v2.items()})
    structured(out / "02_legacy_reproduction/legacy_config.yaml", legacy)
    structured(out / "02_legacy_reproduction/legacy_effective_config.yaml", legacy)
    return data


def independently_validate(graph, ids):
    edges = [graph.by_id.get(edge) for edge in ids]
    if not edges or any(edge is None for edge in edges):
        return math.inf
    vertices = [graph.source]
    for edge in edges:
        if edge.u != vertices[-1]:
            return math.inf
        vertices.append(edge.v)
    domains = [d for d, _ in groupby(edge.domain for edge in edges)]
    if vertices[-1] != graph.target or len(set(vertices)) != len(vertices) or len(set(domains)) != len(domains):
        return math.inf
    return sum(edge.weight for edge in edges)


def load_observations(graphs, data):
    obs = []
    for profile, folder in {"legacy": LEGACY, **V2}.items():
        is_legacy = profile == "legacy"
        records = LEGACY / "runs.csv" if is_legacy else folder / "task_runs.csv"
        methods = ("ektgl","mfea","individual","sto") if is_legacy else ("ea","mfea","ektgl")
        expected = {(m,s,t) for m in methods for s in SEEDS for t in graphs}
        seen = set()
        for r in read(records):
            t,m,s = r["task"],r["method"],int(r["seed"])
            key = (m,s,t)
            if key in seen:
                raise ValueError(f"Duplicate {profile}:{key}")
            seen.add(key)
            raw = r["edge_ids"] if is_legacy else r["best_path_edge_ids"]
            ids = list(map(int, raw.split())) if is_legacy else json.loads(raw)
            cost = float(r["cost" if is_legacy else "best_cost"])
            assert math.isfinite(cost) and independently_validate(graphs[t], ids) == cost, (profile,key)
            obs.append(dict(profile=profile,method=m,seed=s,task=t,cost=cost,feasible=True,
                            edge_ids=json.dumps(ids,separators=(",",":")),
                            subset=Path(data[t]["path"]).parent.name,
                            evaluations=int(r["evaluations" if is_legacy else "objective_evaluation_requests"]),
                            evaluation_definition="legacy GPA requests" if is_legacy else "V2 objective requests",
                            local_search_checks=int(r["local_search_checks"])))
        assert seen == expected, profile
    return obs


def holm(rows):
    prior = 0.0
    for i,row in enumerate(sorted(rows,key=lambda r:r["p_value"])):
        prior = max(prior,(len(rows)-i)*row["p_value"])
        row["p_holm"] = min(1.0, prior)


def outcome(a,b):
    return "tie" if abs(a-b) <= 1e-9 else ("win" if a<b else "loss")


def summarize(obs):
    grouped = defaultdict(list)
    for row in obs:
        grouped[f"{row['profile']}:{row['method']}",row["task"]].append(row)
    result = {}
    for (method, task), rows in sorted(grouped.items()):
        values = np.array([r["cost"] for r in rows])
        result[method,task] = dict(method=method,task=task,n=len(rows),BF=values.min(),
            AVG=values.mean(),STD=values.std(ddof=1) if len(values)>1 else None,
            median=np.median(values),IQR=np.quantile(values,.75)-np.quantile(values,.25),
            worst=values.max(),feasible_rate=1.0)
    return result,grouped


def comparative(out, obs, data):
    d = out / "04_cross_version"
    table(d/"raw_results.csv",obs)
    summaries,groups = summarize(obs)
    table(d/"summary_by_task.csv",summaries.values())
    details=[]
    for pair,a,b in PAIRS:
        for t in data:
            aa,bb=summaries[a,t],summaries[b,t]
            x=[r["cost"] for r in groups[a,t]]
            y=[r["cost"] for r in groups[b,t]]
            stat=mannwhitneyu(x,y,alternative="two-sided",method="asymptotic")
            details.append(dict(comparison=pair,method_a=a,method_b=b,task=t,
                subset=groups[a,t][0]["subset"],AVG_a=aa["AVG"],AVG_b=bb["AVG"],
                BF_a=aa["BF"],BF_b=bb["BF"],STD_a=aa["STD"],STD_b=bb["STD"],
                delta_avg=aa["AVG"]-bb["AVG"],
                relative_to_b_pct=100*(aa["AVG"]-bb["AVG"])/bb["AVG"] if bb["AVG"] else None,
                avg_outcome=outcome(aa["AVG"],bb["AVG"]),bf_outcome=outcome(aa["BF"],bb["BF"]),
                effect_cost_a_higher=2*stat.statistic/(len(x)*len(y))-1,p_value=stat.pvalue))
    holm(details)
    table(d/"cross_version_statistics.csv",details)
    comparisons=[]
    for pair,a,b in PAIRS:
        for scope in ("all","set1","set2"):
            chosen=[r for r in details if r["comparison"]==pair and scope in ("all",r["subset"])]
            for metric in ("avg","bf"):
                c=Counter(r[metric+"_outcome"] for r in chosen)
                comparisons.append(dict(comparison=pair,method_a=a,method_b=b,scope=scope,metric=metric,
                                        **{k:c[k] for k in OUTCOMES}))
    table(d/"win_tie_loss.csv",comparisons)
    table(d/"v2_matched_suite_budgets.csv",[
        dict(profile=profile,**row) for profile,folder in V2.items() for row in read(folder/"matched_suite_budgets.csv")
    ])
    write(d/"cross_version_report.md",
        "# Historical cross-version results\n\n"
        "All 8,400 saved paths passed a separate source/target, edge identity, simple-path, DU and cost audit "
        "on the same 42 instance hashes. Historical suites use different GA, grouping, LS, learning and evaluation protocols. "
        "Legacy GPA request totals omit path-score probes counted by V2. The exact cross-version matched budget is "
        "unrecoverable from these counters. The separate CSV contains the valid V2-only suite-cap comparisons. "
        "Two-sided Mann–Whitney U uses Holm across all ten comparisons × 42 tasks at 0.05. "
        "Historical seed labels do not imply shared random streams or causal matching.\n\n"
        "| Pair | Method A | Method B | AVG W/T/L |\n|---|---|---|---|\n"+
        "".join(f"| {r['comparison']} | {r['method_a']} | {r['method_b']} | {r['win']}/{r['tie']}/{r['loss']} |\n"
                for r in comparisons if r["scope"]=="all" and r["metric"]=="avg"))
    legacy=out/"02_legacy_reproduction"
    table(legacy/"raw_results.csv",(r for r in obs if r["profile"]=="legacy"))
    table(legacy/"summary_by_task.csv",(r for r in summaries.values() if r["method"].startswith("legacy:")))
    table(legacy/"legacy_win_tie_loss.csv",(r for r in comparisons if r["comparison"] in ("E1","ekt_sto","ekt_individual")))


def finite(x):
    return x if math.isfinite(x) else None


def initialize_rows(out, graphs, data, obs):
    frozen={}
    for profile,folder in V2.items():
        rows=defaultdict(dict)
        for r in read(folder/"convergence.csv"):
            if r["generation"]=="0":
                key=r["method"],int(r["seed"]),r["task"]
                rows[key]=r
        frozen[profile]=rows
    final={(r["profile"],r["method"],r["seed"],r["task"]):r["cost"] for r in obs}
    names=list(data)
    packed=[graphs[n] for n in names]
    warmup()
    decoders=[PackedGPA(g) for g in packed]
    local=[limits_for([g]) for g in packed]
    unified=limits_for(packed)
    config=Config()
    n,k=config.population_per_task,len(packed)
    records=[]
    for seed in SEEDS:
        trackers={m: EvaluationTracker(packed, cache_size=config.gpa_cache_per_task,decoders=decoders)
                  for m in ("mfea","ea")}
        rng=random.Random(seed_for(seed,"mfea"))
        genomes=[initialize(unified,rng) for _ in range(n*k)]
        costs=[{} for _ in genomes]
        observations={}
        for j,t in enumerate(names):
            values=[trackers["mfea"].gpa(j,g,"init") for g in genomes]
            for c,s in zip(costs,values):
                c[j]=s.cost
            projected=[project(g,packed[j].n) for g in genomes]
            observations["mfea",j]=(projected,values)
            ea_rng=random.Random(seed_for(seed,t,"ea"))
            random_genomes=[initialize(local[j],ea_rng) for _ in range(n)]
            observations["ea",j]=(random_genomes,[trackers["ea"].gpa(j,g,"init") for g in random_genomes])
        members=[Member(g,c) for g,c in zip(genomes,costs)]
        assign_ranks(members,k)
        for m in ("mfea","ea","ektgl"):
            for j,t in enumerate(names):
                g,s=observations[m if m=="mfea" else "ea",j]
                active=[i for i in range(len(g)) if m!="mfea" or members[i].skill==j]
                costs_at_task=[x.cost for x in s if x.feasible]
                active_costs=[s[i].cost for i in active if s[i].feasible]
                archive=min(costs_at_task,default=math.inf)
                row=dict(method=m,seed=seed,task=t,evaluated=len(g),
                         unique_chromosomes=len({(tuple(i.priority),tuple(i.edge_index)) for i in g}),
                         unique_feasible_paths=len({i.path for i in s if i.feasible}),
                         feasible=len(costs_at_task),active_count=len(active),active_feasible=len(active_costs),
                         initial_archive_cost=finite(archive),
                         initial_active_best=finite(min(active_costs,default=math.inf)),
                         initial_active_mean=float(np.mean(active_costs)) if active_costs else None,
                         archive_outside_active=sum(x.cost==archive for x in s)>0 and
                             (not active_costs or archive<min(active_costs)))
                for p,rows in frozen.items():
                    r=rows[m,seed,t]
                    if finite(archive)!=(float(r["best_cost"]) if r["best_cost"] else None):
                        raise ValueError(f"Reconstructed g0 differs {p}/{m}/{seed}/{t}")
                    assert int(r["population_count"])==len(active) and int(r["evaluated"])==len(g)
                    row[p+"_final"]=final[p,m,seed,t]
                    row[p+"_improvement"]=finite(archive-final[p,m,seed,t])
                records.append(row)
        print(f"initialization seed {seed}: 42/42 traces checked",flush=True)
    directory=out/"05_initialization"
    table(directory/"initialization_by_task_seed.csv",records)
    table(directory/"initialization_to_final.csv",({k:r[k] for k in ("method","seed","task","initial_archive_cost",
        "native_final","native_improvement","matched_ls_final","matched_ls_improvement")} for r in records))
    table(directory/"archive_audit.csv",(r for r in records if r["method"]=="mfea"))
    outside=sum(r["archive_outside_active"] for r in records if r["method"]=="mfea")
    indexed={(r["method"],r["seed"],r["task"]):r for r in records}
    same_later=Counter(
        outcome((indexed["mfea",seed,t]["initial_archive_cost"] or 0), indexed["ektgl",seed,t]["matched_ls_final"])
        for seed in SEEDS for t in names if indexed["mfea",seed,t]["initial_archive_cost"] is not None)
    write(directory/"initialization_report.md",
          f"# Initialization audit\n\nAll 2,520 generation-zero archives were reconstructed from their RNG streams "
          f"and checked independently against both V2 traces. MFEA records all 1,260 evaluations per task before "
          f"skill assignment. The archive was strictly better than its active skill group in {outside}/840 entries. "
          f"Against final matched-LS EKT-GL, the MFEA initialization archive had {same_later['win']} wins, "
          f"{same_later['tie']} ties and {same_later['loss']} losses. This is an archive comparison. "
          f"The matched-initialization intervention uses the identical USS source pool for the independent "
          f"populations and charges every one of the 52,920 initialization requests.\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=RESEARCH)
    parser.add_argument("--inventory-only",action="store_true")
    args=parser.parse_args(argv)
    data=verify_manifests(args.out)
    if args.inventory_only:
        return
    graphs={t:CompactIDPCGraph.from_file(ROOT / r["path"]) for t,r in data.items()}
    obs=load_observations(graphs,data)
    comparative(args.out,obs,data)
    initialize_rows(args.out,graphs,data,obs)


if __name__=="__main__":
    main()
