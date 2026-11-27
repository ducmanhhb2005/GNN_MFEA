"""Analyze committed V3 checkpoints and verify the independently rerun legacy archive."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import argparse
import csv
import json
import math
import sqlite3

import numpy as np
from scipy.stats import rankdata, wilcoxon

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import load, read_checkpoint
from experiments._support.idpc_v3_archive import (
    CONFIRMATION, LEGACY, PILOT, RESEARCH, ROOT, SEEDS, independently_validate,
    json_read, outcome, structured, write,
)

REQUESTS = "objective_evaluation_requests"
COMPONENTS = (
    "A0", "A1", "A2_32", "A2_64", "A3", "A4", "A5", "A6",
    "A8_identity", "A8_independent", "A8_shuffled", "A9_legacy_32",
    "D1", "D2", "D3", "G1", "G2", "G3", "G4", "N1", "N2", "N3",
)
PROFILES = ("native", "matched_ls", "matched_init_ls")
TYPES = ("ea", "mfea", "ektgl")
LEGACY_METHODS = ("sto", "mfea", "individual", "ektgl")


def put(path, rows):
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            output = csv.DictWriter(stream, fieldnames=fields)
            output.writeheader()
            output.writerows(rows)


def legacy_check(out):
    folder = out / "02_legacy_reproduction"
    rows = []
    for runner in sorted(folder.glob("*/runs")):
        original = json_read(LEGACY / "manifest.json")
        replay = json_read(runner.parent / "manifest.json")
        norm = lambda doc: {**{k: v for k, v in doc.items() if k != "fingerprint"}, "tasks": [{k: v for k, v in r.items() if k != "path"}
                                             for r in doc["tasks"]]}
        identity = norm(original) == norm(replay)
        for path in sorted(runner.glob("*.json")):
            ref, actual = json_read(LEGACY / "runs" / path.name), json_read(path)
            ref_events, run_events = [[{key: val for key, val in row.items()
                                       if key not in ("training_seconds", "inference_seconds")}
                                      for row in (obj["transfers"] or [])] for obj in (ref, actual)]
            matched = sum(obj == actual["tasks"].get(task) for task, obj in ref["tasks"].items())
            rows.append(dict(method=ref["method"], seed=ref["seed"], tasks_matched=matched,
                             tasks_total=len(ref["tasks"]), event_payload_exact=ref_events == run_events,
                             configuration_exact=identity and ref["fingerprint"] == original["fingerprint"] and actual["fingerprint"] == replay["fingerprint"],
                             directory=path.parent.parent.name))
    put(folder / "replay_verification.csv", rows)
    exact = {(r["method"], r["seed"]) for r in rows
             if r["tasks_matched"] == r["tasks_total"] and r["event_payload_exact"] and r["configuration_exact"]}
    expected = {(m, s) for m in LEGACY_METHODS for s in SEEDS}
    report = (
        "# Legacy reproduction\n\n"
        "Detached worktree at commit 0a4e11d. The original source, inputs, effective configuration "
        "and per-suite grouping were checked against their archived fingerprints. Compare every "
        "task payload and every non-timing transfer payload. Timing is hardware dependent.\n\n"
        f"Verified exact suite-runs independently rerun **{len(exact)}/{len(expected)}**. "
        f"Found {len(rows)} replay artifacts. See [seed and method ledger](replay_verification.csv).\n\n"
        "Original aggregate outcomes are available in [task summaries](legacy_summary_by_task.csv) "
        "and [W/T/L](legacy_win_tie_loss.csv).\n"
    )
    for name in ("legacy_reproduction_report.md", "reproduction_report.md"):
        write(folder / name, report)
    return len(exact)


def read_runs(out, verify):
    sources = out / "11_checkpoints"
    tasks = json_read(out / "01_baseline_audit/baseline_freeze_manifest.json")["profiles"]["native"]["tasks"]
    graphs = {entry["task"]: CompactIDPCGraph.from_file(ROOT / entry["path"])
              for entry in tasks} if verify else {}
    catalog = {entry["task"]: entry["path"].replace("\\", "/").split("/")[-2] for entry in tasks}
    manifests = sorted(sources.rglob("manifest.json"),
                       key=lambda p: (bool({"budget_extensions", "cap_continuations"}.intersection(p.parts)), p))
    detailed = {p.name for path in manifests if "detailed_diagnostics" in path.parts
                for p in path.parent.glob("*.sqlite")}
    for path in manifests:
        info = json_read(path)
        phase = info["phase"]
        if (phase == "smoke" or "shared_prefix" in path.parts
                or "post_replay_online" in path.parts
                or info.get("prefix_verification") or info.get("performance_validation")):
            continue
        for db in sorted(path.parent.glob("*.sqlite")):
            if info["diagnostic"] and not info.get("full_diagnostics") and db.name in detailed:
                continue
            method, seed_str = db.stem.rsplit("_", 1)
            seed = int(seed_str)
            identity = dict(phase=phase, profile=info["comparison_profile"], arm=info["arm"],
                            method=method, seed=seed, diagnostic=info["diagnostic"])
            try:
                meta, traces = read_checkpoint(db, info["fingerprint"], method, seed)
            except sqlite3.OperationalError as error:
                yield identity, dict(status="temporarily_unreadable", error=str(error)), [], catalog
                continue
            result = meta.get("result")
            identity["artifact"] = db.relative_to(out).as_posix()
            if result:
                if len(result["tasks"]) != len(catalog) or {r["task"] for r in result["tasks"]} != set(catalog):
                    raise ValueError(f"Incomplete result {db}")
                if traces[-1]["generation"] != result["actual_generations"]:
                    raise ValueError(f"Invalid generation ledger {db}")
                for position, task in enumerate(result["tasks"]):
                    count = task["counters"]
                    requests = count.get(REQUESTS, 0)
                    if requests != sum(count.get(f"{s}_evaluations", 0)
                                       for s in ("init", "offspring", "ls", "transfer", "other")):
                        raise ValueError(f"Objective ledger does not reconcile {db}:{task['task']}")
                    if requests != count.get("cache_hits", 0) + count.get("actual_objective_computations", 0):
                        raise ValueError(f"Cache ledger does not reconcile {db}:{task['task']}")
                    if requests != traces[-1]["evaluations"][position]:
                        raise ValueError(f"Task trace does not reconcile {db}:{task['task']}")
                    if verify and task["feasible"]:
                        g = graphs[task["task"]]
                        if independently_validate(g, task["best_path_edge_ids"]) != task["best_cost"]:
                            raise ValueError(f"Independent validation failed {db}:{task['task']}")
                counts = result["counters"]
                if counts[REQUESTS] != sum(t["counters"][REQUESTS] for t in result["tasks"]):
                    raise ValueError(f"Suite ledger does not reconcile {db}")
                if identity["profile"] != "native":
                    if counts[REQUESTS] > 379260 or result["evaluation_budget"] != 379260:
                        raise ValueError(f"Invalid matched budget {db}")
                    if counts[REQUESTS] < 379260:
                        meta = dict(meta, status="below_matched_cap", below_cap=counts[REQUESTS],
                                    result=None)
            yield identity, meta, traces, catalog


def replay_diagnostics(event, base, outputs):
    for row in event["rows"]:
        ident = {**base, "task": row["task"]}
        scalar = {k: v for k, v in row.items() if k not in ("paths", "replay", "learning", "task")}
        outputs["transfer_events"].append({**ident, **scalar})
        for i, path in enumerate(row.get("paths", ()), 1):
            outputs["candidate_quality"].append({**ident, **scalar, "candidate_id": i, **path})
        for trial in row.get("replay", ()):
            outputs["decoder_replay"].append({**ident, **trial})
        if "learning" in row:
            learning = dict(row["learning"])
            sample = learning.pop("edge_sample")
            outputs["learning_metrics"].append({
                **ident, **learning, "train_loss": event["train_loss"],
                "positive_labels": row["positive_labels"], "negative_labels": row["negative_labels"],
                "train_time_s": event["train_time_s"], "inference_time_s": event["inference_time_s"],
                "positive_examples_suite": event.get("positive_examples"),
                "negative_examples_suite": event.get("negative_examples"),
                **event.get("gradients", {}),
            })
            for edge, prob, label in sample:
                outputs["edge_predictions"].append(dict(**ident, edge_id=edge, score=prob, label=label))
    attention = event.get("attention")
    if attention is not None:
        names = [r["task"] for r in event["rows"]]
        for task, weights in zip(names, attention):
            entropy = -sum(w * math.log(w) for w in weights if w)
            outputs["attention_diagnostics"].extend(
                dict(**base, task=task, source_task=source, weight=weight, entropy=entropy)
                for source, weight in zip(names, weights))


def aggregate(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in key)].append(row)
    result = []
    for group, items in sorted(groups.items()):
        finite = [i["cost"] for i in items if i["feasible"]]
        n = len(items)
        full = len(finite) == n
        result.append(dict(zip(key, group), runs=n, BF=min(finite) if finite else math.inf,
                           AVG=float(np.mean(finite)) if full else math.inf,
                           STD=float(np.std(finite, ddof=1)) if full and n > 1 else None,
                           median=float(np.median(finite)) if full else None,
                           IQR=float(np.diff(np.quantile(finite, (.25, .75)))[0]) if full else None,
                           worst=max(finite) if full else math.inf, feasible_runs=len(finite),
                           feasible_rate=len(finite)/n,
                           mean_objective_requests=np.mean([r.get(REQUESTS, 0) for r in items]),
                           mean_init_requests=np.mean([r.get("init_evaluations", 0) for r in items]),
                           mean_improvement=np.mean([r["initial_cost"] - r["cost"] for r in items])
                           if full and all(math.isfinite(r["initial_cost"]) for r in items) else None))
    return result


def paired(rows, contrasts, expected):
    index = defaultdict(dict)
    for row in rows:
        index[row["profile"], row["arm"], row["method"]][row["seed"], row["task"]] = row["cost"]
    stats, detailed = [], []
    rng = np.random.default_rng(4199)
    for label, left, right, family in contrasts:
        aa, bb = index[left], index[right]
        seeds = sorted({s for s, _ in aa} & {s for s, _ in bb})
        if not seeds:
            continue
        tasks = sorted({t for _, t in aa} & {t for _, t in bb})
        x = np.array([[aa[s, t] for t in tasks] for s in seeds])
        y = np.array([[bb[s, t] for t in tasks] for s in seeds])
        complete = seeds == list(expected) and len(tasks) == 42
        finite = np.isfinite(x).all() and np.isfinite(y).all()
        wtl = Counter(outcome(float(a), float(b)) for a, b in zip(x.mean(0), y.mean(0)))
        means = ((y - x) / np.maximum(np.abs(y), 1) * 100).mean(1) if finite else None
        p = (1. if np.all(means == 0) else float(wilcoxon(means).pvalue)) if complete and finite else None
        ci = np.quantile(rng.choice(means, size=(10000, len(seeds))).mean(1), (.025, .975)) \
            if complete and finite else [None, None]
        denominator = sum(range(1, int(np.count_nonzero(means)) + 1)) if means is not None else 0
        nonzero = means[means != 0] if denominator else []
        rank_delta = float(np.sum(rankdata(abs(nonzero)) * np.sign(nonzero)) / denominator) \
            if denominator else (0. if means is not None else None)
        record = dict(comparison=label, family=family, profile=left[0], seeds=len(seeds),
                      seed_ids=" ".join(map(str, seeds)), tasks=len(tasks), complete=complete,
                      wins=wtl["win"], ties=wtl["tie"], losses=wtl["loss"],
                      left_AVG=float(x.mean()), right_AVG=float(y.mean()),
                      effect_pct=float(means.mean()) if means is not None else None,
                      rank_biserial=rank_delta, lower_95=ci[0], upper_95=ci[1], p=p, p_holm=None)
        stats.append(record)
        for task, a, b in zip(tasks, x.T, y.T):
            detailed.append(dict(comparison=label, profile=left[0], task=task, seeds=len(seeds),
                                 left_AVG=a.mean(), right_AVG=b.mean(), outcome=outcome(a.mean(), b.mean())))
    for family in sorted({r["family"] for r in stats}):
        members = [r for r in stats if r["family"] == family]
        tested = sorted((r for r in members if r["p"] is not None), key=lambda r: r["p"])
        n = sum(f == family for _, _, _, f in contrasts)
        previous = 0
        for i, row in enumerate(tested):
            previous = max(previous, min(1., (n-i) * row["p"]))
            row["p_holm"] = previous
    return stats, detailed


def completed_populations(db):
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute("SELECT value FROM meta WHERE key='state'").fetchone()
        return load(row[0])["evolution"]["populations"]


def matched_initialization_checks(rows):
    init_pool = defaultdict(dict)
    for row in rows:
        if row["profile"] == "matched_init_ls":
            init_pool[row["seed"], row["task"]][row["method"]] = row
    checks = []
    for (seed, task), cohort in sorted(init_pool.items()):
        if set(cohort) == {"ea", "mfea", "ektgl"}:
            costs = {method: row["initial_cost"] for method, row in cohort.items()}
            requests = {method: row["init_evaluations"] for method, row in cohort.items()}
            if set(requests.values()) != {1260} or len(set(costs.values())) != 1:
                raise ValueError(f"Matched initialization diverged at {seed}:{task}")
            checks.append(dict(seed=seed, task=task, requests_per_method=1260,
                               shared_initial_archive_cost=costs["mfea"]))
    return checks


def analyze(out, verify):
    raw, suites, run_coverage, convergence = [], [], [], []
    online, learning_work, diversity = [], [], []
    diagnostic = defaultdict(list)
    for base, meta, entries, catalog in read_runs(out, verify):
        run_coverage.append({**base, "status": meta.get("status"), "error": meta.get("error"),
                             "generation": entries[-1]["generation"] if entries else None,
                             "below_cap": meta.get("below_cap")})
        result = meta.get("result")
        if base["diagnostic"]:
            for entry in entries:
                event = entry["transfer"]
                if event:
                    replay_diagnostics(event, {**base, "generation": entry["generation"]}, diagnostic)
        if result is None:
            continue
        if base["diagnostic"]:
            for row in result["tasks"]:
                counts = row["counters"]
                diagnostic["local_search_diagnostics"].append(
                    dict(**base, task=row["task"], ls_no_improvement=counts.get("ls_calls", 0) -
                         counts.get("ls_accepted", 0),
                         **{k: v for k, v in counts.items()
                            if k.startswith(("ls_", "ordinary_ls_", "pre_transfer_ls_"))
                            and k != "ls_no_improvement" or "roundtrip" in k}))
            continue
        if base["method"] == "ektgl":
            for entry in entries:
                event = entry["transfer"]
                if event:
                    learning_work.append(dict(**base, generation=entry["generation"],
                                              **{k: event.get(k) for k in (
                                                  "train_loss", "train_time_s", "inference_time_s",
                                                  "positive_examples", "negative_examples")}))
                    online.extend(dict(**base, **{k: v for k, v in event_row.items() if k != "paths"})
                                  for event_row in event["rows"])
            populations = completed_populations(out / base["artifact"])
            for row, members in zip(result["tasks"], populations):
                distinct_paths = {m.solution.path for m in members if m.solution.feasible}
                distinct_chromosomes = {(tuple(m.chromosome.priority), tuple(m.chromosome.edge_index))
                                        for m in members}
                diversity.append(dict(**base, task=row["task"], active_total=len(members),
                                      active_feasible=sum(m.solution.feasible for m in members),
                                      unique_feasible_paths=len(distinct_paths),
                                      unique_chromosomes=len(distinct_chromosomes)))
        for index, row in enumerate(result["tasks"]):
            initial = entries[0]["best_costs"][index]
            raw.append(dict(**base, task=row["task"], set=catalog[row["task"]],
                            cost=row["best_cost"], feasible=row["feasible"], initial_cost=initial,
                            best_generation=row["best_generation"], actual_generations=result["actual_generations"],
                            best_path_edge_ids=json.dumps(row["best_path_edge_ids"]),
                            independently_validated=verify and row["feasible"], **row["counters"]))
            if base["phase"] == "confirmation" or base["arm"] in ("A0", "A1", "D1", "D2", "D3"):
                for entry in entries:
                    convergence.append(dict(**base, task=row["task"], generation=entry["generation"],
                                            suite_requests=entry["suite_evaluations"],
                                            task_requests=entry["evaluations"][index],
                                            cost=entry["best_costs"][index],
                                            active_feasible=entry["populations"][index][1]))
        suite_fields = ("actual_generations", "wall_clock_s", "train_time_s", "inference_time_s",
                        "local_search_time_s", "cpu_peak_process_bytes", "gpu_peak_allocated_bytes",
                        "started_at", "completed_at")
        suites.append(dict(**base, **{k: result[k] for k in suite_fields}, **result["counters"]))

    # The matched control retains the identical canonical MFEA result for every seed.
    for bucket in (raw, suites, run_coverage, convergence):
        canonical = [r for r in bucket if r["profile"] == "matched_ls" and r["method"] == "mfea"]
        existing = {r["phase"] + str(r["seed"]) for r in bucket
                    if r["profile"] == "matched_init_ls" and r["method"] == "mfea"}
        bucket.extend({**r, "profile": "matched_init_ls", "reused_canonical_result": True}
                      for r in canonical if r["phase"] + str(r["seed"]) not in existing)

    p = out / "06_ablation"
    pilot = [r for r in raw if r["phase"] == "pilot"]
    final = [r for r in raw if r["phase"] == "confirmation"]
    key = ("profile", "arm", "method", "set", "task")
    put(p / "raw_results.csv", pilot)
    put(p / "ablation_raw_results.csv", pilot)
    summary = aggregate(pilot, key)
    put(p / "summary_by_task.csv", summary)
    put(p / "ablation_summary.csv", summary)
    put(p / "learning_work_by_event.csv", [r for r in learning_work if r["phase"] == "pilot"])
    put(p / "online_transfer_outcomes.csv", [r for r in online if r["phase"] == "pilot"])
    put(p / "final_population_diversity.csv", [r for r in diversity if r["phase"] == "pilot"])
    put(p / "suite_workload.csv", [r for r in suites if r["phase"] == "pilot"])
    contrasts = [(f"{arm} vs A0", ("matched_ls", arm, "ektgl"), ("matched_ls", "A0", "ektgl"),
                  "neighborhood" if arm.startswith("N") or arm == "A1" else
                  "decoder" if arm.startswith("D") or arm == "A5" else
                  "learning" if arm.startswith(("G", "A2", "A3", "A4", "A8")) else "other")
                 for arm in COMPONENTS if arm != "A0"]
    contrasts.extend((f"Matched-init vs standard {m}", ("matched_init_ls", "A0", m),
                      ("matched_ls", "A0", m), "initialization") for m in ("ektgl", "ea"))
    contrasts.extend((f"{a} vs {b}", ("matched_ls", a, "ektgl"), ("matched_ls", b, "ektgl"),
                      "decoder") for a, b in (("D3", "D1"), ("D3", "D2"), ("D2", "D1")))
    contrasts += [
        ("A3 vs G2", ("matched_ls", "A3", "ektgl"), ("matched_ls", "G2", "ektgl"), "learning"),
        ("A9_legacy_32 vs A1", ("matched_ls", "A9_legacy_32", "ektgl"),
         ("matched_ls", "A1", "ektgl"), "interaction"),
        ("A9_legacy_32 vs A2_32", ("matched_ls", "A9_legacy_32", "ektgl"),
         ("matched_ls", "A2_32", "ektgl"), "interaction"),
    ]
    for profile in ("matched_ls", "matched_init_ls"):
        scope = "standard" if profile == "matched_ls" else "matched-init"
        for m in ("ektgl", "ea"):
            contrasts.append((f"{m} vs canonical MFEA ({scope})", (profile, "A0", m),
                              (profile, "A0", "mfea"), "baseline_context"))
    stats, detailed = paired(pilot, contrasts, PILOT)
    put(p / "ablation_statistics.csv", stats)
    put(p / "ablation_effects_by_task.csv", detailed)
    p8 = out / "08_v3_pilot"
    put(p8 / "pilot_results.csv", stats)
    for folder, family in (("beam_search", "decoder"), ("graph_learning", "learning"), ("local_search", "neighborhood")):
        put(p8 / folder / "comparisons.csv", [r for r in stats if r["family"] == family])
    matched_init = [r for r in pilot if r["profile"] == "matched_init_ls"]
    put(out / "05_initialization/matched_init_results.csv", matched_init)
    put(out / "05_initialization/matched_init_pool_audit.csv", matched_initialization_checks(pilot))
    put(out / "05_initialization/matched_init_statistics.csv", [r for r in stats if r["family"] == "initialization"])
    for k, rows in diagnostic.items():
        put((p8 / "beam_search" if k == "decoder_replay" else out / "07_transfer_diagnostics") / (k + ".csv"), rows)
    p9 = out / "09_final_benchmark"
    put(p9 / "full_raw_results.csv", final)
    put(p9 / "matched_init_ls" / "initialization_pool_audit.csv", matched_initialization_checks(final))
    put(p9 / "summary_by_task.csv", aggregate(final, key))
    put(p9 / "learning_work_by_event.csv", [r for r in learning_work if r["phase"] == "confirmation"])
    put(p9 / "online_transfer_outcomes.csv", [r for r in online if r["phase"] == "confirmation"])
    put(p9 / "final_population_diversity.csv", [r for r in diversity if r["phase"] == "confirmation"])
    put(p9 / "suite_workload.csv", [r for r in suites if r["phase"] == "confirmation"])
    pairs = []
    for profile in PROFILES:
        for left, right in (("ektgl", "mfea"), ("ektgl", "ea"), ("mfea", "ea")):
            a, b = ((profile, "V3" if m == "ektgl" else "A0", m) for m in (left, right))
            pairs.append((f"{left} vs {right}", a, b, "confirmation_" + profile))
        subset = [r for r in final if r["profile"] == profile]
        put(p9 / profile / "raw_results.csv", subset)
        put(p9 / profile / "summary_by_task.csv", aggregate(subset, key))
    results, by_task = paired(final, pairs, CONFIRMATION)
    put(p9 / "win_tie_loss.csv", results)
    put(p9 / "statistical_tests.csv", results)
    put(p9 / "comparisons_by_task.csv", by_task)
    put(out / "14_reports/FULL_RESULTS.csv", aggregate(final, key))
    put(out / "10_figures/convergence_source.csv", convergence)
    return run_coverage, stats, detailed, diagnostic


def available(out):
    for folder, prefix in (("02_legacy_reproduction", "legacy"), ("04_cross_version", "cross_version")):
        for name in ("raw_results", "summary_by_task", "win_tie_loss"):
            if folder.startswith("02") and name == "win_tie_loss":
                continue
            path = out / folder / (name + ".csv")
            if path.is_file():
                (path.parent / (prefix + "_" + path.name)).write_bytes(path.read_bytes())


def write_status(out, run_coverage, legacy_exact, stats, diagnostic):
    found = {(r["phase"], r["profile"], r["arm"], r["method"], r["seed"], r["diagnostic"]): r
             for r in run_coverage}
    expected = [
        ("pilot", "matched_ls", arm, "ektgl", seed, False) for arm in COMPONENTS for seed in PILOT
    ] + [
        ("pilot", profile, "A0", m, seed, False) for seed in PILOT
        for profile, methods in (("matched_ls", ("ea", "mfea")), ("matched_init_ls", ("ea", "ektgl", "mfea")))
        for m in methods
    ] + [
        ("pilot", "matched_ls", "A0", "ektgl", seed, True) for seed in PILOT
    ] + [
        ("confirmation", profile, "V3" if m == "ektgl" else "A0", m, seed, False)
        for profile in PROFILES for m in TYPES for seed in CONFIRMATION
    ]
    fields = ("phase", "profile", "arm", "method", "seed", "diagnostic")
    coverage = [found.get(k, dict(zip(fields, k), status="missing")) for k in expected]
    counts = {p: dict(Counter(r["status"] for r in coverage if r["phase"] == p))
              for p in ("pilot", "confirmation")}
    structured(out / "coverage.json", coverage)
    structured(out / "experiment_status.json",
               dict(status="PARTIAL", legacy_exact=legacy_exact, legacy_expected=80,
                    checkpoints=counts, confirmation_frozen=(out/"13_configs/frozen_v3.json").is_file(),
                    note="Complete only after scientific acceptance gates are reviewed."))
    registry = []
    for key in dict.fromkeys((r["phase"], r["profile"], r["arm"], r["method"], r["diagnostic"])
                             for r in coverage):
        entries = [r for r in coverage if tuple(r[f] for f in (*fields[:4], "diagnostic")) == key]
        registry.append(dict(zip((*fields[:4], "diagnostic"), key), expected=len(entries),
                             completed=sum(r["status"] == "complete" for r in entries),
                             seeds=" ".join(str(r["seed"]) for r in entries)))
    put(out / "06_ablation/experiment_registry.csv", registry)
    put(out / "06_ablation/ablation_registry.csv", registry)
    effect_rows = [r for r in stats if r["complete"] and r["effect_pct"] is not None]
    desc = (
        "# Controlled ablation results\n\n"
        "Positive effect denotes lower cost for the first arm. Each complete contrast uses 42 tasks "
        "and three paired pilot seeds at 379,260 requests per suite. The inference unit is the seed. "
        "The task-average relative effect is computed before testing. Incomplete contrasts are descriptive. "
        "The Holm denominator includes the planned contrasts in each family.\n\n"
        f"Completed pilot checkpoints {counts['pilot'].get('complete', 0)}/{sum(counts['pilot'].values())}. "
        f"Completed contrasts {len(effect_rows)}.\n\n"
    )
    if effect_rows:
        desc += "| Comparison | Mean relative gain (%) | W/T/L | Holm p |\n|---|---:|---|---:|\n"
        desc += "".join(f"| {r['comparison']} | {r['effect_pct']:.3f} | {r['wins']}/{r['ties']}/{r['losses']} | {r['p_holm']:.3g} |\n"
                        for r in effect_rows)
    desc += "\n[Checkpoint coverage](../coverage.json) · [All paired effects](ablation_statistics.csv)\n"
    write(out / "06_ablation/ablation_report.md", desc)
    write(out / "14_reports/ABLATION_STUDY.md", desc.replace("../coverage.json", "../coverage.json")
          .replace("(ablation_statistics.csv)", "(../06_ablation/ablation_statistics.csv)"))
    events = diagnostic["transfer_events"]
    attempts = [r for r in events if r["attempt"]]
    if attempts:
        accepted = sum(r["accepted"] for r in attempts)
        improved = sum(r["improvement"] for r in attempts)
        dead = sum(r["outcome"] == "dead_end" for r in attempts)
        report = (
            "# Transfer diagnostics\n\n"
            f"Recorded {len(attempts)} A0 attempts, {accepted} accepted, {improved} archive "
            f"improvements and {dead} construction dead ends. Counter denominators exclude "
            "tasks skipped by the budget or absent labels. Learning metrics use current elite labels and are "
            "in-sample. The replay table uses a separate budget ledger for the same frozen populations and scores.\n\n"
            "[Attempts](transfer_events.csv) · [Learning](learning_metrics.csv) · "
            "[Decoder replay](../08_v3_pilot/beam_search/decoder_replay.csv)\n"
        )
        for name in ("transfer_diagnostic_report.md", "diagnostic_report.md"):
            write(out / "07_transfer_diagnostics" / name, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=type(RESEARCH), default=RESEARCH)
    parser.add_argument("--verify-paths", action="store_true")
    args = parser.parse_args()
    out = args.out
    available(out)
    legacy_exact = legacy_check(out)
    coverage, stats, _, diagnostic = analyze(out, args.verify_paths)
    write_status(out, coverage, legacy_exact, stats, diagnostic)
    print(json_read(out / "experiment_status.json"))


if __name__ == "__main__":
    main()
