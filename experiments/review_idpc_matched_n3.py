"""Apply all Matched-N3 research gates to the published artifacts."""
from collections import Counter
from pathlib import Path
import argparse
import re

from PIL import Image

from experiments._support.idpc_v3_archive import json_read, read, sha, structured, write
from experiments.run_idpc_matched_n3 import BUDGET, INIT, METHODS, OUT, RESEARCH, ROOT


def rows(out, name):
    path = out / (name + ".csv")
    return list(read(path)) if path.exists() else []


def review(out):
    coverage = json_read(out/"coverage.json")
    manifest = json_read(out/"experiment_manifest.json")
    seeds = manifest["seeds"]
    raw = rows(out,"full_raw_results")
    workload = rows(out,"workload")
    key_counts = Counter((r["method"], int(r["seed"])) for r in raw)
    tasks = rows(out,"publication_table_all42")
    wtl = rows(out,"win_tie_loss")
    test = rows(out,"statistical_tests")
    operator = rows(out,"comparison_with_v3_2opt")
    tests = (out/"logs/targeted_tests.err").read_text(encoding="utf-8",errors="strict")
    figure_paths = sorted((out/"figures").glob("*.png"))
    fig_ok = len(figure_paths)==8
    for path in figure_paths:
        with Image.open(path) as image:
            fig_ok &= round(image.info.get("dpi",(0,0))[0])==300
            image.verify()
    links = re.findall(r"Source CSV\]\(([^)]+)\)",(out/"figures/index.md").read_text(encoding="utf-8"))
    expected_sources = all(sha(ROOT/p)==v for p,v in manifest["sources"].items())
    v3_intact = all(sha(RESEARCH/p)==v for p,v in manifest["reference_artifacts"].items())
    plan = json_read(out/"configs/statistical_plan.json")
    checks = {
        "Only the three allowed N3 arms": set(r["configuration"] for r in raw)=={m+"_n3" for m in METHODS},
        "Archived V3 comparator and source fingerprints preserved": expected_sources and v3_intact,
        "Original initialization archive and charges independently reconciled": len(workload)==60 and all(
            r["init_archive_exact"]=="True" and int(r["init_evaluations"])==42*INIT[r["method"]] for r in workload),
        "One shared frozen N3 implementation, cache policy and caps": manifest["local_search"]=="V3 Native replace_segment"
            and all(ref["variant"]["ls"]=="subpath" and not ref["variant"]["matched_init"]
                    for ref in manifest["references"].values()),
        "Original ceiling and 379,260 requests for each suite": len(workload)==60 and all(
            r["stop_rule"]=="budget" and int(r["objective_evaluation_requests"])==BUDGET and
            int(r["actual_generations"])<=manifest["generation_ceiling"] for r in workload),
        "60 unique suites and 2,520 task outcomes": key_counts=={(m,s):42 for m in METHODS for s in seeds}
            and len({(r["method"],r["seed"],r["task"]) for r in raw})==2520,
        "Independent DU path checks": len(raw)==2520 and all(
            r["independently_validated"]=="True" if r["feasible"]=="True" else r["cost"]=="inf" for r in raw),
        "Full BF, AVG and sample STD": len(tasks)==42 and all(
            t[f"{m}_runs"]=="20" and all(t[f"{m}_{key}"] for key in ("BF","AVG","STD"))
            for t in tasks for m in METHODS),
        "18 combat tallies across AVG, BF and the three subsets": len(wtl)==18 and all(
            sum(int(r[x]) for x in ("wins","ties","losses"))=={"set1":24,"set2":18,"all42":42}[r["scope"]]
            for r in wtl),
        "All three paired 2-opt contrasts and suite-level inference": len(operator)==3 and all(
            r["paired"]=="True" for r in operator) and len(test)==10 and all(
            r["seeds"]=="20" and r["p_holm"] and r["ci_low"] and r["ci_high"] for r in test)
            and plan==manifest["statistics"],
        "N3 event, transfer, workload and convergence ledgers": all(rows_exist(out,k) for k in (
            "local_search_events","transfer_events","workload","convergence")),
        "Smoke, targeted tests and reproducibility instructions": json_read(out/"logs/smoke_validation.json")["passed"]
            and "Ran 3 tests" in tests and "OK" in tests and (out/"README.md").is_file(),
        "Eight 300-DPI figures and their source CSVs": fig_ok and len(links)==8 and all(
            (out/"figures"/link).is_file() for link in links),
    }
    done = all(checks.values())
    coverage.update(status="COMPLETE" if done else "PARTIAL", acceptance=checks)
    structured(out/"coverage.json",coverage)
    lines = ["# Acceptance review", "", f"**{coverage['status']}**", ""]
    lines.extend(f"- [{'x' if value else ' '}] {label}" for label,value in checks.items())
    write(out/"ACCEPTANCE.md","\n".join(lines)+"\n")
    print(coverage["status"])
    if not done:
        print("\n".join(label for label,value in checks.items() if not value))
    return 0 if done else 2


def rows_exist(out, name):
    path = out / (name + ".csv")
    return path.is_file() and path.stat().st_size > 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    raise SystemExit(review(parser.parse_args().out))
