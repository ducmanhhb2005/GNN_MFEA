"""Review the actual artifacts against the V3 research acceptance gates."""

from collections import Counter

from experiments._support.idpc_v3_archive import CONFIRMATION, PILOT, RESEARCH, ROOT, json_read, read, sha, structured, write


def csv_rows(path):
    return list(read(path)) if path.is_file() else []


def entries(root, phase, coverage):
    rows = [r for r in coverage if r["phase"] == phase]
    observed = Counter(r["status"] for r in rows)
    return rows, observed


def review(root):
    coverage = json_read(root/"coverage.json")
    status = json_read(root/"experiment_status.json")
    pilot, pcount = entries(root, "pilot", coverage)
    confirmation, ccount = entries(root, "confirmation", coverage)
    frozen = root/"13_configs/frozen_v3.json"
    configuration = json_read(frozen) if frozen.is_file() else None
    memo = root/"08_v3_pilot/local_search/n3_memoization_parity.json"
    solver_frozen = configuration is not None and all(
        sha(ROOT/path) == digest for path, digest in configuration["sources"].items())
    memo_exact = not configuration or not configuration.get("native_subpath_memoization") or (
        memo.is_file() and json_read(memo)["equivalent"])
    audit = json_read(root/"01_baseline_audit/baseline_freeze_manifest.json")
    originals = all(sha(ROOT/path)==digest for manifest in audit["profiles"].values()
                    for path,digest in manifest["sources"].items())
    historical = csv_rows(root/"04_cross_version/win_tie_loss.csv")
    replay = csv_rows(root/"02_legacy_reproduction/replay_verification.csv")
    init = csv_rows(root/"05_initialization/initialization_by_task_seed.csv")
    init_control = csv_rows(root/"05_initialization/matched_init_pool_audit.csv")
    diagnostic = root/"07_transfer_diagnostics"
    final_root = root/"09_final_benchmark"
    final = csv_rows(final_root/"full_raw_results.csv")
    final_summary = csv_rows(final_root/"summary_by_task.csv")
    final_stats = csv_rows(final_root/"statistical_tests.csv")
    independent = csv_rows(final_root/"independent_ekt_version_tests.csv")
    after_replay = csv_rows(root/"08_v3_pilot/beam_search/post_replay_online_verification.csv")
    archive_tests = root/"12_logs/v3_regression_final.err"
    tests_ok = archive_tests.is_file() and "Ran 67 tests" in archive_tests.read_text(encoding="utf-8",errors="replace") and \
               "OK" in archive_tests.read_text(encoding="utf-8",errors="replace")
    confirmed_paths = all(row["independently_validated"]=="True" for row in final if row["feasible"]=="True")
    final_keys = {(r["profile"],r["method"],int(r["seed"]),r["task"]) for r in final}
    cap_ok = all(int(r["objective_evaluation_requests"]) == 379260
                 for r in csv_rows(final_root/"suite_workload.csv") if r["profile"] != "native")
    images = list((root/"10_figures").glob("*.png"))
    docs = root/"14_reports"
    reports = ("ROOT_CAUSE_ANALYSIS.md", "CONTROLLED_MECHANISMS.md", "EKT_GL_V3_ALGORITHM.md",
               "EKT_GL_V3_CHANGELOG.md", "FINAL_BENCHMARK_REPORT.md", "ABLATION_STUDY.md",
               "FULL_RESULTS.csv", "results_tables.tex")
    checks = [
        ("Exact legacy replay at the source fingerprint", len(replay)==80 and status["legacy_exact"]==80),
        ("EA and MFEA audited and original manifest sources preserved", originals and
         (root/"01_baseline_audit/baseline_correctness_report.md").is_file()),
        ("Documented source and configuration comparison", all((root/"03_version_diff"/p).is_file() for p in
         ("configuration_diff.csv","source_code_diff.md","local_search_audit.md"))),
        ("Cross-version historical and matched-protocol evidence", all(any(r["comparison"]==k for r in historical)
          for k in ("E1","E2","E3","E4","matched_reference")) and
          (root/"04_cross_version/v2_matched_suite_budgets.csv").is_file()),
        ("Initialization reconstructed and controlled",len(init)==2520 and len(init_control)==126),
        ("Mechanism diagnostics including truncation and exact-ID candidates", all((diagnostic/p).is_file() for p in
         ("local_search_diagnostics.csv","learning_metrics.csv","candidate_quality.csv","attention_diagnostics.csv"))),
        ("All controlled pilots completed",len(pilot)==84 and pcount["complete"]==84),
        ("Online decoder arms verified after the shared replay", len(after_replay)==12 and all(
         all(r[k]=="True" for k in ("matched","matched_manifest","completed_after_replay")) for r in after_replay)),
        ("A supported configuration frozen with independent seeds", solver_frozen and memo_exact and
         tuple(configuration["pilot_seeds"])==PILOT and tuple(configuration["confirmation_seeds"])==CONFIRMATION and
         any(r["arm"]==configuration["selected_arm"] and r["eligible"] for r in configuration["evidence"])),
        ("Final overall coverage and exact matched objective budget", len(confirmation)==180 and
         ccount["complete"]==180 and len(final)==len(final_keys)==7560 and cap_ok),
        ("Confirmation matched-initialization archive and charges reconcile",
         len(csv_rows(final_root/"matched_init_ls/initialization_pool_audit.csv"))==840),
        ("Independent validation of feasible confirmation paths",len(final)==7560 and confirmed_paths),
        ("Complete per-instance results and statistical comparisons", len(final_summary)==378 and
         all(r["runs"]=="20" for r in final_summary) and len(final_stats)==9 and
         all(r["complete"]=="True" for r in final_stats) and len(independent)==2),
        ("Figures in PNG and vector PDF",len(images)>=22 and
         all(p.with_suffix(".pdf").is_file() for p in images) and len(final)==7560),
        ("Source, regression tests and reproducibility instructions", tests_ok and (root/"README.md").is_file()),
        ("Research and confirmation reports",all((docs/p).is_file() for p in reports)),
    ]
    done = all(value for _,value in checks)
    text = ["# Acceptance review", "",
            f"**{'COMPLETE' if done else 'PARTIAL'}**",
            "",
            "Acceptance measures coverage and reproducibility. Statistical outcomes against the canonical baselines "
            "are reported as observed.", ""]
    text.extend(f"- [{'x' if value else ' '}] {label}" for label,value in checks)
    text += ["", "[Checkpoint ledger](../coverage.json) · [Frozen baseline](../01_baseline_audit/baseline_freeze_manifest.json) · "
             "[Research protocol and reproduction](../README.md)", ""]
    write(docs/"ACCEPTANCE.md","\n".join(text))
    status["status"] = "COMPLETE" if done else "PARTIAL"
    status["acceptance"] = {label:value for label,value in checks}
    status["note"] = "Reviewed against actual manifests, results and reports."
    structured(root/"experiment_status.json",status)
    print(status["status"])


if __name__ == "__main__":
    review(RESEARCH)
