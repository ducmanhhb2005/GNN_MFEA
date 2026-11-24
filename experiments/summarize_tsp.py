"""Summarize a completed run_tsp.py study and its independent diagnostic.

python -m experiments.summarize_tsp
"""
import json
import hashlib
from pathlib import Path

import numpy as np

from run_tsp import ROOT, source_digest


def main():
    directory = ROOT / "results" / "tsp"
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != 18 or {r["benchmark"] for r in rows} != {"kro", "pr"}:
        raise ValueError("Both 9-task benchmarks must finish before final summary")
    diagnostic = json.loads((directory / "local_search_diagnostic.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("Result audit must pass before writing the summary")
    if (audit["source_sha256"] != source_digest()
            or audit["summary_sha256"] != hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest()):
        raise ValueError("Source or summary changed after audit; rerun audit_tsp_results")
    runs = rows[0]["runs"]
    if audit["runs_per_mode_per_benchmark"] != runs or audit["validated_tours"] != 36 * runs:
        raise ValueError(f"Rerun audit_tsp_results --runs {runs} for this study")
    lines = ["# EKT-GL cho TSP: kết quả thực nghiệm", "",
             "**[Mở báo cáo thực nghiệm đầy đủ](results/tsp/REPORT.md)** — bảng chi tiết từng task, mean/std, gap, kiểm định thống kê và đồ thị hội tụ.", "",
             f"Đã chạy **18 task TSPLIB, {runs} seed mỗi phương pháp, 300 thế hệ**, chia TSP-kro/TSP-pr (9 task/bộ), với EKT-GL và STO. Dùng lõi trong `ekt_gl/`, tham khảo runner CVRP `run_vrp.py`. Đây là số đo thực tế của bản triển khai trong repo, chưa phải tái lập đủ 20 seed của paper.", "",
             "## Kết quả chính", "",
             "Mọi objective đều cần tối thiểu hóa. Các tỷ lệ dưới đây là trung bình không trọng số trên 9 task, không phải tỷ lệ của tổng objective.", "",
             "| Benchmark | Δmean so EKT-GL paper | Mean gain so STO | Best-of-N gap so LKH paper | Task mean tốt hơn STO | Thắng sau Holm |", "|---|---:|---:|---:|---:|---:|"]
    for benchmark in ("kro", "pr"):
        subset = [r for r in rows if r["benchmark"] == benchmark]
        average = lambda key: np.mean([r[key] for r in subset])
        better = sum(r["improvement_sto_pct"] > 0 for r in subset)
        holm = sum(r["p_holm_within_benchmark"] < .05 and r["improvement_sto_pct"] > 0 for r in subset)
        lines.append(f"| TSP-{benchmark} | {average('delta_paper_pct'):+.2f}% | {average('improvement_sto_pct'):+.2f}% | {average('gap_best_lkh_pct'):.2f}% | {better}/9 | {holm}/9 |")
    overall = np.mean([r["delta_paper_pct"] for r in rows])
    if overall > 0:
        lines += ["", "**Chưa tái lập được chất lượng nghiệm trong paper.** Chi phí trung bình cao hơn số công bố. Lợi ích của transfer so STO và mức khớp paper là hai câu hỏi khác nhau; không suy ra tái lập thành công chỉ vì EKT-GL có thể thắng STO."]
    lines += ["", "![Đối chiếu mean](results/tsp/comparison.png)", "",
              "Bảng chi tiết 18 task (mean/std, best, p-value, gap và thời gian) tại [báo cáo đầy đủ](results/tsp/REPORT.md); số liệu máy đọc được tại [CSV](results/tsp/summary.csv) và [JSON](results/tsp/summary.json). Đường hội tụ: [TSP-kro](results/tsp/kro_convergence.png), [TSP-pr](results/tsp/pr_convergence.png).", "",
              "## Khác biệt đã xác định", "",
              "**Sửa OX cho TSP.** Bản OX dùng trong CVRP cũ quét cha mẹ thứ hai từ đầu nhưng điền con từ sau điểm cắt. Vì hai vị trí không trùng nhau, ngay cả lai hai tour giống nhau cũng có thể phá cấu trúc tour. TSP nay quét và điền theo cùng thứ tự vòng, bảo đảm hai cha mẹ giống nhau sinh chính tour đó. Đã thêm test quy luật này và ví dụ kiểm tra thứ tự OX. Luồng CVRP giữ cách quét cũ; các lượt TSP trước khi sửa được tách khỏi kết quả cuối.", "",
              "**Chuẩn hóa tọa độ.** Đã sửa riêng TSP sang `(xy − xy.min()) / (xy.max() − xy.min())`, dùng một cặp min/max chung cho toàn ma trận, bảo toàn tỷ lệ hình học. Cách chuẩn hóa riêng từng trục của bản CVRP làm biến dạng TSP. Kiểm tra độc lập: optimum TSPLIB nguyên chia khoảng tọa độ chung khớp 18/18 giá trị LKH của Table IX sau làm tròn 2 chữ số. Đây là suy luận về thang đo của paper, vì tác giả không ghi công thức chuẩn hóa cụ thể. CVRP giữ nguyên hành vi cũ; các lượt TSP thang đo cũ được tách khỏi kết quả chính.", "",
              "**Cường độ local search.** Bản CVRP nhận tối đa một bước first-improvement 2-opt mỗi lần gọi; benchmark chính giữ cấu hình này. Kiểm tra riêng dưới đây dùng cùng tour nearest-neighbor, chỉ thay số bước 2-opt; không dùng nhãn tối ưu để tạo nghiệm. Đây là kiểm tra trên tour nearest-neighbor; hậu xử lý trực tiếp nghiệm EKT-GL được báo ở mục riêng phía trên.", "",
              "| Task kiểm tra | Nearest-neighbor | Sau 1 bước 2-opt | 2-opt đến hội tụ |", "|---|---:|---:|---:|"]
    for item in diagnostic["results"]:
        if item["task"] in ("kroA100", "kroA200", "pr76", "pr299"):
            lines.append(f"| {item['task']} | {item['nearest_neighbor_cost']:.3f} | {item['one_improvement_cost']:.3f} | {item['up_to_10000_improvements_cost']:.3f} |")
    lines += ["", "Kết quả chẩn đoán đủ 18 task và các route: [local_search_diagnostic.json](results/tsp/local_search_diagnostic.json). Tất cả đạt điều kiện không có bước first-improvement kế tiếp. Không coi heuristic này là baseline EMTO của paper.", "",
              "**Huấn luyện online.** `OnlineTrainer` gộp loss tất cả task rồi cập nhật Adam một lần cho mỗi epoch/transfer. Với 300 thế hệ, λ=10, epoch=1, mạng nhận 30 optimizer step. Paper ghi epoch=1 nhưng không đủ chi tiết để xác nhận batching/optimizer step trùng khớp. Đây là điểm cần ablation tiếp, không phải nguyên nhân đã chứng minh của độ lệch.", "",
              "## Giới hạn diễn giải", "",
              f"- Chỉ {runs} seed so với 20 trong paper. Best-of-{runs} khác ngân sách best-of-20; mean/std chưa ổn định. Rank-sum dùng xấp xỉ chuẩn, nên p-value với mẫu nhỏ chỉ mang tính thăm dò. Holm áp dụng riêng 9 task/bộ; chưa kết luận transfer tốt hơn một cách chắc chắn khi không qua ngưỡng đã hiệu chỉnh.",
              "- LKH và các số EKT-GL/MTEA-AST của paper được chép từ Table II/IX; không chạy lại LKH hoặc 5 baseline EMTO của tác giả. Khoảng cách với paper là mô tả, không phải kiểm định thống kê giữa hai triển khai.",
              "- STO dùng cùng toán tử nhưng có 300 thế hệ EA, EKT-GL có 270 thế hệ EA và 30 transfer. Chưa cân bằng số đánh giá hàm mục tiêu hoặc thời gian.",
              "- CPU không có CUDA. Một tiến trình tính trước STO chạy trong lúc EKT-GL thực thi, vì vậy thời gian có tranh chấp tài nguyên; không dùng để suy ra speedup so với máy RTX 3090 của paper.",
              "- Chưa chạy TSP-kro-syn, TSP-pr-syn hoặc TSP-all. Chưa có ablation về local search trong từng thế hệ và batching để giải thích hết độ lệch.", "",
              "## Chạy lại", "", "```powershell",
              "python -m pip install -r requirements-tsp.txt",
              f"python run_tsp.py --runs {runs} --generations 300",
              "python -m experiments.check_tsp_local_search",
              f"python -m experiments.refine_tsp_results --runs {runs}",
              f"python -m experiments.audit_tsp_results --runs {runs}",
              "python -m experiments.summarize_tsp",
              "python -m unittest discover -s tests -q", "```", "",
              "Chạy tiếp đủ ngân sách 20 seed của paper:", "", "```powershell",
              "python run_tsp.py --runs 20 --generations 300",
              "python -m experiments.refine_tsp_results --runs 20",
              "python -m experiments.audit_tsp_results --runs 20",
              "python -m experiments.summarize_tsp", "```", "",
              "Cache lưu route, objective, history, loss, attention, seed, config, phiên bản thư viện, hash dữ liệu và source hash. Runner kiểm tra route, tính lại chi phí và kiểm tra hội tụ trước khi dùng lại. Tăng số seed tiếp tục các seed còn thiếu nếu mã/config/dữ liệu giữ nguyên.", "",
              f"Kiểm chứng độc lập: **{audit['seed_files']} file seed, {audit['validated_tours']} tour bảng chính và {audit.get('post_validated_tours', 0)} tour hậu xử lý**; tính lại độ dài trực tiếp từ tọa độ TSPLIB float64, không dùng distance matrix hay evaluate() của solver. Sai số lớn nhất: {audit['max_independent_cost_error']:.3g} cho bảng chính và {audit.get('post_max_independent_cost_error', 0):.3g} cho hậu xử lý. Kết quả: [audit.json](results/tsp/audit.json).", ""]
    post_path = directory / "post_2opt" / "summary.json"
    if post_path.exists():
        if (audit.get("post_validated_tours") != 18*runs
                or audit.get("post_summary_sha256") != hashlib.sha256(post_path.read_bytes()).hexdigest()):
            raise ValueError("Postprocessing audit is missing or stale")
        post = json.loads(post_path.read_text(encoding="utf-8"))
        section = ["## EKT-GL + 2-opt hội tụ sau chạy", "",
                   "Lấy đúng tour EKT-GL cuối cùng của từng seed rồi chạy thêm first-improvement 2-opt đến hội tụ (tối đa 10.000 cải thiện). Đây là biến thể hậu xử lý có thêm ngân sách local search, không sửa bảng chính và không phải xác nhận tái lập cùng thiết lập paper. Không dùng kết quả này để so thắng/thua với STO chưa hậu xử lý.", "",
                   "| Benchmark | Δmean bản chính so paper | Δmean sau 2-opt so paper | Giảm mean nhờ hậu xử lý | Gap best sau 2-opt so LKH |", "|---|---:|---:|---:|---:|"]
        for benchmark in ("kro", "pr"):
            subset = [r for r in post if r["benchmark"] == benchmark]
            avg = lambda key: np.mean([r[key] for r in subset])
            section.append(f"| TSP-{benchmark} | {avg('original_delta_paper_pct'):+.2f}% | {avg('refined_delta_paper_pct'):+.2f}% | {avg('gain_postprocessing_pct'):.2f}% | {avg('best_refined_gap_lkh_pct'):.2f}% |")
        section += ["", "Bảng mean/std đủ 18 task: [post_2opt/summary.csv](results/tsp/post_2opt/summary.csv). Các tour đã tinh chỉnh và thời gian bổ sung nằm trong các JSON cùng thư mục. Tất cả nghiệm hậu xử lý cũng được kiểm chứng độc lập trong audit.json. Kết quả định lượng được phần cải thiện trên các tour đã có; chưa cho biết hiệu quả nếu thay đổi cường độ 2-opt ngay trong vòng tiến hóa.", ""]
        position = lines.index("## Khác biệt đã xác định")
        lines[position:position] = section
    legacy = ROOT / "results" / "tsp_legacy_ox" / "summary.json"
    if legacy.exists():
        before = json.loads(legacy.read_text(encoding="utf-8"))
        before = [r for r in before if r["benchmark"] == "kro"]
        after = [r for r in rows if r["benchmark"] == "kro"]
        if len(before) == 9 and all(r["runs"] == runs for r in before):
            previous = {r["task"]: r for r in before}
            gain = np.mean([100 * (1 - r["ektgl_mean"] / previous[r["task"]]["ektgl_mean"]) for r in after])
            old_delta = np.mean([r["delta_paper_pct"] for r in before])
            new_delta = np.mean([r["delta_paper_pct"] for r in after])
            section = [f"Đối chiếu hai phiên bản trên TSP-kro, cùng {runs} seed và cấu hình: sửa thứ tự quét OX làm mean EKT-GL giảm trung bình **{gain:.2f}%**; Δmean so paper từ **{old_delta:+.2f}%** xuống **{new_delta:+.2f}%**. Đây là so sánh cài đặt trước/sau sửa, không phải baseline của paper. Các file trước sửa chỉ giữ trong thư mục chẩn đoán, không được trộn vào bảng chính.", ""]
            position = next(i for i, line in enumerate(lines) if line.startswith("**Chuẩn hóa tọa độ."))
            lines[position:position] = section
    plot_comparison(rows, directory)
    target = ROOT / "ket_qua_EKT-GL_TSP.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


def plot_comparison(rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["task"] for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key, title in ((axes[0], "delta_paper_pct", "Mean cost vs paper: lower is better"),
                             (axes[1], "improvement_sto_pct", "Gain vs STO: higher is better")):
        values = [r[key] for r in rows]
        colors = ["#007f73" if (v < 0 if key == "delta_paper_pct" else v > 0) else "#bb553b" for v in values]
        axis.barh(y, values, color=colors)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.axvline(0, color="#333", linewidth=.8)
        axis.axhline(8.5, color="#888", linewidth=.7, linestyle="--")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Percent (%)")
        axis.grid(axis="x", alpha=.15)
    fig.suptitle(f"EKT-GL TSP: {rows[0]['runs']} seeds x 300 generations; exploratory study")
    fig.savefig(directory / "comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
