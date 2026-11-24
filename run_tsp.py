"""Reproducible TSP experiments using the same solver/operators as run_vrp.py.

python run_tsp.py --runs 20 --generations 300
Completed seed files are validated and reused; a larger --runs resumes the study.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import urllib.request

import numpy as np
import scipy
from scipy.stats import ranksums
import torch

from ekt_gl import EKTGL, EKTGLConfig, TSPTask
from ekt_gl.evolution import Population

ROOT = Path(__file__).resolve().parent
SOURCE = "https://raw.githubusercontent.com/mastqe/tsplib/master/{name}.tsp"
BENCHMARKS = {
    "kro": ["kroA100", "kroA150", "kroA200", "kroB100", "kroB150", "kroB200", "kroC100", "kroD100", "kroE100"],
    "pr": ["pr76", "pr107", "pr124", "pr136", "pr144", "pr152", "pr226", "pr264", "pr299"],
}
# Transcribed from the supplied PDF, Table II (p. 7), Table IX (p. 14).
# Each list follows Table I's instance order, NOT alphabetical order.
PAPER = {
    "kro": {
        "mean": [5.70, 7.32, 8.27, 5.97, 7.19, 8.37, 5.66, 5.80, 5.89],
        "std": [.13, .14, .18, .13, .13, .21, .15, .10, .10],
        "mtea_ast": [5.65, 7.12, 8.02, 5.93, 6.96, 8.06, 5.53, 5.78, 5.95],
        "best": [5.52, 7.06, 8., 5.71, 6.99, 8., 5.38, 5.53, 5.74],
        "lkh": [5.41, 6.69, 7.44, 5.63, 6.64, 7.47, 5.28, 5.43, 5.56],
        "gap": [2.09, 5.48, 7.57, 1.48, 5.34, 7.16, 1.98, 1.83, 3.29],
        "seconds": 88.25,
        "tsplib_rounded_optimum": [21282, 26524, 29368, 22141, 26130, 29437, 20749, 21294, 22068],
    },
    "pr": {
        "mean": [5.68, 3.94, 5.23, 8.30, 4.72, 5.45, 5.55, 7.09, 7.88],
        "std": [.07, .09, .09, .13, .11, .08, .16, .24, .37],
        "mtea_ast": [5.70, 3.93, 5.27, 8.45, 4.75, 5.47, 5.62, 6.96, 7.79],
        "best": [5.55, 3.82, 5.08, 8.07, 4.54, 5.29, 5.37, 6.59, 7.38],
        "lkh": [5.52, 3.78, 5.01, 7.77, 4.54, 5.23, 5.28, 6.20, 6.64],
        "gap": [.57, 1.10, 1.42, 3.91, .05, 1.07, 1.73, 6.29, 11.18],
        "seconds": 118.36,
        "tsplib_rounded_optimum": [108159, 44303, 59030, 96772, 58537, 73682, 80369, 49135, 48191],
    },
}


def parse_tsp(path: Path) -> np.ndarray:
    """Read EUC_2D TSPLIB coordinates; never use rounded TSPLIB tour lengths."""
    text = path.read_text(encoding="utf-8")
    dimension = re.search(r"DIMENSION\s*:\s*(\d+)", text)
    if not dimension or not re.search(r"EDGE_WEIGHT_TYPE\s*:\s*EUC_2D", text):
        raise ValueError(f"Expected EUC_2D TSPLIB data: {path}")
    if "NODE_COORD_SECTION" not in text:
        raise ValueError(f"Missing coordinate section: {path}")
    nodes = {}
    for line in text.split("NODE_COORD_SECTION", 1)[1].splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "EOF":
            break
        if len(parts) != 3:
            raise ValueError(f"Invalid coordinate row in {path}: {line}")
        node = int(parts[0])
        if node in nodes:
            raise ValueError(f"Duplicate node {node}: {path}")
        nodes[node] = (float(parts[1]), float(parts[2]))
    count = int(dimension.group(1))
    if set(nodes) != set(range(1, count + 1)):
        raise ValueError(f"Coordinate IDs do not match DIMENSION: {path}")
    xy = np.asarray([nodes[i] for i in range(1, count + 1)], dtype=np.float32)
    if not np.isfinite(xy).all():
        raise ValueError(f"Non-finite coordinates: {path}")
    return xy


def load_tasks(benchmark: str, directory: Path, download: bool = False):
    directory.mkdir(parents=True, exist_ok=True)
    tasks, inputs = [], {}
    for index, name in enumerate(BENCHMARKS[benchmark]):
        path = directory / f"{name}.tsp"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path}: use --download to fetch TSPLIB instances")
            with urllib.request.urlopen(SOURCE.format(name=name), timeout=60) as response:
                content = response.read()
            path.write_bytes(content)
        xy = parse_tsp(path)
        tasks.append(TSPTask(name, xy))
        scalar_range = float(xy.max() - xy.min())
        inputs[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source": SOURCE.format(name=name),
                        "normalization": "global_min_max",
                        "coordinate_range": scalar_range,
                        "tsplib_rounded_optimum": PAPER[benchmark]["tsplib_rounded_optimum"][index],
                        "normalized_optimum_scale_check": PAPER[benchmark]["tsplib_rounded_optimum"][index] / scalar_range}
    return tasks, inputs


def source_digest() -> str:
    sources = [Path(__file__), *sorted((ROOT / "ekt_gl").glob("*.py"))]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_seed(tasks, inputs, benchmark, mode, seed, config, output):
    identity = {"benchmark": benchmark, "mode": mode, "seed": seed,
                "config": asdict(config), "inputs": inputs, "source_sha256": source_digest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__, "scipy": scipy.__version__}}
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    path = output / "raw" / f"{benchmark}_{mode}_{seed}_{cache_key}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError(f"Cache identity mismatch: {path}")
        validate_record(record, tasks, config)
        print(f"cached {benchmark}/{mode}/seed={seed}", flush=True)
        return record
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    start = time.perf_counter()
    def progress(generation):
        if generation % 50 == 0:
            print(f"  {benchmark}/{mode}/seed={seed}: generation {generation}/{config.generations}, "
                  f"{time.perf_counter()-start:.0f}s", flush=True)
    if mode == "ektgl":
        result = EKTGL(config).solve(tasks, seed=seed, progress=progress)
        best, history = result.best, result.history
        transfers = [{"loss": r.loss, "attention": r.attention.tolist()} for r in result.transfer_reports]
    else:
        # Same initialization and operators as run_vrp.py's STO; all 300
        # generations evolve, while EKT-GL replaces migration generations.
        rng = np.random.default_rng(seed)
        populations = [Population(t, config.population_size, rng) for t in tasks]
        history = {t.name: [p.best.cost] for t, p in zip(tasks, populations)}
        for generation in range(config.generations):
            for task, population in zip(tasks, populations):
                population.evolve(config.mutation_rate, config.local_search_passes)
                history[task.name].append(population.best.cost)
            progress(generation + 1)
        best = {t.name: p.best for t, p in zip(tasks, populations)}
        transfers = []
    record = {"identity": identity, "seconds": time.perf_counter() - start,
              "best": {name: {"cost": item.cost, "route": list(item.route)} for name, item in best.items()},
              "history": history, "transfers": transfers}
    validate_record(record, tasks, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    print(f"done {benchmark}/{mode}/seed={seed}: {record['seconds']:.1f}s", flush=True)
    return record


def validate_record(record, tasks, config):
    for task in tasks:
        item = record["best"][task.name]
        cost = task.evaluate(tuple(item["route"]))  # also validates permutation
        if not np.isclose(cost, item["cost"], rtol=0, atol=1e-7):
            raise ValueError(f"Invalid saved cost: {task.name}")
        history = np.asarray(record["history"][task.name])
        if len(history) != config.generations + 1 or not np.isfinite(history).all():
            raise ValueError(f"Invalid convergence history: {task.name}")
        if np.any(np.diff(history) > 1e-7) or abs(history[-1] - cost) > 1e-7:
            raise ValueError(f"Non-elitist or inconsistent history: {task.name}")
    expected = len(range(0, config.generations, config.transfer_interval)) if record["identity"]["mode"] == "ektgl" else 0
    if len(record["transfers"]) != expected:
        raise ValueError("Incorrect number of transfers")
    for transfer in record["transfers"]:
        attention = np.asarray(transfer["attention"])
        if (not np.isfinite(transfer["loss"]) or attention.shape != (len(tasks), len(tasks))
                or not np.isfinite(attention).all() or np.any(attention < 0)
                or not np.allclose(attention.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid training loss or task attention")


def holm_adjust(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def summarize(benchmark, records):
    rows = []
    paper = PAPER[benchmark]
    for index, name in enumerate(BENCHMARKS[benchmark]):
        a = np.asarray([r["best"][name]["cost"] for r in records["ektgl"]])
        b = np.asarray([r["best"][name]["cost"] for r in records["sto"]])
        p = float(ranksums(a, b).pvalue)
        rows.append({"benchmark": benchmark, "task": name, "runs": len(a),
                     "ektgl_mean": float(a.mean()), "ektgl_std": float(a.std(ddof=0)),
                     "ektgl_best": float(a.min()), "sto_mean": float(b.mean()),
                     "sto_std": float(b.std(ddof=0)), "paper_mean": paper["mean"][index],
                     "paper_std": paper["std"][index], "paper_mtea_ast": paper["mtea_ast"][index],
                     "delta_paper_pct": float(100 * (a.mean() / paper["mean"][index] - 1)),
                     "improvement_sto_pct": float(100 * (1 - a.mean() / b.mean())),
                     "p_rank_sum": p, "paper_lkh": paper["lkh"][index],
                     "gap_best_lkh_pct": float(100 * (a.min() / paper["lkh"][index] - 1)),
                     "gap_mean_lkh_pct": float(100 * (a.mean() / paper["lkh"][index] - 1)),
                     "paper_best": paper["best"][index], "paper_gap_best20_pct": paper["gap"][index]})
    for row, adjusted in zip(rows, holm_adjust([r["p_rank_sum"] for r in rows])):
        row["p_holm_within_benchmark"] = float(adjusted)
    return rows


def write_report(all_records, output, config, runs):
    rows = [row for benchmark, records in all_records.items() for row in summarize(benchmark, records)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Áp dụng EKT-GL cho TSP và đối chiếu paper", "",
             "Nguồn tham chiếu: Lv, Liu, Li, *Evolutionary Knowledge Transfer via Graph Learning for Many-task Heterogeneous Routing Problems*, DOI 10.1109/TEVC.2025.3649317. Table I/II/IX, trang 6/7/14 của PDF người dùng cung cấp.", "",
             f"Chạy thực tế **{runs} seed (0–{runs-1}), {config.generations} thế hệ**, trên {len(rows)} task; các task cùng benchmark được giải đồng thời qua một mạng EKT-GL. Paper chạy 20 seed, 300 thế hệ.", "",
             "## Thiết lập và cách tính", "",
             f"- Population={config.population_size}; transfer interval={config.transfer_interval}; elite={config.elite_count}; epochs={config.train_epochs}; embedding={config.embedding_dim}; GCN layers={config.gcn_layers}; Adam lr={config.learning_rate}; swap mutation={config.mutation_rate}; 2-opt passes={config.local_search_passes}.",
             "- Dùng lõi `ekt_gl`, swap và first-improvement 2-opt từ thí nghiệm CVRP; sửa quét OX riêng cho TSP theo thứ tự vòng bắt đầu sau điểm cắt thứ hai. OX cũ quét từ đầu cha mẹ thứ hai nhưng điền từ điểm cắt, khiến lai hai cha mẹ giống nhau vẫn phá tour. Luồng CVRP giữ cách quét cũ để không thay đổi thí nghiệm đã có. Mỗi lần gọi 2-opt vẫn chỉ nhận tối đa một cải thiện với cấu hình mặc định.",
             "- Tối ưu tốc độ 2-opt riêng TSP: dùng delta bốn cạnh để loại các phép đảo chắc chắn không cải thiện, kèm chặn sai số tổng float32; mọi ứng viên còn lại vẫn tính objective gốc, giữ thứ tự first-improvement. Test đối chiếu với vòng lặp gốc trên tour ngẫu nhiên, tọa độ trùng, và tour vòng tròn (3–299 nút). Không thay đổi nhánh CVRP.",
             "- TSP chuẩn hóa bằng một min và max chung trên toàn bộ ma trận tọa độ của từng task: (xy − xy.min()) / (xy.max() − xy.min()). Phép biến đổi bảo toàn hình học. Đây là suy luận từ đối chiếu thang đo: optimum TSPLIB chia khoảng tọa độ chung khớp 18/18 giá trị LKH Table IX sau làm tròn 2 chữ số. Paper chỉ nói chuẩn hóa tọa độ, không ghi công thức cụ thể. CVRP trong repo vẫn dùng chuẩn hóa riêng từng trục.",
             "- Chi phí dùng Euclid liên tục trên tọa độ chuẩn hóa, gồm cạnh quay về điểm đầu. Optimum TSPLIB nguyên chỉ dùng để kiểm tra thang đo, không dùng để tính gap: gap luôn dùng cột LKH Table IX.",
             "- STO cùng khởi tạo và toán tử, không transfer. STO có 300 lượt EA; EKT-GL có 270 lượt EA + 30 lượt transfer với cấu hình paper. Đây là cùng ngân sách thế hệ, chưa phải cùng số lần đánh giá hàm mục tiêu hoặc cùng thời gian.",
             "- Seed cả NumPy và Torch; CPU, 1 Torch thread mỗi solver. Runner mặc định chạy tuần tự; trong đợt thực nghiệm này một tiến trình phụ tính trước STO trong khi EKT-GL đang chạy. Thời gian có tranh chấp tài nguyên, chỉ để tham khảo. Tính từ khởi tạo solver đến kết quả, gồm train/inference, không gồm import, tải dữ liệu hay viết báo cáo.",
             "- Std dùng ddof=0 để nhất quán báo cáo CVRP. Paper không mô tả rõ quy ước ddof.",
             "- Δpaper = 100 × (mean bản này / mean Table II − 1); âm là chi phí thấp hơn paper. Improve STO = 100 × (1 − mean EKT-GL / mean STO).",
             "- Gap best/mean = 100 × (best/mean EKT-GL / LKH Table IX − 1). LKH là số tham chiếu lấy từ paper, không phải một lần chạy LKH tại máy này. Table IX làm tròn obj đến 2 chữ số nên không tính lại cột gap paper từ các obj đã làm tròn.",
             "- Wilcoxon rank-sum hai phía (scipy.stats.ranksums), ngưỡng 0,05 như paper; bổ sung Holm cho 9 task trong từng benchmark. Seed ghép giữa hai phương pháp nhưng dùng rank-sum để bám paper. Không kiểm định với số liệu paper vì không có kết quả từng seed của tác giả.", ""]
    if runs != 20 or config.generations != 300:
        lines += ["**Đây là thí nghiệm thăm dò, chưa tái lập đủ thiết lập paper.** Số seed ít làm mean/std và p-value chưa ổn định. Đặc biệt best-of-N của bản này không cùng ngân sách với best-of-20 trong Table IX.", ""]
    for benchmark, records in all_records.items():
        subset = [r for r in rows if r["benchmark"] == benchmark]
        lines += [f"## TSP-{benchmark}", "",
                  "| Task | EKT-GL mean ± std | Paper mean ± std | Δpaper % | STO mean ± std | Improve STO % | p rank-sum | p Holm |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in subset:
            lines.append(f"| {r['task']} | {r['ektgl_mean']:.3f} ± {r['ektgl_std']:.3f} | {r['paper_mean']:.2f} ± {r['paper_std']:.2f} | {r['delta_paper_pct']:+.2f} | {r['sto_mean']:.3f} ± {r['sto_std']:.3f} | {r['improvement_sto_pct']:+.2f} | {r['p_rank_sum']:.4g} | {r['p_holm_within_benchmark']:.4g} |")
        lines += ["", "| Task | Best bản này | LKH paper | Gap best % | Gap mean % | Best paper (20 seed) | Gap paper % |", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in subset:
            lines.append(f"| {r['task']} | {r['ektgl_best']:.3f} | {r['paper_lkh']:.2f} | {r['gap_best_lkh_pct']:.2f} | {r['gap_mean_lkh_pct']:.2f} | {r['paper_best']:.2f} | {r['paper_gap_best20_pct']:.2f} |")
        mean_delta = np.mean([r["delta_paper_pct"] for r in subset])
        mean_gap = np.mean([r["gap_best_lkh_pct"] for r in subset])
        better = sum(r["improvement_sto_pct"] > 0 for r in subset)
        significant = sum(r["p_rank_sum"] < .05 and r["improvement_sto_pct"] > 0 for r in subset)
        holm = sum(r["p_holm_within_benchmark"] < .05 and r["improvement_sto_pct"] > 0 for r in subset)
        lines += ["", f"Trung bình Δpaper: **{mean_delta:+.2f}%**; gap best so LKH: **{mean_gap:.2f}%** (paper: {np.mean(PAPER[benchmark]['gap']):.2f}%). EKT-GL có mean tốt hơn STO ở **{better}/9** task, tốt hơn có ý nghĩa ở **{significant}/9** theo p thô và **{holm}/9** sau Holm.", ""]
        for mode in ("ektgl", "sto"):
            times = [r["seconds"] for r in records[mode]]
            lines.append(f"- {mode.upper()}: {np.mean(times):.1f} ± {np.std(times):.1f} giây/seed cho cả 9 task; tổng {sum(times):.1f} giây.")
        lines += [f"- Paper Table IX báo {PAPER[benchmark]['seconds']:.2f} giây cho EKT-GL trên benchmark; khác máy (paper dùng i7-8700K + RTX 3090, 64 GB RAM) và chi tiết triển khai nên không suy ra speedup từ tỷ lệ thời gian.", "",
                  f"![Hội tụ TSP-{benchmark}]({benchmark}_convergence.png)", ""]
    lines += ["## Nhận định và giới hạn", "",
              "- Chi phí càng thấp càng tốt. Kết quả đo được trong bảng quyết định mức khớp với paper; không coi chạy ra nghiệm hợp lệ là đã tái lập chất lượng thuật toán.",
              "- Đây là bản triển khai trong repo, không phải source chính thức đã xác minh của tác giả. Paper chưa nêu đủ chi tiết để khẳng định khớp: elite ε, cường độ 2-opt và một số toán tử tiến hóa. Chuẩn hóa TSP được chọn theo kiểm tra thang đo độc lập với bộ giải, không dò theo objective đầu ra của EKT-GL. Không điều chỉnh các tham số tiến hóa/huấn luyện sau khi xem kết quả.",
              "- Nếu gap lớn, cần kiểm tra cường độ local search, hội tụ quần thể và chất lượng heatmap trước khi kết luận phương pháp EKT-GL không hiệu quả. Chưa làm ablation nên các nguyên nhân này chỉ là giả thuyết.",
              "- Chưa chạy TSP-kro-syn, TSP-pr-syn, TSP-all hoặc tái cài đặt 5 baseline EMTO. Cột MTEA-AST công bố được lưu trong CSV để tham khảo; không phải kết quả chạy lại.",
              "- Best route được kiểm tra hoán vị, tính lại objective; history phải không tăng và khớp final cost. Mỗi seed lưu history, loss và attention của mọi lần transfer.", "",
              "Nguồn bảng optimum TSPLIB dùng cho kiểm tra thang đo: [tài liệu TSPLIB95 gốc, bản lưu trong TSPLib.Net](https://github.com/pdrozdowski/TSPLib.Net/blob/master/TSPLIB95/DOC.PS). Chia optimum nguyên chỉ là kiểm tra xấp xỉ vì Euclid liên tục khác EUC_2D làm tròn.", "",
              "## Tái lập và chạy tiếp", "", "```powershell",
              "python -m pip install numpy scipy torch matplotlib",
              "python run_tsp.py --runs 3 --generations 300",
              "python run_tsp.py --runs 20 --generations 300",
              "```", "",
              "Dữ liệu trong `tsp_data/`; nguồn và SHA-256 từng file trong `raw/*.json`. Mặc định không tải qua mạng; thêm `--download` nếu thiếu dữ liệu. Cache tách theo config, source hash, phiên bản thư viện và hash dữ liệu; chỉ tái dùng seed đã qua kiểm tra. Tăng --runs sẽ tiếp tục các seed còn thiếu.", "",
              "`summary.csv`/`summary.json`: toàn bộ chỉ số; `raw/*.json`: kết quả từng seed. Biểu đồ hiển thị mean và dải ±1 std của best-so-far từng task, không phải khoảng tin cậy.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def plot_convergence(all_records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for benchmark, records in all_records.items():
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        for index, (axis, name) in enumerate(zip(axes.flat, BENCHMARKS[benchmark])):
            for mode, color in (("ektgl", "#006d77"), ("sto", "#d97706")):
                traces = np.asarray([r["history"][name] for r in records[mode]])
                mean, std = traces.mean(axis=0), traces.std(axis=0)
                axis.plot(mean, label=mode.upper(), color=color, linewidth=1.4)
                axis.fill_between(np.arange(len(mean)), mean - std, mean + std, color=color, alpha=.15)
            axis.axhline(PAPER[benchmark]["mean"][index], color="#777", linestyle="--", label="Paper EKT-GL mean")
            axis.set_title(name)
            axis.set_xlabel("Generation")
            axis.set_ylabel("Normalized tour cost")
            axis.grid(alpha=.15)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"TSP-{benchmark}: {len(records['ektgl'])} seeds, mean +/- std")
        fig.savefig(output / f"{benchmark}_convergence.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tsp_data")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tsp")
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be positive")
    config = EKTGLConfig(generations=arguments.generations)
    all_records = {}
    for benchmark in dict.fromkeys(arguments.benchmarks):
        tasks, inputs = load_tasks(benchmark, arguments.data_dir, arguments.download)
        records = {mode: [] for mode in ("ektgl", "sto")}
        for seed in range(arguments.runs):
            for mode in records:
                print(f"start {benchmark}/{mode}/seed={seed}", flush=True)
                records[mode].append(run_seed(tasks, inputs, benchmark, mode, seed, config, arguments.output))
        all_records[benchmark] = records
        # A finished benchmark gets a report even if the next one is interrupted.
        write_report(all_records, arguments.output, config, arguments.runs)
        plot_convergence(all_records, arguments.output)
    print(f"Report: {arguments.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
