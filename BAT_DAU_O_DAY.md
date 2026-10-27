# Đọc paper song song với code

| Thứ tự | Paper | File cần đọc |
|---|---|---|
| 1 | Section II | `idpc_du/graph.py`, `validation.py`: đồ thị, chi phí, DU |
| 2 | Sections IV-A/B/C, Algorithm 2 | `mfea/chromosome.py`: π/σ, USS, Si, khởi tạo |
| 3 | Algorithm 1, Section IV-F | `idpc_du/gpa.py`: giải mã từng tác vụ và dựng đường đi |
| 4 | Algorithms 3/4 | `mfea/operators.py`: PMX, TPX, swap/reset |
| 5 | Section III và tài liệu [4] | `mfea/ranking.py`: rank, skill factor, scalar fitness |
| 6 | Khung EA/MFEA | `mfea/solver.py`: `solve_ea`, `solve_mfea`, vòng tiến hóa |
| 7 | Section V, Tables I–IV | `experiments/run_mfea.py`, `metrics.py`: chạy và đối chiếu |
| 8 | C-experiment, Figures 7/8 | `experiments/run_paper.py`, `_support/paper_report.py`: hội tụ, chuẩn hóa, đồ thị |

Bốn Algorithm là bốn thành phần cùng được dùng trong EA/MFEA. Hai phương pháp đem so sánh là EA đơn tác vụ và MFEA đa tác vụ.

```powershell
python -B -m app ea data/demo1.json data/demo2.json
python -B -m app mfea data/demo1.json data/demo2.json
```

Chạy đủ bài thực nghiệm:

```powershell
python -m pip install -r requirements-plots.txt
python -B -m experiments.run_paper --workers 2
```

Đọc [kiến trúc](docs/ARCHITECTURE.md), [các giả định](docs/REPRODUCE_PAPER.md) và [báo cáo thực đo](experiments/paper_run/REPORT.md). `exact.py` là công cụ kiểm chứng; `gnn/` và `integration/` là phần mở rộng, đọc sau khi đã hiểu EA/MFEA.
