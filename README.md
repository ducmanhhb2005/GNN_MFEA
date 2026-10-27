# Triển khai paper MFEA cho IDPC-DU

Repo triển khai bốn thành phần của paper CEC 2020: GPA, khởi tạo cá thể, lai ghép PMX/TPX và đột biến swap/reset. Hai phương pháp được đánh giá là **EA đơn tác vụ** và **MFEA đa tác vụ**. GNN là phần mở rộng riêng của đồ án.

## Chạy toàn bộ thực nghiệm paper

Tại `D:\GNN_MFEA`, Python 3.10+:

```powershell
# Chỉ cần cho việc vẽ biểu đồ; thuật toán EA/MFEA dùng thư viện chuẩn
python -m pip install -r requirements-plots.txt

# Cả EA/MFEA trên 42 bộ × 30 seed và C-experiment đúng cặp trong paper
python -B -m experiments.run_paper --workers 2
```

Đầu ra tập trung ở `experiments/paper_run/`: Tables II–IV, đồ thị tương ứng Figures 7–8, đường đi, hội tụ, cấu hình và kiểm chứng độc lập. Sau khi hoàn tất, đọc [báo cáo tổng](experiments/paper_run/REPORT.md).

Tiếp tục đợt chạy bị ngắt với cùng mã, dữ liệu và tham số:

```powershell
python -B -m experiments.run_paper --workers 2 --resume
```

**Triển khai đủ thí nghiệm không đồng nghĩa tái lập thành công số liệu.** Báo cáo ghi rõ những dòng khớp BF/AVG và khoảng cách với paper. Không cắt cạnh, sửa đường đi, dùng nghiệm biết trước hay GNN trong luồng này.

## Đọc code theo paper

1. [Bắt đầu ở đây](BAT_DAU_O_DAY.md), [ánh xạ paper → code](docs/ARCHITECTURE.md).
2. `idpc_du/graph.py` → `validation.py`: bài toán và ràng buộc DU.
3. `mfea/chromosome.py` → `idpc_du/gpa.py` → `mfea/operators.py`: Algorithms 1–4.
4. `mfea/ranking.py` → `mfea/solver.py`: EA/MFEA và vòng tiến hóa.
5. `experiments/run_mfea.py` → `metrics.py` → `run_paper.py`: thí nghiệm, thống kê và báo cáo.

[Hướng dẫn tái lập](docs/REPRODUCE_PAPER.md) công bố các chi tiết paper chưa mô tả đủ và mâu thuẫn giữa văn xuôi/mã giả. [Trạng thái kết quả](docs/KET_QUA_TAI_LAP_2020.md) phân biệt số đo thực tế với số tham chiếu.

## Chạy từng phần

```powershell
python -B -m app ea data/demo1.json data/demo2.json
python -B -m app mfea data/demo1.json data/demo2.json
python -B -m experiments.run_mfea --set set1 --out-dir experiments/my_set1
python -B -m unittest discover -s tests -v
```

`app ea` tối ưu từng đồ thị độc lập; `app mfea` tối ưu chung. Mặc định 100 cá thể, 500 thế hệ. `run_mfea` mặc định chạy cả EA và MFEA, 30 seed. Dữ liệu gốc gồm 42 bộ được giữ nguyên.

## GNN

GNN nằm trong `gnn/` và `integration/`. Checkpoint hiện dùng là `models/edge_gnn_synthetic.pt`, được huấn luyện trên dữ liệu tổng hợp; chưa chứng minh cải thiện benchmark của bản MFEA hiện tại. Khởi tạo GNN có điểm chạy riêng `integration.solver.solve_guided_mfea`; `solve_mfea` luôn khởi tạo ngẫu nhiên.

```powershell
python -m pip install -r requirements-gnn.txt
python -m experiments.train_gnn
python -m experiments.compare_gnn data/demo1.json data/demo2.json --runs 3
```

`exact.py` chỉ phục vụ kiểm chứng bài toán nhỏ và tạo nhãn synthetic.
