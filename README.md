# Triển khai paper MFEA cho IDPC-DU

Repo triển khai bốn thành phần của paper CEC 2020: GPA, khởi tạo cá thể, lai ghép PMX/TPX và đột biến swap/reset. Hai phương pháp được đánh giá là **EA đơn tác vụ** và **MFEA đa tác vụ**.

## EKT-GL cho TSP/CVRP

Mã EKT-GL nằm trong [`ekt_gl/`](ekt_gl/README.md). Benchmark TSP dùng 18 instance
TSPLIB, tách TSP-kro và TSP-pr theo paper:

```powershell
python -m pip install -r requirements-tsp.txt
python run_tsp.py --runs 3 --generations 300
# Chạy tiếp đủ 20 seed theo paper, tái dùng cache đã kiểm chứng:
python run_tsp.py --runs 20 --generations 300
```

Báo cáo: [`results/tsp/REPORT.md`](results/tsp/REPORT.md); số liệu:
[`results/tsp/summary.csv`](results/tsp/summary.csv). Đối chiếu Table II/IX,
kèm STO, kiểm định, thời gian và đường hội tụ. Tham khảo kết quả CVRP tại
[`ket_qua_EKT-GL_CVRP.md`](ket_qua_EKT-GL_CVRP.md).

## Chạy toàn bộ thực nghiệm paper

Tại `D:\GNN_MFEA`, Python 3.10+:

```powershell
# Chỉ cần cho việc vẽ biểu đồ; thuật toán EA/MFEA dùng thư viện chuẩn
python -m pip install -r requirements-plots.txt

# Cả EA/MFEA trên 42 bộ × 30 seed và C-experiment đúng cặp trong paper
python -B -m experiments.run_paper --workers 2 --out-dir experiments/my_paper_run
```

[Báo cáo tổng](experiments/paper_run/REPORT.md) của lần chạy đã lưu gồm Tables II–IV, Figures 7–8, đường đi, hội tụ và cấu hình. Có thể [kiểm chứng bộ kết quả đã lưu qua commit Git gốc](docs/REPRODUCE_PAPER.md#kiểm-chứng-bộ-kết-quả-đã-lưu).

Tiếp tục đợt chạy bị ngắt với cùng mã, dữ liệu và tham số:

```powershell
python -B -m experiments.run_paper --workers 2 --out-dir experiments/my_paper_run --resume
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

## EKT-GL cho IDPC-DU

Mỗi đồ thị có một quần thể EA riêng. Định kỳ, mạng chung học từ các đường đi elite, dự đoán điểm cạnh và tạo ứng viên chuyển giao cho từng quần thể.

```powershell
python -m pip install -r requirements-ektgl.txt
python -m ekt_gl.idpc data/demo1.json data/demo2.json --population 6 --generations 5 --transfer-interval 2
```

Lệnh trả JSON theo từng tác vụ, gồm đường đi tốt nhất, chi phí, hội tụ, số lượt đánh giá và kết quả từng lần chuyển giao. Dùng `--set all` để nạp 42 đồ thị, hoặc thêm `--no-transfer` để chạy STO trên cùng cấu hình EA và local search. Xem [thiết kế và checklist thực nghiệm](docs/KE_HOACH_EKT_GL_IDPC_DU.md).

Để chạy benchmark bốn phương pháp với seed 100–119:

```powershell
python -m experiments.run_ektgl_idpc --set all --runs 20 --seed 100 --out-dir results/ektgl_idpc
```

Thêm `--resume` để tiếp tục từ các checkpoint đã kiểm chứng. Runner lưu đường đi, thống kê, hội tụ và đối chiếu chi phí tại ngân sách GPA chung. Đọc [báo cáo 20 seed](results/ektgl_idpc/REPORT.md).
