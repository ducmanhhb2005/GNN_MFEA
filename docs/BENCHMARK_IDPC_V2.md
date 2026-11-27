# Chạy và kiểm tra benchmark IDPC-DU V2

Ba phương pháp chính là EA thuần, MFEA-EDU và EKT-GL-IDPC-DU. Hồ sơ `native` dùng local search cho EKT-GL. Hồ sơ `matched_ls` dùng cùng DU-aware 2-opt cho cả ba phương pháp và dừng ở trần objective requests hoặc số thế hệ. Hai hồ sơ có manifest và báo cáo riêng. [Audit](benchmark_audit.md) · [Thiết kế](design_decisions.md)

## Môi trường và chạy thử

Chạy từ thư mục gốc repo với Python hỗ trợ PyTorch. Lượt chính dùng GPA Numba và PyTorch 2.7.0+cu126 trên RTX 3050 Laptop 4 GiB. EA và MFEA tìm kiếm trên CPU; phần học và chấm điểm cạnh của EKT-GL chạy trên GPU. `--threads`, `--device` và `--gpa-backend` là một phần fingerprint và phải giữ nguyên khi resume.

```powershell
python -m pip install -r requirements-idpc-v2.txt
python -m unittest discover -s tests -q
python -m experiments.run_idpc_v2 --profile native --debug --instances data/demo1.json data/demo2.json --generations 12 --only-seeds 100 --out-dir results/idpc_v2/debug/tiny
python -m experiments.run_idpc_v2 --profile native --debug --instances data/benchmark/set1/idpc_20x10x2492.idpc data/benchmark/set1/idpc_35x35x42875.idpc data/benchmark/set2/idpc_70x140x923343.idpc data/benchmark/set2/idpc_100x200x2296097.idpc --generations 30 --only-seeds 100 --out-dir results/idpc_v2/debug/real4
python -m experiments.report_idpc_v2 results/idpc_v2/debug/real4 --preliminary
```

Các lượt đo tài nguyên cũ nằm trong `results/idpc_v2/native` (GPA Python, CPU) và `results/idpc_v2/native_cuda` (GPA Numba, CUDA, chưa cache beam). Hai bộ này có fingerprint riêng. Lượt chính bên dưới dùng code cuối với cache beam và báo lại toàn bộ seed. Trong bước mở rộng từ seed pilot, ba shard baseline và một shard EKT-GL có thể chạy đồng thời trên cùng máy; runtime mô tả điều kiện đo này.

## Pilot và lần chạy chính

Pilot dùng seed 100 trong danh sách đã đóng băng. Resume dùng lại pilot chỉ khi fingerprint code, dữ liệu và cấu hình trùng khớp. Có thể dừng sau một generation bằng `--pause-after` để kiểm tra resume; kết quả paused mang trạng thái running và không vào aggregate. Hai hoặc nhiều tiến trình chỉ được chạy song song khi tác vụ cùng manifest và tập `(method, seed)` hoàn toàn rời nhau. Nếu cần so sánh thời gian, dùng điều kiện tải thống nhất.

```powershell
python -m experiments.run_idpc_v2 --profile native --only-seeds 100 --device cuda --gpa-backend numba --out-dir results/idpc_v2/native_all42_v2
python -m experiments.run_idpc_v2 --profile native --device cuda --gpa-backend numba --out-dir results/idpc_v2/native_all42_v2 --resume
python -m experiments.report_idpc_v2 results/idpc_v2/native_all42_v2
python -m experiments.run_idpc_v2 --profile matched_ls --only-seeds 100 --device cuda --gpa-backend numba --out-dir results/idpc_v2/matched_ls_all42_v2
python -m experiments.run_idpc_v2 --profile matched_ls --device cuda --gpa-backend numba --out-dir results/idpc_v2/matched_ls_all42_v2 --resume
python -m experiments.report_idpc_v2 results/idpc_v2/matched_ls_all42_v2
```

Mỗi profile cần 60 suite runs, tổng 2.520 task runs. Mặc định script báo cáo từ chối tạo bảng khi thiếu checkpoint complete. Dùng `--preliminary` để xem coverage và số liệu tạm. Các SQLite checkpoint nội bộ dùng serialization của Python/PyTorch và chỉ nên đọc từ lần chạy của mình. Báo cáo xuất CSV raw, hội tụ, so sánh cặp, matched suite budgets và hình PNG 240 dpi. Thời gian solver được đo một lần trên cả suite và không cộng theo số task.

## Các tệp mã chính

| Tệp | Vai trò |
|---|---|
| [config.py](../ekt_gl/idpc/v2/config.py) | Thiết lập đóng băng và chính sách các profile |
| [gpa_fast.py](../ekt_gl/idpc/v2/gpa_fast.py) | Backend GPA Numba t?y ch?n, d?ng chung quy ??c CEC |
| [core.py](../ekt_gl/idpc/v2/core.py) | Gene bounds, sentinel, USS projection/lift, path encoding, tracker |
| [evolution.py](../ekt_gl/idpc/v2/evolution.py) | Binary tournament, EA và MFEA one-population |
| [local_search.py](../ekt_gl/idpc/v2/local_search.py) | Bounded DU-aware directed 2-opt |
| [learning.py](../ekt_gl/idpc/v2/learning.py) | Đặc trưng đồ thị, GCN, attention, full-edge training và greedy |
| [transfer.py](../ekt_gl/idpc/v2/transfer.py) | Lượt chuyển giao, round-trip, LS và elitist insertion |
| [checkpoint.py](../ekt_gl/idpc/v2/checkpoint.py) | Giao dịch lưu state và trace, resume theo fingerprint |
| [run_idpc_v2.py](../experiments/run_idpc_v2.py) | Nạp manifest, điều phối, kiểm chứng path cuối |
| [report_idpc_v2.py](../experiments/report_idpc_v2.py) | Coverage, thống kê và xuất CSV |
| [idpc_v2_figures.py](../experiments/_support/idpc_v2_figures.py) | Hình vẽ tái tạo từ dữ liệu báo cáo |
| [test_idpc_v2.py](../tests/test_idpc_v2.py) | Gate A cho v2 và kiểm tra resume |

Báo cáo native đo hiệu năng tổng thể khi phương pháp dùng local search khác nhau. Matched-LS giữ cùng operator và cap, nhưng MFEA vẫn tiêu ngân sách ban đầu cho 30K×K lần chấm và phân bổ phần sau theo skill factor. Trục budget trong báo cáo lấy snapshot cuối còn trong trần chung của suite. Operator 2-opt chỉ đảo thứ tự các đỉnh đang có nên không thêm hoặc bỏ đỉnh.

## Kiểm tra và tăng tốc

Gate A gần nhất có 61 kiểm thử đạt. Đã chạy smoke hai đồ thị ở 12 thế hệ và bốn đồ thị thật ở 30 thế hệ. Backend GPA Numba được đối chiếu về cost và exact IDs với GPA Python trên các multigraph sinh ngẫu nhiên, rồi chạy 336 probe độc lập trước lượt all42. Probe không tham gia khởi tạo, tối ưu hoặc huấn luyện. Cache giữ nguyên objective requests, cuộn theo thế hệ và tách beam expansions theo thuật toán khỏi số expansions thực sự phải tính.

Dùng [PyTorch CUDA 12.6](https://download.pytorch.org/whl/cu126/) để tái lập phiên bản CUDA. Có thể chọn CPU và GPA Python khi chạy nghiệm thu ở môi trường khác với output riêng. Manifest ngăn resume bằng môi trường khác hoặc dùng checkpoint cũ.
