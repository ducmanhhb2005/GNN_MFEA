# Thực nghiệm trong paper

Lệnh đầy đủ:

```powershell
python -m pip install -r requirements-plots.txt
python -B -m experiments.run_paper --workers 2
```

| Thành phần | Chức năng |
|---|---|
| `run_paper.py` | Điều phối toàn bộ Tables II–IV và Figures 7–8 |
| `run_mfea.py` | Chạy một tập EA/MFEA, checkpoint từng seed, kiểm tra đường đi |
| `metrics.py` | NIB/NIE/RPD và chuẩn hóa hội tụ theo tài liệu [4] |
| `paper_reference.json` | Số liệu/thiết lập paper và checksum PDF; không truyền đáp án vào solver |
| `_support/paper_report.py` | Xuất bảng, đồ thị và báo cáo thực đo |
| `_support/validate_artifacts.py` | Kiểm chứng đường đi, trace, thống kê, số đánh giá và hash |
| `paper_run/benchmark/` | 42 bộ × 2 phương pháp × 30 seed = 2.520 kết quả |
| `paper_run/convergence/` | Cặp 100×100 + 50×100: hai EA độc lập và một MFEA chung, 120 kết quả |
| `paper_run/REPORT.md` | Đọc kết quả tổng, chênh với paper và giới hạn tái lập |
| `train_gnn.py`, `compare_gnn.py` | Phần mở rộng GNN riêng, ngoài paper |

Cả hai thực nghiệm dùng 100 cá thể, 500 thế hệ, rmp 0,5, mutation rate 0,05, seed 100–129. MFEA chia sẻ quần thể 100 cá thể giữa các tác vụ; mỗi EA có quần thể riêng. Thời gian MFEA là thời gian chung của nhóm.

`--resume` tiếp tục với cùng mã/dữ liệu/cấu hình. `--no-plots` bỏ phụ thuộc Matplotlib nhưng vẫn xuất CSV chuẩn hóa. Thay tham số để thử nhanh thì báo cáo sẽ đánh dấu không phải toàn bộ thiết lập công bố. Không tự gọi kết quả là tái lập thành công nếu BF/AVG còn lệch.

Cách lấy RPD trung bình trong Table II không được paper mô tả đủ: số từ các bảng đã làm tròn không khớp hoàn toàn. Mã xuất cả trung bình toàn bộ và trung bình các bộ được cải thiện, giữ nguyên số paper để đối chiếu.

