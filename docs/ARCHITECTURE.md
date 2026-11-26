# Kiến trúc và ánh xạ paper → code

```text
GNN_MFEA/
├── idpc_du/
│   ├── graph.py                # Đồ thị và parser dữ liệu gốc
│   ├── validation.py           # Kiểm tra đường đi và Domain Uniqueness
│   ├── gpa.py                  # Algorithm 1; projection ở Section IV-F
│   └── exact.py                # Bộ giải chính xác để kiểm chứng đồ thị nhỏ
├── mfea/
│   ├── chromosome.py           # Sections IV-A/B/C, Algorithm 2
│   ├── operators.py            # Algorithms 3/4: PMX, TPX, swap/reset
│   ├── ranking.py              # Factorial rank, skill factor, scalar fitness
│   └── solver.py               # EA/MFEA; luôn khởi tạo ngẫu nhiên
├── ekt_gl/                     # EKT-GL cho TSP/CVRP và IDPC-DU
├── experiments/
│   ├── run_paper.py            # Toàn bộ Tables II–IV và Figures 7–8
│   ├── run_mfea.py             # Chạy EA/MFEA, đối chiếu Tables III/IV
│   ├── metrics.py              # NIB/NIE/RPD và chuẩn hóa hội tụ
│   ├── paper_reference.json    # Số liệu paper, chỉ dùng khi báo cáo
│   └── _support/              # Kiểm chứng, xuất bảng và đồ thị
├── app/                        # python -m app mfea / exact
├── tests/                      # Kiểm thử bài toán, toán tử và tích hợp
├── data/                       # Demo và 42 bộ benchmark gốc
└── docs/                       # PDF, hướng dẫn, nguồn tham khảo
```

## Luồng MFEA

`Graph` → USS và khởi tạo ngẫu nhiên → GPA tính chi phí → xếp hạng → chọn cha mẹ → lai ghép/đột biến → đánh giá trên skill kế thừa → chọn lọc → lặp 500 thế hệ.

GPA giữ thứ tự cạnh trong file, chọn đỉnh chưa thăm có ưu tiên cao nhất và dùng `σ[đỉnh đến] % số cạnh hợp lệ` làm chỉ số Python. Đây là Algorithm 1 dòng 13. Miền chỉ bị cấm sau khi đường đi đã rời miền đó; đi liên tiếp nhiều cạnh cùng miền là hợp lệ.

## Phụ thuộc

- `idpc_du` cung cấp đồ thị, bộ kiểm chứng đường đi và GPA. GPA nhận đối tượng có `priority` và `edge_index`.
- `mfea` dùng đồ thị và GPA để tiến hóa từ quần thể khởi tạo ngẫu nhiên theo Algorithm 2.
- `ekt_gl` chạy TSP/CVRP. `ekt_gl/idpc/` dùng GPA và toán tử IDPC-DU trong các quần thể EA riêng, với một mạng chung học và chuyển giao nghiệm online. Xem [thiết kế EKT-GL-IDPC-DU](KE_HOACH_EKT_GL_IDPC_DU.md).

Chi tiết chọn cha mẹ, xử lý đồng hạng và ghép tác vụ chưa được paper công bố đầy đủ. Các lựa chọn đang dùng được ghi trong [REPRODUCE_PAPER.md](REPRODUCE_PAPER.md); chưa khẳng định là mã gốc tác giả.

## EA, thống kê và C-experiment

`solve_ea(graph, ...)` dùng cùng biểu diễn/toán tử với một tác vụ độc lập. `solve_mfea(graphs, ...)` dùng USS và kế thừa skill để chuyển giao. Hai điểm chạy đều được `run_mfea.py` sử dụng trong đối chứng.

`run_paper.py` chạy đủ benchmark rồi chạy cặp tác vụ được nêu rõ trong Figures 7/8. Thống kê chỉ đọc các kết quả đã sinh: BF/AVG, NIB/NIE/RPD, số đánh giá và min/max mục tiêu để chuẩn hóa. Bộ tối ưu không import module báo cáo và không đọc số paper.

Mỗi lần chạy lưu số đánh giá và cực trị của mọi giá trị mục tiêu hữu hạn. Dữ liệu này chỉ phục vụ công thức chuẩn hóa sau khi tìm kiếm kết thúc. Bảng RPD xuất hai kiểu trung bình vì Table II không nêu rõ tập được lấy trung bình.
