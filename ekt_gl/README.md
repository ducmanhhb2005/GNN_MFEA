# EKT-GL

Triển khai có mô-đun của **Evolutionary Knowledge Transfer via Graph Learning for Many-task Heterogeneous Routing Problems** (Lv, Liu, Li, 2025).

Luồng chính bám theo Algorithm 1 trong bài báo:

1. Mỗi task TSP/CVRP có một quần thể tiến hoá riêng (order crossover, swap mutation, 2-opt).
2. Cứ mỗi `transfer_interval` thế hệ, lấy `elite_count` cá thể tốt nhất của từng task làm nhãn cạnh.
3. GCN cạnh-có-cổng mã hoá đồ thị; self-attention hoà trộn graph embedding giữa mọi task; MLP giải mã heatmap cạnh.
4. Greedy decoder tạo route hợp lệ, 2-opt tinh chỉnh, rồi đưa route sang quần thể đích.

`torch` là dependency bắt buộc của graph-learning module. Mã lõi dùng CPU mặc định; đặt `device="cuda"` trong `EKTGLConfig` nếu có CUDA.

## Ví dụ TSP

```python
import numpy as np
from ekt_gl import EKTGL, EKTGLConfig, TSPTask

rng = np.random.default_rng(7)
tasks = [
    TSPTask("small", rng.random((20, 2))),
    TSPTask("large", rng.random((35, 2))),
]
solver = EKTGL(EKTGLConfig(population_size=30, generations=50, transfer_interval=10))
result = solver.solve(tasks, seed=7)
for name, solution in result.best.items():
    print(name, solution.cost, solution.route)
```

## Cấu trúc

| Module | Vai trò |
| --- | --- |
| `problems.py` | Mô hình TSP/CVRP, chuẩn hoá tọa độ, objective và kiểm tra route. |
| `routing.py` | OX, mutation, 2-opt, greedy heatmap decoder. |
| `evolution.py` | Quần thể độc lập và vòng đời EA cho từng task. |
| `network.py` | GCN encoder, attention fusion, MLP decoder theo Eq. (2)–(10). |
| `training.py` | Nhãn top-ε và online supervised training theo Eq. (11)–(12). |
| `solver.py` | Điều phối EKT-GL / Algorithm 1. |

CVRP dùng một hoán vị khách hàng làm genotype. Khi đánh giá/giải mã, route được tự động tách thành các depot tour hợp lệ theo capacity; do đó crossover không làm mất hay lặp khách hàng.
