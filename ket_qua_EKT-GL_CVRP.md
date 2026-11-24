# Áp dụng EKT-GL vào CVRP và so sánh với paper

Paper: *Evolutionary Knowledge Transfer via Graph Learning for Many-task Heterogeneous Routing Problems* (Lv, Liu, Li, IEEE TEVC 2025).

## 1. Thiết lập thí nghiệm

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Benchmark | VRP-a, 10 task: A-n32-k5 … A-n62-k8 | Bảng V và IX của paper |
| Kích thước quần thể mỗi task | 30 | như paper |
| Số thế hệ | 300 | như paper |
| Chu kỳ transfer λ | 10 | như paper |
| Epoch huấn luyện mỗi lần transfer | 1 | như paper |
| Số elite ε | 5 | paper không nêu, dùng mặc định của code |
| Local search | 2-opt, 1 pass mỗi lần gọi | `local_search_passes=1` |
| Số lần chạy độc lập | 20 | seed 0–19 |
| Chi phí | Tính trên tọa độ chuẩn hoá về [0,1] | như paper |
| Kiểm định | Wilcoxon rank-sum, mức 95% | EKT-GL so với STO |

**Baseline:** STO (single-task optimization) tự cài đặt, dùng cùng toán tử order crossover, swap mutation và 2-opt nhưng không có transfer. Không so với 5 thuật toán EMTO của paper (MFEA, DMFEA-II, MFCGA, AT-MFCGA, MTEA-AST) vì repo không có các thuật toán này.

## 2. Kết quả

Giá trị là mean ± std của 20 lần chạy.

| Task | EKT-GL (bản này) | EKT-GL (paper, Table V) | Δ% so với paper | STO | p | So với STO |
|---|---|---|---|---|---|---|
| A-n32-k5 | 8.88 ± 0.25 | 9.01 | -1.4 | 9.02 ± 0.44 | 0.675 | ≈ |
| A-n34-k5 | 8.82 ± 0.32 | 8.60 | +2.6 | 8.74 ± 0.29 | 0.598 | ≈ |
| A-n36-k5 | 9.29 ± 0.29 | 9.31 | -0.3 | 9.35 ± 0.28 | 0.534 | ≈ |
| A-n38-k5 | 8.79 ± 0.48 | 8.23 | +6.7 | 8.61 ± 0.26 | 0.344 | ≈ |
| A-n44-k6 | 10.45 ± 0.22 | 10.69 | -2.2 | 10.67 ± 0.21 | 0.005 | EKT-GL tốt hơn |
| A-n46-k7 | 11.00 ± 0.48 | 11.23 | -2.1 | 11.33 ± 0.37 | 0.058 | ≈ |
| A-n48-k7 | 12.60 ± 0.55 | 12.64 | -0.3 | 12.98 ± 0.67 | 0.083 | ≈ |
| A-n54-k7 | 14.39 ± 0.60 | 14.45 | -0.4 | 15.36 ± 0.64 | < 0.001 | EKT-GL tốt hơn |
| A-n60-k9 | 17.60 ± 0.61 | 16.28 | +8.1 | 18.53 ± 0.90 | 0.002 | EKT-GL tốt hơn |
| A-n62-k8 | 16.94 ± 0.84 | 15.18 | +11.6 | 18.05 ± 0.87 | 0.001 | EKT-GL tốt hơn |

Độ lệch trung bình so với paper: **+2.2%**.

### Optimality gap so với LKH (best-of-20)

Gap = (nghiệm tốt nhất trong 20 lần chạy − LKH) / LKH, tương ứng cách tính ở Table IX của paper.

| Task | LKH | Best-of-20 (bản này) | Gap bản này | Gap paper (Table IX) |
|---|---|---|---|---|
| A-n32-k5 | 8.08 | 8.32 | 2.9% | 2.57% |
| A-n34-k5 | 7.94 | 8.43 | 6.2% | 3.67% |
| A-n36-k5 | 8.50 | 8.71 | 2.5% | 2.24% |
| A-n38-k5 | 7.60 | 8.11 | 6.6% | 3.89% |
| A-n44-k6 | 9.66 | 9.92 | 2.7% | 6.21% |
| A-n46-k7 | 9.52 | 9.76 | 2.5% | 7.76% |
| A-n48-k7 | 10.95 | 11.74 | 7.2% | 12.16% |
| A-n54-k7 | 12.41 | 13.48 | 8.6% | 10.83% |
| A-n60-k9 | 14.10 | 16.44 | 16.6% | 10.18% |
| A-n62-k8 | 13.01 | 15.21 | 16.9% | 12.30% |
| **Trung bình** | | | **7.3%** | **7.2%** |

## 3. Nhận xét

**Mức độ tái lập.** Kết quả gần với số liệu paper: độ lệch trung bình so với Table V là +2,2%, và 7/10 task lệch trong khoảng ±3% (A-n32, 34, 36, 44, 46, 48, 54), trong đó 5 task thấp hơn paper nhẹ. Ba task lệch rõ là A-n38 (+6,7%), A-n60 (+8,1%) và A-n62 (+11,6%). Optimality gap trung bình so với LKH là 7,3%, gần bằng 7,2% của Table IX. Phân bố khác nhau: bản này tốt hơn paper ở nhóm task vừa (A-n44 đến A-n54) và kém hơn ở hai task lớn nhất (A-n60, A-n62).

**Hiệu quả của transfer.** So với STO, EKT-GL có mean thấp hơn ở 8/10 task. Khác biệt có ý nghĩa thống kê (Wilcoxon rank-sum, 95%) ở bốn task: A-n44, A-n54, A-n60 và A-n62, với mức cải thiện khoảng 2% ở A-n44 và 5–6% ở ba task còn lại. Ở các task nhỏ (A-n32 đến A-n38) không có khác biệt đáng kể, STO nhỉnh hơn nhẹ ở A-n34 và A-n38. Xu hướng này nhất quán với nhận định của paper rằng lợi ích của transfer rõ hơn ở task kích thước lớn.

**Nguyên nhân lệch so với paper (giả thuyết, chưa kiểm chứng bằng ablation).**

1. 2-opt trong cài đặt này chỉ thực hiện tối đa một bước cải thiện mỗi lần gọi, yếu hơn local search chạy đến hội tụ. Điều này ảnh hưởng nhiều nhất ở task lớn, đúng nơi độ lệch lớn nhất (gap A-n60, A-n62 là 16,6% và 16,9%, so với 10,2% và 12,3% của paper).
2. Paper không nêu số lượng elite ε; ở đây dùng mặc định 5.
3. Chi tiết chuẩn hoá tọa độ (min-max riêng từng trục) và khởi tạo mạng có thể khác paper.

## 4. Giới hạn

- Baseline là STO tự cài đặt, không phải 5 thuật toán EMTO của paper, nên các cột +/−/≈ của paper không so trực tiếp được. STO ở đây cũng có thể mạnh hơn các baseline trong paper (ví dụ A-n32: STO 9.02 so với MTEA-AST 9.27 trong paper), nên không nên suy ra "EKT-GL kém paper" từ việc STO thắng nhẹ ở task nhỏ.
- Chưa chạy các benchmark có task phụ (VRP-a-syn, VRP-all) nên chưa kiểm chứng nhận định rằng nhiều task hơn thì EKT-GL tốt hơn.
- Mười phép kiểm định chưa hiệu chỉnh đa so sánh. Với ngưỡng Bonferroni (0,005), A-n54, A-n60, A-n62 vẫn đạt, còn A-n44 (p = 0,005) ở sát ngưỡng.
- Việc gán lợi ích cho hiệu ứng "curriculum learning" như paper chỉ là cách giải thích phù hợp, chưa được kiểm chứng ở đây (chưa chạy ablation).
- Không báo cáo thời gian chạy: các lần chạy dùng cache, chạy song song với số worker khác nhau, nên không so sánh được với Table VIII.

## 5. Tái lập

```powershell
cd D:\GNN_MFEA
python run_vrp.py 20 300 4 6
```

- Số liệu thô từng seed: thư mục `results\` (`ektgl_0_300_<seed>.json`, `sto_0_300_<seed>.json`).
- Bảng tổng hợp: `results_vrp_a.txt`.
- Dữ liệu: Augerat Set A trong `vrp_data\`.
