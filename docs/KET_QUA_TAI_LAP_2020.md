# Kết quả thực nghiệm đầy đủ CEC 2020

Đã hoàn tất bốn Algorithm thành phần và cả hai phần thực nghiệm của paper:

- Benchmark: EA đơn tác vụ và MFEA trên 42 bộ gốc, mỗi trường hợp 30 seed 100–129, tổng cộng 2.520 kết quả tác vụ.
- C-experiment: đúng cặp 100×100 + 50×100, hai EA độc lập và một MFEA chung, thêm 120 kết quả tác vụ.
- Thiết lập công bố: 100 cá thể, 500 thế hệ, rmp 0,5, mutation rate 0,05. Không cắt cạnh hay dùng GNN/oracle trong tìm kiếm.

| Phương pháp | Khớp đồng thời BF và AVG của paper | Đường đi hợp lệ |
|---|---:|---:|
| EA | 8/42 bộ | 1260/1260 |
| MFEA | 7/42 bộ | 1260/1260 |

**Đã triển khai và chạy đủ thực nghiệm, nhưng chưa tái lập thành công toàn bộ số liệu paper.** Việc dùng đủ Algorithm 1–4 và đường đi hợp lệ không bảo đảm BF/AVG khớp.

## So sánh EA và MFEA trong bản triển khai hiện tại

| Tập | MFEA tốt hơn (NIB) | Bằng nhau (NIE) | MFEA kém hơn (NIW) | RPD trung bình toàn bộ (%) |
|---|---:|---:|---:|---:|
| set1 | 14 | 9 | 1 | 0.867 |
| set2 | 12 | 0 | 6 | 0.946 |

RPD trung bình trong Table II không được mô tả đủ để khôi phục duy nhất từ các AVG đã làm tròn. Báo cáo giữ cả trung bình toàn bộ và trung bình các bộ được cải thiện, cùng số paper để đối chiếu.

## Đọc kết quả và kiểm chứng

- [Báo cáo tổng](../experiments/paper_run/REPORT.md).
- [Table II](../experiments/paper_run/table_II.csv), [Table III](../experiments/paper_run/table_III.csv), [Table IV](../experiments/paper_run/table_IV.csv).
- [Figure 7 PNG](../experiments/paper_run/figure_7.png), [SVG](../experiments/paper_run/figure_7.svg); [Figure 8 PNG](../experiments/paper_run/figure_8.png), [SVG](../experiments/paper_run/figure_8.svg).
- [BF/AVG/STD và chênh lệch](../experiments/paper_run/benchmark/summary.csv), [đường đi benchmark](../experiments/paper_run/benchmark/paths.json).
- [Báo cáo C-experiment](../experiments/paper_run/convergence/REPORT.md), [đường hội tụ thô](../experiments/paper_run/convergence/convergence.csv), [cực trị và quy tắc chuẩn hóa](../experiments/paper_run/convergence/normalization.json).
- [Kiểm chứng độc lập](../experiments/paper_run/validation.json), [chỉ số tổng](../experiments/paper_run/metrics.json).

Đã kiểm chứng đủ **2.640 đường đi hợp lệ và 2.640 trace**, thống kê BF/AVG/STD, ngân sách đánh giá, SHA256 dữ liệu 42 bộ và mã nguồn hiện tại. Các biểu đồ được xuất từ số đo thật, không làm trơn hay chọn seed có lợi.

Trước khi chạy, 39 kiểm thử đạt và 1 kiểm thử GNN bỏ qua vì thiếu PyTorch. Bản GPA tăng tốc được kiểm tra với bản chép trực tiếp Algorithm 1.

Paper còn thiếu seed, nhóm tác vụ, thứ tự cạnh và chi tiết đầy đủ của vòng tiến hóa. Văn xuôi/mã giả mâu thuẫn về gene σ. Những điểm này chưa cho phép khẳng định đây là mã gốc tác giả; các giả định nằm trong [REPRODUCE_PAPER.md](REPRODUCE_PAPER.md).

Bộ kết quả hiện hành được lưu tại `experiments/paper_run/`.
