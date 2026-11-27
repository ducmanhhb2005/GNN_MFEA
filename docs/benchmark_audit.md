# Audit benchmark IDPC-DU trước V2

Mốc `0a4e11d`. Hai bộ slide in nhiều slide trên một trang PDF. Vòng GA dựa vào slide 45–48, 53–55 của Chương 2. Mating, ranking, tournament và elitism dựa vào slide 14–20, 33–44 của Chương 6. P1 CEC 2020 mục IV, Algorithms 1–4 cung cấp genotype, GPA, toán tử và USS. P2 TEVC mục III, Equations 2–12 và Algorithm 1 cung cấp vòng EKT-GL; mục IV cung cấp thiết lập so sánh.

| Mục | Hành vi trước V2 | Bằng chứng |
|---|---|---|
| EA CEC | Một task qua vòng MFEA, 100 cá thể, 500 thế hệ, chọn cha mẹ đều | [solver](../mfea/solver.py#L40), [runner](../experiments/run_paper.py#L22) |
| STO cũ (`ea_ls_legacy`) | Tắt transfer, 30 cá thể/task, 300 thế hệ, cha mẹ chọn đều, giữ top N; LS thay cạnh song song/đi tắt, không đảo đoạn | [runner](../experiments/run_ektgl_idpc.py#L95), [population](../ekt_gl/idpc/population.py#L53), [LS](../ekt_gl/idpc/local_search.py#L24) |
| MFEA cũ | 21 cặp suy luận kề theo cỡ trong nhóm tỉ lệ miền/đỉnh; 100 cá thể tổng/cặp, 500 thế hệ, RMP 0,5, mutation 0,05, không LS; init chấm cả hai task, con chấm task thừa kế | [runner](../experiments/run_ektgl_idpc.py#L177), [grouping](../experiments/run_mfea.py#L40), [solver](../mfea/solver.py#L73) |
| Mating cũ | Cha mẹ chọn đều, tie seeded; áp p=0,05 cả sau nhánh mutation bắt buộc; định nghĩa ranking sau có hiệu lực | [mating](../mfea/solver.py#L87), [ranking](../mfea/ranking.py#L188) |
| EKT-GL cũ | 42 task dùng một mạng, pop 30/task, 300 vòng, transfer g=0,10,…,290, fallback EA khi không có elite; benchmark thêm individual | [runner](../experiments/run_ektgl_idpc.py#L29), [vòng lặp](../ekt_gl/idpc/solver.py#L64) |
| GNN cũ | 2 lớp/32 chiều, LayerNorm và một logit, sampling 512 cạnh âm và cân bằng hai lớp; greedy tie ưu tiên weight trước ID; refine trước encode | [config](../ekt_gl/idpc/config.py#L9), [network](../ekt_gl/idpc/network.py#L13), [training](../ekt_gl/idpc/training.py#L58), [decode](../ekt_gl/idpc/decoding.py#L31), [transfer](../ekt_gl/idpc/solver.py#L110) |
| Core dùng lại | Parser giữ đúng edge IDs và hướng; GPA lọc π theo giá trị ≤N và lấy σ ở đỉnh đến. Đỉnh không có cạnh ra dùng 1 làm placeholder. Validator DU độc lập | [parser](../idpc_du/graph.py#L17), [GPA](../idpc_du/gpa.py#L38), [chromosome](../mfea/chromosome.py#L21), [validator](../idpc_du/validation.py#L6) |
| Bộ đếm cũ | MFEA 25.100 GPA requests/task trung bình từ (100×2+100×500)/2 và 0 LS. STO khoảng 9.371 GPA requests gồm init, offspring, encode lại và khoảng 9.478 LS checks gồm đường gốc và các proposal được validator kiểm tra. Hai đơn vị khác nhau, gồm cache hit | [MFEA](../mfea/solver.py#L62), [STO](../ekt_gl/idpc/population.py#L48), [LS](../ekt_gl/idpc/local_search.py#L12), [artifact](../results/ektgl_idpc/REPORT.md) |
| Seeds, checkpoint | 100–119, chỉ checkpoint cuối suite-run, đối chiếu budget GPA hậu kiểm, chưa lưu state mỗi generation | [runner](../experiments/run_ektgl_idpc.py#L141), [checkpoint](../experiments/run_ektgl_idpc.py#L208) |

P1 văn xuôi mô tả chỉ số ở đỉnh đi, dòng 13 của Algorithm 1 dùng đỉnh đến. Giữ quy ước giả mã. P2 viết g=0 và thiếu bước tăng g trong giả mã. V2 dùng 1–300 và chuyển giao ở 10,20,…,300. P1 công bố cấu hình Java 100/500/30, P2 công bố 30 cá thể/task, 300 thế hệ và 20 lượt trên TSP/CVRP. Không áp kết quả lịch sử sang V2.

## Đối chiếu sau triển khai

| Hạng mục | Cách V2 xử lý |
|---|---|
| Vòng EA/MFEA | [evolution.py:43](../ekt_gl/idpc/v2/evolution.py#L43) chọn binary tournament, elitism, mutation-only đúng một lần khi chặn RMP, archive riêng theo task |
| Fitness và USS | [core.py:53](../ekt_gl/idpc/v2/core.py#L53) chiếu/lift, [core.py:118](../ekt_gl/idpc/v2/core.py#L118) đếm chung và áp trần, [evolution.py:22](../ekt_gl/idpc/v2/evolution.py#L22) rank có mask |
| CEC operators | [operators.py:6](../mfea/operators.py#L6) PMX dùng lookup thay dò tuyến tính, giữ cut, thứ tự kết quả và RNG. V2 chỉ resample gene σ có bound dương |
| DU-aware 2-opt | [local_search.py:102](../ekt_gl/idpc/v2/local_search.py#L102) đảo đoạn, giữ exact prefix/suffix, beam cạnh có hướng, chấm đường đủ, round-trip; native và matched-LS gọi cùng hàm |
| GNN V2 | [learning.py:108](../ekt_gl/idpc/v2/learning.py#L108) shared GCN/QKV/MLP; [learning.py:178](../ekt_gl/idpc/v2/learning.py#L178) cross entropy trên tất cả cạnh, tích gradient theo chunk |
| EKT-GL V2 | [run_idpc_v2.py:214](../experiments/run_idpc_v2.py#L214) transfer theo g ≥ 1; [transfer.py:11](../ekt_gl/idpc/v2/transfer.py#L11) chọn cạnh theo điểm, mã hóa, round-trip, 2-opt, đưa vào quần thể |
| Profiles và kết quả | [config.py:23](../ekt_gl/idpc/v2/config.py#L23) đóng băng tham số, [checkpoint.py:19](../ekt_gl/idpc/v2/checkpoint.py#L19) lưu state mỗi thế hệ, [report_idpc_v2.py:47](../experiments/report_idpc_v2.py#L47) kiểm tra coverage và integrity |

Encoder node dùng trung bình feature trên từng cặp đỉnh có hướng và trọng số multiplicity khi cộng message. Đây là thích nghi bộ nhớ; gate trên embedding của cặp không đồng nhất về mặt toán học với gate riêng trên từng cạnh song song. Edge embedding ở decoder kết hợp trạng thái cặp với feature từng cạnh. Cả supervised loss, score, đường đi và GPA đều giữ nguyên edge ID gốc.
