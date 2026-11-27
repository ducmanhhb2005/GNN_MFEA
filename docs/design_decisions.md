# Quyết định thiết kế benchmark V2

| Hạng mục | Thiết lập |
|---|---|
| Suite | 42 file gốc theo thứ tự ổn định, giữ hash, cạnh, source/target |
| Cấu hình | 30/task, 300 thế hệ, 20 seed 100–119 theo manifest cũ, cùng seed giữa ba phương pháp. MFEA một quần thể tổng 30K |
| EA | Binary tournament theo cost, sinh N con, elitism hợp cha mẹ và con. Hòa ưu tiên thứ tự quần thể |
| Shared core | P1 Algorithms 1–4 và §IV-F. π lọc theo giá trị khi chiếu USS, σ prefix. Sentinel −1 được đọc như 1 chỉ ở GPA theo quy ước lịch sử. P1 dùng modulo theo file order |
| MFEA | RMP 0,5, lai PMX/TPX, thừa kế skill, con chỉ chấm task thừa kế. Nhánh chặn lai chỉ đột biến bắt buộc một lần. Các con lai có p=0,05 đột biến hai đoạn. Hòa rank theo thứ tự ứng viên, hòa skill ưu tiên skill cũ |
| LS | Đảo đoạn nội bộ, 16 cuts, beam 8, 4096 edge expansions/cut, 4 paths/cut, 1 pass. Giữ exact prefix/suffix, chọn cải thiện nghiêm ngặt đã round-trip. Lift USS giữ gene khác task và bỏ factorial cost stale |
| Profiles | Native EA và MFEA không LS; EKT-GL ordinary và pre-transfer LS. Matched-LS cùng operator cho ba phương pháp, chặn 30K×301 objective requests hoặc 300 thế hệ |
| EKT-GL | 4 lớp, 128 chiều, BatchNorm, shared attention, MLP 2 logits. Adam 0,001, 1 epoch/update, 5 elite khả thi mỗi task, gồm bản trùng; mean loss theo cạnh rồi theo graph, không sampling |
| Lịch | Chuyển giao 10,20,…,300 thay ordinary generation. Greedy tie theo edge ID. Insertion elitist ưu tiên incumbent khi hòa |
| Trace | Cache GPA giới hạn theo task và generation; mỗi cache hit vẫn là request. Checkpoint SQLite giao dịch nguyên tử, fingerprint nguồn, cấu hình và bộ dữ liệu. Archive ghi ứng viên đã chấm và kiểm chứng |
| Thống kê | Mann–Whitney U hai phía, alpha 0,05, Holm trên toàn bộ task × cặp mỗi profile. Rank-biserial định hướng A tốt hơn. Ghi feasibility riêng và đánh dấu task thiếu mẫu khả thi |
