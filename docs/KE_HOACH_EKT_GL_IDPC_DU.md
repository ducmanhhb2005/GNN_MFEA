# Kế hoạch chuyển sang EKT-GL cho IDPC-DU

Mục tiêu là thay phương pháp dùng GNN để khởi tạo quần thể bằng phương pháp học và chuyển giao nghiệm trong quá trình tiến hóa. Mỗi đồ thị IDPC-DU có một quần thể EA riêng. Một mạng chung học từ các đường đi tốt đang có ở từng quần thể và sinh ứng viên chuyển giao cho từng tác vụ.

Thiết kế dựa trên Algorithm 1 (trang 4), kiến trúc mạng và cách tạo nhãn từ elite (trang 5) của [paper EKT-GL](https://doi.org/10.1109/TEVC.2025.3649317). Luồng TSP/CVRP trong [ekt_gl/](../ekt_gl/README.md) là điểm tham chiếu triển khai. EA và MFEA của CEC 2020 dùng làm đối chứng cho IDPC-DU.

## 1. Dọn phương pháp khởi tạo GNN cũ

**Trạng thái:** Đã gỡ pipeline cũ và cập nhật dependency, package, tài liệu và kiểm thử.

| Thành phần | Việc cần làm |
|---|---|
| `gnn/` | Xóa mô hình chấm điểm cạnh, bộ sinh dữ liệu tổng hợp và pipeline huấn luyện offline cũ. |
| `integration/` | Xóa bước đổi điểm cạnh thành ưu tiên đỉnh để tạo quần thể ban đầu và hàm `solve_guided_mfea`. |
| `models/edge_gnn_synthetic.pt` | Xóa checkpoint cũ. |
| `experiments/train_gnn.py`, `experiments/compare_gnn.py` | Xóa runner huấn luyện và đối chứng khởi tạo cũ. Xóa đầu ra riêng của hai runner nếu còn. |
| `tests/test_integration.py` | Thay các ca kiểm thử khởi tạo GNN bằng kiểm thử luồng chuyển giao IDPC-DU. Cập nhật kiểm thử ranh giới import trong `tests/test_architecture.py`. |
| `requirements-gnn.txt`, `pyproject.toml` | Cập nhật dependency cho EKT-GL-IDPC và khai báo package mới. PyTorch phục vụ luồng học online. |
| `README.md`, `BAT_DAU_O_DAY.md`, `docs/ARCHITECTURE.md`, `docs/REPRODUCE_PAPER.md`, `experiments/README.md` | Rà các chỉ dẫn về phương pháp cũ và thay bằng hướng dẫn chạy phương pháp mới ở những chỗ liên quan. |

Phạm vi dọn tập trung vào GNN khởi tạo. Bộ giải IDPC-DU, baseline EA/MFEA, bộ kiểm chứng chính xác cho đồ thị nhỏ, dữ liệu gốc và các thí nghiệm TSP/CVRP tiếp tục phục vụ đối chứng và kiểm tra hồi quy.

- [x] Xác định commit `e1b78cf` khớp toàn bộ SHA256 mã nguồn trong cả hai manifest CEC 2020, gồm hai runner GNN cũ.
- [x] Bổ sung `--source-commit` để kiểm chứng mã nguồn lịch sử. Xem [lệnh kiểm chứng](REPRODUCE_PAPER.md#kiểm-chứng-bộ-kết-quả-đã-lưu).

## 2. Triển khai EA và vòng chuyển giao

Mã đã được triển khai trong [ekt_gl/idpc/](../ekt_gl/idpc). Điểm vào là `IDPCEKTGL.solve(tasks, seed)`, với `tasks` là ánh xạ từ tên sang đồ thị. Chạy thử hai đồ thị từ thư mục repo bằng lệnh sau.

```powershell
python -m pip install -r requirements-ektgl.txt
python -m ekt_gl.idpc data/demo1.json data/demo2.json --population 6 --generations 5 --transfer-interval 2
```

Dùng `--set all` để chọn cả 42 đồ thị. Đầu ra JSON in thứ tự task theo các khóa trong `best` và dùng cùng thứ tự cho các hàng và cột attention. Các trường hợp chưa tìm được đường đi có chi phí `null` và danh sách cạnh rỗng.

1. Khởi tạo ngẫu nhiên một quần thể cho mỗi đồ thị. Với 42 tác vụ và 30 cá thể mỗi tác vụ, có 42 quần thể và tổng cộng 1.260 cá thể.
2. Ở thế hệ tiến hóa thường, mỗi quần thể dùng chromosome `(π | σ)`, các toán tử PMX/TPX, đột biến swap/reset và GPA của IDPC-DU để sinh và đánh giá con. Tinh chỉnh các đường đi hợp lệ bằng local search, rồi chọn theo chi phí trên chính tác vụ đó.
3. Ở thế hệ chuyển giao, lấy tối đa ε đường đi elite hợp lệ của từng quần thể. Theo lịch trong Algorithm 1 của EKT-GL, nhánh chuyển giao chạy khi `g % λ == 0`, với bộ đếm `g` bắt đầu từ 0.
4. Tạo nhãn cho từng cạnh gốc. Cạnh xuất hiện trong ít nhất một đường đi elite có nhãn 1. Các cạnh khác có nhãn 0. Nếu một tác vụ chưa có elite hợp lệ, bỏ tác vụ đó khỏi loss ở lượt này. Nếu cả nhóm chưa có elite hợp lệ, bỏ lượt học và tiếp tục tiến hóa.
5. Huấn luyện encoder đồ thị, attention giữa các tác vụ và bộ dự đoán cạnh trên các nhãn hiện tại. Duy trì mạng và trạng thái Adam qua các lần chuyển giao trong cùng một seed. Mỗi seed bắt đầu với mạng mới.
6. Dùng điểm cạnh của từng tác vụ để sinh đường đi chuyển giao cho chính đồ thị đó. Sau local search và kiểm tra tính hợp lệ, xét đưa ứng viên vào quần thể đích. Duy trì kích thước quần thể và lưu riêng đường đi tốt nhất đã tìm được cho từng task.

Nhãn đến từ các quần thể đang tiến hóa trong cùng lượt chạy. Nghiệm EA đã lưu và số liệu tham chiếu chỉ dùng khi phân tích kết quả. Thiết kế này đánh giá hiệu quả chuyển giao online trên các tác vụ đang giải.

## 3. Thích nghi cho đồ thị IDPC-DU

| Phần | Thiết kế |
|---|---|
| Đầu vào mạng | Dùng cấu trúc có hướng, cờ nguồn/đích, bậc vào/ra và đặc trưng trọng số, miền. Đặc trưng cạnh gồm trọng số nén log, tần suất miền và trung bình trọng số nén log của miền. Điểm và nhãn gắn với `edge_id` để giữ riêng các cạnh song song. |
| Suy luận | Tận dụng module attention của EKT-GL. Dự đoán điểm cho cạnh có thật, với đặc trưng riêng cho từng cạnh. |
| Giải mã | Bắt đầu từ nguồn. Chọn cạnh có điểm cao trong số cạnh đi đến đỉnh chưa thăm và thuộc miền chưa rời. Dừng khi tới đích. Nếu bị kẹt, ghi nhận thất bại chuyển giao của tác vụ đó. |
| Local search | Thử thay bằng cạnh song song rẻ hơn hoặc rút ngắn đoạn đường bằng một cạnh có hướng phù hợp. Chấp nhận khi đường đi mới vẫn thỏa DU và giảm chi phí. Áp dụng cùng quy tắc cho con sinh ra trong EKT-GL và STO. Nếu đường đi tinh chỉnh không thể mã hóa thành chromosome, giữ con trước tinh chỉnh. |
| Đưa nghiệm vào EA | Mã hóa đường đi thành `(π | σ)`. Chạy GPA kiểm tra lại danh sách ID cạnh và chi phí. Loại và đếm riêng ứng viên không thể biểu diễn đúng trong giới hạn gene hiện tại. Chỉ tiếp nhận ứng viên hợp lệ, khác nghiệm đã có và tốt hơn cá thể kém nhất. |
| Bộ nhớ | Toàn bộ 42 đồ thị có khoảng 12,27 triệu cạnh. Một tensor FP32 kích thước số cạnh × 128 đã chiếm khoảng 5,85 GiB. Encoder gộp thông tin cạnh song song theo cặp đỉnh có hướng. Decoder vẫn chấm từng cạnh gốc theo khối khi giải mã. Mỗi lượt học giữ toàn bộ cạnh dương và lấy tối đa 512 cạnh âm ngẫu nhiên mỗi tác vụ, rồi cân bằng trọng số hai lớp. Trình chạy lưu các cạnh và hai chỉ mục trong mảng NumPy. Mỗi đồ thị được đọc lần lượt bằng parser hiện có, sau đó chuyển sang mảng, giữ ID, trọng số và thứ tự các cạnh song song. |

Mô-đun `task.py` cung cấp `CompactIDPCGraph`. Cả `Graph` và `CompactIDPCGraph` dùng được với `population.py` và các toán tử IDPC-DU sẵn có. `features.py`, `network.py`, `training.py` xử lý học online. `decoding.py` và `local_search.py` tạo, cải thiện và mã hóa ứng viên. `solver.py` điều phối lịch chạy, `config.py` đặt tham số và `__main__.py` chạy một seed từ terminal. Cấu hình mạng khởi đầu dùng hai lớp, embedding 32 chiều. Local search áp dụng cho ứng viên chuyển giao và con ở các thế hệ tiến hóa. Thêm `--no-transfer` để chạy đối chứng STO với cùng toán tử và local search. Runner nhiều seed và đối chứng học riêng từng tác vụ được mô tả ở mục 4.

### Số đo chạy thử

Các lượt dưới đây chạy trên CPU với một luồng PyTorch, seed 7, 4 cá thể mỗi tác vụ và 2 thế hệ. Lượt đầu chuyển giao và lượt sau tiến hóa. RAM cực đại là của cả tiến trình, gồm Python, PyTorch và giai đoạn đọc dữ liệu.

| Nhóm | Số cạnh | Mảng cạnh và chỉ mục (MiB) | RAM cực đại (MiB) | Nạp dữ liệu (s) | Giải (s) |
|---|---:|---:|---:|---:|---:|
| `idpc_10x5x425` | 425 | 0,015 | 259,8 | < 0,01 | 1,00 |
| `idpc_100x200x2296097` | 2.296.097 | 78,8 | 1.010,6 | 7,62 | 1,54 |
| Cả 42 đồ thị | 12.268.131 | 421,2 | 1.157,7 | 93,0 | 14,2 |

Lượt chung 42 tác vụ trả đủ 42 đường đi hợp lệ và 42 ứng viên được chấp nhận ở lần chuyển giao. Một ứng viên được chấp nhận khi hợp lệ, khác các nghiệm đang có và tốt hơn cá thể kém nhất. Kết quả kiểm tra này dùng để xác nhận khả năng chạy hết tập dữ liệu trước benchmark nhiều seed.

```powershell
python -m ekt_gl.idpc --set all --population 4 --generations 2 --seed 7
```

## 4. Thực nghiệm và kiểm thử

Cấu hình xuất phát là **30 cá thể mỗi tác vụ, 300 thế hệ, λ = 10, 1 epoch mỗi lần chuyển giao, 20 seed**. Đề xuất **ε = 5** để bắt đầu. Đây là lựa chọn cấu hình cho bản thích nghi, cần lưu cùng kết quả.

| Phương pháp | Mục đích |
|---|---|
| EA độc lập (STO) | Đo hiệu quả của cùng quần thể, toán tử và local search khi tắt chuyển giao. |
| Học online riêng từng tác vụ | Đo tác dụng của hướng dẫn học được trên chính tác vụ đó. Dùng cùng lịch học, cách tạo nhãn và tiếp nhận nghiệm. |
| EKT-GL-IDPC-DU | Đo hiệu quả học và hợp nhất thông tin qua mạng chung cho nhiều tác vụ. |
| MFEA CEC 2020 | Đối chứng với cơ chế chuyển giao qua lai ghép trong quần thể chung. Ghi riêng cấu hình và ngân sách đánh giá khi so sánh. |

Runner benchmark nằm ở [experiments/run_ektgl_idpc.py](../experiments/run_ektgl_idpc.py). Chạy từ thư mục repo.

```powershell
python -m pip install -r requirements-ektgl.txt
python -m experiments.run_ektgl_idpc --set all --runs 20 --seed 100 --out-dir results/ektgl_idpc
```

Thêm `--resume` để kiểm tra và dùng lại các checkpoint đã xong, với cùng tập đồ thị, mã nguồn, phiên bản thư viện và cấu hình. Mỗi checkpoint chứa toàn bộ kết quả của một phương pháp trên một seed. Runner xuất báo cáo sau mỗi seed đủ các phương pháp yêu cầu.

- EKT-GL, STO và học riêng dùng cùng seed khởi tạo cho mỗi tác vụ, cùng toán tử EA và local search. Học riêng tạo một mạng và optimizer độc lập cho từng đồ thị. EKT-GL chạy mạng chung cho cả nhóm.
- MFEA dùng mặc định 100 cá thể chung cho mỗi cặp, 500 thế hệ, `rmp = 0,5` và mutation rate `0,05`. Cặp tác vụ mặc định lấy theo cách ghép của bộ chạy CEC 2020. Manifest ghi từng cặp và cả hai cấu hình.
- Bảng so sánh theo ngân sách lấy mức GPA chung nhỏ hơn cho từng cặp cùng tác vụ và seed, rồi dùng mốc thế hệ cuối cùng không vượt mức đó. Cột số lượt ở từng mốc cho thấy độ chênh do lấy mẫu theo thế hệ. Kiểm tra local search được thống kê riêng.
- Benchmark mặc định lưu tối đa 2.048 mục cache mỗi tác vụ để tái sử dụng các phép giải mã hoặc tinh chỉnh giống hệt. Số lượt yêu cầu đánh giá và số kiểm tra theo thuật toán vẫn tính cả lượt lặp. Chọn `--cache-size 0` và thư mục mới để đo thời gian khi tắt cache.
- `summary.csv` ghi BF, AVG, độ lệch chuẩn mẫu và tỷ lệ khả thi. AVG để trống nếu có seed không khả thi. `runs.csv` ghi đường đi và kết quả chuyển giao. `convergence.csv` ghi hội tụ theo thế hệ. `matched_budgets.csv` ghi so sánh theo ngân sách. `timings.csv` ghi thời gian tổng, huấn luyện và suy luận kèm giải mã. Chi phí nạp dữ liệu và kiểm chứng đầu ra nằm ngoài thời gian thuật toán.

- [x] Kiểm thử cạnh song song, cạnh có hướng, lặp đỉnh, chuỗi miền `A → A → B` hợp lệ và `A → B → A` không hợp lệ.
- [x] Kiểm thử nhãn từ hợp các đường đi elite, trường hợp chưa có elite hợp lệ, giải mã bị kẹt và mã hóa rồi giải mã lại đúng đường đi.
- [x] Chạy demo nhỏ với ít nhất hai tác vụ. Xác nhận có lượt học và sinh ứng viên thật, lịch chuyển giao đúng, quần thể luôn đủ kích thước sau chọn lọc và chi phí tốt nhất không tăng.
- [x] Đo bộ nhớ và thời gian trên cả đồ thị nhỏ và một đồ thị lớn. Chạy thử đủ 42 tác vụ trong một nhóm với ít thế hệ trước khi chạy nhiều seed.
- [x] Chạy benchmark đủ 42 tác vụ. Lưu đường đi, BF, AVG, độ lệch chuẩn, tỷ lệ khả thi, hội tụ, thời gian gồm huấn luyện và suy luận, số đánh giá và số ứng viên chuyển giao được chấp nhận hoặc bị loại.
- [x] Cài đặt so sánh bằng cùng seed và ghi ngân sách theo từng tác vụ. Vì một thế hệ chuyển giao thay cho một thế hệ EA, đối chiếu thêm theo số lượt đánh giá nghiệm. Ghi cả số lần kiểm tra ứng viên trong local search để làm rõ chi phí tính toán.
- [x] Tích hợp kiểm chứng đường đi cuối bằng `validate_path()` và kiểm thử hồi quy quá trình chạy, lưu, tiếp tục từ checkpoint và đối chiếu ngân sách.

### Kết quả benchmark 42 tác vụ

Đã chạy 20 seed từ 100 đến 119 cho bốn phương pháp. Kết quả gồm 80 checkpoint và 3.360 đường đi hợp lệ. Mỗi dòng dưới đếm 840 so sánh theo cùng tác vụ và seed. Chi phí nhỏ hơn được tính là thắng.

| Đối chứng | EKT-GL thắng / hòa / thua ở kết quả cuối | Tại ngân sách GPA chung |
|---|---:|---:|
| STO | 11 / 811 / 18 | 11 / 811 / 18 |
| Học riêng từng tác vụ | 2 / 824 / 14 | 2 / 824 / 14 |
| MFEA | 128 / 689 / 23 | 250 / 572 / 18 |

EKT-GL tiếp nhận 753 trong 25.200 lượt đề xuất chuyển giao. Phần lớn kết quả hòa với STO và học riêng. Trong số các trường hợp có chênh lệch ở kết quả cuối, STO và học riêng thắng nhiều hơn. Kết quả với MFEA phản ánh cả khác biệt về chuyển giao, cấu trúc quần thể và local search. Đối chứng học riêng giữ cùng cơ chế học và tiến hóa để đánh giá đóng góp của việc học chung giữa các tác vụ.

Các seed được chạy song song. Thời gian ghi nhận có thể chịu ảnh hưởng tranh tài nguyên. Với một số seed MFEA, thời gian là tổng thời gian đo trên các nhóm tác vụ được chạy ở hai tiến trình. Muốn so sánh tốc độ trong điều kiện tuần tự, chạy lại từng phương pháp với cùng môi trường và mức tải.

Kiểm chứng độc lập đã tính lại 3.360 đường đi từ file `.idpc` gốc, đối chiếu 168 dòng thống kê và 2.520 so sánh theo ngân sách GPA. Cả 840 kết quả MFEA khớp kết quả lưu của CEC 2020 về chi phí, ID cạnh, hội tụ và tổng lượt đánh giá. [Kết quả kiểm chứng](../results/ektgl_idpc/validation.json)

Mở [báo cáo benchmark](../results/ektgl_idpc/REPORT.md), [thống kê theo tác vụ](../results/ektgl_idpc/summary.csv) và [bảng theo ngân sách GPA](../results/ektgl_idpc/matched_budgets.csv).

## 5. Phân việc song song và mốc bàn giao

Chốt trước hợp đồng giữa các phần: đầu vào theo đồ thị và `edge_id`, đầu ra mạng là điểm cạnh, ứng viên chuyển giao gồm đường đi và chi phí, runner nhận kết quả theo task và seed.

| Người | Việc chính | Có thể bắt đầu bằng |
|---|---|---|
| P1 | Quần thể EA riêng, giải mã, local search và mã hóa nghiệm IDPC-DU | Đồ thị nhỏ và vector điểm cạnh giả lập. |
| P2 | Đặc trưng, mạng, nhãn elite và huấn luyện online | Đường đi elite mẫu và ID cạnh đã biết. |
| P3 | Điều phối lịch học và chuyển giao, ghi nhận ứng viên | Mạng giả và quần thể thử theo giao diện đã chốt. |
| P4 | Kiểm kê phần cũ, xử lý kiểm chứng lịch sử, runner và báo cáo | Kết quả giả lập theo định dạng runner. |

1. **Ghép demo** sau khi P1, P2 và P3 bàn giao các phần theo giao diện đã chốt. P4 có thể chuẩn bị runner và thống kê từ trước.
2. **Đo tài nguyên** sau khi demo chạy được với mạng thật. Dùng một đồ thị lớn để kiểm tra bộ nhớ trước khi chạy nhóm 42 tác vụ.
3. **Chạy chính thức** sau khi nhóm 42 tác vụ chạy thử thành công. P4 điều phối các lượt chạy, trong khi P1–P3 viết mô tả phương pháp và phân tích các lần chuyển giao thành công hoặc thất bại.
4. **Chốt tài liệu và kết quả** sau khi đối chiếu các baseline, kiểm tra lại mọi đường đi và xác nhận các lệnh cho phương pháp cũ đã được gỡ.
