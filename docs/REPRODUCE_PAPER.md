# Chạy lại MFEA trong paper CEC 2020

Paper: *Multifactorial Evolutionary Algorithm for Inter-Domain Path Computation under Domain Uniqueness Constraint*, DOI 10.1109/CEC48606.2020.9185701. Đọc Section IV để đối chiếu thuật toán, Section V-C cho thiết lập và Tables III/IV ở trang PDF 7 cho số liệu.

## Thành phần được giữ

- Hai đoạn Node Priority π và Edge Index σ; USS lấy kích thước/giới hạn cạnh song song lớn nhất giữa các tác vụ.
- Khởi tạo π bằng hoán vị ngẫu nhiên, σ ngẫu nhiên trong giới hạn Si: Algorithm 2.
- Giải mã tác vụ bằng lọc ổn định các giá trị π ≤ Nj và lấy Nj phần tử σ đầu: Section IV-F.
- GPA: Algorithm 1, theo dõi đỉnh đã thăm và miền đã rời.
- PMX cho π, TPX cho σ với hai lần chọn điểm cắt: Algorithm 3.
- Swap hai vị trí π, reset một vị trí σ: Algorithm 4.
- Factorial rank, skill factor, scalar fitness và chuyển giao giữa tác vụ theo khung MFEA.

Không có cắt cạnh, local search, sửa đường đi, seed từ nghiệm biết trước hay GNN trong lệnh tái lập. Toàn bộ cạnh dữ liệu gốc được giữ.

## Lựa chọn triển khai cần ghi rõ

1. **Mâu thuẫn σ:** phần văn xuôi nói gene của đỉnh đi; Algorithm 1 dòng 13 dùng gene của đỉnh đến. Mã hiện tại cố định theo Algorithm 1: `parallel[sigma[v] % len(parallel)]`.
2. **Thứ tự cạnh:** paper không chỉ rõ thứ tự của tập cạnh song song. Mã giữ thứ tự file gốc; không sắp theo trọng số hay miền.
3. **Si = 0:** Algorithm 2 ghi dấu `'-'`, nhưng Algorithm 1 lại có thể đọc gene của đỉnh đến là một đỉnh không có cạnh đi ra. Mã lưu số 1 làm giá trị thay thế để phép modulo xác định. Đây là quy ước xử lý trường hợp paper chưa giải thích.
4. **Khung tiến hóa:** quần thể đầu được đánh giá trên mọi tác vụ; con chỉ đánh giá trên skill kế thừa. Cha mẹ chọn đều ngẫu nhiên; con lai kế thừa ngẫu nhiên một skill của cha mẹ. Nếu không lai giữa hai skill khác nhau, đột biến từng cha mẹ. Sau đó mỗi con có xác suất đột biến 0,05. Xếp hạng theo chi phí từng tác vụ, phá đồng hạng ngẫu nhiên; chọn 100 cá thể tốt nhất theo scalar fitness từ cha mẹ + con. Paper không cung cấp đủ mã giả để xác nhận mọi chi tiết của vòng này.
5. **Nhóm tác vụ:** `adjacent` ghép kích thước kề nhau, cùng tập dữ liệu và tỉ lệ miền/đỉnh. Ví dụ 10x5 + 15x7. Đây là suy luận từ thời gian giống nhau trong bảng, không phải protocol đã được tác giả xác nhận. Dùng `--instances` để ghi rõ cặp khác.
6. **Ngẫu nhiên:** mặc định seed 100–129. Paper không công bố seed; Python và Java tạo chuỗi ngẫu nhiên khác nhau.

Vì vậy đây là bản triển khai có đối chiếu paper và có công bố giả định. Chưa thể bảo đảm trùng số liệu paper chỉ bằng cách loại bỏ các phần mở rộng.

## Chạy đầy đủ paper

Tại `D:\GNN_MFEA`, Python 3.10+, không cần PyTorch:

```powershell
python -m pip install -r requirements-plots.txt
python -B -m experiments.run_paper --workers 2
```

Lệnh thực hiện:

1. EA trên từng bộ và MFEA trên 21 cặp suy luận: 42 bộ × 2 phương pháp × 30 seed = 2.520 kết quả.
2. C-experiment trên đúng cặp `idpc_100x100x1000000.idpc` + `idpc_50x100x285357.idpc`: hai EA riêng, một MFEA chung, 120 kết quả.
3. Kiểm chứng độc lập đủ đường đi, trace, BF/AVG/STD, số đánh giá và SHA256.
4. Xuất Tables II–IV, CSV hội tụ thô/chuẩn hóa, Figures 7–8 dạng PNG/SVG và báo cáo tổng.

Đầu ra ở `experiments/paper_run/`. `--resume` tiếp tục đúng cấu hình nếu bị ngắt. `--no-plots` chỉ xuất số liệu, không cần Matplotlib. Mặc định 30 seed 100–129, 100 cá thể, 500 thế hệ, rmp 0,5, mutation rate 0,05. Thay tham số sẽ được báo cáo là chưa dùng đủ thiết lập paper.

Chạy riêng benchmark:

```powershell
python -B -m experiments.run_mfea --set all --out-dir experiments/my_benchmark
```

Mặc định chạy cả EA và MFEA. Có thể chỉ định `--methods ea mfea`, `--set set1|set2|all`, `--instances`, `--runs`, `--seed`, `--workers`. Thay mã/dữ liệu/tham số cần đầu ra mới; runner không trộn checkpoint khác cấu hình.

## Chuẩn hóa C-experiment

Paper dẫn tài liệu [4], Gupta/Ong/Feng, *Multifactorial Evolution: Toward Evolutionary Multitasking*, DOI 10.1109/TEVC.2015.2458037. [Bản tác giả, trang 10](https://memetic-computing.org/publication/journal/Multifactorial_Evolution_ONLINE_VERSION.pdf) định nghĩa:

`f_tilde_j = (f_j - f_min_j) / (f_max_j - f_min_j)`.

Cực trị được lấy trên các lần chạy của cả EA và MFEA. Mã ghi min/max của **mọi mục tiêu ứng viên hữu hạn được đánh giá**, gồm khởi tạo; chuẩn hóa best-so-far sau khi chạy, rồi lấy trung bình theo tác vụ và seed. Không dùng số tối ưu trong paper làm cận hướng dẫn tìm kiếm.

Nếu khoảng min–max bằng 0, quy ước đường chuẩn hóa là 0. Lần chưa có nghiệm hợp lệ giữ giá trị thiếu và làm trung bình tương ứng không xác định; không âm thầm bỏ lần thất bại. Mục tiêu vô hạn không đưa vào cực trị hữu hạn, và số lần đó được ghi riêng. Các quy ước này nằm trong `normalization.json`.

Các đồ thị dùng đủ thế hệ 0–500, không làm trơn hay chọn seed có lợi. Thí nghiệm C chỉ có cặp và tham số được paper nêu; không tự thêm một lưới dò cấu hình.

## RPD và Table II

`RPD = 100 × (EA AVG - MFEA AVG) / EA AVG`. NIB đếm bộ MFEA có AVG thấp hơn EA; NIE đếm hai AVG bằng nhau. NIW bổ sung số bộ MFEA kém hơn để thấy đủ kết quả.

Từ AVG đã in trong Tables III/IV: trung bình RPD trên toàn bộ bộ dữ liệu là 9,948% và 4,296%; chỉ lấy các bộ được cải thiện là 13,264% và 4,549%. Table II in 13,2% và 4,6%. Paper không nêu rõ tập lấy trung bình và không cung cấp AVG chưa làm tròn. Vì vậy báo cáo xuất cả hai trung bình cùng số tham chiếu; không ép công thức để khớp.

## Đọc kết quả

- `manifest.json`: tham số, nhóm tác vụ, seed, SHA256 mã và dữ liệu.
- `runs.csv` / `paths.json`: từng lần chạy, chi phí và đường đi dùng ID cạnh gốc.
- `summary.csv`: BF, AVG, độ lệch chuẩn mẫu, số lần có đường đi hợp lệ và chênh với paper.
- `convergence.csv`: best-so-far từ thế hệ 0 đến 500, chưa chuẩn hóa như hình paper.
- `REPORT.md`: bảng thực đo và số dòng khớp BF/AVG.
- `checkpoints/`: phục hồi lần chạy dở, không chứa mã nguồn dự phòng.

AVG không loại bỏ các lần thất bại: JSON ghi null và CSV tổng hợp ghi inf. So sánh AVG theo một chữ số thập phân như bảng. Thời gian Python là thời gian chung của nhóm, không so trực tiếp với Java/i7 trong paper.

Kiểm chứng độc lập đường đi, thống kê, hội tụ và SHA256 sau khi chạy:

```powershell
python -B -m experiments._support.validate_artifacts experiments/paper_run/benchmark experiments/paper_run/convergence --out experiments/paper_run/validation.json
```

Nguồn dữ liệu gốc, mã nguồn tìm được và các điểm khác paper được ghi tại [SOURCE_SEARCH.md](SOURCE_SEARCH.md). `exact.py` chỉ là oracle kiểm chứng bài toán nhỏ và tạo nhãn synthetic cho GNN; không được gọi trong vòng tìm kiếm MFEA.

Báo cáo cấp trên `paper_run/REPORT.md` liên kết bảng và hình; các `REPORT.md` trong từng nhánh là kết quả riêng của benchmark/C-experiment. Thời gian Rs của MFEA là thời gian chung cho nhóm; các số Python không so trực tiếp với Java/i7 trong paper.

Kiểm thử: `python -B -m unittest discover -s tests -v`. GPA được đối chiếu với bản chép trực tiếp Algorithm 1 trên đồ thị thưa/dày; có kiểm thử PMX, DU, USS, ngân sách đánh giá, chuẩn hóa, cả hai thuật toán, checkpoint/resume và kiểm chứng đầu ra.
