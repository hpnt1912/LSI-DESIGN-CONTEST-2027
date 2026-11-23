# Đặc tả mô hình CNN – LSI Design Contest 2027

## Mục tiêu
- Hạng mục: Class 1 – Image Classification, Level 2
- Bài toán: phân loại 4 lớp, thứ tự nhãn **0 cat, 1 dog, 2 deer, 3 horse**
- Input: ảnh RGB 32×32×3, giá trị pixel uint8 0–255
- Output: 4 giá trị, phần cứng lấy argmax (không dùng softmax)

## Dữ liệu
- Huấn luyện: CIFAR-10, lọc 4 lớp (5000 ảnh train, 1000 ảnh test mỗi lớp) – `model/prepare_cifar4.py`
- Tách 10% train làm validation
- Bổ sung ảnh thật resize 32×32 để kiểm tra (ngày thi BTC phát ảnh mới, có camera)
- Chuẩn hoá đầu vào phải làm được trên phần cứng: ưu tiên chia cho lũy thừa của 2 (ví dụ `x/256`) hoặc gộp vào trọng số tầng đầu

## Ràng buộc phần cứng (DS180, xem `docs/hardware.md`)
- Chip: XC7S25CSGA225-1 – 14.600 LUT, 29.200 FF, **80 DSP**, **1.620 Kb BRAM (~202 KB)**
- Clock gốc trên board: 20 MHz (nâng được bằng MMCM)
- Ngân sách: trọng số int8 ≤ ~100 KB; số MAC/ảnh là tiêu chí chính khi chọn kiến trúc (quyết định tốc độ)
- Chỉ dùng các phép: Conv 3×3, ReLU, MaxPool 2×2, Fully connected (theo tài liệu lần 23)
- Lượng tử hóa: trọng số và activation int8, bias int32, scale lũy thừa của 2

## Tiêu chí đánh giá
- Độ chính xác trên tập test và ảnh thật
- Số tham số, số MAC, dung lượng trọng số
- Mục tiêu accuracy: cần thầy xác nhận

## Hạn
- Mô hình Python: 2 tuần (tới khoảng 07/12/2026) – xem `docs/PLAN.md`
- Báo cáo cuối + báo cáo tổng hợp: 31/01/2027
