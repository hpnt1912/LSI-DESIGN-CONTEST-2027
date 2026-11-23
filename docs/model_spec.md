# Đặc tả mô hình CNN – LSI Design Contest 2027

## Mục tiêu
- Hạng mục: Class 1 – Image Classification, Level 2
- Bài toán: phân loại 4 lớp cat, dog, deer, horse
- Input: ảnh RGB 32×32×3
- Output: 4 giá trị, phần cứng lấy argmax (không dùng softmax)

## Dữ liệu
- Huấn luyện: CIFAR-10, lọc 4 lớp (khoảng 5000 ảnh train, 1000 ảnh test mỗi lớp)
- Tách 10% train làm validation
- Bổ sung ảnh thật resize 32×32 để kiểm tra (ngày thi BTC phát ảnh mới, có camera)

## Ràng buộc phần cứng
- Chip: XC7S25CSGA225-1 (khoảng 14.600 LUT, 80 DSP, 1,6 Mb BRAM – cần đối chiếu datasheet)
- Clock gốc trên board: 20 MHz
- Chỉ dùng các phép: Conv 3×3, ReLU, MaxPool 2×2, Fully connected (theo tài liệu lần 23)
- Lượng tử hóa: trọng số và activation int8, bias int32

## Tiêu chí đánh giá
- Độ chính xác trên tập test và ảnh thật
- Số tham số, số MAC, dung lượng trọng số
- Mục tiêu accuracy: cần thầy xác nhận

## Hạn
- Mô hình Python: 2 tuần (tới khoảng 07/12/2026)
- Báo cáo cuối + báo cáo tổng hợp: 31/01/2027
