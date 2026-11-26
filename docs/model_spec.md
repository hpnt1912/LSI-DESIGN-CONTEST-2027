# Đặc tả mô hình CNN – LSI Design Contest 2027

## Mục tiêu
- Hạng mục: Class 1 – Image Classification, Level 2
- Bài toán: phân loại 4 lớp, thứ tự nhãn **0 cat, 1 dog, 2 deer, 3 horse**
- Ảnh gốc (ngày thi, camera, thẻ SD): ảnh chụp thật, **kích thước bất kỳ** (ví dụ 640×480)
- Tiền xử lý trên Raspberry Pi: cắt giữa thành hình vuông → thu nhỏ về 32×32 → RGB uint8
  (giống `model/predict_image.py`; mọi nơi phải dùng cùng một cách xử lý)
- **Đầu vào của CNN** (sau tiền xử lý): 32 × 32 pixel × 3 kênh R, G, B, mỗi giá trị uint8 0–255
  = 3.072 byte mỗi ảnh. Trong PyTorch xếp thành tensor (3, 32, 32) = (kênh, cao, rộng)
- Đầu ra của CNN: 4 logit (cat, dog, deer, horse); phần cứng lấy argmax (không dùng softmax)

## Dữ liệu
- Huấn luyện: CIFAR-10, lọc 4 lớp (5000 ảnh train, 1000 ảnh test mỗi lớp) – `model/prepare_cifar4.py`
- Tách 10% train làm validation
- Ảnh CIFAR đã sẵn 32×32. Kiểm tra thêm bằng ảnh thật kích thước lớn, qua đúng bước tiền xử lý trên (ngày thi BTC phát ảnh mới, có camera)
- Cần hỏi BTC: ảnh ngày thi phát ở kích thước gốc hay đã thu về 32×32
- Chuẩn hoá đầu vào phải làm được trên phần cứng: ưu tiên chia cho lũy thừa của 2 (ví dụ `x/256`) hoặc gộp vào trọng số tầng đầu

## Ràng buộc phần cứng (DS180, xem `docs/hardware.md`)
- Chip: XC7S25CSGA225-1 – 14.600 LUT, 29.200 FF, **80 DSP**, **1.620 Kb BRAM (~202 KB)**
- Clock gốc trên board: 20 MHz (nâng được bằng MMCM)
- Ngân sách: trọng số int8 ≤ ~100 KB; số MAC/ảnh là tiêu chí chính khi chọn kiến trúc (quyết định tốc độ)
- Chỉ dùng các phép: Conv 3×3, ReLU, MaxPool 2×2, Fully connected (theo tài liệu lần 23)
- Lượng tử hóa: trọng số int8, activation uint8 (sau ReLU), bias/tổng tích luỹ int32, thang lũy thừa của 2 theo từng tầng

## Kết quả đã kiểm chứng trên laptop của nhóm (25–26/11/2026)

Test 4.000 ảnh CIFAR-4; "số nguyên" = kết quả golden model = kết quả FPGA. Chi tiết: `docs/so_sanh_5_mo_hinh.md`.

| Mô hình | Vai trò | Float | Số nguyên | MAC/ảnh | Trọng số | Dữ liệu |
|---|---|---|---|---|---|---|
| B_16_32_64 | Cơ bản | 80,23% | 80,00% | 2,8 triệu | 27 KB | CIFAR |
| B3c_c4x4_128+cinic_ft | Nhanh | 87,67% | 87,60% | 4,0 triệu | 104 KB | CIFAR + CINIC-10 |
| E_opt_cnn+cinic_ft | Chính xác cao | 90,45% | 90,65% | 19,8 triệu | 137 KB | CIFAR + CINIC-10 |
| E2_lite+cinic60_kdft | Cân bằng | 90,00% | 89,60% | 12,7 triệu | 137 KB | CIFAR + CINIC-10 |
| E3d_deep5+cinic60_kdft | Thử nghiệm mới | 90,20% | 90,25% | 11,6 triệu | 146 KB | CIFAR + CINIC-10 |

- Golden model của cả 5 mô hình trùng từng bit với cách tính thứ hai bằng PyTorch.
- Các bản dùng CINIC vượt ngân sách trọng số 100 KB ở trên (BRAM ước lượng 55–79%) – cần kiểm tra khi thiết kế RTL.
- Hướng dẫn học: `docs/HuongDan_MoHinh_CNN_Python.docx`; cách kiểm chứng lại: `docs/KIEM_CHUNG.md`.

## Tiêu chí đánh giá
- Độ chính xác trên tập test và ảnh thật
- Số tham số, số MAC, dung lượng trọng số
- Mục tiêu accuracy: cần thầy xác nhận

## Hạn
- Mô hình Python: 2 tuần (tới khoảng 07/12/2026) – xem `docs/PLAN.md`
- Báo cáo cuối + báo cáo tổng hợp: 31/01/2027
