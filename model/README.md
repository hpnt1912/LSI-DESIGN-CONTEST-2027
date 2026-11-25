# Model

Mô hình CNN (PyTorch) và mô hình tham chiếu số nguyên cho RTL. Kế hoạch: [`../docs/PLAN.md`](../docs/PLAN.md),
yêu cầu: [`../docs/model_spec.md`](../docs/model_spec.md).

Tất cả file dưới đây đã viết xong và chạy thử. Hướng dẫn đọc hiểu: `docs/HuongDan_MoHinh_CNN_Python.docx`.

| File | Ngày | Việc |
|---|---|---|
| `config.py` | 1 | Cấu hình chung (đường dẫn, seed, siêu tham số, kiến trúc chọn) |
| `prepare_cifar4.py` | ✅ | Tải CIFAR-10, giữ 4 lớp → `data/cifar4.npz` |
| `data.py` | 2 | DataLoader, chuẩn hoá, augmentation, tách validation |
| `net.py` | 3 | Định nghĩa các kiến trúc CNN |
| `train.py` | 3–5 | Huấn luyện, lưu checkpoint vào `model/checkpoints/` |
| `results.md` | 4–5 | Bảng so sánh kiến trúc (accuracy, tham số, MAC, KB) |
| `evaluate.py` | 6 | Confusion matrix, accuracy từng lớp |
| `quantize.py` | 8–10 | BN folding, lượng tử hóa int8 (PTQ/QAT) |
| `golden_int.py` | 11 | Suy luận chỉ bằng số nguyên (NumPy), khớp từng bit với RTL |
| `export.py` | 12 | Xuất trọng số `.mem` + vector kiểm thử vào `model/export/` |
| `predict_image.py` | 13 | Dự đoán ảnh thật trong `real_images/<lớp>/` |
| `report.py` | – | Ghi một mục vào `results.md` |

Chạy mọi script từ thư mục gốc repo, ví dụ `python model/train.py`.
Dataset, checkpoint (`*.pt`) không commit; `model/export/` thì commit (file nhỏ, RTL cần dùng).

## Kiểm chứng 5 mô hình chính (một lệnh)

```bash
python model/kiem_chung.py nhanh     # vài phút: golden model trên tham số có sẵn
python model/kiem_chung.py tatca     # huấn luyện lại B, B3c, E, E2, E3d (~6 giờ), tự cập nhật + đẩy GitHub
```
Hướng dẫn: [`docs/KIEM_CHUNG.md`](../docs/KIEM_CHUNG.md). Bảng so sánh tự sinh:
[`docs/so_sanh_5_mo_hinh.md`](../docs/so_sanh_5_mo_hinh.md).

## Thí nghiệm tối ưu độ chính xác (nhánh `exp/opt-acc-90`)

Kiến trúc mới nằm trong `net.OPT_ARCHS`, tách khỏi 4 kiến trúc gốc (`--arch all` vẫn chỉ chạy A–D).
Công thức huấn luyện mới chỉ bật khi có cờ `--recipe opt`; không có cờ thì mọi thứ chạy như cũ.

| Tên | Cấu trúc | Ghi chú |
|---|---|---|
| `E_opt_cnn` | Conv32-P · Conv64-Conv64-P · Conv128-P · Linear 2048→4 | VGG-lite, 138.372 tham số (135 KB int8), 19,8 triệu MAC |
| `B2_16_32_64_opt` | giống hệt `B_16_32_64` | đối chứng: cùng phần cứng với B, chỉ khác công thức huấn luyện |

Công thức `opt`: SGD nesterov (lr 0.05, momentum 0.9, wd 5e-4) + warmup 5 epoch + cosine, label smoothing 0.1,
crop + flip + đổi sáng/tương phản ±20% + Cutout 12×12, batch 128, 100 epoch (xem `config.OPT_*`).

```bash
python model/train.py --arch E_opt_cnn --recipe opt   # ~15–20 phút trên 4 CPU
python model/evaluate.py --arch E_opt_cnn
python model/quantize.py --arch E_opt_cnn             # ghi vào model/export_E_opt_cnn/
python model/golden_int.py --arch E_opt_cnn
python model/export.py --arch E_opt_cnn
```

### Dữ liệu bổ sung CINIC-10 (tuỳ chọn)

`prepare_cinic4.py` lấy **84.000 ảnh ImageNet 32×32** (21.000 ảnh/lớp) của 4 lớp từ CINIC-10, bỏ toàn bộ ảnh gốc
CIFAR có trong CINIC (để không lộ tập test) và kiểm tra trùng pixel với tập test. Val/test vẫn là ảnh CIFAR.

```bash
# tải CINIC-10.tar.gz (687 MB) từ https://datashare.ed.ac.uk/handle/10283/3192, rồi:
python model/prepare_cinic4.py --tar ~/Downloads/CINIC-10.tar.gz          # -> data/cinic4_extra.npz
python model/train.py --arch "E2_lite+cinic" --recipe opt --extra-data --cifar-repeat 2 --epochs 30
```

Tên lượt chạy dạng `<kiến trúc>+<nhãn>` dùng chung kiến trúc nhưng có checkpoint / thư mục export riêng.

### Kết luận (4.000 ảnh test CIFAR-4, chọn mô hình theo val, không theo test)

Bảng đầy đủ (kèm ý nghĩa từng thư mục `export_*`): [`docs/so_sanh_mo_hinh.md`](../docs/so_sanh_mo_hinh.md).

| Mô hình | Vai trò | Test số nguyên | Test float | KB trọng số | Triệu MAC | ms/ảnh 16 DSP @100 MHz* |
|---|---|---|---|---|---|---|
| `B_16_32_64` | baseline | 80,00% | 80,23% | 27 | 2,8 | 1,8 |
| `B2_16_32_64_opt` | B + công thức opt | 84,03% | 83,88% | 27 | 2,8 | 1,8 |
| `B3c_c4x4_128+cinic_ft` | ưu tiên tốc độ | 87,20% | 87,08% | 103 | 4,0 | 2,5 |
| **`E2_lite+cinic60_kdft`** | **cân bằng (khuyên dùng)** | **90,30%** | 90,32% | 135 | 12,7 | 7,9 |
| `E_opt_cnn+cinic_ft` | ưu tiên độ chính xác | 90,60% | 90,65% | 135 | 19,8 | 12,4 |

\*Ước lượng lý thuyết, DSP dùng 100%; thực tế chậm hơn 20–40%. Các mô hình `+cinic` dùng `--conv1-bits 16`
(conv1 int16: gộp BN làm trọng số conv1 rất lớn, int8 làm mất 1,5–2 điểm %). Mọi mô hình khớp từng bit với golden.

Cách tạo `E2_lite+cinic60_kdft`: 60 epoch trên CIFAR (x2) + 84k ảnh CINIC → tinh chỉnh 15 epoch chỉ trên CIFAR
(lr 0,005) có distillation từ `E_opt_cnn+cinic_ft` → `quantize.py --conv1-bits 16`. File `.mem`: `model/export_E2_lite+cinic60_kdft/`.

Độ ổn định: 5 lượt E2_lite tinh chỉnh (khác seed / số epoch / có-không KD) cho 89,45–90,30% số nguyên, trung bình
89,86%; nghĩa là E2_lite ≈ 90% chứ chưa chắc chắn > 90% với mọi lần huấn luyện. Distillation không giúp đều
(seed 7: 89,45% có KD so với 89,65% không KD). Dữ liệu CINIC không giúp mạng cỡ B (`B2+cinic_ft` 83,87%).

Kiến trúc gốc A–D vẫn ghi vào `model/export/`; kiến trúc mới ghi vào `model/export_<tên>/` nên không đè bộ `.mem`
baseline. Pipeline lượng tử hoá hỗ trợ thêm tầng conv **không có MaxPool** (khoá `pool` trong `qparams.json`;
file `qparams` cũ không có khoá này được hiểu là có pool, nên kết quả baseline không đổi).
