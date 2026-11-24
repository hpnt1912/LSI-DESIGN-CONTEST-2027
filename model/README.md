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

Kiến trúc gốc A–D vẫn ghi vào `model/export/`; kiến trúc mới ghi vào `model/export_<tên>/` nên không đè bộ `.mem`
baseline. Pipeline lượng tử hoá hỗ trợ thêm tầng conv **không có MaxPool** (khoá `pool` trong `qparams.json`;
file `qparams` cũ không có khoá này được hiểu là có pool, nên kết quả baseline không đổi).
