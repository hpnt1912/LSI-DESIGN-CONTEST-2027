# Model

Mô hình CNN (PyTorch) và mô hình tham chiếu số nguyên cho RTL. Kế hoạch: [`../docs/PLAN.md`](../docs/PLAN.md),
yêu cầu: [`../docs/model_spec.md`](../docs/model_spec.md).

| File | Ngày | Việc |
|---|---|---|
| `prepare_cifar4.py` | ✅ | Tải CIFAR-10, giữ 4 lớp → `data/cifar4.npz` |
| `data.py` | 2 | DataLoader, chuẩn hoá, augmentation, tách validation |
| `net.py` | 3 | Định nghĩa các kiến trúc CNN |
| `train.py` | 3–5 | Huấn luyện, lưu checkpoint vào `model/checkpoints/` |
| `results.md` | 4–5 | Bảng so sánh kiến trúc (accuracy, tham số, MAC, KB) |
| `evaluate.py` | 6 | Confusion matrix, accuracy từng lớp |
| `quantize.py` | 8–10 | BN folding, lượng tử hóa int8 (PTQ/QAT) |
| `golden_int.py` | 11 | Suy luận chỉ bằng số nguyên (NumPy), khớp từng bit với RTL |
| `export.py` | 12 | Xuất trọng số `.mem` + vector kiểm thử vào `model/export/` |
| `real_images/` | 13 | Ảnh thật 32×32 theo thư mục lớp |

Chạy mọi script từ thư mục gốc repo, ví dụ `python model/train.py`.
Dataset, checkpoint (`*.pt`) không commit; `model/export/` thì commit (file nhỏ, RTL cần dùng).
