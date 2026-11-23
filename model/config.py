"""Cấu hình chung cho toàn bộ mô hình (đường dẫn, lớp, siêu tham số).

Mọi script trong model/ import file này, nên muốn đổi gì chung thì sửa ở đây.
"""

from pathlib import Path

# ---- Đường dẫn -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "cifar4.npz"          # tạo bởi prepare_cifar4.py
CKPT_DIR = REPO_ROOT / "model" / "checkpoints"          # trọng số float (không commit)
EXPORT_DIR = REPO_ROOT / "model" / "export"             # tham số số nguyên + file .mem (commit)
FIG_DIR = REPO_ROOT / "model" / "figures"               # hình cho báo cáo (commit)
RESULTS_FILE = REPO_ROOT / "model" / "results.md"       # bảng kết quả thí nghiệm

# ---- Bài toán ----------------------------------------------------------------
CLASSES = ["cat", "dog", "deer", "horse"]               # chỉ số 0, 1, 2, 3
IMG_SIZE = 32

# ---- Huấn luyện ----------------------------------------------------------------
SEED = 2027                 # hạt giống ngẫu nhiên (random seed) để kết quả lặp lại được
VAL_FRACTION = 0.1          # 10% ảnh train dùng làm validation
BATCH_SIZE = 64             # số ảnh mỗi lần cập nhật trọng số
EPOCHS = 30                 # số lần duyệt hết tập train
LR = 1e-3                   # tốc độ học (learning rate) ban đầu
WEIGHT_DECAY = 5e-4         # phạt trọng số lớn (L2 regularization)

# ---- Mô hình chọn để đưa xuống phần cứng ------------------------------------------
# Tên phải có trong net.ARCHS. Đổi sau khi so sánh kiến trúc (xem results.md).
CHOSEN_ARCH = "B_16_32_64"
