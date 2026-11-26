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

# ---- Mô hình mặc định của các script (dùng khi KHÔNG ghi --arch) ----------------------
# Giữ B_16_32_64 – mô hình học theo file hướng dẫn (công thức gốc, 30 epoch). Không đổi sang B3c vì
# `python model/train.py` không --arch sẽ huấn luyện sai công thức (B3c cần --recipe opt + CINIC + tinh chỉnh).
CHOSEN_ARCH = "B_16_32_64"

# ---- ★ MÔ HÌNH CHỌN ĐI THI (chốt 26/11/2026) --------------------------------------------
# Đưa xuống FPGA: tham số + file .mem ở model/export_B3c_c4x4_128+cinic_ft/ ; conv1 dùng trọng số int16.
# Kết quả trên laptop nhóm: test số nguyên 87,06% ± 0,51 (3 seed), ảnh thật 91,7% (108 ảnh), 4,0 triệu MAC/ảnh.
# Huấn luyện lại: python model/kiem_chung.py B3c   (xem docs/KIEM_CHUNG.md)
CONTEST_ARCH = "B3c_c4x4_128+cinic_ft"
CONTEST_CONV1_BITS = 16

# ---- Công thức huấn luyện opt (B3c, E, E2, E3d; ban đầu thử ở nhánh exp/opt-acc-90) -----------
# Chỉ dùng khi chạy  train.py --recipe opt ; công thức gốc ở trên không đổi.
OPT_EPOCHS = 100            # số epoch mặc định của công thức opt
OPT_BATCH_SIZE = 128
OPT_LR = 0.05               # SGD momentum (nesterov) – thường tốt hơn AdamW cho CNN trên CIFAR
OPT_MOMENTUM = 0.9
OPT_WEIGHT_DECAY = 5e-4
OPT_WARMUP_EPOCHS = 5       # tăng tuyến tính LR trong 5 epoch đầu, sau đó giảm theo cosine
OPT_LABEL_SMOOTHING = 0.1
OPT_CUTOUT = 12             # cạnh ô vuông bị xoá (Cutout) trên ảnh 32x32
OPT_JITTER = 0.2            # biên độ đổi độ sáng / độ tương phản ngẫu nhiên (±20%)

# Kiến trúc gốc ghi kết quả vào model/export/; kiến trúc mới ghi vào model/export_<tên>/
# để không đè lên bộ .mem baseline.
BASELINE_ARCHS = ("A_8_16_32", "B_16_32_64", "C_32_64_128", "D_16_32_64_noBN")


def export_dir_for(arch):
    return EXPORT_DIR if arch in BASELINE_ARCHS else REPO_ROOT / "model" / f"export_{arch}"
