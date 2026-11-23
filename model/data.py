"""Ngày 2 – Nạp dữ liệu CIFAR-4 thành DataLoader cho PyTorch.

Chạy thử:  python model/data.py
  -> in kích thước 1 batch và lưu model/figures/samples.png (16 ảnh kèm nhãn).

Quy ước đầu vào (quan trọng cho phần cứng):
  pixel uint8 0..255  ->  x = pixel / 256  (float trong [0, 1))
  Chia cho 256 = dịch phải 8 bit, phần cứng làm "miễn phí".
  Không trừ trung bình / chia độ lệch chuẩn: BatchNorm ở tầng đầu sẽ tự học việc đó.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config


def load_arrays():
    """Đọc cifar4.npz -> (x_train, y_train, x_test, y_test), ảnh dạng (N, 32, 32, 3) uint8."""
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"Chưa có {config.DATA_FILE}. Chạy: python model/prepare_cifar4.py")
    d = np.load(config.DATA_FILE)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def split_train_val(x, y, fraction=config.VAL_FRACTION, seed=config.SEED):
    """Tách ngẫu nhiên (nhưng cố định theo seed) một phần train làm validation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = int(len(x) * fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def to_tensor(img_uint8):
    """(32, 32, 3) uint8 -> tensor (3, 32, 32) float = pixel / 256."""
    return torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 256.0


class Cifar4(Dataset):
    """Bộ dữ liệu: trả về (ảnh tensor 3x32x32, nhãn 0..3).

    augment=True: tăng cường dữ liệu (data augmentation) cho tập train:
      - random crop: đệm 4 pixel 0 quanh ảnh rồi cắt ngẫu nhiên 32x32 (ảnh dịch lệch tối đa 4 pixel)
      - horizontal flip: lật ngang với xác suất 50%
    Mục đích: mô hình thấy nhiều biến thể -> bớt "học thuộc" (overfitting).
    """

    def __init__(self, x, y, augment=False):
        self.x, self.y, self.augment = x, y, augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        if self.augment:
            padded = np.pad(img, ((4, 4), (4, 4), (0, 0)))
            r, c = np.random.randint(0, 9, size=2)
            img = padded[r:r + 32, c:c + 32]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            img = np.ascontiguousarray(img)
        return to_tensor(img), int(self.y[i])


def get_loaders(batch_size=config.BATCH_SIZE):
    """Trả về 3 DataLoader: train (có augment, xáo trộn), val, test."""
    x_tr, y_tr, x_te, y_te = load_arrays()
    x_tr, y_tr, x_va, y_va = split_train_val(x_tr, y_tr)
    g = torch.Generator().manual_seed(config.SEED)
    train = DataLoader(Cifar4(x_tr, y_tr, augment=True), batch_size, shuffle=True, generator=g)
    val = DataLoader(Cifar4(x_va, y_va), 256)
    test = DataLoader(Cifar4(x_te, y_te), 256)
    return train, val, test


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    train, val, test = get_loaders()
    print(f"Số ảnh: train {len(train.dataset)}, val {len(val.dataset)}, test {len(test.dataset)}")
    xb, yb = next(iter(train))
    print(f"1 batch: x {tuple(xb.shape)} (batch, kênh, cao, rộng), y {tuple(yb.shape)}")
    print(f"Giá trị x trong [{xb.min():.3f}, {xb.max():.3f}]")

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for ax, img, lab in zip(axes.flat, xb, yb):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(config.CLASSES[lab], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = config.FIG_DIR / "samples.png"
    fig.savefig(out, dpi=120)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
