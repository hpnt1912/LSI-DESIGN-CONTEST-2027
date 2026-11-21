"""Tải CIFAR-10 và tách 4 lớp của cuộc thi: cat, dog, deer, horse.

Đầu ra: data/cifar4.npz gồm
    x_train (N, 32, 32, 3) uint8, y_train (N,) uint8
    x_test  (M, 32, 32, 3) uint8, y_test  (M,) uint8
Nhãn mới: 0 cat, 1 dog, 2 deer, 3 horse.

Chạy từ thư mục gốc repo:
    python model/prepare_cifar4.py
"""

from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

CLASSES = ["cat", "dog", "deer", "horse"]


def extract(train: bool) -> tuple[np.ndarray, np.ndarray]:
    ds = CIFAR10(root=DATA_DIR, train=train, download=True)
    name_to_cifar = {name: i for i, name in enumerate(ds.classes)}
    cifar_ids = [name_to_cifar[name] for name in CLASSES]

    targets = np.asarray(ds.targets)
    mask = np.isin(targets, cifar_ids)
    remap = {cid: new for new, cid in enumerate(cifar_ids)}
    labels = np.array([remap[t] for t in targets[mask]], dtype=np.uint8)
    return ds.data[mask], labels


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    x_train, y_train = extract(train=True)
    x_test, y_test = extract(train=False)

    out = DATA_DIR / "cifar4.npz"
    np.savez_compressed(out, x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test, classes=np.array(CLASSES))

    print(f"Đã lưu {out}")
    print(f"  train: {x_train.shape}, test: {x_test.shape}")
    for i, name in enumerate(CLASSES):
        print(f"  {i} {name:5s}: train {np.sum(y_train == i)}, test {np.sum(y_test == i)}")


if __name__ == "__main__":
    main()
