"""Dữ liệu bổ sung (tuỳ chọn): ảnh ImageNet 32x32 của 4 lớp cat/dog/deer/horse lấy từ CINIC-10.

CINIC-10 (Darlow và cộng sự, 2018) = CIFAR-10 + ảnh ImageNet đã thu về 32x32, cùng 10 lớp với CIFAR-10.
Mỗi lớp có 21.000 ảnh ImageNet (7.000 trong mỗi thư mục train/valid/test của CINIC) -> 84.000 ảnh cho 4 lớp,
gấp ~4,7 lần 18.000 ảnh train CIFAR-4 hiện có. Đề thi cho phép tự chọn dataset.

QUAN TRỌNG – không để lộ tập test:
  CINIC-10 có chứa cả ảnh gốc CIFAR-10 (kể cả ảnh thuộc tập TEST của CIFAR, tên file "cifar10-...").
  Script này CHỈ lấy ảnh ImageNet (tên file bắt đầu bằng mã synset "n0..."), rồi còn so khớp từng pixel
  với 4.000 ảnh test CIFAR-4 để chắc chắn không có ảnh trùng.

Chạy (từ thư mục gốc repo):
    python model/prepare_cinic4.py --tar đường/dẫn/CINIC-10.tar.gz
File CINIC-10.tar.gz (687 MB): https://datashare.ed.ac.uk/handle/10283/3192
Đầu ra: data/cinic4_extra.npz  gồm x (N, 32, 32, 3) uint8, y (N,) uint8 (0 cat, 1 dog, 2 deer, 3 horse)
"""

import argparse
import io
import tarfile

import numpy as np
from PIL import Image

import config

OUT_FILE = config.REPO_ROOT / "data" / "cinic4_extra.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="đường dẫn tới CINIC-10.tar.gz")
    tar_path = ap.parse_args().tar

    cls_id = {c: i for i, c in enumerate(config.CLASSES)}
    xs, ys, skipped_cifar = [], [], 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".png"):
                continue
            split, cls, fname = m.name.split("/")[-3:]
            if cls not in cls_id:
                continue
            if not fname.startswith("n"):           # "cifar10-train-..." / "cifar10-test-..." -> bỏ
                skipped_cifar += 1
                continue
            img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            assert img.size == (32, 32), m.name
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(cls_id[cls])
    x, y = np.stack(xs), np.array(ys, dtype=np.uint8)

    # Kiểm tra trùng tuyệt đối (từng pixel) với tập test CIFAR-4
    d = np.load(config.DATA_FILE)
    test_hashes = {hash(t.tobytes()) for t in d["x_test"]}
    dup = np.array([hash(t.tobytes()) in test_hashes for t in x])
    x, y = x[~dup], y[~dup]

    np.savez_compressed(OUT_FILE, x=x, y=y)
    print(f"Đã lưu {OUT_FILE}: {len(x)} ảnh ImageNet; bỏ {skipped_cifar} ảnh gốc CIFAR, "
          f"bỏ {dup.sum()} ảnh trùng pixel với tập test")
    for i, c in enumerate(config.CLASSES):
        print(f"  {i} {c:5s}: {np.sum(y == i)}")


if __name__ == "__main__":
    main()
