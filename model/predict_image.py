"""Ngày 13 – Dự đoán ảnh thật (ảnh chụp, ảnh tải từ mạng).

Chạy:
    python model/predict_image.py duong/dan/anh.jpg            # 1 ảnh
    python model/predict_image.py model/real_images            # cả thư mục

Nếu thư mục có dạng real_images/<tên lớp>/*.jpg (cat, dog, deer, horse) thì script tính luôn accuracy.

Tiền xử lý giống lúc thi: cắt giữa ảnh thành hình vuông -> thu nhỏ về 32x32 -> uint8 RGB.
In ra dự đoán của mô hình float và của golden model số nguyên.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import config
import evaluate
import golden_int

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_32x32(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(img.resize((32, 32), Image.BILINEAR), dtype=np.uint8)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = Path(sys.argv[1])
    files = sorted(p for p in target.rglob("*") if p.suffix.lower() in EXTS) if target.is_dir() else [target]
    if not files:
        print("Không tìm thấy ảnh.")
        return

    imgs = np.stack([load_32x32(f) for f in files])
    model = evaluate.load_model(config.CHOSEN_ARCH)
    pred_f = evaluate.predict_all(model, imgs)
    pred_i = golden_int.infer(imgs, golden_int.load_qparams()).argmax(1)

    n_ok, n_lab = 0, 0
    for f, pf, pi in zip(files, pred_f, pred_i):
        truth = f.parent.name if f.parent.name in config.CLASSES else None
        tag = ""
        if truth:
            n_lab += 1
            ok = config.CLASSES[pi] == truth
            n_ok += ok
            tag = "đúng" if ok else f"SAI (thật: {truth})"
        print(f"{f.name:30s} float: {config.CLASSES[pf]:5s}  số nguyên: {config.CLASSES[pi]:5s}  {tag}")
    if n_lab:
        print(f"Accuracy (số nguyên) trên {n_lab} ảnh có nhãn: {n_ok / n_lab:.3f}")


if __name__ == "__main__":
    main()
