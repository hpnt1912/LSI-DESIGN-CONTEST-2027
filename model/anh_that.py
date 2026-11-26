"""Ngày 13 – Đo độ chính xác của cả 5 mô hình chính trên ẢNH THẬT, ghi kết quả vào model/results.md.

Vì sao cần: mọi accuracy trong docs/so_sanh_5_mo_hinh.md đo trên ảnh CIFAR. Ngày thi là ảnh chụp thật, kích thước
bất kỳ; mô hình nào tụt mạnh trên ảnh thật thì không nên chọn, dù accuracy CIFAR cao.

Chuẩn bị: đặt ảnh vào  model/real_images/<lớp>/  với <lớp> là cat, dog, deer, horse (jpg, png, bmp, webp),
kích thước bất kỳ. Nên 20–30 ảnh mỗi lớp, tốt nhất chụp bằng đúng camera sẽ dùng ngày thi.

Chạy (từ thư mục gốc repo, venv đang bật):
    python model/anh_that.py                       # thư mục mặc định model/real_images
    python model/anh_that.py duong/dan/thu_muc     # thư mục khác (cùng cấu trúc <lớp>/)

Tiền xử lý giống predict_image.py: cắt giữa thành hình vuông -> thu về 32x32 -> uint8 RGB.
Cột "số nguyên" (= kết quả FPGA) dùng qparams.npz đã có trên GitHub nên luôn chạy được; cột "float" chỉ có
khi máy có checkpoint model/checkpoints/<lượt chạy>.pt (laptop đã chạy kiem_chung.py thì có).
Script KHÔNG commit / push; cuối cùng in lệnh git để bạn tự làm.
"""

import sys
from pathlib import Path

import numpy as np

import config
import data
import golden_int
from kiem_chung import MODELS, ORDER, fmt, machine, now
from predict_image import EXTS, load_32x32

DEFAULT_DIR = config.REPO_ROOT / "model" / "real_images"
TITLE = "Ảnh thật (model/real_images)"


def load_images(folder):
    """Ảnh trong folder/<lớp>/ -> (ảnh uint8 (N,32,32,3), nhãn (N,), tên file). Bỏ qua ảnh không nằm trong thư mục lớp."""
    imgs, labels, names = [], [], []
    for k, c in enumerate(config.CLASSES):
        for f in sorted((folder / c).glob("*")):
            if f.suffix.lower() in EXTS:
                imgs.append(load_32x32(f))
                labels.append(k)
                names.append(f"{c}/{f.name}")
    if not imgs:
        return None, None, None
    return np.stack(imgs), np.array(labels), names


def int_predict(run, x):
    q = golden_int.load_qparams(config.export_dir_for(run) / "qparams.npz")
    return np.concatenate([golden_int.infer(x[i:i + 500], q).argmax(1) for i in range(0, len(x), 500)])


def float_predict(run, x):
    """None nếu máy không có checkpoint float của lượt chạy này."""
    if not (config.CKPT_DIR / f"{run}.pt").exists():
        return None
    import evaluate
    return evaluate.predict_all(evaluate.load_model(run), x)


def ci95(p, n):
    """Nửa khoảng tin cậy 95% (điểm %) của một tỉ lệ p đo trên n ảnh: 1,96·√(p(1−p)/n)."""
    return 1.96 * np.sqrt(p * (1 - p) / n) * 100


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    x, y, names = load_images(folder)
    if x is None:
        sys.exit(f"✗ Không có ảnh trong {folder}/<lớp>/ (lớp: {', '.join(config.CLASSES)}).\n"
                 f"  Tạo thư mục:  mkdir -p " + " ".join(f"{folder}/{c}" for c in config.CLASSES))
    counts = [int((y == k).sum()) for k in range(4)]
    print(f"{len(x)} ảnh có nhãn: " + ", ".join(f"{c} {n}" for c, n in zip(config.CLASSES, counts)))
    _, _, x_te, y_te = data.load_arrays()

    rows, preds = [], {}
    for k in ORDER:
        run = MODELS[k]["run"]
        if not (config.export_dir_for(run) / "qparams.npz").exists():
            print(f"  {k}: bỏ qua – chưa có qparams.npz")
            continue
        print(f"  {k}: đang chạy golden model ({run}) ...", flush=True)
        pi, pf = int_predict(run, x), float_predict(run, x)
        preds[k] = pi
        acc = float((pi == y).mean())
        cifar = float((int_predict(run, x_te) == y_te).mean()) * 100
        per = ["–" if n == 0 else fmt((pi[y == c] == c).mean() * 100, 0) for c, n in enumerate(counts)]
        rows.append(f"| **{k}** | {fmt(cifar)}% | **{fmt(acc * 100, 1)}%** ± {fmt(ci95(acc, len(y)), 0)} | "
                    f"{'–' if pf is None else fmt((pf == y).mean() * 100, 1) + '%'} | "
                    f"{'+' if acc * 100 >= cifar else ''}{fmt(acc * 100 - cifar, 1)} | {' / '.join(per)} |")

    hard = [n for i, n in enumerate(names) if sum(preds[k][i] != y[i] for k in preds) == len(preds)]
    body = (f"> Ghi bởi `python model/anh_that.py` trên máy `{machine()}` "
            f"({now()}). Thư mục: `{folder.name}/`; "
            f"{len(x)} ảnh: " + ", ".join(f"{c} {n}" for c, n in zip(config.CLASSES, counts)) + ".\n\n"
            "| Mô hình | Test CIFAR (số nguyên) | **Ảnh thật (số nguyên)** ± khoảng tin cậy 95% | Ảnh thật (float) | "
            "Chênh (thật − CIFAR, điểm %) | cat / dog / deer / horse (%) |\n|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n\n"
            f"Với {len(x)} ảnh, khoảng tin cậy rộng (± ở cột 3): chỉ chênh lệch lớn hơn khoảng đó mới đáng tin. "
            "Điều cần xem là mô hình nào tụt mạnh so với CIFAR.\n\n"
            f"Ảnh cả {len(preds)} mô hình đều đoán sai ({len(hard)}): "
            + (", ".join(f"`{n}`" for n in hard[:30]) + (" …" if len(hard) > 30 else "") if hard else "không có") + ".")
    import report
    report.update_section(TITLE, body)
    print("\n" + body.split("\n\n", 1)[1])
    print(f"\n✓ Đã ghi mục '## {TITLE}' vào {config.RESULTS_FILE.relative_to(config.REPO_ROOT)}.")
    print("Đẩy kết quả lên GitHub (chỉ results.md, không đẩy ảnh):")
    print("  git add model/results.md")
    print('  git commit -m "Kết quả 5 mô hình trên ảnh thật"')
    print("  git push")


if __name__ == "__main__":
    main()
