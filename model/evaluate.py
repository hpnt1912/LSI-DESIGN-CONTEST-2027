"""Ngày 6 – Đánh giá chi tiết mô hình float trên tập test.

Chạy:  python model/evaluate.py [--arch B_16_32_64]

Kết quả:
    model/figures/confusion_<arch>.png   ma trận nhầm lẫn (confusion matrix)
    model/figures/errors_<arch>.png      16 ảnh bị đoán sai
    model/results.md                     mục "Đánh giá chi tiết"
"""

import argparse

import numpy as np
import torch

import config
import data
import net
import report


def load_model(arch):
    ckpt = torch.load(config.CKPT_DIR / f"{arch}.pt")
    model = net.build(arch)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def predict_all(model, x_uint8):
    """Dự đoán cho mảng ảnh (N, 32, 32, 3) uint8 -> nhãn (N,)."""
    x = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 256.0
    return torch.cat([model(x[i:i + 500]).argmax(1) for i in range(0, len(x), 500)]).numpy()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=config.CHOSEN_ARCH)
    arch = ap.parse_args().arch

    _, _, x_te, y_te = data.load_arrays()
    model = load_model(arch)
    pred = predict_all(model, x_te)
    acc = (pred == y_te).mean()
    cm = confusion_matrix(y_te, pred, labels=range(4))

    print(f"[{arch}] accuracy test: {acc:.4f}")
    print("Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):")
    print("        " + " ".join(f"{c:>6s}" for c in config.CLASSES))
    for i, c in enumerate(config.CLASSES):
        print(f"{c:>6s}  " + " ".join(f"{v:6d}" for v in cm[i]))

    # Hình ma trận nhầm lẫn
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), config.CLASSES), ax.set_yticks(range(4), config.CLASSES)
    ax.set_xlabel("Dự đoán (predicted)"), ax.set_ylabel("Nhãn thật (true)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{arch} – test acc {acc * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"confusion_{arch}.png", dpi=120)
    plt.close(fig)

    # 16 ảnh bị đoán sai
    wrong = np.flatnonzero(pred != y_te)[:16]
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(x_te[i])
        ax.set_title(f"thật {config.CLASSES[y_te[i]]}\nđoán {config.CLASSES[pred[i]]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"errors_{arch}.png", dpi=120)
    plt.close(fig)

    # Ghi vào results.md
    per_class = cm.diagonal() / cm.sum(1)
    rows = "\n".join(f"| {c} | {per_class[i] * 100:.1f}% |" for i, c in enumerate(config.CLASSES))
    cm_rows = "\n".join(f"| **{c}** | " + " | ".join(str(v) for v in cm[i]) + " |"
                        for i, c in enumerate(config.CLASSES))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    body = (f"Kiến trúc **{arch}**, accuracy test **{acc * 100:.2f}%**.\n\n"
            f"| Lớp | Accuracy |\n|---|---|\n{rows}\n\n"
            f"Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):\n\n"
            f"| | " + " | ".join(config.CLASSES) + " |\n|---|---|---|---|---|\n" + cm_rows + "\n\n"
            f"Nhầm nhiều nhất: **{config.CLASSES[i]} → {config.CLASSES[j]}** ({off[i, j]} ảnh).\n\n"
            f"Hình: `figures/confusion_{arch}.png`, `figures/errors_{arch}.png`.")
    report.update_section("Đánh giá chi tiết (float)", body)
    print(f"Đã lưu hình vào {config.FIG_DIR} và cập nhật {config.RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
