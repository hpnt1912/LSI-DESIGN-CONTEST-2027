"""Ngày 3–5 – Huấn luyện (training) một kiến trúc CNN.

Cách chạy (từ thư mục gốc repo):
    python model/train.py                       # kiến trúc mặc định config.CHOSEN_ARCH
    python model/train.py --arch A_8_16_32      # kiến trúc khác
    python model/train.py --arch all            # lần lượt cả 4 kiến trúc (ngày 4–5)
    python model/train.py --epochs 5            # chạy thử nhanh

Kết quả:
    model/checkpoints/<arch>.pt        trọng số tốt nhất theo validation (không commit)
    model/figures/train_<arch>.png      đường loss / accuracy theo epoch
    model/results.md                    thêm 1 dòng vào bảng so sánh
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

import config
import data
import net


def set_seed(seed=config.SEED):
    """Cố định mọi nguồn ngẫu nhiên -> chạy lại cho cùng kết quả."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate(model, loader):
    """Trả về (loss trung bình, accuracy) trên một tập dữ liệu, không cập nhật trọng số."""
    model.eval()                               # BN dùng thống kê đã học, không dùng của batch
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train_one(arch, epochs, recipe="base"):
    set_seed()
    model = net.build(arch)
    if recipe == "opt":
        train_loader, val_loader, test_loader, loss_fn, opt, sched = opt_recipe(model, epochs)
    else:
        train_loader, val_loader, test_loader = data.get_loaders()
        # Hàm mất mát (loss function): cross-entropy, chuẩn cho bài toán phân loại
        loss_fn = nn.CrossEntropyLoss()
        # Bộ tối ưu (optimizer): AdamW – biến thể của gradient descent tự điều chỉnh bước
        opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        # Lịch giảm tốc độ học (learning-rate schedule) dạng cosine: lớn lúc đầu, nhỏ dần về cuối
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.CKPT_DIR / f"{arch}.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()                          # bật chế độ huấn luyện (BN dùng thống kê của batch)
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            logits = model(x)                  # 1) lan truyền xuôi (forward)
            loss = loss_fn(logits, y)          # 2) đo sai số
            opt.zero_grad()                    # 3) xoá gradient cũ
            loss.backward()                    # 4) lan truyền ngược (backpropagation) tính gradient
            opt.step()                         # 5) cập nhật trọng số
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()

        val_loss, val_acc = evaluate(model, val_loader)
        history["train_loss"].append(run_loss / n)
        history["train_acc"].append(correct / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        mark = ""
        if val_acc > best_val:                 # chỉ giữ mô hình tốt nhất trên validation
            best_val = val_acc
            torch.save({"arch": arch, "state_dict": model.state_dict()}, ckpt_path)
            mark = " *"
        print(f"[{arch}] epoch {ep:2d}/{epochs}  train loss {run_loss / n:.3f} acc {correct / n:.3f}"
              f"  |  val loss {val_loss:.3f} acc {val_acc:.3f}{mark}  ({time.time() - t0:.0f}s)")

    # Đánh giá cuối cùng trên tập test bằng checkpoint tốt nhất
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    _, test_acc = evaluate(model, test_loader)
    minutes = (time.time() - t0) / 60
    print(f"[{arch}] best val {best_val:.4f}  test {test_acc:.4f}  ({minutes:.1f} phút)")

    save_curves(arch, history)
    (config.CKPT_DIR / f"{arch}_history.json").write_text(json.dumps(history))
    append_result(arch, model, best_val, test_acc, epochs, minutes)
    return test_acc


def opt_recipe(model, epochs):
    """Công thức huấn luyện tối ưu (chỉ dùng khi --recipe opt):
      - tăng cường mạnh: crop + flip + đổi sáng/tương phản + Cutout (data.strong_augment)
      - label smoothing 0.1: nhãn đích 0.925/0.025 thay vì 1/0 -> mô hình bớt "quá tự tin"
      - SGD momentum (nesterov), weight decay chỉ áp cho trọng số conv/linear (không cho BN, bias)
      - LR tăng tuyến tính trong OPT_WARMUP_EPOCHS epoch đầu (warmup), rồi giảm theo cosine về 0
    Các thay đổi này chỉ ảnh hưởng lúc huấn luyện; mạng suy luận vẫn chỉ gồm Conv/BN/ReLU/Pool/Linear."""
    loaders = data.get_loaders(config.OPT_BATCH_SIZE, strong_aug=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.OPT_LABEL_SMOOTHING)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": config.OPT_WEIGHT_DECAY},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=config.OPT_LR, momentum=config.OPT_MOMENTUM, nesterov=True)
    warm = config.OPT_WARMUP_EPOCHS

    def lr_factor(ep):                      # ep = số epoch đã xong (0, 1, ...)
        if ep < warm:
            return (ep + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * (ep - warm) / max(1, epochs - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    return (*loaders, loss_fn, opt, sched)


def save_curves(arch, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    ep = range(1, len(h["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(ep, h["train_loss"], label="train")
    a1.plot(ep, h["val_loss"], label="val")
    a1.set_xlabel("epoch"), a1.set_ylabel("loss"), a1.legend(), a1.set_title(f"{arch}: loss")
    a2.plot(ep, h["train_acc"], label="train")
    a2.plot(ep, h["val_acc"], label="val")
    a2.set_xlabel("epoch"), a2.set_ylabel("accuracy"), a2.legend(), a2.set_title(f"{arch}: accuracy")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"train_{arch}.png", dpi=120)
    plt.close(fig)


def append_result(arch, model, val_acc, test_acc, epochs, minutes):
    """Ghi 1 dòng vào bảng trong results.md (tạo file nếu chưa có)."""
    header = ("# Kết quả thí nghiệm\n\n## So sánh kiến trúc (float)\n\n"
              "| Kiến trúc | Tham số | MAC/ảnh | KB int8 | Epoch | Val acc | Test acc | Thời gian |\n"
              "|---|---|---|---|---|---|---|---|\n")
    f = config.RESULTS_FILE
    text = f.read_text() if f.exists() else header
    p = net.count_params(model)
    row = (f"| {arch} | {p:,} | {net.count_macs(model):,} | {p / 1024:.1f} | {epochs} | "
           f"{val_acc * 100:.2f}% | {test_acc * 100:.2f}% | {minutes:.1f} phút |\n")
    if "| Kiến trúc |" not in text:                      # file có nhưng chưa có bảng
        text = header + text.split("\n", 1)[1] if text.startswith("# ") else header + text
    lines = [ln for ln in text.splitlines(keepends=True) if not ln.startswith(f"| {arch} |")]
    # chèn dòng mới vào cuối bảng so sánh (bảng bắt đầu ở dòng "| Kiến trúc |")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("| Kiến trúc |")) + 2
    while i < len(lines) and lines[i].startswith("|"):
        i += 1
    lines.insert(i, row)
    f.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=config.CHOSEN_ARCH,
                    help="tên trong net.ARCHS / net.OPT_ARCHS, hoặc 'all' (4 kiến trúc gốc)")
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"mặc định {config.EPOCHS} (base) hoặc {config.OPT_EPOCHS} (opt)")
    ap.add_argument("--recipe", choices=["base", "opt"], default="base",
                    help="base = công thức gốc; opt = SGD+warmup+cosine, label smoothing, Cutout, đổi sáng")
    args = ap.parse_args()
    epochs = args.epochs or (config.OPT_EPOCHS if args.recipe == "opt" else config.EPOCHS)
    archs = list(net.ARCHS) if args.arch == "all" else [args.arch]
    for a in archs:
        train_one(a, epochs, args.recipe)


if __name__ == "__main__":
    main()
