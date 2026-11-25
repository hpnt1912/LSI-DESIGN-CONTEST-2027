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


def distill_loss(logits, teacher_logits, hard_loss, alpha, temp):
    """Chưng cất tri thức (knowledge distillation, Hinton 2015).
    Học sinh vừa học nhãn thật (hard_loss), vừa bắt chước "độ tự tin" của mạng thầy trên cả 4 lớp
    (ví dụ thầy cho rằng ảnh mèo này hơi giống chó) – thông tin mà nhãn 0/1 không có.
    softmax chỉ dùng khi HUẤN LUYỆN; mạch suy luận vẫn chỉ lấy argmax của logit."""
    soft = nn.functional.kl_div(nn.functional.log_softmax(logits / temp, dim=1),
                                nn.functional.softmax(teacher_logits / temp, dim=1),
                                reduction="batchmean") * temp * temp
    return (1 - alpha) * hard_loss + alpha * soft


def cutmix(x, y):
    """CutMix (Yun 2019): dán một ô chữ nhật của ảnh khác (cùng batch) vào ảnh hiện tại.
    Nhãn được trộn theo tỉ lệ diện tích: lam phần nhãn gốc, (1 - lam) phần nhãn ảnh được dán.
    Buộc mạng nhận dạng từ nhiều bộ phận (tai, mõm, chân...) thay vì chỉ một chi tiết. Chỉ dùng khi huấn luyện."""
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(len(x))
    h, w = x.shape[2:]
    rh, rw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    cy, cx = np.random.randint(h), np.random.randint(w)
    y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
    x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[idx, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    return x, y[idx], lam


def train_one(arch, epochs, recipe="base", teacher=None, kd_alpha=0.7, kd_temp=4.0,
              extra=False, cifar_repeat=1, init=None, lr=None, warmup=None, seed=config.SEED,
              ema_decay=0.0, cutmix_prob=0.0):
    set_seed(seed)
    model = net.build(arch)
    if init is not None:                         # tinh chỉnh (fine-tune) từ một lượt chạy đã có
        model.load_state_dict(torch.load(config.CKPT_DIR / f"{init}.pt")["state_dict"])
        print(f"[{arch}] khởi tạo từ checkpoint {init}")
    ema = None
    if ema_decay > 0:
        # EMA (exponential moving average): giữ thêm một bản trọng số = trung bình trượt của các bước gần đây.
        # Bản trung bình thường ổn định và chính xác hơn bản cuối; chỉ khác lúc huấn luyện, suy luận như cũ.
        # Tạo SAU khi nạp --init, để bản trung bình bắt đầu từ trọng số đã học chứ không từ số ngẫu nhiên.
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
    if teacher is not None:
        import evaluate as eval_mod              # nạp checkpoint mạng thầy (đặt tên khác để không
        teacher_model = eval_mod.load_model(teacher)  # che hàm evaluate() của file này)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"[{arch}] distillation từ {teacher} (alpha {kd_alpha}, T {kd_temp})")
    if recipe == "opt":
        train_loader, val_loader, test_loader, loss_fn, opt, sched = opt_recipe(
            model, epochs, extra, cifar_repeat, lr, warmup)
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
            mixed = cutmix_prob > 0 and np.random.rand() < cutmix_prob
            if mixed:                          #    (tuỳ chọn) CutMix
                x, y_b, lam = cutmix(x, y)
            logits = model(x)                  # 1) lan truyền xuôi (forward)
            loss = loss_fn(logits, y)          # 2) đo sai số
            if mixed:
                loss = lam * loss + (1 - lam) * loss_fn(logits, y_b)
            if teacher is not None:            #    (tuỳ chọn) cộng thêm phần bắt chước mạng thầy
                with torch.no_grad():
                    t_logits = teacher_model(x)
                loss = distill_loss(logits, t_logits, loss, kd_alpha, kd_temp)
            opt.zero_grad()                    # 3) xoá gradient cũ
            loss.backward()                    # 4) lan truyền ngược (backpropagation) tính gradient
            opt.step()                         # 5) cập nhật trọng số
            if ema is not None:
                ema.update_parameters(model)
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()

        eval_model = model if ema is None else ema.module   # có EMA: chọn / lưu bản trung bình
        val_loss, val_acc = evaluate(eval_model, val_loader)
        history["train_loss"].append(run_loss / n)
        history["train_acc"].append(correct / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        mark = ""
        if val_acc > best_val:                 # chỉ giữ mô hình tốt nhất trên validation
            best_val = val_acc
            torch.save({"arch": arch, "state_dict": eval_model.state_dict()}, ckpt_path)
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


def opt_recipe(model, epochs, extra=False, cifar_repeat=1, lr=None, warmup=None):
    """Công thức huấn luyện tối ưu (chỉ dùng khi --recipe opt):
      - tăng cường mạnh: crop + flip + đổi sáng/tương phản + Cutout (data.strong_augment)
      - label smoothing 0.1: nhãn đích 0.925/0.025 thay vì 1/0 -> mô hình bớt "quá tự tin"
      - SGD momentum (nesterov), weight decay chỉ áp cho trọng số conv/linear (không cho BN, bias)
      - LR tăng tuyến tính trong OPT_WARMUP_EPOCHS epoch đầu (warmup), rồi giảm theo cosine về 0
    Các thay đổi này chỉ ảnh hưởng lúc huấn luyện; mạng suy luận vẫn chỉ gồm Conv/BN/ReLU/Pool/Linear."""
    loaders = data.get_loaders(config.OPT_BATCH_SIZE, strong_aug=True, extra=extra, cifar_repeat=cifar_repeat)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.OPT_LABEL_SMOOTHING)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": config.OPT_WEIGHT_DECAY},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=lr or config.OPT_LR, momentum=config.OPT_MOMENTUM, nesterov=True)
    warm = config.OPT_WARMUP_EPOCHS if warmup is None else warmup

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
    ap.add_argument("--teacher", default=None,
                    help="(tuỳ chọn) tên kiến trúc đã có checkpoint dùng làm mạng thầy, ví dụ E_opt_cnn")
    ap.add_argument("--kd-alpha", type=float, default=0.7, help="tỉ trọng phần bắt chước thầy (0..1)")
    ap.add_argument("--kd-temp", type=float, default=4.0, help="nhiệt độ làm 'mềm' xác suất của thầy")
    ap.add_argument("--extra-data", action="store_true",
                    help="(chỉ với --recipe opt) ghép thêm ảnh ImageNet CINIC-10 vào tập train "
                         "(tạo bằng prepare_cinic4.py); nên đặt tên lượt chạy dạng <arch>+cinic")
    ap.add_argument("--cifar-repeat", type=int, default=1, help="lặp ảnh train CIFAR k lần khi có --extra-data")
    ap.add_argument("--init", default=None,
                    help="tinh chỉnh: khởi tạo trọng số từ checkpoint của lượt chạy này (ví dụ E2_lite+cinic)")
    ap.add_argument("--lr", type=float, default=None, help=f"LR ban đầu (mặc định {config.OPT_LR})")
    ap.add_argument("--warmup", type=int, default=None, help=f"số epoch warmup (mặc định {config.OPT_WARMUP_EPOCHS})")
    ap.add_argument("--seed", type=int, default=config.SEED, help="hạt giống ngẫu nhiên (để chạy lặp nhiều seed)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="hệ số EMA trọng số, ví dụ 0.999 (0 = tắt, như cũ); chọn/lưu bản trung bình")
    ap.add_argument("--cutmix", type=float, default=0.0, help="xác suất áp CutMix cho mỗi batch (0 = tắt)")
    args = ap.parse_args()
    epochs = args.epochs or (config.OPT_EPOCHS if args.recipe == "opt" else config.EPOCHS)
    archs = list(net.ARCHS) if args.arch == "all" else [args.arch]
    for a in archs:
        train_one(a, epochs, args.recipe, args.teacher, args.kd_alpha, args.kd_temp,
                  args.extra_data, args.cifar_repeat, args.init, args.lr, args.warmup, args.seed,
                  args.ema, args.cutmix)


if __name__ == "__main__":
    main()
