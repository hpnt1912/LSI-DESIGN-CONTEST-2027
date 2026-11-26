"""Đo ảnh GẦN TRÙNG giữa dữ liệu huấn luyện thêm (CINIC, 84.000 ảnh) và 4.000 ảnh test CIFAR-4.

Vì sao: prepare_cinic4.py chỉ loại ảnh trùng TUYỆT ĐỐI từng pixel (dòng 53–57). Một ảnh test có "bản sao" hơi
khác (cắt lệch, nén lại, lật ngang) trong dữ liệu train sẽ làm accuracy test cao hơn thực tế (rò rỉ – leakage).
Để có mốc so sánh, script đo cùng chỉ số giữa CIFAR train (20.000 ảnh) và CIFAR test: chính CIFAR-10 cũng
được biết là có một ít ảnh test gần trùng ảnh train.

Cách đo: với mỗi ảnh test (và bản lật ngang của nó), tìm ảnh train giống nhất theo hệ số tương quan
(correlation) trên ảnh thu nhỏ 16×16, rồi tính lại hệ số tương quan trên ảnh đầy đủ 32×32 cho cặp đó.
Tương quan 1,0 = giống hệt (sai khác chỉ ở độ sáng/tương phản). Ngưỡng nào là "trùng" phải xem bằng mắt:
script lưu hình các cặp giống nhất để kiểm tra.

Chạy từ thư mục gốc repo (cần data/cifar4.npz và data/cinic4_extra.npz, vài phút trên CPU):
    python documentation/tools/check_cinic_leak.py
"""

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
OUT_FIG = REPO / "documentation" / "assets" / "other_visualizations" / "cinic_near_duplicates.png"
CLASSES = ["cat", "dog", "deer", "horse"]
THRESHOLDS = (0.99, 0.98, 0.97, 0.95, 0.90)


def feat(x, size):
    """(N,32,32,3) uint8 -> vector đã trừ trung bình và chuẩn hoá độ dài 1 (tích vô hướng = hệ số tương quan)."""
    t = torch.from_numpy(x).float().permute(0, 3, 1, 2)
    if size != 32:
        t = torch.nn.functional.avg_pool2d(t, 32 // size)
    t = t.flatten(1)
    t = t - t.mean(1, keepdim=True)
    return t / t.norm(dim=1, keepdim=True).clamp_min(1e-6)


def best_match(x_query, x_pool, chunk=8192):
    """Với mỗi ảnh query (xét cả bản lật ngang): chỉ số ảnh giống nhất trong pool và tương quan ở 32×32."""
    q = torch.cat([feat(x_query, 16), feat(x_query[:, :, ::-1].copy(), 16)])   # [ảnh gốc; ảnh lật]
    best_s = torch.full((len(q),), -2.0)
    best_i = torch.zeros(len(q), dtype=torch.long)
    for a in range(0, len(x_pool), chunk):
        s, i = (q @ feat(x_pool[a:a + chunk], 16).T).max(1)
        better = s > best_s
        best_s[better], best_i[better] = s[better], i[better] + a
    n = len(x_query)
    flip = best_s[n:] > best_s[:n]                                       # bản lật giống hơn?
    idx = torch.where(flip, best_i[n:], best_i[:n]).numpy()
    xq = np.where(flip.numpy()[:, None, None, None], x_query[:, :, ::-1], x_query)
    full = (feat(np.ascontiguousarray(xq), 32) * feat(x_pool[idx], 32)).sum(1).numpy()
    return idx, full, flip.numpy()


def report(name, corr, same_label):
    print(f"\n{name}: số ảnh test (trên 4.000) có ảnh train tương quan ≥ ngưỡng (ở 32×32)")
    for t in THRESHOLDS:
        m = corr >= t
        print(f"  ≥ {t:.2f}: {m.sum():4d} ảnh ({m.mean() * 100:.2f}%), trong đó cùng nhãn {(m & same_label).sum()}")


def save_pairs(x_te, y_te, x_tr, y_tr, idx, corr, flip, n=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-corr)[:n]
    fig, axes = plt.subplots(4, 12, figsize=(14, 5.6))
    for k, i in enumerate(order):
        r, c = divmod(k, 6)
        a, b = axes[r, 2 * c], axes[r, 2 * c + 1]
        a.imshow(x_te[i])
        a.set_title(f"test {CLASSES[y_te[i]]}", fontsize=7)
        b.imshow(x_tr[idx[i]][:, ::-1] if flip[i] else x_tr[idx[i]])
        b.set_title(f"CINIC {CLASSES[y_tr[idx[i]]]}\nr={corr[i]:.3f}", fontsize=7)
        a.axis("off"), b.axis("off")
    fig.suptitle("24 cặp (ảnh test CIFAR, ảnh CINIC giống nhất) có tương quan cao nhất – kiểm tra bằng mắt", fontsize=10)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110)
    plt.close(fig)
    print(f"\nĐã lưu {OUT_FIG.relative_to(REPO)} (cặp giống nhất đứng đầu; ảnh CINIC đã lật nếu bản lật giống hơn)")
    return order


def main():
    d = np.load(REPO / "data" / "cifar4.npz")
    c = np.load(REPO / "data" / "cinic4_extra.npz")
    x_tr, y_tr, x_te, y_te = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    torch.set_grad_enabled(False)

    idx_c, corr_c, flip_c = best_match(x_te, c["x"])
    idx_t, corr_t, _ = best_match(x_te, x_tr)
    report("CINIC (84.000 ảnh) ↔ test", corr_c, c["y"][idx_c] == y_te)
    report("Mốc so sánh – CIFAR train (20.000 ảnh) ↔ test", corr_t, y_tr[idx_t] == y_te)
    order = save_pairs(x_te, y_te, c["x"], c["y"], idx_c, corr_c, flip_c)
    print("\nTop 24 (chỉ số ảnh test, chỉ số CINIC, tương quan, cùng nhãn):")
    for i in order:
        print(f"  test {i:4d}  cinic {idx_c[i]:5d}  r={corr_c[i]:.4f}  {'cùng' if c['y'][idx_c[i]] == y_te[i] else 'khác'} nhãn")
    accuracy_by_subset(x_te, y_te, corr_c)


def accuracy_by_subset(x_te, y_te, corr):
    """Accuracy số nguyên (golden, qparams có sẵn trong repo) của 5 mô hình trên từng nhóm ảnh test.
    B không học CINIC nên là ĐỐI CHỨNG: nếu B cũng đúng nhiều hơn ở nhóm 'trùng' thì nhóm đó chỉ là ảnh dễ."""
    import sys
    sys.path.insert(0, str(REPO / "model"))
    import config
    import golden_int
    from kiem_chung import MODELS, ORDER

    groups = [("tất cả", np.ones(len(y_te), bool)), ("r ≥ 0,90", corr >= 0.90),
              ("0,80 ≤ r < 0,90", (corr >= 0.80) & (corr < 0.90)), ("r < 0,90", corr < 0.90), ("r < 0,80", corr < 0.80)]
    print("\nAccuracy số nguyên (%) theo nhóm ảnh test – r = tương quan với ảnh CINIC giống nhất:")
    print(f"  {'':5s}" + "".join(f"{g:>17s}" for g, _ in groups))
    print(f"  {'số ảnh':5s}" + "".join(f"{int(m.sum()):>17d}" for _, m in groups))
    for k in ORDER:
        q = golden_int.load_qparams(config.export_dir_for(MODELS[k]["run"]) / "qparams.npz")
        ok = np.concatenate([golden_int.infer(x_te[i:i + 500], q).argmax(1) for i in range(0, len(x_te), 500)]) == y_te
        print(f"  {k:5s}" + "".join(f"{ok[m].mean() * 100:17.2f}" for _, m in groups))


if __name__ == "__main__":
    main()
