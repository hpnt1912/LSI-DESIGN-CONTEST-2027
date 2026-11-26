"""Sinh sơ đồ (SVG) và biểu đồ cho bộ tài liệu, lấy số liệu TRỰC TIẾP từ source code của dự án.

Nguồn số liệu (không gõ tay con số nào):
  - Cấu trúc 5 mô hình, lệnh huấn luyện: model/net.py, model/kiem_chung.py (MODELS, ORDER)
  - Tham số / MAC / kích thước tensor: chạy net.build(...) của PyTorch
  - Thông số lượng tử hoá: model/export*/qparams.json (đã commit)
  - Accuracy, thời gian huấn luyện, ma trận nhầm lẫn: model/results.md, docs/so_sanh_5_mo_hinh.md

Chạy từ thư mục gốc repo (venv có torch + matplotlib):
    python documentation/tools/make_figures.py
Chỉ ghi vào documentation/assets/. Không sửa file nào khác của dự án.
"""

import json
import re
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import escape

DOC = Path(__file__).resolve().parents[1]
REPO = DOC.parent
sys.path.insert(0, str(REPO / "model"))

import torch  # noqa: E402

import config  # noqa: E402
import kiem_chung  # noqa: E402
import net  # noqa: E402

ASSETS = DOC / "assets"
ARCH_DIR = ASSETS / "architecture_diagrams"
PIPE_DIR = ASSETS / "pipeline_diagrams"
VIS_DIR = ASSETS / "other_visualizations"
FIG_DIR = ASSETS / "project_figures"

ORDER = kiem_chung.ORDER                         # ["B", "B3c", "E", "E2", "E3d"]
MODELS = kiem_chung.MODELS
ARCH_NO = {k: i + 1 for i, k in enumerate(ORDER)}

# Bảng màu (light surface) – ô 1..4 của bảng màu chuẩn, đã chạy validate_palette.js (PASS)
BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
INK, INK2, GRID = "#191919", "#666666", "#d9d9d9"
FILL = {"input": "#efefef", "conv": "#cce3fd", "bn": "#e6f0fb", "relu": "#e6f0fb", "pool": "#fde3d3",
        "nopool": "#f7f7f7", "flat": "#efefef", "fc": "#fde3d3", "out": "#efefef"}
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"


# ---------------------------------------------------------------------------------------------------------
# Số liệu từ code
# ---------------------------------------------------------------------------------------------------------
def layer_table(run):
    """Mỗi tầng conv: (C_in, C_out, H_in, H_out_conv, pool, H_out), tham số PyTorch/phần cứng, MAC."""
    m = net.build(run)
    rows, c_in, size = [], 3, 32
    for c_out, pool in net._layer_cfg(m):
        out = size // 2 if pool else size
        rows.append(dict(c_in=c_in, c_out=c_out, h_in=size, h_conv=size, pool=pool, h_out=out,
                         p_torch=c_in * c_out * 9 + 2 * c_out, p_hw=c_in * c_out * 9 + c_out,
                         mac=size * size * c_out * c_in * 9))
        c_in, size = c_out, out
    n_flat = c_in * size * size
    fc = dict(n_in=n_flat, p=n_flat * 4 + 4, mac=n_flat * 4, c=c_in, h=size)
    # đối chiếu với PyTorch thật
    assert sum(r["p_torch"] for r in rows) + fc["p"] == sum(t.numel() for t in m.parameters())
    assert sum(r["p_hw"] for r in rows) + fc["p"] == net.count_params(m)
    assert sum(r["mac"] for r in rows) + fc["mac"] == net.count_macs(m)
    assert tuple(m(torch.zeros(1, 3, 32, 32)).shape) == (1, 4)
    return rows, fc


def qparams(run):
    return json.loads((config.export_dir_for(run) / "qparams.json").read_text())


def summary_numbers():
    """Đọc bảng tự sinh docs/so_sanh_5_mo_hinh.md (số chạy thật trên laptop của nhóm)."""
    text = (REPO / "docs" / "so_sanh_5_mo_hinh.md").read_text()

    def row(label):
        ln = next(x for x in text.splitlines() if x.startswith(f"| {label}"))
        return [c.strip().strip("*") for c in ln.strip().strip("|").split("|")[1:]]

    num = lambda s: float(re.sub(r"[^\d,]", "", s).replace(",", "."))
    ints = [num(c) for c in row("**Test số nguyên**")]
    floats = [num(c) for c in row("Test float")]
    per = [[float(v.replace(",", ".")) for v in c.split("/")] for c in row("Accuracy từng lớp")]
    tmin = [num(c) for c in row("Thời gian huấn luyện")]
    return {k: dict(int=ints[i], float=floats[i], per_class=per[i], train_min=tmin[i]) for i, k in enumerate(ORDER)}


# ---------------------------------------------------------------------------------------------------------
# SVG tiện ích
# ---------------------------------------------------------------------------------------------------------
class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.items = w, h, []
        self.title = title

    def rect(self, x, y, w, h, fill, stroke="#8c8c8c", r=6, sw=1):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=13, anchor="middle", weight="normal", color=INK, italic=False):
        st = ' font-style="italic"' if italic else ""
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" fill="{color}"{st}>{escape(s)}</text>')

    def arrow(self, x1, y1, x2, y2, color=INK2, dash=False):
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"'
                          f' marker-end="url(#ah)"{d}/>')

    def path(self, d, color=INK2, dash=False):
        ds = ' stroke-dasharray="5 4"' if dash else ""
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" marker-end="url(#ah)"{ds}/>')

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}" role="img">\n'
                f'<title>{escape(self.title)}</title>\n'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>\n'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n')
        path.write_text(head + "\n".join(self.items) + "\n</svg>\n", encoding="utf-8")


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


# ---------------------------------------------------------------------------------------------------------
# 1) Sơ đồ kiến trúc từng mô hình
# ---------------------------------------------------------------------------------------------------------
def arch_diagram(key):
    run = MODELS[key]["run"]
    rows, fc = layer_table(run)
    q = qparams(run)["layers"]
    W, row_h, gap = 900, 54, 34
    n = len(rows) + 4                                   # input + conv rows + flatten + fc + output
    H = 92 + n * row_h + (n - 1) * gap + 40
    s = Svg(W, H, f"Kiến trúc {ARCH_NO[key]} – {key} ({run})")
    s.text(W / 2, 30, f"Kiến trúc {ARCH_NO[key]}: {key} – {MODELS[key]['vai_tro']}", 18, weight="bold")
    s.text(W / 2, 52, f"Lượt chạy: {run}   ·   nguồn: model/net.py + model/export_*/qparams.json", 12, color=INK2)
    y = 72
    x0, bw = 150, 470                                   # vùng hộp chính
    sx = x0 + bw + 24                                   # cột chú thích bên phải

    def label_left(txt, yy):
        s.text(x0 - 14, yy + row_h / 2 + 5, txt, 13, anchor="end", weight="bold")

    # input
    s.rect(x0, y, bw, row_h, FILL["input"])
    s.text(x0 + bw / 2, y + 22, "Ảnh vào: 3 × 32 × 32 (R, G, B)", 14, weight="bold")
    s.text(x0 + bw / 2, y + 41, "pixel uint8 0..255  →  x = pixel / 256  (data.to_tensor)", 12, color=INK2)
    label_left("Input", y)
    s.text(sx, y + 32, "tensor (N, 3, 32, 32)", 12, anchor="start", color=INK2)
    for i, r in enumerate(rows):
        s.arrow(x0 + bw / 2, y + row_h, x0 + bw / 2, y + row_h + gap - 2)
        y += row_h + gap
        segs = [("conv", f"Conv 3×3  {r['c_in']}→{r['c_out']}", 0.40), ("bn", "BN", 0.13), ("relu", "ReLU", 0.15),
                ("pool" if r["pool"] else "nopool", "MaxPool 2×2" if r["pool"] else "không pool", 0.32)]
        xx = x0
        for kind, txt, frac in segs:
            ww = bw * frac
            s.rect(xx, y, ww - 4, row_h, FILL[kind], stroke="#8c8c8c" if kind != "nopool" else "#c4c4c4")
            s.text(xx + (ww - 4) / 2, y + 23, txt, 13, weight="bold" if kind == "conv" else "normal",
                   color=INK if kind != "nopool" else INK2, italic=kind == "nopool")
            if kind == "conv":
                s.text(xx + (ww - 4) / 2, y + 41, "stride 1, padding 1", 11, color=INK2)
            if kind in ("pool",):
                s.text(xx + (ww - 4) / 2, y + 41, "stride 2", 11, color=INK2)
            xx += ww
        wb = q[i].get("wbits", 8)
        label_left(f"conv{i + 1}", y)
        s.text(sx, y + 18, f"ra: {r['c_out']} × {r['h_out']} × {r['h_out']}"
               + ("" if r["pool"] else "  (không pool)"), 13, anchor="start", weight="bold")
        s.text(sx, y + 35, f"tham số {fmt_int(r['p_hw'])} (sau gộp BN) · MAC {fmt_int(r['mac'])}", 11,
               anchor="start", color=INK2)
        s.text(sx, y + 50, f"trọng số int{wb}, shift {q[i]['shift']}", 11, anchor="start", color=INK2)
    # flatten
    s.arrow(x0 + bw / 2, y + row_h, x0 + bw / 2, y + row_h + gap - 2)
    y += row_h + gap
    s.rect(x0, y, bw, row_h, FILL["flat"])
    s.text(x0 + bw / 2, y + 23, f"Flatten: {fc['c']} × {fc['h']} × {fc['h']} → {fmt_int(fc['n_in'])}", 14,
           weight="bold")
    s.text(x0 + bw / 2, y + 41, "duỗi theo thứ tự (kênh, hàng, cột) – torch.flatten(x, 1)", 12, color=INK2)
    label_left("Flatten", y)
    s.text(sx, y + 32, "không có tham số", 12, anchor="start", color=INK2)
    # fc
    s.arrow(x0 + bw / 2, y + row_h, x0 + bw / 2, y + row_h + gap - 2)
    y += row_h + gap
    s.rect(x0, y, bw, row_h, FILL["fc"])
    s.text(x0 + bw / 2, y + 23, f"Linear (Fully Connected): {fmt_int(fc['n_in'])} → 4", 14, weight="bold")
    s.text(x0 + bw / 2, y + 41, "logit = W·x + b  (không có ReLU, không softmax khi suy luận)", 12, color=INK2)
    label_left("fc", y)
    s.text(sx, y + 25, f"tham số {fmt_int(fc['p'])} = {fmt_int(fc['n_in'])}×4 + 4", 12, anchor="start",
           color=INK2)
    s.text(sx, y + 42, f"MAC {fmt_int(fc['mac'])}", 12, anchor="start", color=INK2)
    # output
    s.arrow(x0 + bw / 2, y + row_h, x0 + bw / 2, y + row_h + gap - 2)
    y += row_h + gap
    s.rect(x0, y, bw, row_h, FILL["out"])
    s.text(x0 + bw / 2, y + 23, "4 logit: [cat, dog, deer, horse]", 14, weight="bold")
    s.text(x0 + bw / 2, y + 41, "lớp dự đoán = argmax(logit)  (phần cứng chỉ cần so sánh 4 số)", 12, color=INK2)
    label_left("Output", y)
    tot_p = sum(r["p_hw"] for r in rows) + fc["p"]
    tot_m = sum(r["mac"] for r in rows) + fc["mac"]
    s.text(W / 2, H - 16, f"Tổng: {fmt_int(tot_p)} tham số (sau gộp BN) · {fmt_int(tot_m)} MAC/ảnh · "
           f"{len(rows)} tầng conv + 1 Linear", 13, weight="bold")
    out = ARCH_DIR / f"arch{ARCH_NO[key]}_{key}.svg"
    s.save(out)
    return out


# ---------------------------------------------------------------------------------------------------------
# 2) Sơ đồ quy trình
# ---------------------------------------------------------------------------------------------------------
def box(s, x, y, w, h, title, sub="", fill="#efefef", bold=True):
    s.rect(x, y, w, h, fill)
    if sub:
        s.text(x + w / 2, y + h / 2 - 4, title, 13, weight="bold" if bold else "normal")
        s.text(x + w / 2, y + h / 2 + 13, sub, 11, color=INK2)
    else:
        s.text(x + w / 2, y + h / 2 + 5, title, 13, weight="bold" if bold else "normal")


def pipeline_overview():
    W, H = 980, 760
    s = Svg(W, H, "Pipeline toàn dự án: dữ liệu → huấn luyện → số nguyên → RTL")
    s.text(W / 2, 30, "Pipeline toàn dự án (mỗi hộp ghi file Python thực hiện)", 18, weight="bold")
    # cột dữ liệu
    box(s, 30, 60, 250, 50, "CIFAR-10 (tải tự động)", "torchvision.datasets.CIFAR10")
    box(s, 30, 140, 250, 50, "prepare_cifar4.py", "giữ 4 lớp cat/dog/deer/horse", "#cce3fd")
    box(s, 30, 220, 250, 50, "data/cifar4.npz", "20.000 train · 4.000 test")
    box(s, 700, 60, 250, 50, "CINIC-10.tar.gz (687 MB)", "tải tay – Đại học Edinburgh")
    box(s, 700, 140, 250, 50, "prepare_cinic4.py", "chỉ lấy ảnh ImageNet, bỏ ảnh CIFAR", "#cce3fd")
    box(s, 700, 220, 250, 50, "data/cinic4_extra.npz", "84.000 ảnh (B3c, E, E2, E3d)")
    s.arrow(155, 110, 155, 138); s.arrow(155, 190, 155, 218)
    s.arrow(825, 110, 825, 138); s.arrow(825, 190, 825, 218)
    # data.py
    box(s, 330, 300, 320, 56, "data.py – get_loaders()", "tách 18.000 train / 2.000 val · augment · batch", "#cce3fd")
    s.path("M155,270 L155,328 L328,328"); s.path("M825,270 L825,328 L652,328", dash=True)
    s.text(815, 300, "--extra-data (4 mô hình +cinic)", 11, anchor="end", color=INK2)
    # net + train
    box(s, 30, 400, 250, 56, "net.py – build(run)", "SmallCNN (B) / VGGLite (4 mô hình)", "#cce3fd")
    box(s, 330, 400, 320, 56, "train.py – train_one()", "forward → loss → backward → SGD/AdamW", "#cce3fd")
    s.arrow(490, 356, 490, 398); s.arrow(280, 428, 328, 428)
    box(s, 700, 400, 250, 56, "checkpoints/<run>.pt", "trọng số float tốt nhất theo val")
    s.arrow(650, 428, 698, 428)
    s.text(490, 474, "B3c/E/E2/E3d: 2 lượt train (+cinic → tinh chỉnh _ft/_kdft)", 11, color=INK2)
    # evaluate & quantize
    box(s, 700, 500, 250, 56, "evaluate.py", "accuracy, ma trận nhầm lẫn, hình", "#cce3fd")
    box(s, 330, 500, 320, 56, "quantize.py", "gộp BN · hiệu chuẩn · int8/int16, shift", "#cce3fd")
    s.arrow(825, 456, 825, 498); s.path("M760,456 L760,488 L490,488 L490,498")
    box(s, 30, 500, 250, 56, "export_<run>/qparams.npz/.json", "tham số số nguyên")
    s.arrow(330, 528, 282, 528)
    box(s, 30, 600, 250, 56, "golden_int.py", "suy luận chỉ bằng số nguyên", "#cce3fd")
    box(s, 330, 600, 320, 56, "export.py", "weights/*.mem · testvec/*.mem · README", "#cce3fd")
    box(s, 700, 600, 250, 56, "RTL Verilog (rtl/, tb/)", "chưa có trong repo – bước sau")
    s.arrow(155, 556, 155, 598); s.arrow(280, 628, 328, 628); s.arrow(650, 628, 698, 628, dash=True)
    # kết quả
    box(s, 30, 690, 920, 50, "results.md · figures/*.png · docs/so_sanh_5_mo_hinh.md · kiem_chung_log.json",
        "kiem_chung.py gọi lần lượt train → evaluate → quantize → golden → export cho 5 mô hình và tự cập nhật các file này",
        "#fde3d3")
    s.save(PIPE_DIR / "pipeline_overview.svg")


def training_loop():
    W, H = 900, 668
    s = Svg(W, H, "Vòng lặp huấn luyện trong train.py (train_one)")
    s.text(W / 2, 30, "Vòng lặp huấn luyện – model/train.py, hàm train_one() (dòng 77–164)", 17, weight="bold")
    steps = [
        ("Khởi tạo", "set_seed · net.build · (--init) · (thầy) · loss/optimizer/scheduler", "dòng 80–108"),
        ("Lấy 1 batch", "for x, y in train_loader  (augment ngẫu nhiên mỗi lần lấy)", "dòng 119"),
        ("1) Lan truyền xuôi", "logits = model(x)    – (thầy: t_logits = teacher(x))", "dòng 123, 129"),
        ("2) Tính loss", "CrossEntropy(+label smoothing)  · (+ distill_loss)", "dòng 124, 130"),
        ("3) Xoá gradient cũ", "opt.zero_grad()", "dòng 131"),
        ("4) Lan truyền ngược", "loss.backward()  → ∂loss/∂w cho mọi trọng số", "dòng 132"),
        ("5) Cập nhật trọng số", "opt.step()   (AdamW cho B · SGD nesterov cho 4 mô hình còn lại)", "dòng 133"),
    ]
    x, w, h, y = 120, 520, 50, 56
    ys = []
    for i, (t, sub, ln) in enumerate(steps):
        box(s, x, y, w, h, t, sub, "#cce3fd" if i else "#efefef")
        s.text(x + w + 12, y + h / 2 + 4, ln, 11, anchor="start", color=INK2)
        ys.append(y)
        if i < len(steps) - 1:
            s.arrow(x + w / 2, y + h, x + w / 2, y + h + 14)
        y += h + 16
    batch_y = ys[1] + h / 2
    # vòng lặp batch (bên trái): từ bước 5 quay về "Lấy 1 batch"
    s.path(f"M{x},{ys[-1] + h / 2} L{x - 40},{ys[-1] + h / 2} L{x - 40},{batch_y} L{x - 2},{batch_y}")
    s.text(x - 46, (batch_y + ys[-1] + h / 2) / 2, "batch tiếp", 11, anchor="end", color=INK2)
    s.arrow(x + w / 2, y - 16, x + w / 2, y + 6)
    y += 6
    box(s, x, y, w, h, "Hết 1 epoch", "sched.step() · evaluate(model, val) · nếu val tốt nhất → torch.save", "#fde3d3")
    s.text(x + w + 12, y + h / 2 + 4, "dòng 139–151", 11, anchor="start", color=INK2)
    # vòng lặp epoch (bên phải): từ "Hết 1 epoch" quay về "Lấy 1 batch"
    rx = x + w + 110
    s.path(f"M{x + w},{y + h / 2 - 10} L{rx},{y + h / 2 - 10} L{rx},{batch_y - 12} L{x + w + 2},{batch_y - 12}")
    s.text(rx + 6, (batch_y + y) / 2, "epoch tiếp", 11, anchor="start", color=INK2)
    s.arrow(x + w / 2, y + h, x + w / 2, y + h + 14)
    y += h + 16
    box(s, x, y, w, h, "Kết thúc", "nạp checkpoint tốt nhất → đo test 1 lần → vẽ đường học → ghi results.md",
        "#efefef")
    s.text(x + w + 12, y + h / 2 + 4, "dòng 155–163", 11, anchor="start", color=INK2)
    s.save(PIPE_DIR / "training_loop.svg")


def quant_flow():
    W, H = 900, 560
    s = Svg(W, H, "Từ mô hình float sang số nguyên cho phần cứng")
    s.text(W / 2, 30, "Từ mô hình float đến file .mem cho RTL – model/quantize.py, golden_int.py, export.py", 16,
           weight="bold")
    steps = [
        ("checkpoints/<run>.pt", "mạng float: Conv(bias=False) → BN → ReLU → [MaxPool] … → Linear", "#efefef", ""),
        ("1. Gộp BN – fold_bn()", "W' = k·W,  b' = k·(b − μ) + β,  k = γ/√(σ²+ε)", "#cce3fd", "quantize.py 34–65"),
        ("2. Hiệu chuẩn – calib_amax()", "2.000 ảnh train → phân vị 99,99% activation sau ReLU mỗi tầng",
         "#cce3fd", "quantize.py 84–96"),
        ("3. Chọn thang lũy thừa 2 – frac_bits_for()", "f = ⌊log2(q_max / max|v|)⌋  → fw, f_out, shift",
         "#cce3fd", "quantize.py 99–130"),
        ("4. Làm tròn", "Wq = round(W·2^fw) int8 (conv1 int16) · bq = round(b·2^(f_in+fw)) int32",
         "#cce3fd", "quantize.py 113–117"),
        ("5. Golden model số nguyên – golden_int.infer()", "acc = ΣWq·xq + bq ; y = clip((acc + 2^(s−1)) >> s, 0, 255)",
         "#cce3fd", "golden_int.py 43–84"),
        ("6. So khớp từng bit", "torch_int_forward() (cách tính thứ hai) phải ra logit giống hệt",
         "#cce3fd", "quantize.py 147–161, 199–207"),
        ("7. Xuất cho RTL – export.py", "weights/*.mem (hex bù 2) · testvec/*.mem · labels.csv · README.md",
         "#fde3d3", "export.py 43–118"),
    ]
    x, w, h, y = 40, 600, 50, 52
    for i, (t, sub, fill, ln) in enumerate(steps):
        box(s, x, y, w, h, t, sub, fill)
        if ln:
            s.text(x + w + 14, y + h / 2 + 4, ln, 11, anchor="start", color=INK2)
        if i < len(steps) - 1:
            s.arrow(x + w / 2, y + h, x + w / 2, y + h + 10)
        y += h + 12
    s.save(PIPE_DIR / "quantization_flow.svg")


def run_order():
    W, H = 900, 390
    s = Svg(W, H, "Thứ tự chạy 5 mô hình trong kiem_chung.py")
    s.text(W / 2, 30, "kiem_chung.py: các bước của 5 mô hình và phụ thuộc mạng thầy", 16, weight="bold")
    y = 60
    for k in ORDER:
        m = MODELS[k]
        stages = [c[0].replace(".py", "") + ("" if c[0] != "train.py" else f" {c[2]}") for c in m["stages"]]
        trains = [c for c in m["stages"] if c[0] == "train.py"]
        x = 30
        s.text(x, y + 22, f"{k}", 14, anchor="start", weight="bold")
        x += 52
        for c in trains:
            run = c[2]
            ep = c[c.index("--epochs") + 1] if "--epochs" in c else str(config.EPOCHS)
            extra = "+CINIC" if "--extra-data" in c else ("CIFAR" if "--recipe" in c or k == "B" else "")
            kd = " +thầy E" if "--teacher" in c else ""
            box(s, x, y, 230, 40, f"train {run}", f"{ep} epoch · {extra}{kd}", "#cce3fd", bold=False)
            s.arrow(x + 230, y + 20, x + 244, y + 20)
            x += 246
        for t in ["evaluate", "quantize", "golden", "export"]:
            box(s, x, y, 66 if t != "quantize" else 74, 40, t, "", "#efefef", bold=False)
            x += (66 if t != "quantize" else 74) + 6
        y += 58
        del stages
    s.text(30, 350, "Mạng thầy: ở lượt tinh chỉnh _kdft, E2 và E3d dùng checkpoint E_opt_cnn+cinic_ft (--teacher).",
           12, anchor="start", color=INK2)
    s.text(30, 368, "Vì vậy kiem_chung.py luôn xếp E chạy trước E2 và E3d (kiem_chung.py dòng 576–580).", 12,
           anchor="start", color=INK2)
    s.save(PIPE_DIR / "run_order.svg")


# ---------------------------------------------------------------------------------------------------------
# 3) Biểu đồ (matplotlib → SVG)
# ---------------------------------------------------------------------------------------------------------
def charts(nums):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#8c8c8c", "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    macs = {k: net.count_macs(net.build(MODELS[k]["run"])) / 1e6 for k in ORDER}
    params = {k: net.count_params(net.build(MODELS[k]["run"])) for k in ORDER}

    # (a) accuracy số nguyên theo MAC – một chuỗi, nhãn trực tiếp
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs, ys = [macs[k] for k in ORDER], [nums[k]["int"] for k in ORDER]
    ax.scatter(xs, ys, s=70, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
    offs = {"B": (8, -4), "B3c": (8, -4), "E": (-10, -14), "E2": (8, -12), "E3d": (-6, 8)}
    for k in ORDER:
        dx, dy = offs[k]
        ax.annotate(f"{k}  {nums[k]['int']:.2f}%".replace(".", ","), (macs[k], nums[k]["int"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=10, color=INK,
                    ha="right" if dx < 0 else "left")
    ax.set_xlabel("Triệu phép nhân–cộng (MAC) mỗi ảnh  –  càng ít càng nhanh")
    ax.set_ylabel("Test số nguyên (%)")
    ax.set_title("Độ chính xác (kết quả FPGA) theo khối lượng tính – 5 kiến trúc", fontsize=12, color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlim(0, 22)
    ax.set_ylim(78, 92)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "acc_vs_mac.svg")
    plt.close(fig)

    # (b) tham số & MAC theo tầng – small multiples, một màu
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6), sharey="row")
    for j, k in enumerate(ORDER):
        rows, fc = layer_table(MODELS[k]["run"])
        names = [f"conv{i + 1}" for i in range(len(rows))] + ["fc"]
        p = [r["p_hw"] / 1000 for r in rows] + [fc["p"] / 1000]
        m = [r["mac"] / 1e6 for r in rows] + [fc["mac"] / 1e6]
        for ax, vals, col in ((axes[0, j], p, BLUE), (axes[1, j], m, ORANGE)):
            ax.bar(names, vals, color=col, width=0.65, edgecolor="white", linewidth=2)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
        axes[0, j].set_title(f"{ARCH_NO[k]}. {k}", fontsize=11, color=INK)
    axes[0, 0].set_ylabel("Nghìn tham số\n(sau gộp BN)")
    axes[1, 0].set_ylabel("Triệu MAC")
    fig.suptitle("Tham số (hàng trên) và phép nhân–cộng (hàng dưới) theo từng tầng", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "params_mac_per_layer.svg")
    plt.close(fig)

    # (c) accuracy từng lớp (số nguyên) – 4 chuỗi, 4 ô màu đầu, có chú giải
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    cls = config.CLASSES
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    xpos = np.arange(len(ORDER))
    bw = 0.2
    for i, c in enumerate(cls):
        ax.bar(xpos + (i - 1.5) * bw, [nums[k]["per_class"][i] for k in ORDER], bw, color=cols[i], label=c,
               edgecolor="white", linewidth=1.5)
    ax.set_xticks(xpos, [f"{ARCH_NO[k]}. {k}" for k in ORDER])
    ax.set_ylim(60, 100)
    ax.set_ylabel("Accuracy số nguyên từng lớp (%)")
    ax.set_title("Lớp nào khó? Accuracy từng lớp (số nguyên) – cat luôn thấp nhất", fontsize=12, color=INK)
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "per_class_accuracy.svg")
    plt.close(fig)

    # (d) tổng tham số và thời gian huấn luyện – hai biểu đồ riêng (không dùng 2 trục y)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = [f"{ARCH_NO[k]}. {k}" for k in ORDER]
    a1.barh(labels, [params[k] / 1000 for k in ORDER], color=BLUE, edgecolor="white", linewidth=2)
    for i, k in enumerate(ORDER):
        a1.text(params[k] / 1000 + 2, i, fmt_int(params[k]), va="center", fontsize=9, color=INK)
    a1.set_xlabel("Nghìn tham số (sau gộp BN)")
    a1.invert_yaxis()
    a1.set_xlim(0, 175)
    a2.barh(labels, [nums[k]["train_min"] for k in ORDER], color=BLUE, edgecolor="white", linewidth=2)
    for i, k in enumerate(ORDER):
        a2.text(nums[k]["train_min"] + 1.5, i, f"{nums[k]['train_min']:.0f} phút", va="center", fontsize=9,
                color=INK)
    a2.set_xlabel("Phút huấn luyện trên laptop (chưa tính thời gian của thầy E)")
    a2.invert_yaxis()
    a2.set_xlim(0, 120)
    for a in (a1, a2):
        a.grid(axis="x", color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "params_and_training_time.svg")
    plt.close(fig)

    # (e) lịch learning rate – công thức y hệt train.py (CosineAnnealingLR và opt_recipe.lr_factor)
    import math
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))

    def base(e, T=30):
        return 0.5 * config.LR * (1 + math.cos(math.pi * e / T))

    def opt(e, E, warm, lr0):
        return lr0 * ((e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, E - warm))))

    cases = [("B – gốc: AdamW, 30 epoch", [base(e) for e in range(30)]),
             ("+cinic (B3c, E): SGD, 30 epoch", [opt(e, 30, config.OPT_WARMUP_EPOCHS, config.OPT_LR) for e in range(30)]),
             ("+cinic60 (E2, E3d): SGD, 60 epoch", [opt(e, 60, config.OPT_WARMUP_EPOCHS, config.OPT_LR) for e in range(60)]),
             ("tinh chỉnh _ft/_kdft: 15 epoch", [opt(e, 15, 0, 0.005) for e in range(15)])]
    for ax, (title, ys) in zip(axs, cases):
        ax.plot(range(1, len(ys) + 1), ys, color=BLUE, linewidth=2)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("epoch")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel("learning rate")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "lr_schedules.svg")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
# 4) Chép hình do chính dự án sinh ra (đã commit trong model/figures/)
# ---------------------------------------------------------------------------------------------------------
def copy_project_figures():
    names = ["samples.png", "so_sanh_5_mo_hinh.png"]
    for k in ORDER:
        m = MODELS[k]
        names += [f"confusion_{m['run']}.png", f"errors_{m['run']}.png"]
        names += [f"train_{r}.png" for r in m["train_runs"]]
    copied = []
    for n in names:
        src = config.FIG_DIR / n
        if src.exists():
            shutil.copy2(src, FIG_DIR / n)
            copied.append(n)
        else:
            print(f"  (không có {src.relative_to(REPO)})")
    return copied


def main():
    for d in (ARCH_DIR, PIPE_DIR, VIS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    nums = summary_numbers()
    for k in ORDER:
        print("✓", arch_diagram(k).relative_to(REPO))
    pipeline_overview()
    training_loop()
    quant_flow()
    run_order()
    charts(nums)
    copied = copy_project_figures()
    print(f"✓ sơ đồ quy trình, biểu đồ, {len(copied)} hình của dự án -> {ASSETS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
