"""Kiểm chứng 5 mô hình chính trên máy của bạn – một lệnh làm tất cả, tự cập nhật kết quả và đẩy lên GitHub.

Cách dùng (từ thư mục gốc repo, venv đang bật):
    python model/kiem_chung.py nhanh           # KIỂM TRA NHANH (vài phút): chạy golden model số nguyên trên
                                               #   tham số có sẵn của cả 5 mô hình, so với số liệu đã ghi
    python model/kiem_chung.py B               # huấn luyện lại + đánh giá + lượng tử hoá + golden + xuất .mem
    python model/kiem_chung.py B3c E E2        # nhiều mô hình (tự xếp đúng thứ tự phụ thuộc)
    python model/kiem_chung.py tatca           # cả 5 mô hình (nhiều giờ – nên chạy qua đêm)
    python model/kiem_chung.py bang            # chỉ vẽ lại bảng + hình so sánh từ kết quả hiện có

Tuỳ chọn:
    --khong-push   chỉ commit trên máy, không đẩy lên GitHub
    --lam-lai      bỏ qua phần đã xong ở lần chạy trước (mặc định: chạy TIẾP từ bước bị dừng)
    --thu          chạy thử: mọi lượt huấn luyện chỉ 1 epoch, không commit/push (để kiểm tra script)

Mỗi mô hình xong là: model/results.md, model/figures/, model/export*/ được ghi đè bằng kết quả của máy bạn,
bảng docs/so_sanh_5_mo_hinh.md + hình model/figures/so_sanh_5_mo_hinh.png được vẽ lại, trạng thái mô hình
chuyển thành "đã kiểm chứng trên <tên máy>", rồi tự git commit + push.
"""

import argparse
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import config

REPO = config.REPO_ROOT
LOG_FILE = REPO / "model" / "kiem_chung_log.json"
DOC_FILE = REPO / "docs" / "so_sanh_5_mo_hinh.md"
FIG_FILE = config.FIG_DIR / "so_sanh_5_mo_hinh.png"
CINIC_NPZ = REPO / "data" / "cinic4_extra.npz"
PY = sys.executable

# Công thức tinh chỉnh (fine-tune) chung cho các bản "+cinic...": 15 epoch chỉ trên ảnh CIFAR, LR 0,005.
FT = ["--recipe", "opt", "--lr", "0.005", "--warmup", "0", "--epochs", "15"]
EXTRA = ["--recipe", "opt", "--extra-data", "--cifar-repeat", "2"]
TEACHER = "E_opt_cnn+cinic_ft"


def stages_after_train(run, conv1_bits):
    q = ["quantize.py", "--arch", run] + (["--conv1-bits", "16"] if conv1_bits == 16 else [])
    return [["evaluate.py", "--arch", run], q, ["golden_int.py", "--arch", run], ["export.py", "--arch", run]]


# ---- 5 mô hình được chọn --------------------------------------------------------------------------------
# run        : tên lượt chạy cuối cùng (checkpoint, thư mục export, dòng trong results.md)
# train_runs : các lượt huấn luyện tạo ra mô hình (để cộng thời gian huấn luyện)
# ai_ref     : số liệu đã ghi trước đây (B: laptop của nhóm; B3c/E/E2: AI chạy trên máy cloud, nhánh
#              exp/opt-acc-90) – để so với kết quả kiểm chứng trên máy bạn
MODELS = {
    "B": dict(
        run="B_16_32_64", vai_tro="Cơ bản (~80%)", conv1_bits=8, cinic=False, needs_teacher=False,
        train_runs=["B_16_32_64"],
        stages=[["train.py", "--arch", "B_16_32_64"]] + stages_after_train("B_16_32_64", 8),
        ai_ref=dict(float=80.23, int=80.00, phut=4.3), du_kien="khoảng 5–10 phút"),
    "B3c": dict(
        run="B3c_c4x4_128+cinic_ft", vai_tro="Nhanh", conv1_bits=16, cinic=True, needs_teacher=False,
        train_runs=["B3c_c4x4_128+cinic", "B3c_c4x4_128+cinic_ft"],
        stages=[["train.py", "--arch", "B3c_c4x4_128+cinic", *EXTRA, "--epochs", "30"],
                ["train.py", "--arch", "B3c_c4x4_128+cinic_ft", "--init", "B3c_c4x4_128+cinic", *FT]]
        + stages_after_train("B3c_c4x4_128+cinic_ft", 16),
        ai_ref=dict(float=87.08, int=87.20, phut=45.5), du_kien="khoảng 1 giờ"),
    "E": dict(
        run="E_opt_cnn+cinic_ft", vai_tro="Chính xác cao (+ mạng thầy cho E2, E3d)", conv1_bits=16,
        cinic=True, needs_teacher=False,
        train_runs=["E_opt_cnn+cinic", "E_opt_cnn+cinic_ft"],
        stages=[["train.py", "--arch", "E_opt_cnn+cinic", *EXTRA, "--epochs", "30"],
                ["train.py", "--arch", "E_opt_cnn+cinic_ft", "--init", "E_opt_cnn+cinic", *FT]]
        + stages_after_train("E_opt_cnn+cinic_ft", 16),
        ai_ref=dict(float=90.65, int=90.60, phut=74.7), du_kien="khoảng 1,5 giờ"),
    "E2": dict(
        run="E2_lite+cinic60_kdft", vai_tro="Cân bằng ★ (khuyên dùng)", conv1_bits=16, cinic=True,
        needs_teacher=True,
        train_runs=["E2_lite+cinic60", "E2_lite+cinic60_kdft"],
        stages=[["train.py", "--arch", "E2_lite+cinic60", *EXTRA, "--epochs", "60"],
                ["train.py", "--arch", "E2_lite+cinic60_kdft", "--init", "E2_lite+cinic60", *FT,
                 "--teacher", TEACHER]]
        + stages_after_train("E2_lite+cinic60_kdft", 16),
        ai_ref=dict(float=90.33, int=90.30, phut=99.6), du_kien="khoảng 2 giờ (cần E làm thầy)"),
    "E3d": dict(
        run="E3d_deep5+cinic60_kdft", vai_tro="Thử nghiệm mới (5 conv)", conv1_bits=16, cinic=True,
        needs_teacher=True,
        train_runs=["E3d_deep5+cinic60", "E3d_deep5+cinic60_kdft"],
        stages=[["train.py", "--arch", "E3d_deep5+cinic60", *EXTRA, "--epochs", "60"],
                ["train.py", "--arch", "E3d_deep5+cinic60_kdft", "--init", "E3d_deep5+cinic60", *FT,
                 "--teacher", TEACHER]]
        + stages_after_train("E3d_deep5+cinic60_kdft", 16),
        ai_ref=None, du_kien="khoảng 2 giờ (ước tính, cần E làm thầy)"),
}
ORDER = ["B", "B3c", "E", "E2", "E3d"]


# ---- Nhật ký trạng thái ---------------------------------------------------------------------------------
def load_log():
    return json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else {}


def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def machine():
    return platform.node() or "may-khong-ten"


# ---- Chạy một lệnh con, in ra màn hình và ghi vào file nhật ký ------------------------------------------
def run_step(cmd, logf, smoke):
    cmd = list(cmd)
    if smoke and cmd[0] == "train.py":                 # chạy thử: chỉ 1 epoch
        if "--epochs" in cmd:
            cmd[cmd.index("--epochs") + 1] = "1"
        else:
            cmd += ["--epochs", "1"]
    full = [PY, str(REPO / "model" / cmd[0]), *cmd[1:]]
    line = "$ python model/" + " ".join(f'"{c}"' if "+" in c else c for c in cmd)
    print("\n" + line, flush=True)
    with open(logf, "a", encoding="utf-8") as lf:
        lf.write(f"\n[{now()}] {line}\n")
        p = subprocess.Popen(full, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for out in p.stdout:
            print(out, end="", flush=True)
            lf.write(out)
        p.wait()
    if p.returncode != 0:
        sys.exit(f"\n✗ Lỗi ở bước: {line}\n  Xem nhật ký: {logf}\n"
                 f"  Sửa xong chạy lại đúng lệnh cũ – script sẽ chạy TIẾP từ bước này.")


# ---- Dữ liệu CINIC-10 -----------------------------------------------------------------------------------
def ensure_cinic(smoke):
    if CINIC_NPZ.exists():
        return
    if smoke:
        sys.exit("Chạy thử cần data/cinic4_extra.npz (có thể tạo tạm). Bỏ --thu hoặc chuẩn bị CINIC trước.")
    cands = [Path.home() / "Downloads" / "CINIC-10.tar.gz", REPO / "data" / "CINIC-10.tar.gz"]
    cands += [Path(p) for p in glob.glob("/mnt/c/Users/*/Downloads/CINIC-10.tar.gz")]
    tar = next((c for c in cands if c.exists()), None)
    if tar is None:
        sys.exit("✗ Chưa có dữ liệu CINIC-10 (cần cho B3c, E, E2, E3d).\n"
                 "  1. Tải CINIC-10.tar.gz (687 MB) tại https://datashare.ed.ac.uk/handle/10283/3192\n"
                 "     (lưu vào thư mục Downloads của Windows là được).\n"
                 "  2. Chạy lại đúng lệnh này – script tự tìm file và chuẩn bị dữ liệu.\n"
                 "  Xem chi tiết: docs/KIEM_CHUNG.md")
    print(f"Tìm thấy {tar} -> chuẩn bị dữ liệu CINIC (vài phút)...")
    subprocess.run([PY, str(REPO / "model" / "prepare_cinic4.py"), "--tar", str(tar)], cwd=REPO, check=True)


# ---- Git ------------------------------------------------------------------------------------------------
def git(*args, check=False):
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=check)


def commit_push(message, runs, push):
    paths = ["model/results.md", "model/figures", "model/kiem_chung_log.json", str(DOC_FILE.relative_to(REPO))]
    for r in runs:
        d = config.export_dir_for(r)
        if d.exists():
            paths.append(str(d.relative_to(REPO)))
    git("add", "--", *[p for p in paths if (REPO / p).exists()])
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("(Không có gì thay đổi để commit.)")
        return
    git("commit", "-q", "-m", message, check=True)
    print(f"✓ Đã commit: {message}")
    if not push:
        return
    if git("push", "-q").returncode != 0:
        git("pull", "--no-edit", "-q")                  # có commit mới trên GitHub -> gộp rồi đẩy lại
        if git("push", "-q").returncode != 0:
            print("⚠ Không đẩy được lên GitHub. Kết quả vẫn đã commit trên máy; chạy `git pull` rồi `git push`.")
            return
    print("✓ Đã đẩy lên GitHub")


# ---- Số liệu cho bảng so sánh ---------------------------------------------------------------------------
def results_rows():
    """Đọc bảng 'So sánh kiến trúc' trong results.md -> {tên lượt: (val, test, phút)}."""
    rows = {}
    if not config.RESULTS_FILE.exists():
        return rows
    for ln in config.RESULTS_FILE.read_text().splitlines():
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) == 8 and c[5].endswith("%") and c[7].endswith("phút"):
            rows[c[0]] = (float(c[5][:-1]), float(c[6][:-1]), float(c[7].split()[0]))
    return rows


def hw_numbers(run, conv1_bits):
    """Tham số, MAC, bộ nhớ trọng số thật (byte) và ước lượng BRAM."""
    import net
    m = net.build(run)
    cfg = net._layer_cfg(m)
    params, macs = net.count_params(m), net.count_macs(m)
    w_bytes, c_in, size, buf = 0, 3, 32, 0
    for i, (c_out, pool) in enumerate(cfg):
        wb = 2 if (i == 0 and conv1_bits == 16) else 1
        w_bytes += c_in * c_out * 9 * wb + c_out * 4          # trọng số int8/int16 + bias int32
        out = size // 2 if pool else size
        buf = max(buf, c_in * size * size + c_out * out * out)  # bản đồ vào + ra (uint8, pool làm luôn)
        c_in, size = c_out, out
    n_flat = c_in * size * size
    w_bytes += n_flat * 4 + 4 * 4
    return dict(layers=len(cfg), params=params, macs=macs, w_kb=w_bytes / 1024,
                bram_pct=(w_bytes + buf) / (1620 * 1024 / 8) * 100, buf_kb=buf / 1024)


def golden_check(run):
    """Accuracy số nguyên + accuracy từng lớp, chạy golden_int trên qparams hiện có (None nếu chưa có)."""
    import data
    import golden_int
    f = config.export_dir_for(run) / "qparams.npz"
    if not f.exists():
        return None
    _, _, x_te, y_te = data.load_arrays()
    acc, pred = golden_int.accuracy(golden_int.load_qparams(f), x_te, y_te)
    per = [float((pred[y_te == k] == k).mean() * 100) for k in range(4)]
    return dict(int=float(acc * 100), per_class=per)


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}".replace(".", ",")


def status_text(key, log):
    st = log.get(key, {})
    if st.get("trang_thai") == "laptop":
        return f"✅ Đã kiểm chứng trên máy `{st.get('may')}` ({st.get('ngay')})"
    if MODELS[key]["ai_ref"] is None:
        return "⛔ Chưa chạy – chỉ là đề xuất, chưa có số liệu"
    return "⚠️ Chưa kiểm chứng – số liệu do AI chạy trên máy cloud"


def build_table(log, quick=False):
    """Tính lại mọi số liệu, viết docs/so_sanh_5_mo_hinh.md và vẽ hình so sánh."""
    rows = results_rows()
    info = {}
    for k in ORDER:
        m = MODELS[k]
        hw = hw_numbers(m["run"], m["conv1_bits"])
        g = golden_check(m["run"])
        r = rows.get(m["run"])
        train_min = sum(rows[t][2] for t in m["train_runs"] if t in rows) if r else None
        info[k] = dict(hw=hw, g=g, float=r[1] if r else None, val=r[0] if r else None, train_min=train_min)
        if quick and g is not None:
            ref = m["ai_ref"]["int"] if m["ai_ref"] else None
            ok = ref is not None and abs(g["int"] - ref) < 0.01
            log.setdefault(k, {})["nhanh"] = dict(ngay=now(), may=machine(), int=round(g["int"], 2), khop=ok)
    write_doc(info, log)
    draw_figure(info)
    return info


def write_doc(info, log):
    cols = ORDER
    head = "| | " + " | ".join(f"**{k}**<br>{MODELS[k]['vai_tro']}" for k in cols) + " |"
    sep = "|---|" + "---|" * len(cols)

    def row(name, fn):
        return f"| {name} | " + " | ".join(fn(k) for k in cols) + " |"

    def ms(k, dsp, mhz):
        return fmt(info[k]["hw"]["macs"] / (dsp * mhz * 1e6) * 1000, 1) + " ms"

    I = info
    table = "\n".join([
        head, sep,
        row("Tên lượt chạy (thư mục `model/export*`)", lambda k: f"`{MODELS[k]['run']}`"),
        row("**Test số nguyên** (đúng kết quả FPGA)", lambda k: f"**{fmt(I[k]['g']['int'])}%**" if I[k]["g"] else "–"),
        row("Test float (PyTorch)", lambda k: f"{fmt(I[k]['float'])}%" if I[k]["float"] else "–"),
        row("Val float (dùng để chọn)", lambda k: f"{fmt(I[k]['val'])}%" if I[k]["val"] else "–"),
        row("Số đã ghi trước đây (float / số nguyên)",
            lambda k: f"{fmt(MODELS[k]['ai_ref']['float'])}% / {fmt(MODELS[k]['ai_ref']['int'])}%"
            if MODELS[k]["ai_ref"] else "–"),
        row("Accuracy từng lớp (số nguyên)<br>cat / dog / deer / horse",
            lambda k: " / ".join(fmt(v, 1) for v in I[k]["g"]["per_class"]) if I[k]["g"] else "–"),
        row("Số tầng conv", lambda k: str(I[k]["hw"]["layers"])),
        row("Tham số", lambda k: f"{I[k]['hw']['params']:,}".replace(",", ".")),
        row("Trọng số thật (W int8 – conv1 int16 nếu có; bias int32)",
            lambda k: f"{fmt(I[k]['hw']['w_kb'], 1)} KB"),
        row("BRAM ước lượng (trọng số + bộ đệm ảnh)", lambda k: f"~{I[k]['hw']['bram_pct']:.0f}%"),
        row("Phép nhân–cộng (MAC) / ảnh", lambda k: f"{fmt(I[k]['hw']['macs'] / 1e6, 1)} triệu"),
        row("Thời gian/ảnh – 16 DSP @ 20 MHz*", lambda k: ms(k, 16, 20)),
        row("Thời gian/ảnh – 16 DSP @ 100 MHz*", lambda k: ms(k, 16, 100)),
        row("Thời gian/ảnh – 64 DSP @ 100 MHz*", lambda k: ms(k, 64, 100)),
        row("Conv1", lambda k: f"int{MODELS[k]['conv1_bits']}"),
        row("Dữ liệu cần", lambda k: "CIFAR + CINIC-10" if MODELS[k]["cinic"] else "CIFAR"),
        row("Thời gian huấn luyện (theo results.md)",
            lambda k: f"{fmt(I[k]['train_min'], 0)} phút" if I[k]["train_min"] else MODELS[k]["du_kien"]),
        row("Cần mạng thầy E", lambda k: "có" if MODELS[k]["needs_teacher"] else "không"),
        row("**Trạng thái**", lambda k: status_text(k, log)),
    ])
    quick_lines = []
    for k in cols:
        n = log.get(k, {}).get("nhanh")
        if n:
            mark = "khớp số đã ghi ✅" if n["khop"] else ("chưa có số để so" if not MODELS[k]["ai_ref"]
                                                         else "KHÁC số đã ghi ⚠️")
            quick_lines.append(f"| {k} | {n['ngay']} | `{n['may']}` | {fmt(n['int'])}% | {mark} |")
    quick = ("Chưa chạy `python model/kiem_chung.py nhanh`." if not quick_lines else
             "| Mô hình | Ngày | Máy | Test số nguyên | So với số đã ghi |\n|---|---|---|---|---|\n"
             + "\n".join(quick_lines))
    doc = f"""# So sánh 5 mô hình chính – LSI Design Contest 2027

> File này **tự sinh** bởi `python model/kiem_chung.py` – không sửa tay (sửa sẽ bị ghi đè).
> Cập nhật lần cuối: {now()} trên máy `{machine()}`.

Mọi accuracy đo trên **4.000 ảnh test CIFAR-4**. "Test số nguyên" = kết quả của `golden_int.py`, tức đúng kết
quả mạch FPGA sẽ cho ra. Quy trình kiểm chứng: [`docs/KIEM_CHUNG.md`](KIEM_CHUNG.md).

## Bảng so sánh

{table}

\\*Ước lượng lý thuyết = MAC ÷ (số DSP × tần số), giả sử DSP bận 100%; mạch thật chậm hơn khoảng 20–40%.
Clock gốc của board là 20 MHz; muốn 100 MHz phải dùng MMCM. Camera ~30 khung/giây = 33 ms/khung.
BRAM: chip có 1.620 Kb ≈ 202 KB; ước lượng = trọng số + bản đồ vào/ra lớn nhất (uint8), chưa tính phần
BRAM bị lẻ khối 18 Kb và các bộ đệm dòng – thực tế cao hơn. Trên ~80% là chật.

![So sánh độ chính xác và số phép tính](../model/figures/so_sanh_5_mo_hinh.png)

## Ý nghĩa trạng thái

| Ký hiệu | Nghĩa |
|---|---|
| ✅ Đã kiểm chứng | Huấn luyện lại từ đầu trên máy của nhóm bằng `kiem_chung.py`; mọi số trong cột là của máy đó |
| ⚠️ Chưa kiểm chứng | Số liệu do AI (Claude) chạy trên máy cloud ở nhánh `exp/opt-acc-90`; chưa chạy lại trên máy nhóm |
| ⛔ Chưa chạy | Kiến trúc mới chỉ được đề xuất, chưa huấn luyện lần nào – chưa có số liệu |

## Kiểm tra nhanh (golden model trên tham số có sẵn)

{quick}

## Ghi chú từng mô hình

- **B – Cơ bản:** mô hình nhóm đã học (3 conv). Dễ hiểu, nhỏ, nhanh; độ chính xác ~80%.
- **B3c – Nhanh:** thân B + 1 conv 64→128 ở 4×4. Ít phép tính nhất trong nhóm >85%. Cần CINIC.
- **E – Chính xác cao:** VGG rút gọn, cặp conv kép ở 16×16. Chậm nhất; **cũng là mạng thầy** (distillation) cho
  E2 và E3d, nên phải chạy E trước.
- **E2 – Cân bằng ★:** như E nhưng cặp conv kép ở 8×8 → cùng tham số, ít hơn 36% phép tính. Khuyên dùng.
  Lưu ý: 5 lần huấn luyện trước đây cho 89,45–90,30% (≈90%, chưa chắc lần nào cũng >90%).
- **E3d – Thử nghiệm:** 5 conv hẹp hơn (24/48/64/64/112), ít phép tính hơn E2. Nếu đạt ~90% thì nhanh hơn E2;
  đổi lại RTL thêm 1 tầng.
- Các bản `+cinic` học thêm 84.000 ảnh ImageNet 32×32 (CINIC-10), đã loại ảnh gốc CIFAR và ảnh trùng tập test;
  sau đó tinh chỉnh 15 epoch chỉ trên CIFAR. Conv1 dùng trọng số int16 (int16 × uint8 vẫn vừa 1 DSP48E1).
- Lệnh huấn luyện các bản `+cinic` được dựng lại từ ghi chú commit của nhánh exp (`--cifar-repeat 2`,
  tinh chỉnh `--lr 0.005 --warmup 0`, distillation alpha 0,7 / T 4). Kết quả trên máy nhóm có thể lệch
  vài phần mười % so với số AI.

## Hai kiến trúc mới không chọn (chưa chạy – chỉ là đánh giá trên giấy)

| Kiến trúc | Trọng số / MAC | Đánh giá |
|---|---|---|
| `E3a_w112` (E2 rộng hơn) | ~150 KB / 13,7 triệu | Tiềm năng thấp–vừa: mở rộng thêm cho lợi ích giảm dần (E chỉ hơn E2 0,3 điểm), BRAM càng chật |
| `T_wide_teacher` | 522 KB / 49 triệu | Không vừa chip (202 KB BRAM) – chỉ làm mạng thầy; distillation trước đây cho kết quả không ổn định; tốn thời gian nhất |
"""
    DOC_FILE.write_text(doc)
    print(f"✓ Đã cập nhật {DOC_FILE.relative_to(REPO)}")


def draw_figure(info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    chua = []
    for k in ORDER:
        if not info[k]["g"]:
            chua.append(k)
            continue
        acc, mac = info[k]["g"]["int"], info[k]["hw"]["macs"] / 1e6
        ax.scatter(mac, acc, s=info[k]["hw"]["w_kb"] * 3, alpha=0.6)
        ax.annotate(f"{k}\n{acc:.2f}%".replace(".", ","), (mac, acc), textcoords="offset points",
                    xytext=(8, -4), fontsize=9)
    if chua:
        ax.text(0.99, 0.02, "Chưa có số liệu: " + ", ".join(chua), transform=ax.transAxes,
                ha="right", fontsize=8, color="0.4")
    ax.set_xlabel("Triệu phép nhân–cộng (MAC) mỗi ảnh  (càng ít càng nhanh)")
    ax.set_ylabel("Test số nguyên (%)")
    ax.set_title("5 mô hình: độ chính xác và khối lượng tính\n(kích thước chấm ~ dung lượng trọng số)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_FILE, dpi=120)
    plt.close(fig)


# ---- Chạy lại một mô hình ------------------------------------------------------------------------------
def verify_model(key, log, args):
    m = MODELS[key]
    st = log.setdefault(key, {})
    done = [] if args.lam_lai else st.get("buoc_xong", [])
    logf = config.CKPT_DIR / f"kiem_chung_{key}.txt"
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\n▶ Mô hình {key} – {m['vai_tro']}  ({m['run']}), dự kiến {m['du_kien']}\n{'=' * 78}")
    if done:
        print(f"(Chạy tiếp: bỏ qua {len(done)} bước đã xong ở lần trước. Muốn làm lại từ đầu: --lam-lai)")
    t0 = time.time()
    for i, cmd in enumerate(m["stages"]):
        if i in done:
            continue
        run_step(cmd, logf, args.thu)
        if not args.thu:
            done.append(i)
            st["buoc_xong"] = done
            save_log(log)
    if args.thu:
        print(f"✓ Chạy thử {key} xong ({(time.time() - t0) / 60:.1f} phút)")
        return
    st.update(trang_thai="laptop", ngay=now(), may=machine(), buoc_xong=[], ghi_chu="Chạy bằng kiem_chung.py")
    save_log(log)
    info = build_table(log)
    g = info[key]["g"]
    msg = (f"Kiểm chứng {key} ({m['run']}) trên {machine()}: float {fmt(info[key]['float'])}%, "
           f"số nguyên {fmt(g['int'] if g else None)}%")
    commit_push(msg, [m["run"], *m["train_runs"]], not args.khong_push)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("viec", nargs="+", help="nhanh | bang | tatca | B B3c E E2 E3d")
    ap.add_argument("--khong-push", action="store_true")
    ap.add_argument("--lam-lai", action="store_true")
    ap.add_argument("--thu", action="store_true")
    args = ap.parse_args()
    log = load_log()
    if not config.DATA_FILE.exists():
        sys.exit("✗ Chưa có data/cifar4.npz – chạy: python model/prepare_cifar4.py")

    if args.viec in (["nhanh"], ["bang"]):
        quick = args.viec == ["nhanh"]
        print("Đang chạy golden model số nguyên cho các mô hình có tham số sẵn (vài phút)..." if quick
              else "Đang vẽ lại bảng so sánh...")
        info = build_table(log, quick=quick)
        save_log(log)
        print(f"\n{'Mô hình':6s} {'Số nguyên':>10s} {'Số đã ghi':>10s}  Trạng thái")
        for k in ORDER:
            g, ref = info[k]["g"], MODELS[k]["ai_ref"]
            pct = lambda v: "–" if v is None else fmt(v) + "%"
            print(f"{k:6s} {pct(g['int'] if g else None):>10s} {pct(ref['int'] if ref else None):>10s}  "
                  f"{status_text(k, log)}")
        if not args.thu:
            commit_push("Kiểm tra nhanh 5 mô hình" if quick else "Cập nhật bảng so sánh 5 mô hình",
                        [MODELS[k]["run"] for k in ORDER], not args.khong_push)
        return

    keys = ORDER if args.viec == ["tatca"] else args.viec
    bad = [k for k in keys if k not in MODELS]
    if bad:
        sys.exit(f"✗ Không biết mô hình {bad}. Chọn trong: {', '.join(ORDER)} (hoặc nhanh / bang / tatca)")
    teacher_ckpt = config.CKPT_DIR / f"{TEACHER}.pt"
    if any(MODELS[k]["needs_teacher"] for k in keys) and "E" not in keys and not teacher_ckpt.exists():
        print(f"ℹ E2/E3d cần mạng thầy {TEACHER} mà máy chưa có checkpoint -> chạy thêm E trước.")
        keys = ["E", *keys]
    keys = [k for k in ORDER if k in keys]               # đúng thứ tự phụ thuộc
    if any(MODELS[k]["cinic"] for k in keys):
        ensure_cinic(args.thu)
    total = ", ".join(f"{k} ({MODELS[k]['du_kien']})" for k in keys)
    print(f"Sẽ kiểm chứng: {total}")
    for k in keys:
        verify_model(k, log, args)
    print("\n✓ Xong. Xem bảng: docs/so_sanh_5_mo_hinh.md")


if __name__ == "__main__":
    main()
