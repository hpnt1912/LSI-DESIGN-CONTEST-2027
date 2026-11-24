"""Ngày 12 – Xuất tham số số nguyên và vector kiểm thử cho RTL (Verilog).

Chạy:  python model/export.py [--n 20]

Định dạng .mem: mỗi dòng 1 số ở hệ 16 (hex), dạng bù 2 (two's complement), đọc bằng $readmemh trong Verilog.
    int8  -> 2 chữ số hex   (ví dụ -3 -> "fd")
    uint8 -> 2 chữ số hex   (ví dụ 200 -> "c8")
    int32 -> 8 chữ số hex   (ví dụ -5 -> "fffffffb")

Thứ tự phần tử (rất quan trọng khi viết RTL):
    trọng số conv  W[o][c][ky][kx]   (kênh ra, kênh vào, hàng, cột của bộ lọc)
    trọng số fc    W[o][k]           k = c*16 + h*4 + w  (giống torch.flatten)
    ảnh / feature  X[c][h][w]        (kênh, hàng, cột)

Kết quả trong model/export/:
    weights/<tầng>_w.mem, weights/<tầng>_b.mem
    testvec/img<i>_input.mem, img<i>_conv1.mem, ..., img<i>_fc.mem  (đầu ra từng tầng)
    testvec/labels.csv           nhãn thật, dự đoán của golden model
    README.md                    mô tả các tầng, shift, kích thước
"""

import argparse
import json

import numpy as np

import config
import data
import golden_int


def to_hex(arr, bits):
    """Mảng số nguyên -> list chuỗi hex bù 2 với số bit cho trước."""
    mask = (1 << bits) - 1
    width = bits // 4
    return [format(int(v) & mask, f"0{width}x") for v in np.asarray(arr).ravel()]


def write_mem(path, arr, bits):
    path.write_text("\n".join(to_hex(arr, bits)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="số ảnh test làm vector kiểm thử")
    ap.add_argument("--arch", default=config.CHOSEN_ARCH,
                    help="A–D: ghi vào model/export/ ; kiến trúc mới: model/export_<arch>/")
    args = ap.parse_args()
    n, out_dir = args.n, config.export_dir_for(args.arch)

    layers = golden_int.load_qparams(out_dir / "qparams.npz")
    wdir, tdir = out_dir / "weights", out_dir / "testvec"
    wdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)

    # 1) Trọng số và bias
    _, _, x_te, y_te = data.load_arrays()
    logits, trace = golden_int.infer(x_te[:n], layers, keep=True)
    out_shape = {name: list(t.shape[1:]) for name, t in trace}
    lines = []
    for L in layers:
        write_mem(wdir / f"{L['name']}_w.mem", L["w"], 8)
        write_mem(wdir / f"{L['name']}_b.mem", L["b"], 32)
        shift = L.get("shift", "–")
        pool = ("có" if L.get("pool", True) else "không") if L["type"] == "conv" else "–"
        lines.append(f"| {L['name']} | {L['type']} | {list(L['w'].shape)} | {L['w'].size} | {len(L['b'])} | "
                     f"{L['f_in']} | {L['fw']} | {L.get('f_out', '–')} | {shift} | {pool} | "
                     f"{out_shape[L['name']]} |")
    last_conv = [t for name, t in trace if name.startswith("conv")][-1]
    c_last, h_last, w_last = last_conv.shape[1:]

    # 2) Vector kiểm thử: n ảnh đầu của tập test
    pred = logits.argmax(1)
    for name, t in trace:
        bits = 32 if name == "fc" else 8
        for i in range(n):
            write_mem(tdir / f"img{i:02d}_{name}.mem", t[i], bits)
    csv = ["index,label,label_name,pred,pred_name"] + [
        f"{i},{y_te[i]},{config.CLASSES[y_te[i]]},{pred[i]},{config.CLASSES[pred[i]]}" for i in range(n)]
    (tdir / "labels.csv").write_text("\n".join(csv) + "\n")

    meta = json.loads((out_dir / "qparams.json").read_text())
    conv_names = ", ".join(f"`img<i>_{L['name']}.mem`" for L in layers if L["type"] == "conv")
    readme = f"""# Tham số số nguyên cho RTL

Sinh tự động bởi `model/quantize.py` và `model/export.py` – không sửa tay.
Kiến trúc: **{meta['arch']}**. Ảnh vào: {meta['input']}.

| Tầng | Loại | Kích thước W | Số W | Số bias | f_in | fw | f_out | shift | MaxPool | Đầu ra [C,H,W] |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(lines)}

## Phép tính mỗi tầng conv (padding 1, stride 1)
```
acc = sum(W[o][c][ky][kx] * X[c][h+ky-1][w+kx-1]) + B[o]     (int32; X ngoài biên = 0)
y   = (acc + (1 << (shift-1))) >>> shift                    (dịch phải số học, làm tròn)
y   = min(max(y, 0), 255)                                    (ReLU + bão hoà, uint8)
out = max của mỗi ô 2x2                                       (MaxPool – chỉ ở tầng có cột MaxPool = "có")
```
Tầng fc: `logit[o] = sum(W[o][k] * X[k]) + B[o]` (int32), lớp = argmax(logit).

## Thứ tự phần tử trong file .mem
- `W` conv: `[o][c][ky][kx]`; `W` fc: `[o][k]` với `k = c*{h_last * w_last} + h*{w_last} + w`
  (đầu vào fc là {c_last}×{h_last}×{w_last}).
- Ảnh / feature map: `[c][h][w]`. Ảnh vào là pixel 0..255, kênh R, G, B.
- int8/uint8: 2 chữ số hex; int32: 8 chữ số hex; số âm ở dạng bù 2.

## Vector kiểm thử (`testvec/`)
{n} ảnh đầu của tập test CIFAR-4. Với ảnh `i`: `img<i>_input.mem` (3×32×32), {conv_names}
(đầu ra mỗi tầng conv, sau pool nếu tầng có pool), `img<i>_fc.mem` (4 logit int32).
`labels.csv`: nhãn thật và lớp golden model dự đoán. Testbench RTL phải ra đúng từng giá trị này.
"""
    (out_dir / "README.md").write_text(readme)
    total = sum(L["w"].size for L in layers)
    print(f"Đã xuất {len(layers)} tầng ({total:,} trọng số) vào {wdir}")
    print(f"Đã xuất vector kiểm thử cho {n} ảnh vào {tdir} (golden đúng {np.sum(pred == y_te[:n])}/{n})")


if __name__ == "__main__":
    main()
