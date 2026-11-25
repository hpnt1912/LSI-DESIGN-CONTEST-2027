"""Ngày 8–10 – Gộp BatchNorm và lượng tử hoá (quantization) về số nguyên 8 bit.

Chạy:  python model/quantize.py [--arch B_16_32_64]

Các bước:
  1. BN folding: gộp BatchNorm vào trọng số/bias của conv đứng trước -> mạng chỉ còn Conv, ReLU, Pool, Linear.
  2. Hiệu chuẩn (calibration): chạy 2000 ảnh train qua mạng float, đo độ lớn activation từng tầng.
  3. Chọn số bit phần lẻ (frac bits) là lũy thừa của 2 cho trọng số và activation từng tầng.
  4. Làm tròn trọng số về int8, bias về int32; tính lượng dịch bit (shift) mỗi tầng.
  5. Kiểm tra: accuracy số nguyên (golden_int.py) và so khớp từng bit với một cách tính thứ hai bằng PyTorch.

Kết quả:
  model/export/qparams.npz    tham số số nguyên (golden_int.py và export.py đọc file này)
  model/export/qparams.json   bản dễ đọc (frac bits, shift, khoảng giá trị)
  model/results.md            mục "Lượng tử hoá"
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import config
import data
import evaluate
import golden_int
import report

PERCENTILE = 99.99      # bỏ 0,01% giá trị lớn bất thường khi chọn thang activation


def fold_bn(model):
    """Trả về list tầng float (numpy float64) sau khi gộp BN.

    BN (lúc suy luận):  y = gamma * (z - mean) / sqrt(var + eps) + beta,   z = W*x + b
    Đặt  k = gamma / sqrt(var + eps)   (mỗi kênh một số)
      ->  y = (k*W) * x + (k*(b - mean) + beta)
      ->  W' = k * W,   b' = k * (b - mean) + beta
    """
    mods = list(model.features)
    layers, i, idx = [], 0, 1
    while i < len(mods):
        conv = mods[i]
        w = conv.weight.detach().double().numpy()
        b = conv.bias.detach().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
        i += 1
        if isinstance(mods[i], torch.nn.BatchNorm2d):
            bn = mods[i]
            k = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
            w = w * k[:, None, None, None]
            b = k * (b - bn.running_mean.detach().double().numpy()) + bn.bias.detach().double().numpy()
            i += 1
        i += 1                                    # bỏ qua ReLU (không có tham số)
        # MaxPool (nếu có) ngay sau ReLU. Kiến trúc gốc A–D: luôn có; VGGLite (E_...): có thể không.
        pool = i < len(mods) and isinstance(mods[i], torch.nn.MaxPool2d)
        if pool:
            i += 1
        layers.append({"name": f"conv{idx}", "type": "conv", "w": w, "b": b, "pool": pool})
        idx += 1
    fc = model.classifier
    layers.append({"name": "fc", "type": "linear",
                   "w": fc.weight.detach().double().numpy(), "b": fc.bias.detach().double().numpy()})
    return layers


def float_forward(layers, x):
    """Chạy mạng đã gộp BN bằng số thực. x: tensor (N,3,32,32) float64.
    Trả về logits và list activation sau ReLU (trước pool) của từng tầng conv."""
    acts = []
    for L in layers:
        w, b = torch.from_numpy(L["w"]), torch.from_numpy(L["b"])
        if L["type"] == "conv":
            x = F.relu(F.conv2d(x, w, b, padding=1))
            acts.append(x)
            if L.get("pool", True):
                x = F.max_pool2d(x, 2)
        else:
            x = torch.flatten(x, 1) @ w.T + b
    return x, acts


def calib_amax(layers, x):
    """Phân vị PERCENTILE của activation sau ReLU từng tầng conv, tính lần lượt từng tầng và bỏ ngay
    activation của tầng trước (cho cùng kết quả như giữ tất cả, nhưng tốn ít RAM hơn nhiều với mạng lớn)."""
    out = []
    with torch.no_grad():
        for L in layers:
            if L["type"] != "conv":
                break
            y = F.relu(F.conv2d(x, torch.from_numpy(L["w"]), torch.from_numpy(L["b"]), padding=1))
            out.append(float(np.percentile(y.numpy(), PERCENTILE)))
            x = F.max_pool2d(y, 2) if L.get("pool", True) else y
            del y
    return out


def frac_bits_for(max_abs, q_max):
    """Số bit phần lẻ f lớn nhất sao cho max_abs * 2^f <= q_max (càng lớn càng chính xác)."""
    return int(np.floor(np.log2(q_max / max_abs)))


def quantize(layers, calib_x, conv1_bits=8):
    """calib_x: tensor ảnh float (N,3,32,32). Trả về list tầng số nguyên + thông tin.
    conv1_bits=16: riêng tầng đầu dùng trọng số int16 (chỉ vài trăm trọng số; DSP48E1 nhân 25x18 bit
    nên int16 x uint8 vẫn vừa 1 DSP). Dùng khi gộp BN làm trọng số conv1 quá lớn -> fw nhỏ, sai số lớn."""
    a_maxes = calib_amax(layers, calib_x.double())
    q_layers, f_in = [], 8                                   # ảnh vào: pixel/256 -> 8 bit phần lẻ
    for i, L in enumerate(layers):
        wbits = conv1_bits if i == 0 else 8
        w_max = 2 ** (wbits - 1) - 1                         # 127 (int8) hoặc 32767 (int16)
        fw = frac_bits_for(np.abs(L["w"]).max(), w_max)
        wq = np.clip(np.round(L["w"] * 2.0 ** fw), -w_max - 1, w_max)
        wq = wq.astype(np.int8 if wbits == 8 else np.int16)
        f_acc = f_in + fw
        bq = np.round(L["b"] * 2.0 ** f_acc).astype(np.int64)
        assert np.abs(bq).max() < 2 ** 31, "bias tràn int32"
        q = {"name": L["name"], "type": L["type"], "f_in": f_in, "fw": fw, "f_acc": f_acc,
             "w": wq, "b": bq.astype(np.int32)}
        if wbits != 8:
            q["wbits"] = wbits                               # thiếu khoá này = int8 (tương thích file cũ)
        if L["type"] == "conv":
            a_max = a_maxes[i]
            f_out = frac_bits_for(a_max, 255)
            q.update(f_out=f_out, shift=f_acc - f_out, a_max=a_max, pool=bool(L.get("pool", True)))
            assert q["shift"] > 0
            f_in = f_out
        q_layers.append(q)
    return q_layers


def save_qparams(q_layers, arch, out_dir=config.EXPORT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arch": arch, "input": "pixel uint8, f=8 (x = pixel/256), thứ tự kênh R,G,B",
            "layers": [{k: v for k, v in q.items() if k not in ("w", "b")} for q in q_layers]}
    arrays = {}
    for i, q in enumerate(q_layers):
        arrays[f"w{i}"], arrays[f"b{i}"] = q["w"], q["b"]
        meta["layers"][i].update(w_shape=list(q["w"].shape),
                                 w_range=[int(q["w"].min()), int(q["w"].max())],
                                 b_range=[int(q["b"].min()), int(q["b"].max())])
    np.savez(out_dir / "qparams.npz", meta=json.dumps(meta), **arrays)
    (out_dir / "qparams.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def torch_int_forward(q_layers, x_uint8):
    """Cách tính thứ hai (độc lập với golden_int.py) bằng conv2d của PyTorch trên float64.
    Số nguyên < 2^53 biểu diễn chính xác trong float64 -> kết quả phải trùng từng bit."""
    x = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).double()
    for q in q_layers:
        w, b = torch.from_numpy(q["w"].astype(np.float64)), torch.from_numpy(q["b"].astype(np.float64))
        if q["type"] == "conv":
            acc = F.conv2d(x, w, b, padding=1)
            s = q["shift"]
            x = torch.floor((acc + 2 ** (s - 1)) / 2 ** s).clamp(0, 255)
            if q.get("pool", True):
                x = F.max_pool2d(x, 2)
        else:
            x = torch.flatten(x, 1) @ w.T + b
    return x.numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=config.CHOSEN_ARCH)
    ap.add_argument("--conv1-bits", type=int, choices=[8, 16], default=8,
                    help="số bit trọng số của tầng conv1 (mặc định 8 như cũ)")
    args = ap.parse_args()
    arch = args.arch
    out_dir = config.export_dir_for(arch)        # A–D: model/export/ ; kiến trúc mới: model/export_<arch>/

    model = evaluate.load_model(arch)
    x_tr, _, x_te, y_te = data.load_arrays()
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).double() / 256.0

    # 1) Gộp BN, kiểm tra mạng gộp cho kết quả như mạng gốc
    layers = fold_bn(model)
    with torch.no_grad():
        ref = torch.cat([model(to_t(x_te[i:i + 500]).float()) for i in range(0, 4000, 500)]).double()
        # chạy từng nhóm 500 ảnh: mạng lớn (E, E2…) chạy cả 4.000 ảnh float64 một lúc sẽ hết RAM
        fold = torch.cat([float_forward(layers, to_t(x_te[i:i + 500]))[0] for i in range(0, 4000, 500)])
    diff = (ref - fold).abs().max().item()
    acc_float = (fold.argmax(1).numpy() == y_te).mean()
    print(f"[1] BN folding: sai khác logit lớn nhất {diff:.2e}, accuracy float {acc_float:.4f}")

    # 2–4) Hiệu chuẩn + lượng tử hoá
    rng = np.random.default_rng(config.SEED)
    calib = to_t(x_tr[rng.choice(len(x_tr), 2000, replace=False)])
    q_layers = quantize(layers, calib, args.conv1_bits)
    save_qparams(q_layers, arch, out_dir)
    print("[2-4] Tham số từng tầng:")
    for q in q_layers:
        extra = (f"f_out {q['f_out']:2d} shift {q['shift']:2d}{' +pool' if q['pool'] else ''}"
                 if q["type"] == "conv" else "(logit int32)")
        wb = f" (int{q['wbits']})" if "wbits" in q else ""
        print(f"   {q['name']:6s} f_in {q['f_in']:2d} fw {q['fw']:2d}{wb} -> {extra}")

    # 5) Kiểm tra
    gl = golden_int.load_qparams(out_dir / "qparams.npz")
    acc_int, pred_int = golden_int.accuracy(gl, x_te, y_te)
    logits_g = np.concatenate([golden_int.infer(x_te[i:i + 500], gl) for i in range(0, 4000, 500)])
    logits_t = np.concatenate([torch_int_forward(q_layers, x_te[i:i + 500]) for i in range(0, 4000, 500)])
    bit_exact = np.array_equal(logits_g, logits_t)
    agree = (pred_int == fold.argmax(1).numpy()).mean()
    print(f"[5] accuracy số nguyên {acc_int:.4f} (float {acc_float:.4f}, chênh {(acc_int - acc_float) * 100:+.2f} điểm %)")
    print(f"    golden_int.py và PyTorch trùng từng bit: {bit_exact}; trùng dự đoán với float: {agree * 100:.1f}%")

    rows = "\n".join(
        f"| {q['name']} | {q['f_in']} | {q['fw']} | {q.get('f_out', '–')} | {q.get('shift', '–')} | "
        f"{list(q['w'].shape)} | [{q['w'].min()}, {q['w'].max()}] |" for q in q_layers)
    w_note = "int8" if args.conv1_bits == 8 else "int8 (riêng conv1: int16)"
    body = (f"Kiến trúc **{arch}**. Trọng số {w_note}, activation uint8, bias/tổng tích luỹ int32, "
            f"thang là lũy thừa của 2 (theo từng tầng).\n\n"
            f"| Bước | Kết quả |\n|---|---|\n"
            f"| BN folding – sai khác logit lớn nhất | {diff:.1e} |\n"
            f"| Accuracy float (sau gộp BN) | {acc_float * 100:.2f}% |\n"
            f"| Accuracy số nguyên (golden model) | {acc_int * 100:.2f}% |\n"
            f"| Chênh lệch (số nguyên − float) | {(acc_int - acc_float) * 100:+.2f} điểm % |\n"
            f"| Trùng dự đoán float ↔ số nguyên | {agree * 100:.1f}% |\n"
            f"| golden_int.py ↔ PyTorch trùng từng bit | {'Có' if bit_exact else 'KHÔNG'} |\n\n"
            f"| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |\n|---|---|---|---|---|---|---|\n{rows}")
    title = "Lượng tử hoá (int8)" if arch in config.BASELINE_ARCHS else f"Lượng tử hoá (int8) – {arch}"
    report.update_section(title, body)
    print(f"Đã lưu {out_dir / 'qparams.npz'} và cập nhật results.md")


if __name__ == "__main__":
    main()
