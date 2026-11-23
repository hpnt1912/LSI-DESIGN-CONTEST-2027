"""Ngày 11 – Mô hình tham chiếu chỉ dùng số nguyên (golden model).

Đây là "đáp án chuẩn" cho phần cứng: RTL phải cho ra KẾT QUẢ GIỐNG HỆT TỪNG BIT với file này.
Chỉ dùng NumPy và số nguyên: nhân, cộng, dịch bit, so sánh. Không có số thực nào.

Quy ước số (fixed-point):
    Một số thực v được lưu bằng số nguyên q với  v ≈ q × 2^(-f)   (f = số bit phần lẻ, "frac bits")
    Ảnh vào:      q = pixel (uint8 0..255),         f = 8      (v = pixel/256)
    Trọng số:     int8  (-128..127),                f = fw     (riêng mỗi tầng)
    Bias:         int32,                             f = f_in + fw  (cùng thang với tổng tích luỹ)
    Activation:   uint8 (0..255) sau ReLU,           f = f_out  (riêng mỗi tầng)

Một tầng conv (cho mỗi kênh ra o, vị trí h, w):
    acc  = Σ_{c, ky, kx} Wq[o, c, ky, kx] × xq[c, h+ky-1, w+kx-1]  +  bq[o]     (số nguyên 32 bit)
    y    = (acc + 2^(s-1)) >> s          với s = f_in + fw - f_out   (dịch phải = chia 2^s, làm tròn)
    y    = min(max(y, 0), 255)           (ReLU + bão hoà về 8 bit không dấu)
    sau đó MaxPool 2x2: lấy max của mỗi ô 2x2.
Tầng Linear cuối: logit = Σ Wq × xq + bq (int32), lớp dự đoán = argmax(logit).

Chạy:  python model/golden_int.py      -> accuracy số nguyên trên tập test
(cần model/export/qparams.npz, tạo bởi: python model/quantize.py)
"""

import json

import numpy as np

import config

QPARAMS_FILE = config.EXPORT_DIR / "qparams.npz"


def load_qparams(path=QPARAMS_FILE):
    """Đọc tham số số nguyên. Trả về list các tầng (dict)."""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    layers = []
    for i, m in enumerate(meta["layers"]):
        layers.append({**m, "w": d[f"w{i}"].astype(np.int64), "b": d[f"b{i}"].astype(np.int64)})
    return layers


def requant_relu(acc, shift):
    """Dịch phải có làm tròn, rồi ReLU + bão hoà về 0..255."""
    if shift > 0:
        acc = (acc + (1 << (shift - 1))) >> shift   # >> trên số âm của NumPy là dịch số học (giữ dấu)
    return np.clip(acc, 0, 255)


def conv3x3_int(x, w, b):
    """x: (N, C, H, W) int64, w: (O, C, 3, 3), b: (O,)  ->  acc (N, O, H, W) int64. Padding 1 bằng số 0."""
    n, c, h, wd = x.shape
    xp = np.zeros((n, c, h + 2, wd + 2), dtype=np.int64)
    xp[:, :, 1:-1, 1:-1] = x
    acc = np.zeros((n, w.shape[0], h, wd), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            patch = xp[:, :, ky:ky + h, kx:kx + wd]                 # (N, C, H, W)
            acc += np.einsum("nchw,oc->nohw", patch, w[:, :, ky, kx])
    return acc + b[None, :, None, None]


def maxpool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def infer(images_uint8, layers, keep=False):
    """images_uint8: (N, 32, 32, 3) uint8.  Trả về logits int64 (N, 4)
    và (nếu keep=True) danh sách đầu ra từng tầng để làm vector kiểm thử cho RTL."""
    x = images_uint8.astype(np.int64).transpose(0, 3, 1, 2)          # NHWC -> NCHW
    trace = [("input", x)]
    for L in layers:
        if L["type"] == "conv":
            acc = conv3x3_int(x, L["w"], L["b"])
            x = maxpool2(requant_relu(acc, L["shift"]))
            trace.append((L["name"], x))
        else:                                                         # linear
            flat = x.reshape(len(x), -1)                              # thứ tự (kênh, hàng, cột)
            x = flat @ L["w"].T + L["b"]
            trace.append((L["name"], x))
    return (x, trace) if keep else x


def accuracy(layers, x_uint8, y, batch=500):
    preds = np.concatenate([infer(x_uint8[i:i + batch], layers).argmax(1)
                            for i in range(0, len(x_uint8), batch)])
    return (preds == y).mean(), preds


def main():
    import data
    _, _, x_te, y_te = data.load_arrays()
    layers = load_qparams()
    acc, _ = accuracy(layers, x_te, y_te)
    print(f"Golden model số nguyên – accuracy test: {acc:.4f}")
    _, trace = infer(x_te[:1], layers, keep=True)
    for name, t in trace:
        print(f"  {name:8s} shape {str(t.shape[1:]):14s} min {t.min():8d} max {t.max():8d}")


if __name__ == "__main__":
    main()
