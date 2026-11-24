# Tham số số nguyên cho RTL

Sinh tự động bởi `model/quantize.py` và `model/export.py` – không sửa tay.
Kiến trúc: **E_opt_cnn**. Ảnh vào: pixel uint8, f=8 (x = pixel/256), thứ tự kênh R,G,B.

| Tầng | Loại | Kích thước W | Số W | Số bias | f_in | fw | f_out | shift | MaxPool | Đầu ra [C,H,W] |
|---|---|---|---|---|---|---|---|---|---|---|
| conv1 | conv | [32, 3, 3, 3] | 864 | 32 | 8 | 4 | 5 | 7 | có | [32, 16, 16] |
| conv2 | conv | [64, 32, 3, 3] | 18432 | 64 | 5 | 8 | 6 | 7 | không | [64, 16, 16] |
| conv3 | conv | [64, 64, 3, 3] | 36864 | 64 | 6 | 8 | 6 | 8 | có | [64, 8, 8] |
| conv4 | conv | [128, 64, 3, 3] | 73728 | 128 | 6 | 8 | 6 | 8 | có | [128, 4, 4] |
| fc | linear | [4, 2048] | 8192 | 4 | 6 | 8 | – | – | – | [4] |

## Phép tính mỗi tầng conv (padding 1, stride 1)
```
acc = sum(W[o][c][ky][kx] * X[c][h+ky-1][w+kx-1]) + B[o]     (int32; X ngoài biên = 0)
y   = (acc + (1 << (shift-1))) >>> shift                    (dịch phải số học, làm tròn)
y   = min(max(y, 0), 255)                                    (ReLU + bão hoà, uint8)
out = max của mỗi ô 2x2                                       (MaxPool – chỉ ở tầng có cột MaxPool = "có")
```
Tầng fc: `logit[o] = sum(W[o][k] * X[k]) + B[o]` (int32), lớp = argmax(logit).

## Thứ tự phần tử trong file .mem
- `W` conv: `[o][c][ky][kx]`; `W` fc: `[o][k]` với `k = c*16 + h*4 + w`
  (đầu vào fc là 128×4×4).
- Ảnh / feature map: `[c][h][w]`. Ảnh vào là pixel 0..255, kênh R, G, B.
- int8/uint8: 2 chữ số hex; int32: 8 chữ số hex; số âm ở dạng bù 2.

## Vector kiểm thử (`testvec/`)
20 ảnh đầu của tập test CIFAR-4. Với ảnh `i`: `img<i>_input.mem` (3×32×32), `img<i>_conv1.mem`, `img<i>_conv2.mem`, `img<i>_conv3.mem`, `img<i>_conv4.mem`
(đầu ra mỗi tầng conv, sau pool nếu tầng có pool), `img<i>_fc.mem` (4 logit int32).
`labels.csv`: nhãn thật và lớp golden model dự đoán. Testbench RTL phải ra đúng từng giá trị này.
