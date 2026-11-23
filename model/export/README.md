# Tham số số nguyên cho RTL

Sinh tự động bởi `model/quantize.py` và `model/export.py` – không sửa tay.
Kiến trúc: **B_16_32_64**. Ảnh vào: pixel uint8, f=8 (x = pixel/256), thứ tự kênh R,G,B.

| Tầng | Loại | Kích thước W | Số W | Số bias | f_in | fw | f_out | shift |
|---|---|---|---|---|---|---|---|---|
| conv1 | conv | [16, 3, 3, 3] | 432 | 16 | 8 | 4 | 5 | 7 |
| conv2 | conv | [32, 16, 3, 3] | 4608 | 32 | 5 | 8 | 5 | 8 |
| conv3 | conv | [64, 32, 3, 3] | 18432 | 64 | 5 | 8 | 5 | 8 |
| fc | linear | [4, 1024] | 4096 | 4 | 5 | 8 | – | – |

## Phép tính mỗi tầng conv (padding 1, stride 1)
```
acc = sum(W[o][c][ky][kx] * X[c][h+ky-1][w+kx-1]) + B[o]     (int32; X ngoài biên = 0)
y   = (acc + (1 << (shift-1))) >>> shift                    (dịch phải số học, làm tròn)
y   = min(max(y, 0), 255)                                    (ReLU + bão hoà, uint8)
out = max của mỗi ô 2x2                                       (MaxPool)
```
Tầng fc: `logit[o] = sum(W[o][k] * X[k]) + B[o]` (int32), lớp = argmax(logit).

## Thứ tự phần tử trong file .mem
- `W` conv: `[o][c][ky][kx]`; `W` fc: `[o][k]` với `k = c*16 + h*4 + w`.
- Ảnh / feature map: `[c][h][w]`. Ảnh vào là pixel 0..255, kênh R, G, B.
- int8/uint8: 2 chữ số hex; int32: 8 chữ số hex; số âm ở dạng bù 2.

## Vector kiểm thử (`testvec/`)
20 ảnh đầu của tập test CIFAR-4. Với ảnh `i`: `img<i>_input.mem` (3×32×32), `img<i>_conv1.mem`,
`img<i>_conv2.mem`, `img<i>_conv3.mem` (sau pool), `img<i>_fc.mem` (4 logit int32).
`labels.csv`: nhãn thật và lớp golden model dự đoán. Testbench RTL phải ra đúng từng giá trị này.
