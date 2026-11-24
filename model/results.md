# Kết quả thí nghiệm

## So sánh kiến trúc (float)

| Kiến trúc | Tham số | MAC/ảnh | KB int8 | Epoch | Val acc | Test acc | Thời gian |
|---|---|---|---|---|---|---|---|
| A_8_16_32 | 8,084 | 813,056 | 7.9 | 30 | 74.55% | 73.47% | 2.7 phút |
| B_16_32_64 | 27,684 | 2,805,760 | 27.0 | 30 | 81.55% | 80.23% | 4.3 phút |
| C_32_64_128 | 101,444 | 10,330,112 | 99.1 | 30 | 86.25% | 82.83% | 8.8 phút |
| D_16_32_64_noBN | 27,684 | 2,805,760 | 27.0 | 30 | 75.10% | 73.88% | 3.5 phút |

## Đánh giá chi tiết (float)

Kiến trúc **B_16_32_64**, accuracy test **80.23%**.

| Lớp | Accuracy |
|---|---|
| cat | 71.1% |
| dog | 78.0% |
| deer | 86.8% |
| horse | 85.0% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 711 | 177 | 78 | 34 |
| **dog** | 141 | 780 | 45 | 34 |
| **deer** | 53 | 38 | 868 | 41 |
| **horse** | 48 | 52 | 50 | 850 |

Nhầm nhiều nhất: **cat → dog** (177 ảnh).

Hình: `figures/confusion_B_16_32_64.png`, `figures/errors_B_16_32_64.png`.

## Lượng tử hoá (int8)

Kiến trúc **B_16_32_64**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 6.2e-06 |
| Accuracy float (sau gộp BN) | 80.23% |
| Accuracy số nguyên (golden model) | 80.00% |
| Chênh lệch (số nguyên − float) | -0.22 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 4 | 5 | 7 | [16, 3, 3, 3] | [-90, 78] |
| conv2 | 5 | 8 | 5 | 8 | [32, 16, 3, 3] | [-116, 92] |
| conv3 | 5 | 8 | 5 | 8 | [64, 32, 3, 3] | [-72, 94] |
| fc | 5 | 9 | – | – | [4, 1024] | [-113, 122] |
