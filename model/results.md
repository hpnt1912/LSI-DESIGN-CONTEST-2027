# Kết quả thí nghiệm

## So sánh kiến trúc (float)

| Kiến trúc | Tham số | MAC/ảnh | KB int8 | Epoch | Val acc | Test acc | Thời gian |
|---|---|---|---|---|---|---|---|
| A_8_16_32 | 8,084 | 813,056 | 7.9 | 30 | 74.55% | 73.47% | 2.7 phút |
| B_16_32_64 | 27,684 | 2,805,760 | 27.0 | 30 | 81.55% | 80.23% | 4.3 phút |
| C_32_64_128 | 101,444 | 10,330,112 | 99.1 | 30 | 86.25% | 82.83% | 8.8 phút |
| D_16_32_64_noBN | 27,684 | 2,805,760 | 27.0 | 30 | 75.10% | 73.88% | 3.5 phút |
| E_opt_cnn | 138,372 | 19,767,296 | 135.1 | 100 | 89.10% | 88.58% | 15.7 phút |
| B2_16_32_64_opt | 27,684 | 2,805,760 | 27.0 | 100 | 84.95% | 83.88% | 9.8 phút |

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
## Đánh giá chi tiết (float) – E_opt_cnn

Kiến trúc **E_opt_cnn**, accuracy test **88.58%**.

| Lớp | Accuracy |
|---|---|
| cat | 83.6% |
| dog | 85.8% |
| deer | 93.2% |
| horse | 91.7% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 836 | 101 | 42 | 21 |
| **dog** | 91 | 858 | 30 | 21 |
| **deer** | 36 | 15 | 932 | 17 |
| **horse** | 21 | 33 | 29 | 917 |

Nhầm nhiều nhất: **cat → dog** (101 ảnh).

Hình: `figures/confusion_E_opt_cnn.png`, `figures/errors_E_opt_cnn.png`.

## Lượng tử hoá (int8) – E_opt_cnn

Kiến trúc **E_opt_cnn**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.8e-06 |
| Accuracy float (sau gộp BN) | 88.58% |
| Accuracy số nguyên (golden model) | 88.52% |
| Chênh lệch (số nguyên − float) | -0.05 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.3% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 4 | 5 | 7 | [32, 3, 3, 3] | [-81, 84] |
| conv2 | 5 | 8 | 6 | 7 | [64, 32, 3, 3] | [-82, 69] |
| conv3 | 6 | 8 | 6 | 8 | [64, 64, 3, 3] | [-77, 89] |
| conv4 | 6 | 8 | 6 | 8 | [128, 64, 3, 3] | [-65, 107] |
| fc | 6 | 8 | – | – | [4, 2048] | [-54, 71] |

## Lượng tử hoá (int8) – B2_16_32_64_opt

Kiến trúc **B2_16_32_64_opt**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.4e-06 |
| Accuracy float (sau gộp BN) | 83.88% |
| Accuracy số nguyên (golden model) | 84.03% |
| Chênh lệch (số nguyên − float) | +0.15 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.6% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 5 | 5 | 8 | [16, 3, 3, 3] | [-109, 124] |
| conv2 | 5 | 8 | 6 | 7 | [32, 16, 3, 3] | [-106, 88] |
| conv3 | 6 | 7 | 6 | 7 | [64, 32, 3, 3] | [-59, 69] |
| fc | 6 | 8 | – | – | [4, 1024] | [-69, 78] |
