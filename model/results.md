# Kết quả thí nghiệm

## So sánh kiến trúc (float)

| Kiến trúc | Tham số | MAC/ảnh | KB int8 | Epoch | Val acc | Test acc | Thời gian |
|---|---|---|---|---|---|---|---|
| A_8_16_32 | 8,084 | 813,056 | 7.9 | 30 | 74.55% | 73.47% | 2.7 phút |
| C_32_64_128 | 101,444 | 10,330,112 | 99.1 | 30 | 86.25% | 82.83% | 8.8 phút |
| D_16_32_64_noBN | 27,684 | 2,805,760 | 27.0 | 30 | 75.10% | 73.88% | 3.5 phút |
| E_opt_cnn | 138,372 | 19,767,296 | 135.1 | 100 | 89.10% | 88.58% | 15.7 phút |
| B2_16_32_64_opt | 27,684 | 2,805,760 | 27.0 | 100 | 84.95% | 83.88% | 9.8 phút |
| B3a_c4x4_64 | 64,612 | 3,395,584 | 63.1 | 100 | 87.15% | 84.35% | 12.2 phút |
| B3_kd_16_32_64 | 27,684 | 2,805,760 | 27.0 | 100 | 86.45% | 83.58% | 24.5 phút |
| B3b_dbl8_64 | 64,612 | 5,165,056 | 63.1 | 100 | 87.15% | 85.08% | 14.1 phút |
| B3c_c4x4_128 | 105,636 | 3,989,504 | 103.2 | 100 | 86.50% | 85.30% | 13.0 phút |
| E2_lite+cinic | 138,372 | 12,689,408 | 135.1 | 30 | 87.70% | 89.22% | 43.9 phút |
| E2_lite+cinic_ft | 138,372 | 12,689,408 | 135.1 | 15 | 89.45% | 89.83% | 10.6 phút |
| E_opt_cnn+cinic | 138,372 | 19,767,296 | 135.1 | 30 | 89.35% | 90.20% | 63.1 phút |
| E_opt_cnn+cinic_ft | 138,372 | 19,767,296 | 135.1 | 15 | 90.10% | 90.65% | 11.6 phút |
| B3c_c4x4_128+cinic | 105,636 | 3,989,504 | 103.2 | 30 | 86.65% | 86.22% | 43.7 phút |
| B3c_c4x4_128+cinic_ft | 105,636 | 3,989,504 | 103.2 | 15 | 87.80% | 87.08% | 1.8 phút |
| B2_16_32_64_opt+cinic | 27,684 | 2,805,760 | 27.0 | 30 | 83.45% | 82.53% | 23.1 phút |
| B2_16_32_64_opt+cinic_ft | 27,684 | 2,805,760 | 27.0 | 15 | 85.70% | 84.15% | 1.8 phút |
| E2_lite+cinic60 | 138,372 | 12,689,408 | 135.1 | 60 | 89.80% | 89.35% | 96.3 phút |
| E2_lite+cinic_s2 | 138,372 | 12,689,408 | 135.1 | 30 | 87.95% | 89.30% | 44.9 phút |
| E2_lite+cinic60_ft | 138,372 | 12,689,408 | 135.1 | 15 | 90.15% | 90.12% | 3.5 phút |
| E2_lite+cinic_s2_ft | 138,372 | 12,689,408 | 135.1 | 15 | 89.00% | 89.72% | 3.8 phút |
| E2_lite+cinic60_kdft | 138,372 | 12,689,408 | 135.1 | 15 | 90.20% | 90.33% | 3.3 phút |
| E2_lite+cinic_s2_kdft | 138,372 | 12,689,408 | 135.1 | 15 | 89.15% | 89.50% | 3.5 phút |
| B_16_32_64 | 27,684 | 2,805,760 | 27.0 | 30 | 81.55% | 80.23% | 3.1 phút |

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

## Đánh giá chi tiết (float) – B2_16_32_64_opt

Kiến trúc **B2_16_32_64_opt**, accuracy test **83.88%**.

| Lớp | Accuracy |
|---|---|
| cat | 75.3% |
| dog | 80.8% |
| deer | 89.5% |
| horse | 89.9% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 753 | 148 | 64 | 35 |
| **dog** | 116 | 808 | 48 | 28 |
| **deer** | 55 | 24 | 895 | 26 |
| **horse** | 28 | 36 | 37 | 899 |

Nhầm nhiều nhất: **cat → dog** (148 ảnh).

Hình: `figures/confusion_B2_16_32_64_opt.png`, `figures/errors_B2_16_32_64_opt.png`.

## Lượng tử hoá (int8) – B3c_c4x4_128

Kiến trúc **B3c_c4x4_128**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.9e-06 |
| Accuracy float (sau gộp BN) | 85.30% |
| Accuracy số nguyên (golden model) | 85.17% |
| Chênh lệch (số nguyên − float) | -0.12 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 5 | 5 | 8 | [16, 3, 3, 3] | [-110, 124] |
| conv2 | 5 | 7 | 5 | 7 | [32, 16, 3, 3] | [-88, 91] |
| conv3 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-69, 74] |
| conv4 | 6 | 8 | 6 | 8 | [128, 64, 3, 3] | [-73, 80] |
| fc | 6 | 8 | – | – | [4, 2048] | [-106, 91] |

## Lượng tử hoá (int8) – B3b_dbl8_64

Kiến trúc **B3b_dbl8_64**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.6e-06 |
| Accuracy float (sau gộp BN) | 85.08% |
| Accuracy số nguyên (golden model) | 84.62% |
| Chênh lệch (số nguyên − float) | -0.45 điểm % |
| Trùng dự đoán float ↔ số nguyên | 97.1% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 5 | 5 | 8 | [16, 3, 3, 3] | [-124, 113] |
| conv2 | 5 | 7 | 5 | 7 | [32, 16, 3, 3] | [-68, 65] |
| conv3 | 5 | 7 | 5 | 7 | [64, 32, 3, 3] | [-61, 72] |
| conv4 | 5 | 8 | 6 | 7 | [64, 64, 3, 3] | [-66, 62] |
| fc | 6 | 8 | – | – | [4, 1024] | [-61, 85] |

## Lượng tử hoá (int8) – B3a_c4x4_64

Kiến trúc **B3a_c4x4_64**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.2e-06 |
| Accuracy float (sau gộp BN) | 84.35% |
| Accuracy số nguyên (golden model) | 84.65% |
| Chênh lệch (số nguyên − float) | +0.30 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.4% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 5 | 5 | 8 | [16, 3, 3, 3] | [-112, 127] |
| conv2 | 5 | 7 | 6 | 6 | [32, 16, 3, 3] | [-68, 65] |
| conv3 | 6 | 8 | 6 | 8 | [64, 32, 3, 3] | [-123, 121] |
| conv4 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-39, 75] |
| fc | 5 | 8 | – | – | [4, 1024] | [-112, 90] |

## Lượng tử hoá (int8) – B3_kd_16_32_64

Kiến trúc **B3_kd_16_32_64**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.3e-06 |
| Accuracy float (sau gộp BN) | 83.58% |
| Accuracy số nguyên (golden model) | 83.95% |
| Chênh lệch (số nguyên − float) | +0.38 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.1% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 5 | 5 | 8 | [16, 3, 3, 3] | [-112, 120] |
| conv2 | 5 | 7 | 5 | 7 | [32, 16, 3, 3] | [-74, 51] |
| conv3 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-65, 84] |
| fc | 6 | 8 | – | – | [4, 1024] | [-63, 92] |

## Đánh giá chi tiết (float) – B3c_c4x4_128

Kiến trúc **B3c_c4x4_128**, accuracy test **85.30%**.

| Lớp | Accuracy |
|---|---|
| cat | 78.2% |
| dog | 83.7% |
| deer | 88.9% |
| horse | 90.4% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 782 | 141 | 55 | 22 |
| **dog** | 104 | 837 | 35 | 24 |
| **deer** | 42 | 40 | 889 | 29 |
| **horse** | 32 | 35 | 29 | 904 |

Nhầm nhiều nhất: **cat → dog** (141 ảnh).

Hình: `figures/confusion_B3c_c4x4_128.png`, `figures/errors_B3c_c4x4_128.png`.

## Đánh giá chi tiết (float) – B3b_dbl8_64

Kiến trúc **B3b_dbl8_64**, accuracy test **85.08%**.

| Lớp | Accuracy |
|---|---|
| cat | 73.7% |
| dog | 84.5% |
| deer | 91.5% |
| horse | 90.6% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 737 | 183 | 51 | 29 |
| **dog** | 88 | 845 | 37 | 30 |
| **deer** | 33 | 32 | 915 | 20 |
| **horse** | 22 | 42 | 30 | 906 |

Nhầm nhiều nhất: **cat → dog** (183 ảnh).

Hình: `figures/confusion_B3b_dbl8_64.png`, `figures/errors_B3b_dbl8_64.png`.

## Lượng tử hoá (int8) – E2_lite+cinic

Kiến trúc **E2_lite+cinic**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.0e-06 |
| Accuracy float (sau gộp BN) | 89.22% |
| Accuracy số nguyên (golden model) | 89.25% |
| Chênh lệch (số nguyên − float) | +0.02 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.4% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 5 | 14 | [32, 3, 3, 3] | [-27268, 20066] |
| conv2 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-46, 67] |
| conv3 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-54, 70] |
| conv4 | 5 | 8 | 6 | 7 | [128, 64, 3, 3] | [-50, 75] |
| fc | 6 | 8 | – | – | [4, 2048] | [-53, 75] |

## Lượng tử hoá (int8) – E2_lite+cinic_ft

Kiến trúc **E2_lite+cinic_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.1e-06 |
| Accuracy float (sau gộp BN) | 89.83% |
| Accuracy số nguyên (golden model) | 89.95% |
| Chênh lệch (số nguyên − float) | +0.12 điểm % |
| Trùng dự đoán float ↔ số nguyên | 99.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 5 | 14 | [32, 3, 3, 3] | [-26377, 19742] |
| conv2 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-46, 66] |
| conv3 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-54, 71] |
| conv4 | 5 | 8 | 6 | 7 | [128, 64, 3, 3] | [-59, 79] |
| fc | 6 | 8 | – | – | [4, 2048] | [-53, 74] |

## Lượng tử hoá (int8) – E_opt_cnn+cinic

Kiến trúc **E_opt_cnn+cinic**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.0e-06 |
| Accuracy float (sau gộp BN) | 90.20% |
| Accuracy số nguyên (golden model) | 90.38% |
| Chênh lệch (số nguyên − float) | +0.18 điểm % |
| Trùng dự đoán float ↔ số nguyên | 99.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 4 | 15 | [32, 3, 3, 3] | [-17404, 15799] |
| conv2 | 4 | 8 | 5 | 7 | [64, 32, 3, 3] | [-78, 88] |
| conv3 | 5 | 8 | 6 | 7 | [64, 64, 3, 3] | [-107, 104] |
| conv4 | 6 | 8 | 6 | 8 | [128, 64, 3, 3] | [-89, 118] |
| fc | 6 | 8 | – | – | [4, 2048] | [-67, 77] |

## Lượng tử hoá (int8) – E_opt_cnn+cinic_ft

Kiến trúc **E_opt_cnn+cinic_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.6e-06 |
| Accuracy float (sau gộp BN) | 90.65% |
| Accuracy số nguyên (golden model) | 90.60% |
| Chênh lệch (số nguyên − float) | -0.05 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.7% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 4 | 15 | [32, 3, 3, 3] | [-16702, 15556] |
| conv2 | 4 | 8 | 6 | 6 | [64, 32, 3, 3] | [-77, 91] |
| conv3 | 6 | 8 | 6 | 8 | [64, 64, 3, 3] | [-101, 100] |
| conv4 | 6 | 7 | 6 | 7 | [128, 64, 3, 3] | [-45, 68] |
| fc | 6 | 8 | – | – | [4, 2048] | [-70, 76] |

## Đánh giá chi tiết (float) – E_opt_cnn+cinic_ft

Kiến trúc **E_opt_cnn+cinic_ft**, accuracy test **90.65%**.

| Lớp | Accuracy |
|---|---|
| cat | 86.3% |
| dog | 87.4% |
| deer | 94.4% |
| horse | 94.5% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 863 | 97 | 31 | 9 |
| **dog** | 88 | 874 | 32 | 6 |
| **deer** | 25 | 13 | 944 | 18 |
| **horse** | 16 | 26 | 13 | 945 |

Nhầm nhiều nhất: **cat → dog** (97 ảnh).

Hình: `figures/confusion_E_opt_cnn+cinic_ft.png`, `figures/errors_E_opt_cnn+cinic_ft.png`.

## Lượng tử hoá (int8) – B3c_c4x4_128+cinic_ft

Kiến trúc **B3c_c4x4_128+cinic_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.0e-06 |
| Accuracy float (sau gộp BN) | 87.08% |
| Accuracy số nguyên (golden model) | 87.20% |
| Chênh lệch (số nguyên − float) | +0.12 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.2% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 4 | 15 | [16, 3, 3, 3] | [-16961, 10830] |
| conv2 | 4 | 6 | 5 | 5 | [32, 16, 3, 3] | [-65, 76] |
| conv3 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-62, 99] |
| conv4 | 6 | 7 | 6 | 7 | [128, 64, 3, 3] | [-43, 67] |
| fc | 6 | 8 | – | – | [4, 2048] | [-91, 72] |

## Đánh giá chi tiết (float) – B3c_c4x4_128+cinic_ft

Kiến trúc **B3c_c4x4_128+cinic_ft**, accuracy test **87.08%**.

| Lớp | Accuracy |
|---|---|
| cat | 78.3% |
| dog | 85.3% |
| deer | 92.5% |
| horse | 92.2% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 783 | 138 | 53 | 26 |
| **dog** | 92 | 853 | 34 | 21 |
| **deer** | 23 | 23 | 925 | 29 |
| **horse** | 21 | 38 | 19 | 922 |

Nhầm nhiều nhất: **cat → dog** (138 ảnh).

Hình: `figures/confusion_B3c_c4x4_128+cinic_ft.png`, `figures/errors_B3c_c4x4_128+cinic_ft.png`.

## Lượng tử hoá (int8) – B2_16_32_64_opt+cinic_ft

Kiến trúc **B2_16_32_64_opt+cinic_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.5e-06 |
| Accuracy float (sau gộp BN) | 84.15% |
| Accuracy số nguyên (golden model) | 83.88% |
| Chênh lệch (số nguyên − float) | -0.28 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 4 | 15 | [16, 3, 3, 3] | [-17241, 10604] |
| conv2 | 4 | 7 | 5 | 6 | [32, 16, 3, 3] | [-94, 69] |
| conv3 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-59, 88] |
| fc | 6 | 8 | – | – | [4, 1024] | [-56, 78] |

## Lượng tử hoá (int8) – E2_lite+cinic60_ft

Kiến trúc **E2_lite+cinic60_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.0e-06 |
| Accuracy float (sau gộp BN) | 90.12% |
| Accuracy số nguyên (golden model) | 89.95% |
| Chênh lệch (số nguyên − float) | -0.18 điểm % |
| Trùng dự đoán float ↔ số nguyên | 99.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 5 | 14 | [32, 3, 3, 3] | [-30337, 20776] |
| conv2 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-57, 120] |
| conv3 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-56, 95] |
| conv4 | 5 | 8 | 6 | 7 | [128, 64, 3, 3] | [-73, 114] |
| fc | 6 | 8 | – | – | [4, 2048] | [-50, 66] |

## Lượng tử hoá (int8) – E2_lite+cinic_s2_ft

Kiến trúc **E2_lite+cinic_s2_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.3e-06 |
| Accuracy float (sau gộp BN) | 89.72% |
| Accuracy số nguyên (golden model) | 89.65% |
| Chênh lệch (số nguyên − float) | -0.08 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.9% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 5 | 14 | [32, 3, 3, 3] | [-18071, 14716] |
| conv2 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-57, 68] |
| conv3 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-68, 59] |
| conv4 | 5 | 8 | 6 | 7 | [128, 64, 3, 3] | [-59, 72] |
| fc | 6 | 8 | – | – | [4, 2048] | [-49, 82] |

## Lượng tử hoá (int8) – E2_lite+cinic60_kdft

Kiến trúc **E2_lite+cinic60_kdft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.4e-06 |
| Accuracy float (sau gộp BN) | 90.33% |
| Accuracy số nguyên (golden model) | 90.30% |
| Chênh lệch (số nguyên − float) | -0.02 điểm % |
| Trùng dự đoán float ↔ số nguyên | 99.3% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 5 | 14 | [32, 3, 3, 3] | [-28011, 19777] |
| conv2 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-58, 118] |
| conv3 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-56, 97] |
| conv4 | 5 | 8 | 6 | 7 | [128, 64, 3, 3] | [-64, 99] |
| fc | 6 | 8 | – | – | [4, 2048] | [-45, 64] |

## Lượng tử hoá (int8) – E2_lite+cinic_s2_kdft

Kiến trúc **E2_lite+cinic_s2_kdft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 2.9e-06 |
| Accuracy float (sau gộp BN) | 89.50% |
| Accuracy số nguyên (golden model) | 89.45% |
| Chênh lệch (số nguyên − float) | -0.05 điểm % |
| Trùng dự đoán float ↔ số nguyên | 99.3% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 5 | 14 | [32, 3, 3, 3] | [-18243, 14594] |
| conv2 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-56, 70] |
| conv3 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-68, 60] |
| conv4 | 5 | 8 | 6 | 7 | [128, 64, 3, 3] | [-51, 71] |
| fc | 6 | 8 | – | – | [4, 2048] | [-48, 80] |

## Đánh giá chi tiết (float) – E2_lite+cinic60_kdft

Kiến trúc **E2_lite+cinic60_kdft**, accuracy test **90.33%**.

| Lớp | Accuracy |
|---|---|
| cat | 87.5% |
| dog | 85.4% |
| deer | 94.4% |
| horse | 94.0% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 875 | 81 | 35 | 9 |
| **dog** | 107 | 854 | 32 | 7 |
| **deer** | 29 | 13 | 944 | 14 |
| **horse** | 20 | 22 | 18 | 940 |

Nhầm nhiều nhất: **dog → cat** (107 ảnh).

Hình: `figures/confusion_E2_lite+cinic60_kdft.png`, `figures/errors_E2_lite+cinic60_kdft.png`.
