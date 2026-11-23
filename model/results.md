# Kết quả thí nghiệm

## So sánh kiến trúc (float)

| Kiến trúc | Tham số | MAC/ảnh | KB int8 | Epoch | Val acc | Test acc | Thời gian |
|---|---|---|---|---|---|---|---|
| A_8_16_32 | 8,084 | 813,056 | 7.9 | 30 | 74.50% | 73.88% | 2.2 phút |
| B_16_32_64 | 27,684 | 2,805,760 | 27.0 | 30 | 81.80% | 79.67% | 3.9 phút |
| C_32_64_128 | 101,444 | 10,330,112 | 99.1 | 30 | 85.40% | 83.27% | 7.2 phút |
| D_16_32_64_noBN | 27,684 | 2,805,760 | 27.0 | 30 | 74.85% | 73.90% | 3.0 phút |

## Đánh giá chi tiết (float)

Kiến trúc **B_16_32_64**, accuracy test **79.67%**.

| Lớp | Accuracy |
|---|---|
| cat | 75.5% |
| dog | 72.5% |
| deer | 87.4% |
| horse | 83.3% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 755 | 130 | 85 | 30 |
| **dog** | 175 | 725 | 65 | 35 |
| **deer** | 70 | 25 | 874 | 31 |
| **horse** | 53 | 44 | 70 | 833 |

Nhầm nhiều nhất: **dog → cat** (175 ảnh).

Hình: `figures/confusion_B_16_32_64.png`, `figures/errors_B_16_32_64.png`.

## Lượng tử hoá (int8)

Kiến trúc **B_16_32_64**. Trọng số int8, activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 6.8e-06 |
| Accuracy float (sau gộp BN) | 79.67% |
| Accuracy số nguyên (golden model) | 79.83% |
| Chênh lệch (số nguyên − float) | +0.15 điểm % |
| Trùng dự đoán float ↔ số nguyên | 97.9% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 4 | 5 | 7 | [16, 3, 3, 3] | [-83, 75] |
| conv2 | 5 | 8 | 5 | 8 | [32, 16, 3, 3] | [-105, 95] |
| conv3 | 5 | 8 | 5 | 8 | [64, 32, 3, 3] | [-69, 85] |
| fc | 5 | 8 | – | – | [4, 1024] | [-49, 66] |
