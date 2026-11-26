# Kết quả thí nghiệm

## 5 mô hình chính (đang dùng)

> Mục này **tự sinh** bởi `python model/kiem_chung.py` (cập nhật 2026-11-26 08:33). Bảng đầy đủ: [`docs/so_sanh_5_mo_hinh.md`](../docs/so_sanh_5_mo_hinh.md); cách kiểm chứng: [`docs/KIEM_CHUNG.md`](../docs/KIEM_CHUNG.md).
> Các bảng và mục bên dưới là nhật ký mọi thí nghiệm (kể cả mô hình không dùng).

| Mã | Vai trò | Lượt chạy (tên mục chi tiết bên dưới) | Test float | Test số nguyên (FPGA) | MAC/ảnh | Trọng số | Trạng thái |
|---|---|---|---|---|---|---|---|
| **B** | Cơ bản (~80%) | `B_16_32_64` | 80,23% | **80,00%** | 2,8 triệu | 27,4 KB | ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 20:34) |
| **B3c** | Nhanh | `B3c_c4x4_128+cinic_ft` | 87,67% | **87,60%** | 4,0 triệu | 104,3 KB | ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 21:06) |
| **E** | Chính xác cao (+ mạng thầy cho E2, E3d) | `E_opt_cnn+cinic_ft` | 90,45% | **90,65%** | 19,8 triệu | 136,8 KB | ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 22:20) |
| **E2** | Cân bằng | `E2_lite+cinic60_kdft` | 90,00% | **89,60%** | 12,7 triệu | 136,8 KB | ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 23:59) |
| **E3d** | Thử nghiệm mới (5 conv) | `E3d_deep5+cinic60_kdft` | 90,20% | **90,25%** | 11,6 triệu | 145,6 KB | ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-26 01:38) |

## So sánh kiến trúc (float)

> ✅ **Chạy thật trên máy nhóm**: `A_8_16_32`, `C_32_64_128`, `D_16_32_64_noBN`, `B_16_32_64`, `B3c_c4x4_128+cinic`, `B3c_c4x4_128+cinic_ft`, `E_opt_cnn+cinic`, `E_opt_cnn+cinic_ft`, `E2_lite+cinic60`, `E2_lite+cinic60_kdft`, `E3d_deep5+cinic60`, `E3d_deep5+cinic60_kdft`.

> ⚠️ **Số do AI chạy trên máy cloud** – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo: `E_opt_cnn`, `B2_16_32_64_opt`, `B3a_c4x4_64`, `B3_kd_16_32_64`, `B3b_dbl8_64`, `B3c_c4x4_128`, `E2_lite+cinic`, `E2_lite+cinic_ft`, `B2_16_32_64_opt+cinic`, `B2_16_32_64_opt+cinic_ft`, `E2_lite+cinic_s2`, `E2_lite+cinic60_ft`, `E2_lite+cinic_s2_ft`, `E2_lite+cinic_s2_kdft`.

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
| B2_16_32_64_opt+cinic | 27,684 | 2,805,760 | 27.0 | 30 | 83.45% | 82.53% | 23.1 phút |
| B2_16_32_64_opt+cinic_ft | 27,684 | 2,805,760 | 27.0 | 15 | 85.70% | 84.15% | 1.8 phút |
| E2_lite+cinic_s2 | 138,372 | 12,689,408 | 135.1 | 30 | 87.95% | 89.30% | 44.9 phút |
| E2_lite+cinic60_ft | 138,372 | 12,689,408 | 135.1 | 15 | 90.15% | 90.12% | 3.5 phút |
| E2_lite+cinic_s2_ft | 138,372 | 12,689,408 | 135.1 | 15 | 89.00% | 89.72% | 3.8 phút |
| E2_lite+cinic_s2_kdft | 138,372 | 12,689,408 | 135.1 | 15 | 89.15% | 89.50% | 3.5 phút |
| B_16_32_64 | 27,684 | 2,805,760 | 27.0 | 30 | 81.55% | 80.23% | 3.4 phút |
| B3c_c4x4_128+cinic | 105,636 | 3,989,504 | 103.2 | 30 | 86.20% | 86.65% | 26.8 phút |
| B3c_c4x4_128+cinic_ft | 105,636 | 3,989,504 | 103.2 | 15 | 87.30% | 87.67% | 2.1 phút |
| E_opt_cnn+cinic | 138,372 | 19,767,296 | 135.1 | 30 | 89.50% | 89.20% | 64.7 phút |
| E_opt_cnn+cinic_ft | 138,372 | 19,767,296 | 135.1 | 15 | 90.45% | 90.45% | 5.3 phút |
| E2_lite+cinic60 | 138,372 | 12,689,408 | 135.1 | 60 | 89.70% | 89.92% | 89.6 phút |
| E2_lite+cinic60_kdft | 138,372 | 12,689,408 | 135.1 | 15 | 90.20% | 90.00% | 5.7 phút |
| E3d_deep5+cinic60 | 147,524 | 11,582,464 | 144.1 | 60 | 90.80% | 89.65% | 91.0 phút |
| E3d_deep5+cinic60_kdft | 147,524 | 11,582,464 | 144.1 | 15 | 90.70% | 90.20% | 5.2 phút |
| B3c_c4x4_128+cinic_s2 | 105,636 | 3,989,504 | 103.2 | 30 | 87.00% | 85.67% | 27.0 phút |
| B3c_c4x4_128+cinic_ft_s2 | 105,636 | 3,989,504 | 103.2 | 15 | 88.65% | 86.72% | 2.1 phút |
| B3c_c4x4_128+cinic_s3 | 105,636 | 3,989,504 | 103.2 | 30 | 86.75% | 85.72% | 28.9 phút |
| B3c_c4x4_128+cinic_ft_s3 | 105,636 | 3,989,504 | 103.2 | 15 | 87.65% | 86.85% | 2.2 phút |

## Đánh giá chi tiết (float)

> ✅ **Chạy thật trên máy nhóm** (`B_16_32_64`).

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

> ✅ **Chạy thật trên máy nhóm** (`B_16_32_64`).

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

> ⚠️ **Số do AI chạy trên máy cloud** (`E_opt_cnn`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`E_opt_cnn`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B2_16_32_64_opt`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B2_16_32_64_opt`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B3c_c4x4_128`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B3b_dbl8_64`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B3a_c4x4_64`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B3_kd_16_32_64`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B3c_c4x4_128`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`B3b_dbl8_64`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`E2_lite+cinic`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`E2_lite+cinic_ft`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`E_opt_cnn+cinic`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ✅ **Chạy thật trên máy nhóm** (`E_opt_cnn+cinic_ft`).

Kiến trúc **E_opt_cnn+cinic_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.3e-06 |
| Accuracy float (sau gộp BN) | 90.45% |
| Accuracy số nguyên (golden model) | 90.65% |
| Chênh lệch (số nguyên − float) | +0.20 điểm % |
| Trùng dự đoán float ↔ số nguyên | 99.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 4 | 15 | [32, 3, 3, 3] | [-20093, 15746] |
| conv2 | 4 | 7 | 5 | 6 | [64, 32, 3, 3] | [-42, 69] |
| conv3 | 5 | 8 | 6 | 7 | [64, 64, 3, 3] | [-73, 113] |
| conv4 | 6 | 7 | 6 | 7 | [128, 64, 3, 3] | [-37, 79] |
| fc | 6 | 8 | – | – | [4, 2048] | [-69, 85] |

## Đánh giá chi tiết (float) – E_opt_cnn+cinic_ft

> ✅ **Chạy thật trên máy nhóm** (`E_opt_cnn+cinic_ft`).

Kiến trúc **E_opt_cnn+cinic_ft**, accuracy test **90.45%**.

| Lớp | Accuracy |
|---|---|
| cat | 85.5% |
| dog | 86.0% |
| deer | 94.9% |
| horse | 95.4% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 855 | 88 | 39 | 18 |
| **dog** | 88 | 860 | 28 | 24 |
| **deer** | 27 | 9 | 949 | 15 |
| **horse** | 15 | 16 | 15 | 954 |

Nhầm nhiều nhất: **cat → dog** (88 ảnh).

Hình: `figures/confusion_E_opt_cnn+cinic_ft.png`, `figures/errors_E_opt_cnn+cinic_ft.png`.

## Lượng tử hoá (int8) – B3c_c4x4_128+cinic_ft

> ✅ **Chạy thật trên máy nhóm** (`B3c_c4x4_128+cinic_ft`).

Kiến trúc **B3c_c4x4_128+cinic_ft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.7e-06 |
| Accuracy float (sau gộp BN) | 87.67% |
| Accuracy số nguyên (golden model) | 87.60% |
| Chênh lệch (số nguyên − float) | -0.08 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.0% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 4 | 15 | [16, 3, 3, 3] | [-16537, 10642] |
| conv2 | 4 | 6 | 5 | 5 | [32, 16, 3, 3] | [-48, 66] |
| conv3 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-52, 98] |
| conv4 | 6 | 7 | 6 | 7 | [128, 64, 3, 3] | [-42, 64] |
| fc | 6 | 8 | – | – | [4, 2048] | [-83, 74] |

## Đánh giá chi tiết (float) – B3c_c4x4_128+cinic_ft

> ✅ **Chạy thật trên máy nhóm** (`B3c_c4x4_128+cinic_ft`).

Kiến trúc **B3c_c4x4_128+cinic_ft**, accuracy test **87.67%**.

| Lớp | Accuracy |
|---|---|
| cat | 79.8% |
| dog | 86.4% |
| deer | 92.8% |
| horse | 91.7% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 798 | 129 | 55 | 18 |
| **dog** | 90 | 864 | 30 | 16 |
| **deer** | 26 | 24 | 928 | 22 |
| **horse** | 22 | 30 | 31 | 917 |

Nhầm nhiều nhất: **cat → dog** (129 ảnh).

Hình: `figures/confusion_B3c_c4x4_128+cinic_ft.png`, `figures/errors_B3c_c4x4_128+cinic_ft.png`.

## Lượng tử hoá (int8) – B2_16_32_64_opt+cinic_ft

> ⚠️ **Số do AI chạy trên máy cloud** (`B2_16_32_64_opt+cinic_ft`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`E2_lite+cinic60_ft`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ⚠️ **Số do AI chạy trên máy cloud** (`E2_lite+cinic_s2_ft`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ✅ **Chạy thật trên máy nhóm** (`E2_lite+cinic60_kdft`).

Kiến trúc **E2_lite+cinic60_kdft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.2e-06 |
| Accuracy float (sau gộp BN) | 90.00% |
| Accuracy số nguyên (golden model) | 89.60% |
| Chênh lệch (số nguyên − float) | -0.40 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.2% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 11 | 5 | 14 | [32, 3, 3, 3] | [-29645, 20646] |
| conv2 | 5 | 6 | 6 | 5 | [64, 32, 3, 3] | [-35, 74] |
| conv3 | 6 | 7 | 5 | 8 | [64, 64, 3, 3] | [-55, 85] |
| conv4 | 5 | 8 | 6 | 7 | [128, 64, 3, 3] | [-60, 82] |
| fc | 6 | 9 | – | – | [4, 2048] | [-83, 125] |

## Lượng tử hoá (int8) – E2_lite+cinic_s2_kdft

> ⚠️ **Số do AI chạy trên máy cloud** (`E2_lite+cinic_s2_kdft`) – chưa huấn luyện lại trên máy nhóm, chỉ để tham khảo.

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

> ✅ **Chạy thật trên máy nhóm** (`E2_lite+cinic60_kdft`).

Kiến trúc **E2_lite+cinic60_kdft**, accuracy test **90.00%**.

| Lớp | Accuracy |
|---|---|
| cat | 82.0% |
| dog | 87.2% |
| deer | 95.4% |
| horse | 95.4% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 820 | 117 | 45 | 18 |
| **dog** | 83 | 872 | 27 | 18 |
| **deer** | 18 | 11 | 954 | 17 |
| **horse** | 9 | 16 | 21 | 954 |

Nhầm nhiều nhất: **cat → dog** (117 ảnh).

Hình: `figures/confusion_E2_lite+cinic60_kdft.png`, `figures/errors_E2_lite+cinic60_kdft.png`.

## Đánh giá chi tiết (float) – E3d_deep5+cinic60_kdft

> ✅ **Chạy thật trên máy nhóm** (`E3d_deep5+cinic60_kdft`).

Kiến trúc **E3d_deep5+cinic60_kdft**, accuracy test **90.20%**.

| Lớp | Accuracy |
|---|---|
| cat | 83.9% |
| dog | 87.0% |
| deer | 94.1% |
| horse | 95.8% |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):

| | cat | dog | deer | horse |
|---|---|---|---|---|
| **cat** | 839 | 111 | 36 | 14 |
| **dog** | 84 | 870 | 29 | 17 |
| **deer** | 26 | 11 | 941 | 22 |
| **horse** | 13 | 18 | 11 | 958 |

Nhầm nhiều nhất: **cat → dog** (111 ảnh).

Hình: `figures/confusion_E3d_deep5+cinic60_kdft.png`, `figures/errors_E3d_deep5+cinic60_kdft.png`.

## Lượng tử hoá (int8) – E3d_deep5+cinic60_kdft

> ✅ **Chạy thật trên máy nhóm** (`E3d_deep5+cinic60_kdft`).

Kiến trúc **E3d_deep5+cinic60_kdft**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.9e-06 |
| Accuracy float (sau gộp BN) | 90.20% |
| Accuracy số nguyên (golden model) | 90.25% |
| Chênh lệch (số nguyên − float) | +0.05 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.9% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (int8) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 12 | 5 | 15 | [24, 3, 3, 3] | [-31559, 26464] |
| conv2 | 5 | 7 | 5 | 7 | [48, 24, 3, 3] | [-91, 92] |
| conv3 | 5 | 6 | 5 | 6 | [64, 48, 3, 3] | [-37, 69] |
| conv4 | 5 | 8 | 5 | 8 | [64, 64, 3, 3] | [-81, 100] |
| conv5 | 5 | 8 | 6 | 7 | [112, 64, 3, 3] | [-67, 83] |
| fc | 6 | 8 | – | – | [4, 1792] | [-78, 57] |

## Ảnh thật (model/real_images)

> Ghi bởi `python model/anh_that.py` trên máy `LAPTOP-2RM30BH4` (2026-11-26 19:12). Thư mục: `real_images/`; 108 ảnh: cat 26, dog 26, deer 27, horse 29.

| Mô hình | Test CIFAR (số nguyên) | **Ảnh thật (số nguyên)** ± khoảng tin cậy 95% | Ảnh thật (float) | Chênh (thật − CIFAR, điểm %) | cat / dog / deer / horse (%) |
|---|---|---|---|---|---|
| **B** | 80,00% | **82,4%** ± 7 | 81,5% | +2,4 | 81 / 73 / 96 / 79 |
| **B3c** | 87,60% | **91,7%** ± 5 | 92,6% | +4,1 | 85 / 88 / 96 / 97 |
| **E** | 90,65% | **95,4%** ± 4 | 95,4% | +4,7 | 88 / 100 / 100 / 93 |
| **E2** | 89,60% | **97,2%** ± 3 | 95,4% | +7,6 | 92 / 96 / 100 / 100 |
| **E3d** | 90,25% | **96,3%** ± 4 | 94,4% | +6,0 | 92 / 96 / 100 / 97 |

Với 108 ảnh, khoảng tin cậy rộng (± ở cột 3): chỉ chênh lệch lớn hơn khoảng đó mới đáng tin. Điều cần xem là mô hình nào tụt mạnh so với CIFAR.

Ảnh cả 5 mô hình đều đoán sai (0): không có.

## Lượng tử hoá (int8) – B3c_c4x4_128+cinic_ft_s2

Kiến trúc **B3c_c4x4_128+cinic_ft_s2**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 3.6e-06 |
| Accuracy float (sau gộp BN) | 86.72% |
| Accuracy số nguyên (golden model) | 86.98% |
| Chênh lệch (số nguyên − float) | +0.25 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.5% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (số nguyên) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 12 | 5 | 15 | [16, 3, 3, 3] | [-28983, 23775] |
| conv2 | 5 | 7 | 5 | 7 | [32, 16, 3, 3] | [-79, 79] |
| conv3 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-49, 97] |
| conv4 | 6 | 8 | 6 | 8 | [128, 64, 3, 3] | [-84, 109] |
| fc | 6 | 8 | – | – | [4, 2048] | [-68, 70] |

## Lượng tử hoá (int8) – B3c_c4x4_128+cinic_ft_s3

Kiến trúc **B3c_c4x4_128+cinic_ft_s3**. Trọng số int8 (riêng conv1: int16), activation uint8, bias/tổng tích luỹ int32, thang là lũy thừa của 2 (theo từng tầng).

| Bước | Kết quả |
|---|---|
| BN folding – sai khác logit lớn nhất | 4.8e-06 |
| Accuracy float (sau gộp BN) | 86.85% |
| Accuracy số nguyên (golden model) | 86.60% |
| Chênh lệch (số nguyên − float) | -0.25 điểm % |
| Trùng dự đoán float ↔ số nguyên | 98.4% |
| golden_int.py ↔ PyTorch trùng từng bit | Có |

| Tầng | f_in | fw | f_out | shift | Kích thước W | Khoảng W (số nguyên) |
|---|---|---|---|---|---|---|
| conv1 | 8 | 12 | 4 | 16 | [16, 3, 3, 3] | [-28507, 30302] |
| conv2 | 4 | 7 | 5 | 6 | [32, 16, 3, 3] | [-68, 96] |
| conv3 | 5 | 7 | 6 | 6 | [64, 32, 3, 3] | [-56, 108] |
| conv4 | 6 | 7 | 6 | 7 | [128, 64, 3, 3] | [-40, 67] |
| fc | 6 | 8 | – | – | [4, 2048] | [-91, 74] |

## Nhiều seed – B3c

> Ghi bởi `python model/nhieu_seed.py B3c` trên máy `LAPTOP-2RM30BH4` (2026-11-26 20:22). Mỗi seed chạy đúng công thức của lượt chính (`B3c_c4x4_128+cinic_ft`), chỉ khác `--seed`.

| Seed | Lượt chạy cuối | Val float | Test float | Test số nguyên | cat / dog / deer / horse (số nguyên) | Thời gian huấn luyện |
|---|---|---|---|---|---|---|
| 2027 (lượt chính) | `B3c_c4x4_128+cinic_ft` | 87,30% | 87,67% | **87,60%** | 79,7 / 85,2 / 93,1 / 92,4 | 28,9 phút |
| 2 | `B3c_c4x4_128+cinic_ft_s2` | 88,65% | 86,72% | **86,98%** | 77,0 / 85,1 / 92,5 / 93,3 | 29,1 phút |
| 3 | `B3c_c4x4_128+cinic_ft_s3` | 87,65% | 86,85% | **86,60%** | 77,7 / 84,8 / 91,5 / 92,4 | 31,1 phút |

**3 lần chạy – test số nguyên: 87,06% ± 0,51** (trung bình ± độ lệch chuẩn), thấp nhất 86,60%, cao nhất 87,60%. Test float: 87,08% ± 0,52.

Cách dùng khi so sánh: dùng **trung bình** thay cho số của một lần chạy; hai mô hình chênh nhau ít hơn khoảng độ lệch chuẩn thì coi như tương đương.
