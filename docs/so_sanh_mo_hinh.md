# So sánh mô hình – nên dùng bản nào?

> ⚠️ **Tài liệu tham khảo: mọi số trong file này do AI chạy trên máy cloud (nhánh `exp/opt-acc-90`), chưa phải
> số chạy thật trên máy nhóm.** Nhóm đã chốt 5 mô hình chính và chạy thật trên laptop – **số chính thức** nằm ở
> [`so_sanh_5_mo_hinh.md`](so_sanh_5_mo_hinh.md) (tự cập nhật). Cách kiểm chứng: [`KIEM_CHUNG.md`](KIEM_CHUNG.md).

Nhánh `exp/opt-acc-90`. Mọi con số đo trên **4.000 ảnh test CIFAR-4** (cat/dog/deer/horse), chọn mô hình theo tập
validation, không theo test. "Số nguyên" = accuracy của `golden_int.py`, tức đúng kết quả phần cứng sẽ cho ra
(mọi bản đều khớp từng bit với cách tính thứ hai bằng PyTorch).

## Kết luận nhanh

| Nếu nhóm muốn… | Dùng thư mục | Test số nguyên |
|---|---|---|
| **Cân bằng tốc độ – độ chính xác (khuyên dùng)** | **`model/export_E2_lite+cinic60_kdft/`** | **90,30%** |
| Chính xác nhất, chấp nhận chậm hơn | `model/export_E_opt_cnn+cinic_ft/` | 90,60% |
| Nhanh, gọn, ít BRAM | `model/export_B3c_c4x4_128+cinic_ft/` | 87,20% |
| Giữ nguyên thiết kế B gốc | `model/export/` (B gốc) | 80,00% |

## Bảng so sánh chính

| Mô hình | Vai trò | Test số nguyên | Test float | Trọng số | Triệu MAC/ảnh | 16 DSP @ 20 MHz | 16 DSP @ 100 MHz | 64 DSP @ 100 MHz |
|---|---|---|---|---|---|---|---|---|
| B_16_32_64 | baseline của nhóm | 80,00% | 80,23% | 27 KB | 2,8 | 8,8 ms | 1,8 ms | 0,4 ms |
| B2_16_32_64_opt | B + công thức huấn luyện mới | 84,03% | 83,88% | 27 KB | 2,8 | 8,8 ms | 1,8 ms | 0,4 ms |
| B3c_c4x4_128+cinic_ft | ưu tiên tốc độ | 87,20% | 87,08% | 103 KB | 4,0 | 12,5 ms | 2,5 ms | 0,6 ms |
| **E2_lite+cinic60_kdft** ★ | **cân bằng** | **90,30%** | 90,33% | 135 KB | 12,7 | 40 ms | 7,9 ms | 2,0 ms |
| E_opt_cnn+cinic_ft | ưu tiên độ chính xác | 90,60% | 90,65% | 135 KB | 19,8 | 62 ms | 12,4 ms | 3,1 ms |

Thời gian là **ước tính lý thuyết** = số MAC ÷ (số DSP × tần số), giả sử DSP chạy 100% thời gian; mạch thật chậm
hơn khoảng 20–40%. Clock gốc của board là 20 MHz; muốn 100 MHz phải dùng MMCM.

Cách đọc: camera chạy ~30 khung/giây = 33 ms/khung. Ở 20 MHz, E (62 ms) và E2_lite (40 ms) **chưa kịp thời gian thực**
→ nên nâng clock lên ~100 MHz. Ở 100 MHz với 16 DSP (20% số DSP của chip), E2_lite ~8 ms/ảnh là dư sức.

### Vì sao chọn E2_lite+cinic60_kdft

- Chỉ kém bản chính xác nhất 0,3 điểm % (~12 ảnh / 4.000) nhưng **ít hơn 36% phép tính** → nhanh hơn ~1,5 lần.
- Accuracy từng lớp: cat 87,5% · dog 85,4% · deer 94,4% · horse 94,0%.
- Phần cứng vẫn đơn giản: 4 conv 3×3 + 1 Linear, ReLU, MaxPool, thang lũy thừa của 2. Chỉ khác B ở hai điểm:
  conv3 **không có MaxPool**, và **conv1 dùng trọng số int16** (864 trọng số; int16 × uint8 vẫn vừa 1 DSP48E1).

### Lưu ý trước khi chốt

1. **BRAM:** E và E2_lite cần ~135 KB trọng số, vượt ngân sách ~100 KB trong `docs/model_spec.md`; cộng bộ đệm
   feature map ≈ 75% BRAM của chip – vừa nhưng chật. Nếu không đủ → dùng B3c (103 KB).
2. **Độ ổn định:** 5 lần huấn luyện E2_lite cho 89,45–90,30% (trung bình 89,86%). E2_lite ≈ 90%, chưa chắc
   lần nào cũng trên 90%. Cần biên an toàn → dùng E (90,60%).
3. **Dữ liệu thêm:** các bản `+cinic` học thêm 84.000 ảnh ImageNet 32×32 từ CINIC-10 (đề cho tự chọn dataset).
   Đã bỏ mọi ảnh gốc CIFAR có trong CINIC và kiểm tra không trùng pixel với tập test. Nên ghi rõ trong báo cáo.
4. **Ảnh thật ngày thi** chưa được đo. Chạy:
   `python model/predict_image.py model/real_images --arch "E2_lite+cinic60_kdft"`.

## Ý nghĩa tên thư mục

| Phần trong tên | Nghĩa |
|---|---|
| `B_…`, `B2_…`, `B3a/b/c_…` | Họ mạng nhỏ 16/32/64 kênh (B3 thêm 1 conv ở độ phân giải thấp) |
| `E_opt_cnn` | Mạng lớn: Conv32-P · Conv64-Conv64-P · Conv128-P (cặp conv kép ở 16×16) |
| `E2_lite` | Như E nhưng cặp conv kép ở 8×8 → cùng số tham số, **ít hơn 36% MAC** |
| `+cinic` | Huấn luyện thêm 84.000 ảnh ImageNet (CINIC-10) |
| `_ft` | Sau đó tinh chỉnh (fine-tune) 15 epoch chỉ trên ảnh CIFAR |
| `60` | 60 epoch thay vì 30 |
| `_kd` / `_kdft` | Có distillation (học thêm từ mạng thầy `E_opt_cnn+cinic_ft`) |
| `_s2` | Chạy lại với seed khác (seed 7) để kiểm tra độ ổn định |

## Tất cả thư mục `model/export*`

"Có `.mem`" = có `weights/*.mem` + `testvec/` cho RTL. "Chỉ qparams" = chỉ có tham số số nguyên để kiểm tra bằng
`golden_int.py`, muốn có `.mem` thì chạy `python model/export.py --arch <tên>`.

| Thư mục | Test số nguyên | Trọng số | Triệu MAC | File | Dùng? |
|---|---|---|---|---|---|
| `export/` (B_16_32_64) | 80,00% | 27 KB | 2,8 | có `.mem` | baseline của nhóm (không sửa) |
| `export_B2_16_32_64_opt` | 84,03% | 27 KB | 2,8 | chỉ qparams | dự phòng nếu giữ phần cứng B |
| `export_B2_16_32_64_opt+cinic_ft` | 83,88% | 27 KB | 2,8 | chỉ qparams | tham khảo (dữ liệu thêm không giúp mạng cỡ B) |
| `export_B3_kd_16_32_64` | 83,95% | 27 KB | 2,8 | chỉ qparams | tham khảo (KD không giúp mạng cỡ B) |
| `export_B3a_c4x4_64` | 84,65% | 63 KB | 3,4 | chỉ qparams | tham khảo |
| `export_B3b_dbl8_64` | 84,62% | 63 KB | 5,2 | chỉ qparams | tham khảo |
| `export_B3c_c4x4_128` | 85,17% | 103 KB | 4,0 | có `.mem` | tham khảo (bản có `+cinic_ft` tốt hơn) |
| **`export_B3c_c4x4_128+cinic_ft`** | **87,20%** | 103 KB | 4,0 | có `.mem` | **bản ưu tiên tốc độ** |
| `export_E_opt_cnn` | 88,52% | 135 KB | 19,8 | có `.mem` | tham khảo (chưa có dữ liệu thêm) |
| `export_E_opt_cnn+cinic` | 90,38% | 135 KB | 19,8 | chỉ qparams | tham khảo (bản `_ft` tốt hơn) |
| **`export_E_opt_cnn+cinic_ft`** | **90,60%** | 135 KB | 19,8 | có `.mem` | **bản chính xác nhất** |
| `export_E2_lite+cinic` | 89,25% | 135 KB | 12,7 | chỉ qparams | tham khảo |
| `export_E2_lite+cinic_ft` | 89,95% | 135 KB | 12,7 | có `.mem` | tham khảo |
| `export_E2_lite+cinic60_ft` | 89,95% | 135 KB | 12,7 | chỉ qparams | tham khảo |
| **`export_E2_lite+cinic60_kdft`** ★ | **90,30%** | 135 KB | 12,7 | có `.mem` | **khuyên dùng** |
| `export_E2_lite+cinic_s2_ft` | 89,65% | 135 KB | 12,7 | chỉ qparams | kiểm tra độ ổn định |
| `export_E2_lite+cinic_s2_kdft` | 89,45% | 135 KB | 12,7 | chỉ qparams | kiểm tra độ ổn định |

Các bản có `+cinic` dùng conv1 int16 (thêm 0,4–0,9 KB so với số ở cột "Trọng số"). Kết quả chi tiết từng lượt
chạy (ma trận nhầm lẫn, tham số lượng tử từng tầng): `model/results.md`. Cách chạy lại: `model/README.md`.

Các thư mục "tham khảo" giữ lại để báo cáo có số liệu so sánh; khi nhóm đã chốt mô hình có thể xoá bớt.
