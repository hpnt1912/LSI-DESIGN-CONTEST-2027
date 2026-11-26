# Kế hoạch: mô hình Python (2 tuần, 23/11 – 07/12/2026)

**Code của cả 14 ngày đã hoàn chỉnh và chạy thử** (kết quả trong `model/results.md`). Checklist dưới đây dùng để theo dõi tiến độ *học*: mỗi ngày chạy lại, đọc hiểu code theo file hướng dẫn `docs/HuongDan_MoHinh_CNN_Python.docx`, trả lời câu hỏi tự kiểm tra rồi đánh dấu `[x]`. Chi tiết yêu cầu mô hình xem [`model_spec.md`](model_spec.md).

## Tuần 1: mô hình số thực (float)

- [x] **Chốt yêu cầu.** `docs/model_spec.md`.
- [ ] **Ngày 1 – Python cơ bản.** `git pull`; đọc Phần P.1–P.6 của file hướng dẫn, gõ lại mọi ví dụ trong `python`.
- [ ] **Ngày 2 – NumPy, PyTorch + lý thuyết nền.** Phần P.7–P.10, Phần I.1–I.3, Phần II.
- [ ] **Ngày 3 – Dữ liệu.** `python model/data.py` – đọc hiểu `model/data.py` (Phần III.2).
  *Xong khi:* in được kích thước batch, xem `model/figures/samples.png`.
- [ ] **Ngày 4 – CNN và huấn luyện.** `python model/net.py`, `python model/train.py` (Phần I.4–I.7, III.3).
  *Xong khi:* có accuracy test ~79–80% cho B_16_32_64.
- [ ] **Ngày 5 – So sánh kiến trúc.** `python model/train.py --arch all` (Phần I.9, III.4); đọc bảng trong `model/results.md`.
- [ ] **Ngày 6 – Đánh giá.** Confusion matrix, accuracy từng lớp (`model/evaluate.py`).
- [ ] **Ngày 7 – Dự phòng + báo cáo giữa kỳ** cho thầy (1 trang).

## Tuần 2: chuẩn bị cho phần cứng

- [ ] **Ngày 8 – BN folding.** (Phần IV.1) Gộp BatchNorm vào Conv; output trước/sau gần như trùng.
- [ ] **Ngày 9–10 – Lượng tử hóa int8** (`model/quantize.py`). PTQ trước; sụt quá ~2 điểm % thì làm QAT. Ưu tiên scale là lũy thừa của 2.
- [ ] **Ngày 11 – Golden model số nguyên** (`model/golden_int.py`, NumPy): nhân int8, cộng int32, dịch bit, bão hòa.
  *Xong khi:* accuracy khớp với mô hình lượng tử hóa trong PyTorch.
- [ ] **Ngày 12 – Xuất file** (`model/export.py`): trọng số `.mem` (hex) vào `model/export/`, ~20 ảnh test kèm output từng tầng.
- [ ] **Ngày 13 – Ảnh thật.** Ảnh chụp/ảnh web resize 32×32 trong `model/real_images/<lớp>/`, đo accuracy.
- [ ] **Ngày 14 – Báo cáo cho thầy.**

## Trạng thái (26/11/2026)

- [x] Code mô hình Python đầy đủ: huấn luyện, đánh giá, lượng tử, golden model, xuất `.mem`, ảnh thật, nhiều seed.
- [x] Kiểm chứng 5 mô hình chính trên laptop nhóm (`docs/so_sanh_5_mo_hinh.md`).
- [x] **Chốt mô hình đi thi: B3c_c4x4_128+cinic_ft** (87,1% ± 0,5; ảnh thật 91,7%; 4,0 triệu MAC) – `docs/model_spec.md`.
- [ ] Nhóm đọc hiểu code + tài liệu theo Ngày 1–13 ở trên (đánh dấu từng ngày khi hiểu).
- [ ] Ngày 14: báo cáo cho thầy → xong giai đoạn Python.
- Giai đoạn tiếp: RTL cho B3c (conv 3×3, ReLU, maxpool, linear, requant) – mỗi module có testbench so với `model/golden_int.py`.

## Ghi chú

- Ngày 9–11 là phần khó nhất; khi tới đó nên dùng model Claude mạnh hơn.
- Mỗi ngày: `cd ~/projects/lsi-design-contest-2027 && claude`, giao đúng một việc; cuối ngày `git add -A && git commit -m "..." && git push`.
