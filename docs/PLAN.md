# Kế hoạch: mô hình Python (2 tuần, 23/11 – 07/12/2026)

Mỗi ngày làm một việc, xong thì đánh dấu `[x]` và commit. Chi tiết yêu cầu mô hình xem [`model_spec.md`](model_spec.md).

## Tuần 1: mô hình số thực (float)

- [x] **Ngày 1 – Chốt yêu cầu.** Viết `docs/model_spec.md`.
- [ ] **Ngày 2 – Dữ liệu.** `model/data.py`: DataLoader CIFAR-4 từ `data/cifar4.npz`, chuẩn hoá, augmentation (random crop pad 4, lật ngang), tách 10% train làm validation.
  *Xong khi:* in được kích thước batch, hiển thị 8 ảnh đúng nhãn.
- [ ] **Ngày 3 – CNN cơ sở.** `model/net.py` + `model/train.py`. Mạng Conv3×3(3→8)–ReLU–Pool → Conv(8→16)–ReLU–Pool → Conv(16→32)–ReLU–Pool → FC(512→4).
  *Xong khi:* có accuracy test đầu tiên, lưu checkpoint vào `model/checkpoints/` (không commit).
- [ ] **Ngày 4–5 – So sánh kiến trúc.** Thử 3–4 biến thể (độ rộng, số tầng, có/không BatchNorm).
  *Xong khi:* có bảng accuracy / số tham số / số MAC / KB trọng số int8 trong `model/results.md`.
- [ ] **Ngày 6 – Đánh giá.** Confusion matrix, accuracy từng lớp (`model/evaluate.py`).
- [ ] **Ngày 7 – Dự phòng + báo cáo giữa kỳ** cho thầy (1 trang).

## Tuần 2: chuẩn bị cho phần cứng

- [ ] **Ngày 8 – BN folding.** Gộp BatchNorm vào Conv; output trước/sau gần như trùng.
- [ ] **Ngày 9–10 – Lượng tử hóa int8** (`model/quantize.py`). PTQ trước; sụt quá ~2 điểm % thì làm QAT. Ưu tiên scale là lũy thừa của 2.
- [ ] **Ngày 11 – Golden model số nguyên** (`model/golden_int.py`, NumPy): nhân int8, cộng int32, dịch bit, bão hòa.
  *Xong khi:* accuracy khớp với mô hình lượng tử hóa trong PyTorch.
- [ ] **Ngày 12 – Xuất file** (`model/export.py`): trọng số `.mem` (hex) vào `model/export/`, ~20 ảnh test kèm output từng tầng.
- [ ] **Ngày 13 – Ảnh thật.** Ảnh chụp/ảnh web resize 32×32 trong `model/real_images/<lớp>/`, đo accuracy.
- [ ] **Ngày 14 – Báo cáo cho thầy.**

## Ghi chú

- Ngày 9–11 là phần khó nhất; khi tới đó nên dùng model Claude mạnh hơn.
- Mỗi ngày: `cd ~/projects/lsi-design-contest-2027 && claude`, giao đúng một việc; cuối ngày `git add -A && git commit -m "..." && git push`.
