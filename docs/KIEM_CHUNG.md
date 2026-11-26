# Quy trình kiểm chứng 5 mô hình chính

Mục tiêu: tự chạy lại trên máy của nhóm cả 5 mô hình đã chọn, để mọi con số trong báo cáo là **số của nhóm**,
không phải số do AI chạy. Toàn bộ chỉ cần **một script**: `model/kiem_chung.py`. Mỗi lần chạy xong một mô hình,
script tự ghi kết quả, vẽ lại bảng so sánh và đẩy lên GitHub.

Bảng so sánh tự sinh: [`docs/so_sanh_5_mo_hinh.md`](so_sanh_5_mo_hinh.md).

## 5 mô hình

| Mã | Vai trò | Lượt chạy cuối (thư mục `model/export*`) | Số AI ghi trước khi kiểm chứng | **Kết quả trên laptop (25–26/11)** |
|---|---|---|---|---|
| `B` | Cơ bản (~80%) | `B_16_32_64` (`model/export/`) | 80,00% | ✅ **80,00%** |
| `B3c` | Nhanh | `B3c_c4x4_128+cinic_ft` | 87,20% | ✅ **87,60%** |
| `E` | Chính xác cao + mạng thầy | `E_opt_cnn+cinic_ft` | 90,60% | ✅ **90,65%** |
| `E2` | Cân bằng | `E2_lite+cinic60_kdft` | 90,30% | ✅ **89,60%** |
| `E3d` | Thử nghiệm mới (5 conv) | `E3d_deep5+cinic60_kdft` | – (chưa từng chạy) | ✅ **90,25%** |

Số là test số nguyên (golden model). Trạng thái luôn mới nhất: [`docs/so_sanh_5_mo_hinh.md`](so_sanh_5_mo_hinh.md).

## Bước 0 – Chuẩn bị (mỗi lần mở máy)

```bash
cd ~/projects/lsi-design-contest-2027
source ~/eda-venv/bin/activate
git pull
pip install -r requirements.txt
ls data/                       # phải có cifar4.npz; nếu không: python model/prepare_cifar4.py
```

## Bước 1 – Kiểm tra nhanh (5–10 phút, không cần CINIC)

```bash
python model/kiem_chung.py nhanh
```

Script chạy golden model số nguyên trên tham số **đã có sẵn** của B, B3c, E, E2 và so với số đã ghi.
Kết quả mong đợi (sau khi đã kiểm chứng cả 5 mô hình trên laptop 25–26/11):

```
Mô hình  Số nguyên  Số đã ghi  Trạng thái
B          80,00%     80,00%  ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 20:34)
B3c        87,60%     87,20%  ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 21:06)
E          90,65%     90,60%  ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 22:20)
E2         89,60%     90,30%  ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-25 23:59)
E3d        90,25%          –  ✅ Đã kiểm chứng trên máy `LAPTOP-2RM30BH4` (2026-11-26 01:38)
```

Cột "Số nguyên" là kết quả golden model trên tham số hiện có trên GitHub (của laptop). Cột "Số đã ghi" là số AI ghi **trước** khi
kiểm chứng; chênh vài phần mười là bình thường (xem mục "Đọc kết quả").

Bước này chứng minh các file tham số trên GitHub **cho đúng accuracy đã ghi**, nhưng chưa chứng minh tự huấn
luyện lại được. Muốn vậy làm tiếp Bước 2–3.

## Bước 2 – Tải dữ liệu CINIC-10 (một lần, cần cho B3c, E, E2, E3d)

1. Mở trình duyệt: <https://datashare.ed.ac.uk/handle/10283/3192> → tải **`CINIC-10.tar.gz`** (687 MB).
2. Để nguyên trong thư mục **Downloads** của Windows. Không cần giải nén.
3. Lần đầu chạy mô hình cần CINIC, script tự tìm file ở `C:\Users\<tên>\Downloads\` và tạo
   `data/cinic4_extra.npz` (84.000 ảnh ImageNet 32×32 của 4 lớp, đã loại ảnh gốc CIFAR và ảnh trùng tập test).

CINIC-10 (Darlow và cộng sự, 2018, Đại học Edinburgh) = ảnh CIFAR-10 + ảnh ImageNet thu nhỏ về 32×32, cùng
10 lớp. Đề thi cho phép tự chọn dataset; nhớ ghi rõ trong báo cáo là có dùng thêm dữ liệu này.

## Bước 3 – Huấn luyện lại và kiểm chứng

```bash
python model/kiem_chung.py B          # ~5–10 phút: thử trước cho quen
python model/kiem_chung.py tatca      # cả 5 mô hình, ~6 giờ – nên chạy qua đêm
```

Hoặc chọn từng mô hình: `python model/kiem_chung.py B3c E E2 E3d`. Script tự sắp đúng thứ tự
(E phải chạy trước E2 và E3d vì E là mạng thầy; thiếu E thì script tự thêm E).

| Mô hình | Các bước script tự chạy | Thời gian ước tính* |
|---|---|---|
| B | train 30 epoch → evaluate → quantize → golden → export | ~5–10 phút |
| B3c | train 30 epoch (+CINIC) → tinh chỉnh 15 epoch → evaluate → quantize (conv1 int16) → golden → export | ~1 giờ |
| E | như B3c với mạng E | ~1,5 giờ |
| E2 | train 60 epoch (+CINIC) → tinh chỉnh 15 epoch + học từ thầy E → … | ~2 giờ |
| E3d | như E2 với mạng E3d | ~2 giờ (chưa từng chạy – ước tính) |

\*Ước tính theo thời gian máy cloud đã ghi (máy đó chạy B 30 epoch mất ~4 phút, gần bằng laptop của nhóm).

**Chạy qua đêm:** cắm sạc, đặt Windows **không ngủ** (Settings → System → Power → Sleep: Never), giữ nguyên cửa sổ
Ubuntu (đóng cửa sổ là dừng). Nhật ký đầy đủ từng mô hình: `model/checkpoints/kiem_chung_<mã>.txt`.

**Bị dừng giữa chừng** (mất điện, lỗi, lỡ đóng cửa sổ): chạy lại **đúng lệnh cũ**. Script nhớ bước nào đã xong
(`model/kiem_chung_log.json`) và chạy tiếp. Muốn làm lại từ đầu: thêm `--lam-lai`.

## Bước 4 – Đo trên ảnh thật (1–2 giờ, không cần huấn luyện)

Mọi accuracy ở trên đo trên ảnh CIFAR. Ngày thi là ảnh chụp thật, nên cần biết mô hình nào **tụt mạnh** trên ảnh thật.

1. Hỏi ban tổ chức: ảnh ngày thi phát ở kích thước gốc hay đã thu về 32×32 (`docs/model_spec.md` dòng 17).
2. Tạo thư mục và chép ảnh vào (mỗi lớp 20–30 ảnh, nhiều góc chụp và nền khác nhau; tốt nhất chụp bằng đúng camera dùng ngày thi):
   ```bash
   mkdir -p model/real_images/cat model/real_images/dog model/real_images/deer model/real_images/horse
   cd model/real_images && explorer.exe . && cd ../..
   ```
3. Chạy (khoảng 3 phút):
   ```bash
   python model/anh_that.py
   ```
   Script chạy cả 5 mô hình bằng golden model số nguyên, cùng tiền xử lý như `predict_image.py`. Kết quả gồm accuracy, khoảng tin cậy
   95%, accuracy từng lớp, mức chênh so với CIFAR, và danh sách ảnh mà cả 5 mô hình đều đoán sai. Tất cả được ghi vào mục
   **"Ảnh thật (model/real_images)"** của `model/results.md`.
4. Đẩy kết quả lên (chỉ `results.md`; ảnh giữ trên máy):
   ```bash
   git add model/results.md && git commit -m "Kết quả 5 mô hình trên ảnh thật" && git push
   ```

## Bước 5 – Chạy thêm seed cho mô hình đã chọn (qua đêm)

Mỗi mô hình mới chạy 1 lần. E, E3d, E2 chênh nhau dưới 1 điểm, nhỏ hơn mức dao động giữa các lần chạy. Muốn kết luận chắc thì chạy thêm
seed và báo cáo **trung bình ± độ lệch chuẩn**. Chỉ cần làm cho mô hình định chọn (sau Bước 4).

```bash
python model/nhieu_seed.py E3d
```

- Mặc định thêm seed 2 và 3 (cùng lượt chính seed 2027 là 3 lần). Mỗi seed mất khoảng bằng thời gian huấn luyện lượt chính (E3d khoảng
  96 phút), nên 2 seed khoảng 3,2 giờ. Đổi seed: `--seeds 2 3 4`.
- Mỗi seed chạy **đúng** các lệnh của lượt chính trong `kiem_chung.py`, chỉ thêm `--seed` và đặt tên lượt chạy có đuôi `_s2`, `_s3`,
  nên **không đè** kết quả 5 mô hình chính. Mạng thầy của E2/E3d vẫn là E gốc.
- Bị dừng: chạy lại đúng lệnh, script chạy tiếp từ bước dở.
- Kết quả: mục **"Nhiều seed – <mã>"** trong `model/results.md`. Script **không** tự push; cuối cùng nó in lệnh `git` để bạn chạy.

## Đã kiểm tra: ảnh gần trùng giữa CINIC và tập test

`prepare_cinic4.py` chỉ loại ảnh trùng tuyệt đối. Đo lại bằng `documentation/tools/check_cinic_leak.py` (26/11) cho kết quả:
- Khoảng 5–8% ảnh test có bản gần trùng trong CINIC.
- Mô hình B, không học CINIC, cũng đoán nhóm ảnh đó đúng hơn hẳn (87% so với 79%), tức đó là ảnh dễ chứ không phải bị học thuộc.
- Bỏ các ảnh trùng ra thì mọi mô hình giảm 0,1–0,4 điểm, B giảm nhiều nhất, và thứ tự các mô hình không đổi.

**Không thấy dấu hiệu CINIC làm phồng kết quả.** Chi tiết và bảng số: `documentation/15_CODE_REVIEW_NOTES.md` mục B5.

## Script tự cập nhật những gì

Sau **mỗi mô hình** chạy xong:

| File / thư mục | Nội dung |
|---|---|
| `model/results.md` | Dòng của từng lượt huấn luyện, mục "Đánh giá chi tiết" và "Lượng tử hoá" của mô hình đó |
| `model/figures/` | Đường học `train_*.png`, ma trận nhầm lẫn, ảnh đoán sai, hình so sánh `so_sanh_5_mo_hinh.png` |
| `model/export/` (B) hoặc `model/export_<lượt chạy>/` | `qparams`, file `.mem` trọng số, 20 vector kiểm thử, README cho RTL |
| `docs/so_sanh_5_mo_hinh.md` | Bảng so sánh 5 mô hình, cột trạng thái chuyển thành **✅ Đã kiểm chứng trên máy `<tên máy>` (ngày)** |
| `model/kiem_chung_log.json` | Trạng thái và bước đã xong của từng mô hình |
| GitHub | Tự `git commit` + `git push` (thêm `--khong-push` nếu chỉ muốn lưu trên máy) |

Checkpoint `.pt` và dữ liệu không bao giờ được đẩy lên (đã nằm trong `.gitignore`).

## Đọc kết quả – thế nào là "đạt"

1. Trong phần in ra của `quantize.py`: dòng **`golden_int.py và PyTorch trùng từng bit: True`** – bắt buộc.
2. Accuracy trên máy nhóm lệch số đã ghi **khoảng ±1 điểm %** là bình thường (khác CPU, khác phiên bản thư viện,
   lệnh tinh chỉnh được dựng lại từ ghi chú). Lệch nhiều hơn ~2 điểm thì gửi file nhật ký cho Claude xem.
3. E2 từng cho 89,45–90,30% qua 5 lần huấn luyện → được ~90% là đạt, chưa chắc lần nào cũng > 90%.
4. E3d chưa có số tham khảo: đây là thí nghiệm mới, kết quả nào cũng là thông tin hữu ích cho báo cáo.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Chưa có dữ liệu CINIC-10` | Làm Bước 2, rồi chạy lại đúng lệnh |
| `Không đẩy được lên GitHub` | Kết quả vẫn đã commit trên máy. Chạy `git pull` rồi `git push`; nếu hỏi mật khẩu: `gh auth login` |
| `Killed` khi đang chạy B3c/E/E2/E3d | Thiếu RAM (dữ liệu CINIC ~0,5 GB trong RAM). Đóng bớt chương trình rồi chạy lại đúng lệnh |
| `Lỗi ở bước: ...` | Đọc dòng cuối của khối lỗi, xem `model/checkpoints/kiem_chung_<mã>.txt`; sửa xong chạy lại đúng lệnh |

## Chạy thử script (cho người sửa code)

`python model/kiem_chung.py tatca --thu` chạy mọi bước với 1 epoch, không commit/push – để chắc script không
lỗi trước khi chạy thật. **Đừng** dùng trên repo thật nếu không muốn ghi đè kết quả: nên chạy trong một bản sao.
