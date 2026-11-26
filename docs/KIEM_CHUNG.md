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
Kết quả mong đợi (đã thử trên máy cloud, khớp hoàn toàn):

```
Mô hình  Số nguyên  Số đã ghi  Trạng thái
B           80,00%     80,00%  ✅ Đã kiểm chứng ...
B3c         87,20%     87,20%  ⚠️ Chưa kiểm chứng – số liệu do AI chạy trên máy cloud
E           90,60%     90,60%  ⚠️ ...
E2          90,30%     90,30%  ⚠️ ...
E3d              –          –  ⛔ Chưa chạy ...
```

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
