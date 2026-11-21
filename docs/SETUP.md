# Môi trường làm việc & checklist cài đặt

Cập nhật: 21/11/2026

## 1. Phần mềm nằm ở đâu trên laptop

| Thành phần | Vị trí | Ghi chú |
|---|---|---|
| **Vivado ML Standard 2025.2** | `D:\fpga\2025.2\Vivado` | Chạy trên Windows. Chỉ cài Spartan-7 + Cable Drivers |
| Project Vivado | `D:\LSI2027\` | Đường dẫn không dấu, không cách. Project thử: `D:\LSI2027\vivado_test` |
| Vitis HLS (cài kèm, không bỏ được) | `D:\fpga\2025.2\Vitis` | Dùng nếu viết CNN bằng C/C++ → RTL |
| File tải tạm của bộ cài Vivado | `D:\fpga\Downloads\Vivado_2025.2` | Khoảng 16 GB, xoá được sau khi cài xong |
| Claude Code (Windows) | `%USERPROFILE%\.local\bin\claude.exe` | Lệnh `claude` trong PowerShell |
| **Ubuntu (WSL2)** | File ổ ảo `ext4.vhdx` trong ổ C | Không mở trực tiếp file này |
| Repo này (bản làm việc) | Trong Ubuntu: `~/projects/lsi-design-contest-2027` | Từ Windows: `\\wsl.localhost\Ubuntu\home\<user>\projects\lsi-design-contest-2027` |
| Python venv | Trong Ubuntu: `~/eda-venv` | Tự bật mỗi lần mở Ubuntu |
| Bộ ảnh CIFAR-10 | Trong repo: `data/` | Không commit (có trong `.gitignore`) |

Máy có thêm Ubuntu dual boot từ trước; không dùng cho dự án này, khi khởi động chọn **Windows**.

Dung lượng (21/11/2026, trước khi cài Vivado): ổ C trống ~48 GB, ổ D trống ~165 GB.
Quy tắc: phần mềm nặng cài vào ổ D, giữ ổ C trống ≥ 20 GB.

## 2. Mở thư mục, tài liệu và dùng Claude Code

### 2.1. Hai "thế giới" file trên máy
| Nơi | Ví dụ | Mở từ Windows |
|---|---|---|
| **Trong Ubuntu (WSL2)** | repo, `data/`, `docs/*.pdf` | File Explorer → **Linux** → Ubuntu → home → `<user>` → ... |
| **Ổ Windows** | Vivado `D:\fpga`, project Vivado `D:\LSI2027` | Mở bình thường |

- `<user>` = username Ubuntu (chữ đứng trước `@` trên dấu nhắc lệnh).
- Trong Ubuntu, ổ Windows nằm ở `/mnt/c`, `/mnt/d` (ví dụ `D:\LSI2027` = `/mnt/d/LSI2027`).
- Chạy mô phỏng / lệnh trong Ubuntu, trên thư mục `~/...`; không chạy qua `\\wsl` hay `/mnt/d` (chậm).

### 2.2. Mở thư mục repo
| Cách | Lệnh / thao tác |
|---|---|
| File Explorer từ Ubuntu | `cd ~/projects/lsi-design-contest-2027 && explorer.exe .` |
| File Explorer trực tiếp | Dán vào thanh địa chỉ: `\\wsl.localhost\Ubuntu\home\<user>\projects\lsi-design-contest-2027` |
| VS Code (khuyên dùng) | `cd ~/projects/lsi-design-contest-2027 && code .` |
| Claude Code | `cd ~/projects/lsi-design-contest-2027 && claude` |

Mẹo: chuột phải thư mục repo trong File Explorer → **Pin to Quick access**.

### 2.3. Tài liệu PDF của ban tổ chức
PDF **chỉ lưu trên máy**, không đẩy lên GitHub (`.gitignore` chặn `docs/*.pdf`). Link gốc: xem `docs/README.md`.

| Bản | Vị trí | Dùng để |
|---|---|---|
| Trong repo (Ubuntu) | `~/projects/lsi-design-contest-2027/docs/` | Claude Code đọc khi cần |
| Bản copy trên Windows | `D:\LSI2027\docs\` | Đọc bằng trình xem PDF / WPS / Zotero |

```bash
# Tải lại (chạy ở thư mục gốc repo; trong Claude Code thêm dấu ! ở đầu)
wget -P docs https://www.lsi-contest.com/Initialize_20260618_EN.pdf \
             https://www.lsi-contest.com/ZERO7FPGA_Connection-table_261109_001.pdf
# Copy sang ổ D
mkdir -p /mnt/d/LSI2027/docs && cp docs/*.pdf /mnt/d/LSI2027/docs/
# Mở thư mục docs bằng File Explorer
explorer.exe docs
```

| File | Nội dung |
|---|---|
| `Initialize_20260618_EN.pdf` | Cài Pi + ZERO7, thiết kế và chạy bài mẫu (nạp JTAG bằng Vivado Hardware Manager) |
| `ZERO7FPGA_Connection-table_261109_001.pdf` | Bảng nối chân GPIO Pi ↔ chân FPGA, dùng để viết `.xdc` |

### 2.4. Dùng Claude Code
| Thao tác | Tác dụng |
|---|---|
| `!<lệnh>` | Chạy lệnh shell ngay trong Claude Code, ví dụ `!git pull`, `!ls docs` |
| `Shift+Tab` | Đổi chế độ: hỏi trước từng bước / tự sửa file / **auto mode** (tự chạy lệnh không hỏi) / Plan mode |
| `Esc` | Dừng Claude giữa chừng |
| `/clear` | Xoá ngữ cảnh khi chuyển sang việc khác |
| `/usage` | Xem hạn mức gói Pro đã dùng |
| `@<file>` | Chỉ đích danh file cho Claude đọc |
| `/exit` | Thoát Claude Code |

- Dòng chữ mờ trong ô nhập chỉ là **gợi ý tự động**; không bấm Tab/Enter nếu không muốn làm việc đó.
- Cập nhật checklist: nói với Claude Code "đánh dấu [x] mục ... trong docs/SETUP.md rồi commit, push".
- Lỗi `claude is not recognized` trên Windows: đóng hẳn PowerShell, mở cửa sổ mới (không Admin).
  Nếu vẫn lỗi, kiểm tra `Test-Path "$env:USERPROFILE\.local\bin\claude.exe"`.

### 2.5. Vivado
- Mở: shortcut **Vivado 2025.2** trên Desktop.
- Tạo project mới: đặt trong `D:\LSI2027\...` (không dấu, không cách), Part: `xc7s25csga225-1`.
- Xem tài nguyên: sau **Run Synthesis** → **Open Synthesized Design** → **Reports → Report Utilization**.
- Báo cáo tổng hợp dạng file `.rpt` nằm trong `<project>.runs/synth_1/` (ví dụ `counter_utilization_synth.rpt`).
- Tạo báo cáo bằng script của repo: `vivado -mode batch -source vivado/scripts/synth_xc7s25.tcl -tclargs <top>`
  (chạy trong **Vivado 2025.2 Tcl Shell** trên Windows, tại thư mục repo).

## 3. Checklist cài đặt

### Đã xong
- [x] Git, Claude Code, extension Claude Code cho VS Code (Windows)
- [x] WSL2 + Ubuntu, build-essential, Git, Claude Code trong Ubuntu, extension WSL cho VS Code
- [x] Icarus Verilog, GTKWave, Verilator, Yosys, cocotb (venv `~/eda-venv`)
- [x] Kiểm tra bằng bộ đếm 4 bit (`~/projects/test-eda`)
- [x] Repo GitHub: https://github.com/hpnt1912/lsi-design-contest-2027
- [x] Vivado 2025.2 vào `D:\fpga`
- [x] Kiểm tra Vivado: project thử ở `D:\LSI2027\vivado_test\test_counter`, part `xc7s25csga225-1`,
      Run Synthesis → 4 register
- [x] PyTorch CPU + `requirements.txt` + `python model/prepare_cifar4.py`
      → `data/cifar4.npz`: 5000 train / 1000 test mỗi lớp (tổng 20000 / 4000)

### Còn lại
- [ ] Xoá file tải tạm `D:\fpga\Downloads` (~16 GB)
- [ ] Gửi mail mượn ZERO7 (khaild@uit.edu.vn, tiêu đề `Zero7 board <Trường> <Họ tên>`)
- [x] Tải tài liệu PDF ZERO7 vào `docs/` của repo (xem mục 2.3)
- [ ] Copy PDF sang `D:\LSI2027\docs` (không bắt buộc)

### Khi có phần cứng
- [ ] Raspberry Pi Imager → Raspberry Pi OS 64-bit, bật SSH + Wi-Fi
- [ ] Extension Remote-SSH cho VS Code
- [ ] Chạy bài mẫu ZERO7: `pi_button_receiver.v` + `Zero7.xdc` nạp qua JTAG, `raspi-zer07_button.py` trên Pi

## 4. Phần cứng cần có

| Thiết bị | Trạng thái |
|---|---|
| Board ZERO7 | Chưa mượn |
| Raspberry Pi 5 + nguồn USB-C 27 W | Chưa có |
| microSD ≥ 32 GB + đầu đọc | Chưa có |
| Camera Module + cáp camera cho Pi 5 | Chưa có |
| Cáp micro-HDMI → HDMI, chuột, bàn phím | Chưa có |
| Bộ nạp JTAG (tài liệu ghi "JTAG-H53"; có thể là Digilent JTAG-HS3, cần hỏi lại) + dây jumper | Chưa có |

## 5. Lệnh hằng ngày

```bash
cd ~/projects/lsi-design-contest-2027
git pull                 # lấy bản mới nhất trước khi làm
claude                   # làm việc với Claude Code
git add -A && git commit -m "..." && git push   # lưu lên GitHub
```
