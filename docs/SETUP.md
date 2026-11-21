# Môi trường làm việc & checklist cài đặt

Cập nhật: 21/11/2026

## 1. Phần mềm nằm ở đâu trên laptop

| Thành phần | Vị trí | Ghi chú |
|---|---|---|
| **Vivado ML Standard 2025.2** | `D:\fpga\2025.2\Vivado` | Chạy trên Windows. Chỉ cài Spartan-7 + Cable Drivers |
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

## 2. Mở thư mục repo

| Cách | Lệnh / thao tác |
|---|---|
| File Explorer từ Ubuntu | `cd ~/projects/lsi-design-contest-2027 && explorer.exe .` |
| File Explorer trực tiếp | Thanh bên trái → **Linux** → Ubuntu → home → `<user>` → projects |
| VS Code (khuyên dùng) | `cd ~/projects/lsi-design-contest-2027 && code .` |
| Claude Code | `cd ~/projects/lsi-design-contest-2027 && claude` |

Chạy mô phỏng / lệnh trong Ubuntu, không chạy qua đường `\\wsl` từ Windows (chậm).

## 3. Checklist cài đặt

### Đã xong
- [x] Git, Claude Code, extension Claude Code cho VS Code (Windows)
- [x] WSL2 + Ubuntu, build-essential, Git, Claude Code trong Ubuntu, extension WSL cho VS Code
- [x] Icarus Verilog, GTKWave, Verilator, Yosys, cocotb (venv `~/eda-venv`)
- [x] Kiểm tra bằng bộ đếm 4 bit (`~/projects/test-eda`)
- [x] Repo GitHub: https://github.com/hpnt1912/lsi-design-contest-2027

### Đang làm / còn lại
- [x] Vivado 2025.2 vào `D:\fpga` (đang cài)
- [x] Kiểm tra Vivado: project với `counter.v`, part `xc7s25csga225-1`, Run Synthesis → Utilization phải có 4 FF
- [x] PyTorch CPU + `pip install -r requirements.txt` + `python model/prepare_cifar4.py`
- [ ] Gửi mail mượn ZERO7 (khaild@uit.edu.vn, tiêu đề `Zero7 board <Trường> <Họ tên>`)
- [ ] Tải tài liệu PDF ZERO7 về máy (link trong `docs/README.md`)

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
