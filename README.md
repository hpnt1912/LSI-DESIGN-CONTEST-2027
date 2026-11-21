# LSI Design Contest in Hokkaido 2027 — CNN on ZERO7

Hardware implementation of a CNN image classifier (4 classes: **cat, dog, deer, horse**, 32×32 input) for the
30th LSI Design Contest in Hokkaido, targeting **AMD Spartan-7 XC7S25CSGA225-1** on the Gigafirm **ZERO7** board
(Raspberry Pi 5 HAT).

> Thiết kế phần cứng CNN phân loại ảnh 4 lớp cho LSI Design Contest 2027. Chip mục tiêu: XC7S25CSGA225-1 trên board ZERO7.

## Mốc thời gian

| Việc | Hạn |
|---|---|
| Nộp báo cáo cuối + báo cáo tổng hợp (synthesis report) XC7S25CSGA225-1 | **CN 31/01/2027** |
| Thuyết trình (hybrid, 10 phút / tối đa 10 slide + 5 phút Q&A) | **T6 05/03/2027** |

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `docs/` | Đường dẫn tài liệu cuộc thi, ghi chú thiết kế |
| `model/` | Huấn luyện CNN (PyTorch), lượng tử hóa, xuất trọng số |
| `rtl/` | Mã Verilog của bộ tăng tốc CNN |
| `tb/` | Testbench (cocotb / Icarus / Verilator) |
| `vivado/` | Script tổng hợp, file ràng buộc (`.xdc`) |
| `pi/` | Chương trình Python trên Raspberry Pi (camera, giao tiếp FPGA) |
| `reports/` | Báo cáo tổng hợp xuất từ Vivado (dùng để nộp) |

## Môi trường

- Vivado ML Standard **2025.2** (Windows), part `xc7s25csga225-1`
- WSL2 Ubuntu: Icarus Verilog, Verilator, Yosys, GTKWave, cocotb
- Python: xem `requirements.txt`

```bash
# Trong Ubuntu (WSL2)
source ~/eda-venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python model/prepare_cifar4.py          # tải CIFAR-10, tách 4 lớp
```

## Tạo báo cáo tổng hợp bắt buộc

```bat
:: Trong thư mục repo, trên Windows (Vivado 2025.2)
vivado -mode batch -source vivado/scripts/synth_xc7s25.tcl -tclargs <ten_module_top>
```

Kết quả nằm trong `reports/` (utilization, timing summary).

## Tài liệu cuộc thi

Xem [`docs/README.md`](docs/README.md).
