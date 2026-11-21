# CLAUDE.md

Dự án thi LSI Design Contest in Hokkaido 2027: phần cứng CNN phân loại ảnh trên board ZERO7.

## Ràng buộc cứng
- Chip mục tiêu: **AMD Spartan-7 XC7S25CSGA225-1** (part `xc7s25csga225-1`). Mọi thiết kế phải tổng hợp vừa chip này.
- Công cụ: Vivado ML Standard 2025.2 (chạy trên Windows).
- Đầu vào ảnh 32×32 RGB; 4 lớp theo thứ tự chỉ số: 0 cat, 1 dog, 2 deer, 3 horse.
- Board ZERO7 là HAT gắn GPIO 40 chân của Raspberry Pi 5; Pi lo camera và hiển thị.

## Quy ước code
- RTL: Verilog-2001, một module mỗi file, tên file = tên module.
- Reset đồng bộ, tích cực mức thấp `rst_n`; clock tên `clk`.
- Số học fixed-point; ghi rõ định dạng (ví dụ Q1.7) trong comment đầu module.
- Mỗi module RTL có testbench trong `tb/`, so sánh với mô hình Python tham chiếu trong `model/`.
- Python: PEP 8, dùng `pathlib`.

## Lệnh hay dùng
- Lint: `verilator --lint-only -Wall rtl/<file>.v`
- Mô phỏng nhanh: `iverilog -o sim rtl/*.v tb/<tb>.v && vvp sim`
- Tổng hợp nhanh (ước lượng): `yosys -p "read_verilog rtl/*.v; synth_xilinx -family xc7 -top <top>; stat"`
- Tổng hợp chính thức (Windows): `vivado -mode batch -source vivado/scripts/synth_xc7s25.tcl -tclargs <top>`
- Chuẩn bị dữ liệu: `python model/prepare_cifar4.py`

## Cách làm việc
- Trả lời bằng tiếng Việt.
- Trước khi sửa nhiều file, nêu kế hoạch ngắn.
- Chạy lint + mô phỏng sau khi sửa RTL; chỉ commit khi pass.
- Không commit dataset, file `.vcd`, thư mục build của Vivado.
