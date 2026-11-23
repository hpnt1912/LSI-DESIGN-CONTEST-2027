# CLAUDE.md

Dự án thi LSI Design Contest in Hokkaido 2027: phần cứng CNN phân loại ảnh trên board ZERO7.

## Ràng buộc cứng
- Chip mục tiêu: **AMD Spartan-7 XC7S25CSGA225-1** (part `xc7s25csga225-1`). Mọi thiết kế phải tổng hợp vừa chip này.
- Công cụ: Vivado ML Standard 2025.2 (chạy trên Windows).
- Đầu vào ảnh 32×32 RGB; 4 lớp theo thứ tự chỉ số: 0 cat, 1 dog, 2 deer, 3 horse.
- Board ZERO7 là HAT gắn GPIO 40 chân của Raspberry Pi 5; Pi lo camera và hiển thị.

## Tài liệu dự án (đọc khi cần)
- `docs/PLAN.md`: kế hoạch từng ngày, đánh dấu `[x]` khi xong.
- `docs/HuongDan_MoHinh_CNN_Python.docx`: hướng dẫn chi tiết A–Z mô hình Python (người dùng học theo file này).
- `docs/model_spec.md`: yêu cầu mô hình CNN.
- `docs/hardware.md`: tài nguyên XC7S25 (14.600 LUT, 80 DSP, 1.620 Kb BRAM), bảng chân ZERO7 (clock 20 MHz ở B13).
- `docs/cnn23_reference.md`: lý thuyết CNN của cuộc thi lần 23 (đề yêu cầu tham khảo).
- `docs/SETUP.md`: vị trí phần mềm, đường dẫn, checklist cài đặt (cập nhật khi môi trường thay đổi).

## Môi trường
- Ubuntu WSL2, user `ngoc_thang`, repo ở `~/projects/lsi-design-contest-2027`, venv `~/eda-venv` (PyTorch bản CPU).

## Quy ước code
- RTL: Verilog-2001, một module mỗi file, tên file = tên module.
- Reset đồng bộ, tích cực mức thấp `rst_n`; clock tên `clk`.
- Số học fixed-point; ghi rõ định dạng (ví dụ Q1.7) trong comment đầu module.
- Mỗi module RTL có testbench trong `tb/`, so sánh với mô hình Python tham chiếu trong `model/`.
- Python: PEP 8, dùng `pathlib`. Chạy script từ thư mục gốc repo.
- Mô hình: chỉ dùng Conv 3×3, ReLU, MaxPool 2×2, Linear; output 4 logit, phần cứng lấy argmax.
- Cố định seed (`torch.manual_seed`) để kết quả lặp lại được; ghi kết quả thử nghiệm vào `model/results.md`.
- Lượng tử hóa: int8 trọng số, uint8 activation, int32 bias/accumulator, thang lũy thừa của 2 (xem `model/export/README.md`). `model/golden_int.py` là chuẩn bit-exact cho RTL.

## Lệnh hay dùng
- Lint: `verilator --lint-only -Wall rtl/<file>.v`
- Mô phỏng nhanh: `iverilog -o sim rtl/*.v tb/<tb>.v && vvp sim`
- Tổng hợp nhanh (ước lượng): `yosys -p "read_verilog rtl/*.v; synth_xilinx -family xc7 -top <top>; stat"`
- Tổng hợp chính thức (Windows): `vivado -mode batch -source vivado/scripts/synth_xc7s25.tcl -tclargs <top>`
- Chuẩn bị dữ liệu: `python model/prepare_cifar4.py`

## Cách làm việc
- Trả lời bằng tiếng Việt.
- Trước khi sửa nhiều file, nêu kế hoạch ngắn.
- Người dùng đang học: giải thích ngắn gọn code mới viết (mỗi khối làm gì), không chỉ đưa code.
- Mỗi phiên làm đúng một việc trong `docs/PLAN.md`; xong thì cập nhật checklist.
- Chạy lint + mô phỏng sau khi sửa RTL; chỉ commit khi pass.
- Không commit dataset, file `.vcd`, thư mục build của Vivado.
