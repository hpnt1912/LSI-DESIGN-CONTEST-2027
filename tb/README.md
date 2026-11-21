# Testbench

- Testbench Verilog: `tb_<module>.v`, chạy bằng Icarus Verilog.
- Testbench Python: dùng cocotb, so sánh đầu ra RTL với mô hình tham chiếu trong `model/`.

```bash
iverilog -o sim ../rtl/*.v tb_<module>.v && vvp sim
```
