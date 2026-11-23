# Phần cứng: chip XC7S25 và board ZERO7

Nguồn: AMD DS180 (7 Series FPGAs Data Sheet: Overview, Table 2), "Raspberry Pi and ZERO7: From Setup to Design and Operation",
"ZERO7 FPGA Connection Table" (261109).

## 1. Tài nguyên XC7S25CSGA225-1

| Tài nguyên | Số lượng | Ghi chú |
|---|---|---|
| Logic cells | 23.360 | |
| Slices | 3.650 | Mỗi slice 4 LUT + 8 FF |
| **LUT** | **14.600** | |
| **Flip-flop** | **29.200** | |
| Distributed RAM | 313 Kb | Chỉ một số slice dùng LUT làm RAM/SRL |
| **DSP48E1** | **80** | Pre-adder, nhân 25×18, cộng, tích luỹ |
| **Block RAM** | 90 × 18 Kb = 45 × 36 Kb = **1.620 Kb (~202 KB)** | Dual-port |
| CMT | 3 | Mỗi CMT: 1 MMCM + 1 PLL |
| XADC | 1 | |
| I/O | 3 bank, 150 user I/O (gói CSGA225) | |

### Ngân sách cho CNN

- Trọng số int8 nên ≤ ~100 KB, chừa BRAM cho bộ đệm feature map (32×32×16 int8 = 16 KB).
- Mạng 8/16/32 kênh ≈ 8 nghìn tham số (~8 KB). Mạng 16/32/64 kênh ≈ 28 nghìn tham số (~28 KB), ~2,8 triệu MAC/ảnh.
- Nút thắt là **DSP (80)** → số MAC quyết định tốc độ. Ước lượng thô (bỏ qua truyền ảnh):
  2,8 triệu MAC với 9 DSP @ 20 MHz ≈ 15 ms/ảnh; với 80 DSP @ 100 MHz ≈ 0,35 ms/ảnh.

## 2. Board ZERO7

- HAT cắm lên header 40 chân của Raspberry Pi 5; GPIO/SPI/UART/I2C của Pi nối thẳng vào FPGA.
- Pi lo camera, HDMI, mạng; FPGA làm bộ tăng tốc. Có 2 PMOD, 2 cổng I/F 1.8 V, QSPI flash, LED, nút, DIP switch.

### Quy trình nạp (theo hướng dẫn)
1. Windows 11 + Vivado 2025.2, part `xc7s25csga225-1`.
2. RTL project → thêm `.v` + `Zero7.xdc` → (tuỳ chọn) ILA để debug → Generate Bitstream.
3. Cắm ZERO7 lên Pi, cấp nguồn Pi (LED trên ZERO7 sáng).
4. Nối JTAG adapter ("JTAG-H53" theo tài liệu) vào chân JTAG của ZERO7 và USB vào PC.
5. Hardware Manager → Open New Target → Local server → **JTAG clock 3.000.000 Hz** → Program Device (.bit + .ltx).

Bài mẫu: Pi đọc nút ở GPIO4, gửi qua GPIO27; FPGA đồng bộ 2 tầng rồi bật GPIO2 và GPIO22. File mẫu:
`pi_button_receiver.v`, `Zero7.xdc`, `raspi-zer07_button.py` (đi kèm tài liệu, xin BTC/người cho mượn board).

### Bảng chân FPGA (tín hiệu quan trọng)

| Chức năng | Tín hiệu → chân FPGA |
|---|---|
| Clock | **CLK20M (20 MHz) → B13**; EXT_CLK → H1 |
| SPI0 của Pi | GPIO10 MOSI → A6 (JP9), GPIO9 MISO → A7, GPIO11 SCLK → A8 (JP11), GPIO8 CE0 → D9, GPIO7 CE1 → C10 |
| UART của Pi | GPIO14 → B6, GPIO15 → C7 |
| I2C của Pi | GPIO2 SDA → B1, GPIO3 SCL → B2 (cũng nối U11) |
| GPIO khác | GPIO4 A2, GPIO17 A3, GPIO27 A4, GPIO22 A5, GPIO5 A9, GPIO12 D10, GPIO23 C8, GPIO24 D8, GPIO25 C9, GPIO26 B12 |
| LED dùng chung GPIO | GPIO6 → A10 (D3 xanh dương), GPIO13 → B10 (D9 xanh lá), GPIO19 → B11 (D8 đỏ), GPIO18 → D7 (LED2 WS2812B) |
| LED riêng FPGA | GPIO_LED1 → R3 (D5), GPIO_LED2 → R2 (D6), S7_LED0 → N1 (LED1 WS2812B) |
| Nút / switch | GPIO21 → B15 (SW4 STOP), GPIO20 → C14 (SW5 START), GPIO16 → C13 (SW6 RUN), GPIO_PSW → R4 (SW1), GPIO_DIPSW → M4 (SW3) |
| PMOD1 (J1) | D15, E15, F15, G15, E13, E14, F14, G14 |
| PMOD2 (J2) | G13, G11, E11, D12, G12, F11, E12, D13 |

> ⚠️ JP9/JP11 ghi "CUT" trên đường SPI dùng chung với QSPI flash. Cần kiểm chứng trạng thái jumper với BTC hoặc người cho mượn board trước khi dùng SPI.

### Truyền ảnh Pi → FPGA
Mỗi ảnh 32×32×3 = 3.072 byte. Có thể dùng SPI0 hoặc bus GPIO song song; chọn ở giai đoạn thiết kế RTL.
