# Raspberry Pi 5

Chương trình chạy trên Pi: đọc ảnh từ camera hoặc thẻ SD, resize về 32×32, gửi sang FPGA qua GPIO,
nhận kết quả phân loại và hiển thị kèm thời gian xử lý.

Gói dự kiến (Raspberry Pi OS): `python3-picamera2`, `python3-opencv`, `gpiozero` / `lgpio`
(thêm `spidev` nếu giao tiếp SPI).
