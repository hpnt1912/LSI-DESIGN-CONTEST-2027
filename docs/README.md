# Tài liệu cuộc thi

## Trang chính thức

- Trang chủ: https://www.lsi-contest.com/index_e.html
- Đặc tả thiết kế (Design Specification): https://www.lsi-contest.com/shiyou_1e.html
- Tổng quan CNN của cuộc thi lần 23 (đề yêu cầu tham khảo cấu trúc CNN): https://www.lsi-contest.com/2020/shiyou_3e.html

## Tài liệu ZERO7

- Hướng dẫn cài đặt "Raspberry Pi and ZERO7: From Setup to Design and Operation":
  https://www.lsi-contest.com/Initialize_20260618_EN.pdf
- Bảng chân kết nối FPGA (ZERO7 FPGA Connection Table):
  https://www.lsi-contest.com/ZERO7FPGA_Connection-table_261109_001.pdf

## Datasheet chip

- AMD DS180 – 7 Series FPGAs Data Sheet: Overview (bảng tài nguyên Spartan-7):
  tìm "DS180 7 Series Overview" trên docs.amd.com

Tóm tắt các tài liệu trên: [`hardware.md`](hardware.md), [`cnn23_reference.md`](cnn23_reference.md).
PDF của ban tổ chức không đưa lên repo; tải về máy để đọc.

## Tóm tắt yêu cầu

- Đội 1–3 sinh viên đại học/cao đẳng.
- Ảnh đầu vào 32×32; 4 lớp: cat, dog, deer, horse. Dataset huấn luyện tùy chọn.
- Ảnh kiểm tra do ban tổ chức cung cấp trong ngày thi; hệ thống đọc ảnh từ camera, thẻ SD...
  và trình bày tốc độ xuất kết quả.
- Điều kiện: dữ liệu kiểm tra được chỉ định + camera thời gian thực.
- **Bắt buộc mọi hạng mục:** tổng hợp vừa chip XC7S25CSGA225-1 và nộp báo cáo tổng hợp.

### Hạng mục

| Class | Level | Yêu cầu |
|---|---|---|
| 1. Image Classification | L1 Beginner | CNN đơn giản (ví dụ ảnh 3×3), tổng hợp + implement trên XC7S25 |
| | L2 Intermediate | CNN cho ảnh 32×32, huấn luyện + kiểm tra, tổng hợp + implement trên XC7S25 |
| | L3 Advanced | Tự do board/kiến trúc, vẫn phải có báo cáo tổng hợp XC7S25 |
| 2. ZERO7 | L3 | Board cố định ZERO7, đề tài tự do |

Giám khảo đánh giá tốc độ, diện tích, và cả ý tưởng, tính độc đáo.

## Mượn board ZERO7 (Việt Nam)

PGS. Lâm Đức Khải, ĐH Công nghệ Thông tin (UIT) — khaild@uit.edu.vn
Tiêu đề mail: `Zero7 board <Tên trường> <Họ tên>`

Hỏi đáp / nộp bài: support@LSI-contest.com
