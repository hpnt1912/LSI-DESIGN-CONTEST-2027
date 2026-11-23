# Mã ví dụ CNN của BTC (cuộc thi lần 23, Octave)

- `CNN2019.zip`: bản gốc tải từ trang https://www.lsi-contest.com/2020/shiyou_3-6e.html
- `cnn2019_octave/`: bản giải nén, đã đổi mã hoá Shift-JIS → UTF-8 để đọc được chú thích tiếng Nhật.

## Bài toán ví dụ: nhận dạng "〇" và "×"

Ảnh nhị phân 3×3 (1 = đen, 0 = trắng), 2 lớp. 6 ảnh huấn luyện (3 ảnh 〇, 3 ảnh ×).

| Tầng | File | Cấu hình |
|---|---|---|
| Conv | `Convolutional_layer.m` | 2 bộ lọc 2×2, stride 1, không padding, bias → 2×2×2, **sigmoid** |
| Pooling | `Pooling_layer.m` | Max pooling 2×2, stride 2 → 1×1×2 |
| Output (FC) | `Output_layer.m` | 2 → 2, bias, **sigmoid** |
| Hàm mất mát | `Output_Backprop.m` | Bình phương sai số `½ Σ (t − a)²` |
| Lan truyền ngược | `*_Backprop.m` | Viết tay cho từng tầng |
| Huấn luyện | `OX_Judgment.m` | Gradient descent toàn batch, η = 0.1, 5000 vòng, vẽ đường mất mát |
| Kiểm tra | `Kensho.m` (検証 = kiểm định) | Đưa 1 ảnh vào, so 2 output để kết luận 〇 hay × |

Chạy: mở Octave trong thư mục này, chạy `OX_Judgment` rồi `Kensho`.

## Bài học cho dự án

- Cấu trúc **Conv → MaxPool → FC** giống hướng đi của mô hình 2027; chỉ khác quy mô (32×32×3, 4 lớp, nhiều tầng hơn).
- Mã ví dụ dùng sigmoid; mô hình của mình dùng **ReLU** (đề lần 23 cho phép) vì phần cứng chỉ cần một phép so sánh.
- Lan truyền ngược viết tay chỉ để học; PyTorch tính gradient tự động.
- Phần forward (`Convolutional_layer`, `Pooling_layer`, `Output_layer`) chính là thứ golden model số nguyên (`model/golden_int.py`)
  và RTL sẽ hiện thực: vòng lặp nhân–cộng trên cửa sổ, cộng bias, kích hoạt, lấy max.
- Lưu ý khi đọc: trong `Output_Backprop.m`, `dCda` nhân thêm `a_p·(1−a_p)` dù tầng pooling không có hàm kích hoạt; mạng nhỏ nên vẫn hội tụ, không cần bắt chước.
