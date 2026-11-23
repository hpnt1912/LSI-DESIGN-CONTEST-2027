# Tham khảo: tổng quan CNN của cuộc thi lần 23 (Okinawa 2020)

Đề lần 30 yêu cầu tham khảo cấu trúc CNN ở tài liệu này: https://www.lsi-contest.com/2020/shiyou_3e.html
(các trang 3-1 → 3-6 và 4). Dưới đây là tóm tắt.

## Lý thuyết (mục 3)

- **Mạng nơ-ron:** input → các tầng ẩn → output; output được so với tín hiệu giám sát (supervisor, nhãn one-hot).
- **Một nơ-ron:** `z = Σ w·x + b`, `a = h(z)`. Hàm kích hoạt khả vi bất kỳ, ví dụ ReLU `h(z) = max(0, z)`.
- **Conv (3-4):** bộ lọc 3×3 trượt trên ảnh, cộng bias, qua hàm kích hoạt → feature map:
  `z[u,v,k'] = Σ_{i,j,k} w[i,j,k,k'] · x[u+i-1, v+j-1, k] + b[k']`, `a = h(z)`.
- **Pooling (3-5):** gom 2×2 thành 1. Max / average / LP pooling. Không có trọng số, không có hàm kích hoạt.
- **Huấn luyện (3-6):** hàm mất mát tổng bình phương sai số `C_n = ½ Σ (a_out − t)²`; lan truyền ngược
  `δ_out = (a − t)·h'(z)`, `δ_l = (Σ δ_{l+1} w)·h'(z_l)`; cập nhật `w ← w − η ∂C/∂w`.
- **Lưu đồ:** khởi tạo ngẫu nhiên → đưa nhãn → forward → tính C → gradient → cập nhật → lặp tới khi hội tụ.

## Challenge lần 23 (mục 4)

| Level | Cấu hình |
|---|---|
| 1 Beginner | 3 filter 3×3, max pool 2×2, sigmoid, fully connected |
| 2 Intermediate | input 28×28, 3 filter 5×5, max pooling, sigmoid hoặc ReLU, fully connected |
| Expert | Unlimited |

Mỗi level có outline thiết kế phần cứng và file Verilog để mô phỏng (trên trang web cũ).

## Mã ví dụ

- Trang 3-6 có PDF "CNN example" và zip mã Octave 3.8.
- Repo lưu bản gốc và bản giải nén (UTF-8) trong [`reference/`](reference/README.md): bài nhận dạng 〇/× ảnh 3×3,
  Conv 2×2 → MaxPool → FC, sigmoid, bình phương sai số.

## Áp dụng cho bài 2027

- Giữ đúng các khối conv 3×3, ReLU, max pool 2×2, FC để dễ trình bày trong báo cáo.
- Huấn luyện có thể dùng cross-entropy thay bình phương sai số; phần cứng chỉ lấy argmax nên không ảnh hưởng.
- Khác biệt: ảnh 32×32 RGB (3 kênh), 4 lớp.
