# 14 – Lộ trình học: từ con số 0 đến tự dựng lại 5 kiến trúc

← [13_RUN_PIPELINE_STEP_BY_STEP.md](13_RUN_PIPELINE_STEP_BY_STEP.md) · [Mục lục](README.md) · [15_CODE_REVIEW_NOTES.md](15_CODE_REVIEW_NOTES.md) →

Lộ trình gồm 7 chặng, mỗi chặng có: **mục tiêu**, **đọc gì** (file tài liệu + dòng code), **bài tập** có đáp án kiểm chứng được, và
**tự kiểm tra** (trả lời được thì qua chặng). Mọi đáp án số đều đã được chạy lại bằng code của dự án.

Thời lượng ghi ở mỗi chặng chỉ là **gợi ý** cho sinh viên năm 2 đã biết Python cơ bản, không phải số đo.

```
Chặng 0  Python + NumPy + PyTorch tối thiểu
Chặng 1  Dữ liệu đi vào mô hình thế nào          → 02
Chặng 2  Một tầng CNN làm phép tính gì           → 03, 04
Chặng 3  Dựng 5 kiến trúc bằng tay               → 04–08, 12
Chặng 4  Huấn luyện                              → 09
Chặng 5  Đánh giá và đọc kết quả                 → 11
Chặng 6  Từ float sang số nguyên cho FPGA        → 10
Chặng 7  Tự thiết kế kiến trúc thứ 6             → 12, 15
```

---

## Chặng 0 – Nền tảng tối thiểu (gợi ý: 3–5 buổi)

**Mục tiêu:** đọc được code trong `model/` mà không vấp cú pháp.

| Cần biết | Gặp ở đâu trong dự án |
|---|---|
| NumPy: `shape`, cắt mảng `a[r:r+32]`, `np.pad`, `np.clip`, phép `>>` | `data.py` dòng 59–67, `golden_int.py` dòng 43–47 |
| `torch.Tensor`, `permute`, `float()`, `argmax` | `data.py` dòng 38, `evaluate.py` dòng 29–33 |
| `nn.Module`, `__init__` / `forward`, `nn.Sequential` | `net.py` dòng 27–47 |
| `Dataset` / `DataLoader` | `data.py` dòng 41–68, 91–108 |
| `argparse` (đọc tham số dòng lệnh) | `train.py` dòng 232–262 |

**Bài tập 0.1.** Mở Python và chạy:

```bash
python -c "import numpy as np; d=np.load('data/cifar4.npz'); print(d['x_train'].shape, d['x_train'].dtype, d['y_train'][:10])"
```

Giải thích từng số in ra: `(20000, 32, 32, 3)` = số ảnh, cao, rộng, kênh màu; `uint8` = 0..255; nhãn 0..3 = cat, dog, deer, horse.

---

## Chặng 1 – Dữ liệu (gợi ý: 2 buổi)

**Đọc:** [02_DATA_PREPROCESSING.md](02_DATA_PREPROCESSING.md); code `prepare_cifar4.py`, `data.py`.

**Bài tập 1.1.** Vì sao tập train sau khi tách còn 18.000 ảnh? *(Đáp án: `VAL_FRACTION = 0.1` → 20.000 × 0,1 = 2.000 ảnh val.)*

**Bài tập 1.2.** Với 4 mô hình dùng CINIC và `--cifar-repeat 2`, một epoch có bao nhiêu ảnh và bao nhiêu batch?
*(Đáp án: 18.000 × 2 + 84.000 = 120.000 ảnh; batch 128 → ⌈120.000 / 128⌉ = 938 batch.)*

**Bài tập 1.3.** Pixel 200 được đưa vào mạng là số nào? Vì sao chia 256 chứ không chia 255? *(Đáp án: 200/256 = 0,78125. Chia 256 = 2⁸
để phần cứng coi pixel uint8 là fixed-point với 8 bit phân số, không cần phép chia. Xem [15](15_CODE_REVIEW_NOTES.md) mục D.)*

**Tự kiểm tra:** Ảnh val có bị tăng cường không? Ảnh CINIC có vào tập val/test không? *(Không / không – `data.py` dòng 99–107.)*

---

## Chặng 2 – Một tầng CNN (gợi ý: 3–4 buổi)

**Đọc:** [03_CNN_FOUNDATIONS.md](03_CNN_FOUNDATIONS.md) (convolution, padding, BN, ReLU, MaxPool, Linear, receptive field); mục 3 của
[04_ARCHITECTURE_1_B.md](04_ARCHITECTURE_1_B.md).

**Bài tập 2.1.** Conv 3×3, padding 1, stride 1 trên ảnh 32×32: đầu ra bao nhiêu? Sau MaxPool 2×2? *(Đáp án: (32 + 2·1 − 3)/1 + 1 = 32;
pool → 16.)*

**Bài tập 2.2.** Conv1 của B (3 → 16 kênh): bao nhiêu trọng số, bao nhiêu MAC? *(Đáp án: 3 × 16 × 9 = 432 trọng số; sau gộp BN thêm 16
bias → 448 tham số phần cứng. MAC = 432 × 32 × 32 = 442.368.)*

**Bài tập 2.3.** Vì sao B có 27.796 tham số PyTorch học nhưng chỉ 27.684 tham số trên FPGA? *(Đáp án: mỗi kênh BN có γ, β (2 số); gộp
BN vào conv chỉ còn 1 bias mỗi kênh → bớt 16 + 32 + 64 = 112.)*

**Tự kiểm tra:** Giải thích bằng lời vì sao MaxPool không có tham số, và vì sao Linear cuối không có ReLU.

---

## Chặng 3 – Tự dựng lại 5 kiến trúc (gợi ý: 3 buổi)

**Đọc:** các file [04](04_ARCHITECTURE_1_B.md) → [08](08_ARCHITECTURE_5_E3d.md), `net.py` dòng 27–47 (`SmallCNN`) và 75–94 (`VGGLite`).

**Bài tập 3.1 – viết lại từ đầu, không nhìn `net.py`.** Viết một lớp `MyCNN(cfg)` với `cfg` là danh sách `(số kênh ra, có pool hay không)`.
Mỗi phần tử tạo `Conv2d(3×3, pad 1, bias=False) → BatchNorm2d → ReLU → [MaxPool2d(2)]`, cuối cùng là `Flatten → Linear(…, 4)`.

<details>
<summary>Lời giải tham khảo (đã chạy thử)</summary>

```python
import torch
from torch import nn


class MyCNN(nn.Module):
    """cfg: danh sách (số kênh ra, có MaxPool sau hay không)."""

    def __init__(self, cfg, n_classes=4):
        super().__init__()
        layers, c_in, size = [], 3, 32
        for c_out, pool in cfg:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                       nn.BatchNorm2d(c_out), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool2d(2))
                size //= 2
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(c_in * size * size, n_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


CFG = {
    "B":   [(16, True), (32, True), (64, True)],
    "B3c": [(16, True), (32, True), (64, True), (128, False)],
    "E":   [(32, True), (64, False), (64, True), (128, True)],
    "E2":  [(32, True), (64, True), (64, False), (128, True)],
    "E3d": [(24, True), (48, True), (64, False), (64, False), (112, True)],
}
for name, cfg in CFG.items():
    m = MyCNN(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(name, n, tuple(m(torch.zeros(1, 3, 32, 32)).shape))
```

Kết quả phải là:

```
B 27796 (1, 4)
B3c 105876 (1, 4)
E 138660 (1, 4)
E2 138660 (1, 4)
E3d 147836 (1, 4)
```
</details>

So sánh với `net.py`: lớp `VGGLite` (dòng 75–94) gần như y hệt lời giải trên. `SmallCNN` của B là trường hợp riêng: mọi tầng đều có pool
và có thêm tuỳ chọn tắt BN (`use_bn`, dùng cho kiến trúc D). Điểm khác duy nhất: `SmallCNN` cố định đầu vào Linear là `c_in * 4 * 4`
(dòng 41), còn `VGGLite` tự tính kích thước.

**Bài tập 3.2.** Chỉ đổi vị trí một `True/False` để biến E thành E2. MAC giảm bao nhiêu và vì sao tham số không đổi? *(Đáp án: conv 64→64
chuyển từ 16×16 xuống 8×8: 36.864 × (256 − 64) = 7.077.888 MAC ít hơn, tức 19.767.296 → 12.689.408 (−36%). Tham số = C_in·C_out·9 không
phụ thuộc H·W.)*

**Bài tập 3.3.** Kiểm tra `MyCNN` của bạn bằng hàm đếm của dự án:

```bash
PYTHONPATH=model python -c "import net; m = net.build('E2_lite'); print(net.count_params(m), net.count_macs(m))"
```

*(Đáp án: `138372 12689408`.)* Có thể chạy `python documentation/tools/verify_numbers.py` để in toàn bộ bảng kích thước tensor của 5 mô
hình.

**Tự kiểm tra:** Vẽ lại sơ đồ E3d trên giấy, ghi kích thước tensor sau mỗi tầng, rồi so với `assets/architecture_diagrams/arch5_E3d.svg`.

---

## Chặng 4 – Huấn luyện (gợi ý: 3–4 buổi)

**Đọc:** [09_TRAINING_PROCESS.md](09_TRAINING_PROCESS.md); `train.py` dòng 77–164 (`train_one`) và 167–189 (`opt_recipe`).

**Bài tập 4.1.** Chạy B với 3 epoch để xem vòng lặp hoạt động (vài chục giây đến một phút trên CPU):

```bash
python model/train.py --arch B_16_32_64 --epochs 3
```

> Lệnh này **ghi đè** `model/checkpoints/B_16_32_64.pt` và dòng `B_16_32_64` trong `model/results.md`. Nên làm trong một bản sao repo,
> hoặc đặt tên khác như `--arch "B_16_32_64+thu"` (phần sau dấu `+` chỉ là nhãn) rồi xoá dòng đó khỏi `results.md` sau khi xong.

**Bài tập 4.2.** Với công thức opt (lr 0,05, warmup 5, cosine, 30 epoch), LR ở epoch thứ 1, 5 và 6 là bao nhiêu? *(Đáp án theo
`lr_factor`, `train.py` dòng 183–186: epoch 1 → 0,05 × 1/5 = 0,01; epoch 5 → 0,05 × 5/5 = 0,05; epoch 6 → 0,05 × ½(1 + cos(π·0/25)) = 0,05.)*

**Bài tập 4.3.** Với label smoothing 0,1 và 4 lớp, nhãn đích của ảnh mèo là gì? *(Đáp án: [0,925; 0,025; 0,025; 0,025].)*

**Tự kiểm tra:** Vì sao val loss không so trực tiếp được với train loss ở công thức opt? *(Xem [15](15_CODE_REVIEW_NOTES.md) C2.)*

---

## Chặng 5 – Đánh giá (gợi ý: 1–2 buổi)

**Đọc:** [11_EVALUATION_AND_RESULTS.md](11_EVALUATION_AND_RESULTS.md); `evaluate.py`.

**Bài tập 5.1.** Từ ma trận nhầm lẫn float của E2 (hàng cat: 820 đúng; cột cat: 110 ảnh lớp khác bị đoán là cat), tính Precision,
Recall, F1 của lớp cat. *(Đáp án: P = 820/930 = 88,2%; R = 820/1000 = 82,0%; F1 = 85,0%.)*

**Bài tập 5.2.** Accuracy 90% đo trên 4.000 ảnh có sai số chuẩn bao nhiêu? *(Đáp án: √(0,9 × 0,1 / 4000) ≈ 0,47 điểm %; khoảng tin cậy
95% ≈ ±0,93.)* Từ đó, giải thích vì sao chưa kết luận được E (90,65%) tốt hơn E3d (90,25%).

---

## Chặng 6 – Lượng tử hoá cho FPGA (gợi ý: 3–4 buổi)

**Đọc:** [10_QUANTIZATION_AND_HARDWARE.md](10_QUANTIZATION_AND_HARDWARE.md); `quantize.py` (34–130), `golden_int.py` (43–84).

**Bài tập 6.1.** Trọng số lớn nhất (trị tuyệt đối) của một tầng là 0,73. Lượng tử int8 (q_max = 127) thì số bit phân số f là bao nhiêu?
*(Đáp án: `frac_bits_for` = ⌊log₂(127 / 0,73)⌋ = ⌊7,44⌋ = 7, tức thang 2⁻⁷.)*

**Bài tập 6.2.** Tổng tích luỹ acc = 1000, shift = 4. Đầu ra uint8 là bao nhiêu? *(Đáp án: (1000 + 2³) >> 4 = 1008 >> 4 = 63; kẹp 0..255
→ 63.)*

**Bài tập 6.3.** Chạy golden model của E2 cho ảnh test đầu tiên:

```bash
PYTHONPATH=model python -c "import config, data, golden_int; L = golden_int.load_qparams(config.export_dir_for('E2_lite+cinic60_kdft') / 'qparams.npz'); _, _, x, y = data.load_arrays(); print(golden_int.infer(x[:1], L)[0], y[0])"
```

*(Đáp án: `[114051  12992 -65261 -61425] 0` → argmax = 0 = cat, đúng nhãn.)* Lệnh này chỉ cần `data/cifar4.npz` và file `qparams.npz` đã
có trên GitHub, không cần huấn luyện.

**Tự kiểm tra:** Vì sao conv1 của 4 mô hình +cinic dùng int16? Một phép nhân int16 × uint8 có vừa DSP48E1 không? *(Xem
[10](10_QUANTIZATION_AND_HARDWARE.md).)*

---

## Chặng 7 – Tự thiết kế kiến trúc thứ 6 (dự án nhỏ)

**Mục tiêu:** áp dụng mọi thứ ở trên để làm một bước thiết kế giống nhóm đã làm.

1. **Đặt ràng buộc.** Ví dụ: trọng số ≤ 100 KB (`docs/model_spec.md`), MAC ≤ 8 triệu.
2. **Đề xuất 2–3 cấu hình** bằng `MyCNN`. Tính tham số và MAC trước khi huấn luyện, loại cấu hình vượt ràng buộc.
3. **Thêm vào `OPT_ARCHS`** trong một bản sao repo (đừng sửa repo chính khi đang học), ví dụ `"F_mine": ((24, True), (48, True), (96, True), (96, False))`.
4. **Huấn luyện ngắn** với công thức opt (ví dụ 30 epoch, chưa cần CINIC) rồi so với B2 (cùng công thức, 83,88% sau 100 epoch, số
   AI trên cloud).
5. **Lượng tử hoá + golden** (`quantize.py`, `golden_int.py`), xem mức sụt khi chuyển sang số nguyên.
6. **Ghi kết quả** vào một bảng theo mẫu của [12_ARCHITECTURE_COMPARISON.md](12_ARCHITECTURE_COMPARISON.md) mục 1, và tự trả lời 8 câu hỏi
   ở mục 2 cho mô hình của mình.

Ràng buộc của dự án (`CLAUDE.md`): chỉ dùng Conv 3×3, ReLU, MaxPool 2×2, Linear (BN gộp vào conv khi lượng tử). Không dùng Dropout,
residual, depthwise conv vì RTL của nhóm không hỗ trợ.

---

## Thứ tự đọc code gợi ý

| Thứ tự | File | Vì sao đọc lúc này |
|---|---|---|
| 1 | `model/config.py` (50 dòng) | mọi hằng số và đường dẫn; ngắn |
| 2 | `model/prepare_cifar4.py` | dữ liệu đến từ đâu |
| 3 | `model/data.py` | ảnh đi vào mạng ra sao |
| 4 | `model/net.py` | 5 kiến trúc; `count_params`, `count_macs` |
| 5 | `model/train.py` | vòng lặp huấn luyện, 2 công thức, KD |
| 6 | `model/evaluate.py` | ma trận nhầm lẫn |
| 7 | `model/golden_int.py` (110 dòng) | phép tính số nguyên mà FPGA sẽ làm – **đọc trước** quantize |
| 8 | `model/quantize.py` | gộp BN, hiệu chuẩn, chọn thang |
| 9 | `model/export.py` | file `.mem` cho RTL |
| 10 | `model/kiem_chung.py` | ghép mọi thứ thành một quy trình |

---

← [13_RUN_PIPELINE_STEP_BY_STEP.md](13_RUN_PIPELINE_STEP_BY_STEP.md) · [Mục lục](README.md) · [15_CODE_REVIEW_NOTES.md](15_CODE_REVIEW_NOTES.md) →
