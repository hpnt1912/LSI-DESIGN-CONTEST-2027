# 03 – Kiến thức nền tảng để hiểu 5 kiến trúc CNN

← [02_DATA_PREPROCESSING.md](02_DATA_PREPROCESSING.md) · [Mục lục](README.md) · [04_ARCHITECTURE_1_B.md](04_ARCHITECTURE_1_B.md) →

File này chỉ giải thích những kiến thức **thực sự xuất hiện trong code** của dự án. Mỗi mục gồm 5 phần:
**(1) bản chất · (2) tại sao cần · (3) hoạt động thế nào · (4) ví dụ · (5) nằm ở đâu trong code**.

Quy ước: ví dụ ghi "*minh hoạ*" là số tự chọn để giải thích (đã tính bằng Python để không sai số học); ví dụ ghi
"*số thật*" lấy từ file kết quả của dự án. Đường dẫn và số dòng tính theo nhánh `main` (commit `e64cd23`).

**Mục lục**

1. [Tensor và các chiều dữ liệu](#1-tensor-và-các-chiều-dữ-liệu)
2. [Biểu diễn ảnh (image representation)](#2-biểu-diễn-ảnh-image-representation)
3. [Tích chập (convolution)](#3-tích-chập-convolution)
4. [Kernel và filter](#4-kernel-và-filter)
5. [Feature map](#5-feature-map)
6. [Stride và padding](#6-stride-và-padding)
7. [Pooling](#7-pooling)
8. [Hàm kích hoạt (activation function) và ReLU](#8-hàm-kích-hoạt-activation-function-và-relu)
9. [Flatten và tầng kết nối đầy đủ (fully connected / Linear)](#9-flatten-và-tầng-kết-nối-đầy-đủ-fully-connected--linear)
10. [Logit, softmax và argmax](#10-logit-softmax-và-argmax)
11. [Hàm mất mát (loss function): cross-entropy, label smoothing](#11-hàm-mất-mát-loss-function-cross-entropy-label-smoothing)
12. [Lan truyền xuôi (forward propagation)](#12-lan-truyền-xuôi-forward-propagation)
13. [Lan truyền ngược (backpropagation)](#13-lan-truyền-ngược-backpropagation)
14. [Gradient descent](#14-gradient-descent)
15. [Optimizer: SGD momentum/Nesterov, AdamW, weight decay](#15-optimizer-sgd-momentumnesterov-adamw-weight-decay)
16. [Learning rate và lịch learning rate (warmup, cosine)](#16-learning-rate-và-lịch-learning-rate-warmup-cosine)
17. [Batch size](#17-batch-size)
18. [Epoch](#18-epoch)
19. [Train / validation / test](#19-train--validation--test)
20. [Overfitting và underfitting](#20-overfitting-và-underfitting)
21. [Dropout (dự án KHÔNG dùng)](#21-dropout-dự-án-không-dùng)
22. [Batch Normalization](#22-batch-normalization)
23. [Tăng cường dữ liệu (data augmentation)](#23-tăng-cường-dữ-liệu-data-augmentation)
24. [Chưng cất tri thức (knowledge distillation)](#24-chưng-cất-tri-thức-knowledge-distillation)
25. [Tinh chỉnh (fine-tuning) và dữ liệu bổ sung](#25-tinh-chỉnh-fine-tuning-và-dữ-liệu-bổ-sung)
26. [Hạt giống ngẫu nhiên (seed) và checkpoint](#26-hạt-giống-ngẫu-nhiên-seed-và-checkpoint)
27. [MAC – thước đo khối lượng tính](#27-mac--thước-đo-khối-lượng-tính)
28. [Trường tiếp nhận (receptive field)](#28-trường-tiếp-nhận-receptive-field)
29. [Số dấu phẩy tĩnh (fixed-point) và lượng tử hoá](#29-số-dấu-phẩy-tĩnh-fixed-point-và-lượng-tử-hoá)
30. [Có trong code nhưng 5 mô hình không dùng: EMA, CutMix](#30-có-trong-code-nhưng-5-mô-hình-không-dùng-ema-cutmix)

---

## 1. Tensor và các chiều dữ liệu

1. **Bản chất.** Tensor là mảng số nhiều chiều. Số chiều gọi là *rank*: vô hướng (0 chiều), vector (1), ma trận (2),
   tensor 3 hay 4 chiều.
2. **Tại sao cần.** Mạng nơ-ron xử lý cùng lúc nhiều ảnh, mỗi ảnh nhiều kênh, mỗi kênh là một lưới điểm. Tensor gói tất cả vào
   một khối để GPU/CPU tính song song.
3. **Hoạt động.** PyTorch dùng thứ tự **(N, C, H, W)** = (số ảnh trong batch, số kênh, chiều cao, chiều rộng). Trong khi đó NumPy và
   file ảnh dùng (H, W, C). Đổi giữa hai dạng bằng `permute` / `transpose`.
4. **Ví dụ (số thật).** Một batch huấn luyện của B có `x.shape = (64, 3, 32, 32)` (64 ảnh, 3 kênh RGB, 32×32) và `y.shape = (64,)`.
   Sau conv1 của B: `(64, 16, 32, 32)`; sau Flatten: `(64, 1024)`; đầu ra: `(64, 4)`.
5. **Trong code.**
   - `model/data.py` dòng 38: `torch.from_numpy(img_uint8).permute(2, 0, 1)` đổi (32, 32, 3) → (3, 32, 32).
   - `model/golden_int.py` dòng 71: `transpose(0, 3, 1, 2)` đổi NHWC → NCHW.
   - `model/data.py` dòng 128 in kích thước một batch.

## 2. Biểu diễn ảnh (image representation)

1. **Bản chất.** Ảnh màu là 3 lưới số (R, G, B). Mỗi số là độ sáng 0..255, lưu bằng 8 bit không dấu (uint8).
2. **Tại sao cần.** Mạng chỉ tính trên số. Cách biến pixel thành số phải **giống nhau** khi huấn luyện và khi chạy trên FPGA.
3. **Hoạt động trong dự án.** Ảnh CIFAR đã sẵn 32×32. Giá trị vào mạng là `x = pixel / 256`, thuộc [0; 0,996]. Dự án **không** trừ
   trung bình hay chia độ lệch chuẩn, vì chia cho 256 = 2⁸ trên phần cứng chỉ là dịch phải 8 bit, gần như "miễn phí".
4. **Ví dụ.** Pixel 255 → 255/256 = 0,99609; pixel 128 → 0,5. Một ảnh = 3·32·32 = 3.072 byte.
5. **Trong code.** `model/data.py` dòng 6–9 (docstring nêu quy ước) và dòng 36–38 (`to_tensor`). Trên phần cứng: `golden_int.py` coi ảnh
   vào là số nguyên `pixel` với thang `f = 8` (dòng 8, 71). Ảnh thật được cắt giữa rồi thu về 32×32: `predict_image.py` dòng 27–32.

## 3. Tích chập (convolution)

1. **Bản chất.** Một bộ lọc nhỏ (3×3) trượt trên ảnh. Tại mỗi vị trí, nhân từng phần tử của bộ lọc với vùng ảnh bên dưới rồi cộng lại
   (cộng thêm bias), được **một** số của ảnh đầu ra.
2. **Tại sao cần.** Đặc trưng ảnh (cạnh, góc, kết cấu lông, hình tai…) mang tính **cục bộ** và **lặp lại** ở nhiều chỗ. Conv dùng **cùng
   một bộ trọng số** ở mọi vị trí (chia sẻ trọng số – weight sharing), nên ít tham số hơn nhiều so với nối đầy đủ. Ví dụ conv1 của B có
   432 trọng số; nếu thay bằng Linear từ 3.072 số vào ra 16×32×32 = 16.384 số thì cần hơn 50 triệu trọng số.
3. **Hoạt động** (công thức của golden model, `golden_int.py` dòng 13–15):

   ```text
   z[o, h, w] = Σ_c Σ_ky Σ_kx  W[o, c, ky, kx] · x[c, h+ky−1, w+kx−1]  +  b[o]
   ```

   - `o`: kênh ra; `c`: kênh vào; `ky, kx ∈ {0,1,2}`: vị trí trong bộ lọc 3×3.
   - Điểm ngoài biên ảnh lấy bằng 0 (padding).
   - PyTorch (và dự án) **không lật** kernel. Về toán học đây là phép tương quan chéo (cross-correlation), nhưng trong học sâu vẫn gọi là
     "convolution".
4. **Ví dụ (minh hoạ, 1 kênh vào, 1 kênh ra).** Ảnh 4×4, kernel phát hiện cạnh dọc, padding 1:

   ```text
   x = [[1,2,0,1],        kernel = [[1,0,-1],
        [0,1,3,1],                  [1,0,-1],
        [2,1,0,0],                  [1,0,-1]]
        [1,0,2,1]]
   Điểm ra (1,1): vùng [[1,2,0],[0,1,3],[2,1,0]] · kernel = (1+0+2) − (0+3+0) = 0
   Toàn bộ đầu ra (chưa bias): [[-3,-2,1,3],[-4,0,2,3],[-2,-2,0,5],[-1,1,0,2]]
   ```

5. **Trong code.**
   - PyTorch: `nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=…)` ở `model/net.py` dòng 34 (B) và 83 (4 mô hình còn lại).
   - Số nguyên: `conv3x3_int` (`golden_int.py` dòng 50–60) cài đặt đúng công thức trên bằng 9 lần `einsum`, mỗi lần cho một vị trí
     (ky, kx).

## 4. Kernel và filter

1. **Bản chất.** **Kernel** là ma trận trọng số 3×3 tác động lên **một** kênh vào. **Filter** là chồng `C_in` kernel (kích thước
   `C_in × 3 × 3`) tạo ra **một** kênh ra. Một tầng conv có `C_out` filter.
2. **Tại sao cần.** Mỗi filter học phát hiện một kiểu đặc trưng. Nhiều filter cho ra nhiều "góc nhìn" khác nhau về ảnh.
3. **Hoạt động.** Trọng số conv có hình `(C_out, C_in, 3, 3)`. Mỗi filter tạo một feature map.
4. **Ví dụ (số thật).** conv2 của E2 = `Conv2d(32, 64, 3)`: 64 filter, mỗi filter 32×3×3 = 288 trọng số, tổng
   64·288 = 18.432 trọng số.
5. **Trong code.** Thứ tự phần tử khi xuất cho RTL là `W[o][c][ky][kx]` (`export.py` dòng 11, dòng 104 trong README sinh ra), đúng
   thứ tự `(C_out, C_in, 3, 3)` của PyTorch.

## 5. Feature map

1. **Bản chất.** Feature map là ảnh đầu ra của một filter, cho biết đặc trưng mà filter tìm thấy mạnh hay yếu ở mỗi vị trí.
2. **Tại sao cần.** Tầng sau "đọc" feature map của tầng trước để ghép đặc trưng đơn giản thành đặc trưng phức tạp: cạnh → góc → bộ
   phận → con vật.
3. **Hoạt động.** Mỗi tầng conv tạo `C_out` feature map. MaxPool làm chúng nhỏ dần. Càng về sau số kênh càng nhiều, kích thước càng nhỏ.
4. **Ví dụ (số thật).** E3d: 24×16×16 → 48×8×8 → 64×8×8 → 64×8×8 → 112×4×4 (đo bằng golden model, xem `08_ARCHITECTURE_5_E3d.md`).
5. **Trong code.** Golden model lưu feature map từng tầng (`infer(..., keep=True)`, `golden_int.py` dòng 68–84) để làm vector kiểm thử
   `testvec/img<i>_conv<k>.mem` cho RTL (`export.py` dòng 75–78).

## 6. Stride và padding

1. **Bản chất.**
   - **Stride (S)** là bước trượt của bộ lọc.
   - **Padding (P)** là số viền 0 thêm quanh ảnh.
2. **Tại sao cần.**
   - Stride > 1 thu nhỏ ảnh.
   - Padding giữ kích thước ảnh và giúp pixel ở biên được "nhìn" đủ số lần.
3. **Hoạt động.**

   ```text
   H_out = floor((H_in + 2·P − K) / S) + 1
   ```

   - `H_in`: kích thước vào.
   - `K`: kích thước kernel (hoặc cửa sổ pool).
   - `P`: số hàng/cột 0 thêm mỗi phía.
   - `S`: bước trượt.
   - `floor`: làm tròn xuống.
4. **Ví dụ (số thật).**
   - Mọi conv của dự án: `K = 3, P = 1, S = 1` → `H_out = floor((32 + 2 − 3)/1) + 1 = 32`, kích thước giữ nguyên.
   - MaxPool: `K = 2, P = 0, S = 2` → `floor((32 − 2)/2) + 1 = 16`.
5. **Trong code.**
   - `padding=1` ở `net.py` dòng 34/83; stride conv mặc định là 1.
   - `nn.MaxPool2d(2)` lấy stride = 2 (mặc định của PyTorch: stride = kernel_size).
   - Golden model đệm viền 0 ở `golden_int.py` dòng 53–54.

## 7. Pooling

1. **Bản chất.** Gom mỗi ô 2×2 thành 1 số. Dự án dùng **max pooling**: lấy số lớn nhất.
2. **Tại sao cần.**
   - Giảm 4 lần số điểm, nên giảm 4 lần phép tính và bộ nhớ của tầng sau.
   - Bền với dịch chuyển nhỏ của vật trong ảnh.
   - Tăng trường tiếp nhận.
   - Trên phần cứng chỉ cần bộ so sánh, không cần nhân.
3. **Hoạt động.** Không có trọng số, không có hàm kích hoạt. Cửa sổ 2×2, bước 2.
4. **Ví dụ (minh hoạ).**
   - Feature map 4×4 sau ReLU `[[0,0,1.5,3.5],[0,0.5,2.5,3.5],[0,0,0.5,5.5],[0,1.5,0.5,2.5]]` → MaxPool → `[[0.5,3.5],[1.5,5.5]]`.
   - Ví dụ, ô trên trái `max(0, 0, 0, 0.5) = 0.5`.
5. **Trong code.**
   - `nn.MaxPool2d(2)` ở `net.py` dòng 38 (B, luôn có) và 86 (4 mô hình còn lại, chỉ khi `pool=True`).
   - Số nguyên: `maxpool2` (`golden_int.py` dòng 63–65) dùng `reshape(n, c, h//2, 2, w//2, 2).max(axis=(3, 5))`.
   - Thứ tự "ReLU rồi pool" hay "pool rồi ReLU" cho cùng kết quả, vì hai phép đều đơn điệu. Golden model làm requant + ReLU rồi mới
     pool (dòng 76–78).

## 8. Hàm kích hoạt (activation function) và ReLU

1. **Bản chất.** Hàm phi tuyến áp lên từng số sau conv. **ReLU**: `ReLU(z) = max(0, z)`.
2. **Tại sao cần.**
   - Không có hàm phi tuyến thì nhiều tầng conv/linear chồng nhau vẫn chỉ tương đương **một** phép tuyến tính, không học được ranh
     giới phức tạp.
   - ReLU được chọn vì: rẻ trên phần cứng (chỉ xét bit dấu), đạo hàm đơn giản (0 hoặc 1) nên không bị "tắt gradient" như sigmoid ở
     vùng bão hoà, và nằm trong danh sách phép tính cho phép của đề (`docs/model_spec.md`).
3. **Hoạt động.** Giá trị âm thành 0, dương giữ nguyên. Đạo hàm: 1 nếu z > 0, 0 nếu z < 0.
4. **Ví dụ (minh hoạ).** Đầu ra conv ở mục 3 cộng bias 0,5 rồi qua ReLU: `-3+0,5 = -2,5 → 0`; `3+0,5 = 3,5 → 3,5`.
5. **Trong code.**
   - `nn.ReLU()` ở `net.py` dòng 37/84.
   - Trên số nguyên, ReLU gộp với bão hoà 8 bit: `np.clip(acc, 0, 255)` (`golden_int.py` dòng 47).
   - Tầng Linear cuối **không** có ReLU (`net.py` dòng 41/90).

## 9. Flatten và tầng kết nối đầy đủ (fully connected / Linear)

1. **Bản chất.**
   - **Flatten** duỗi tensor `C × H × W` thành vector `C·H·W` số.
   - **Linear** (fully connected) nối **mỗi** đầu vào với **mỗi** đầu ra: `y = W·x + b`.
2. **Tại sao cần.** Sau các conv, feature map đã chứa đặc trưng cấp cao. Linear tổng hợp toàn bộ chúng để chấm điểm 4 lớp.
3. **Hoạt động.**
   - `W` có hình `(N_out, N_in)` = `(4, C·H·W)`.
   - Số tham số = `N_in·N_out + N_out`.
   - Thứ tự duỗi là (kênh, hàng, cột): phần tử `k = c·H·W + h·W + w`.
4. **Ví dụ (số thật).**
   - B: Flatten 64×4×4 = 1.024 → Linear 1.024→4, có 1.024·4 + 4 = 4.100 tham số.
   - E3d: 112×4×4 = 1.792 → 7.172 tham số.
5. **Trong code.**
   - `torch.flatten(x, 1)` ở `net.py` dòng 46/94; `nn.Linear(c_in*size*size, 4)` ở dòng 41/90.
   - Golden: `flat = x.reshape(len(x), -1)` rồi `flat @ W.T + b` (`golden_int.py` dòng 81–82).
   - Thứ tự `k` khi xuất `.mem`: `export.py` dòng 104.

## 10. Logit, softmax và argmax

1. **Bản chất.**
   - **Logit** là 4 điểm số thô mà mạng cho ra (có thể âm).
   - **Softmax** biến logit thành xác suất: `p_i = e^{z_i} / Σ_j e^{z_j}`.
   - **Argmax** là chỉ số của logit lớn nhất.
2. **Tại sao cần.**
   - Khi **huấn luyện**, loss cần xác suất (softmax nằm bên trong cross-entropy).
   - Khi **suy luận**, chỉ cần biết lớp nào lớn nhất. Softmax không đổi thứ tự các số (vì hàm mũ tăng), nên argmax(logit) =
     argmax(softmax). Phần cứng bỏ hẳn softmax.
3. **Hoạt động.** Softmax luôn dương, tổng bằng 1.
4. **Ví dụ.**
   - *Minh hoạ*: logit `[2,0; 0,5; −1,0; 0,1]` → softmax `[0,703; 0,157; 0,035; 0,105]` → argmax = 0 (cat).
   - *Số thật*: ảnh test số 0, mô hình E2. Golden model cho logit số nguyên `[114.051; 12.992; −65.261; −61.425]`, thang 2¹⁵, nên logit
     thực ≈ `[3,481; 0,396; −1,992; −1,875]` → argmax = cat (đúng nhãn). Chi tiết ở mục 3.7 của `07_ARCHITECTURE_4_E2.md`.
5. **Trong code.**
   - Dự đoán: `logits.argmax(1)` (`train.py` dòng 44, 137; `evaluate.py` dòng 33; `golden_int.py` dòng 88).
   - Softmax chỉ xuất hiện trong loss distillation (`train.py` dòng 54–56) và ngầm trong `nn.CrossEntropyLoss`.

## 11. Hàm mất mát (loss function): cross-entropy, label smoothing

1. **Bản chất.** Loss là một số đo "mạng sai bao nhiêu" trên một batch. Huấn luyện tức là tìm trọng số làm loss nhỏ nhất.
2. **Tại sao cần.**
   - Cần một đại lượng khả vi để lấy đạo hàm theo trọng số.
   - Accuracy không khả vi (đúng/sai nhảy bậc) nên không dùng trực tiếp để huấn luyện được.
3. **Hoạt động.**
   - **Cross-entropy (CE)** với nhãn đúng `y`: `L = −ln p_y`. Mạng tự tin đúng (`p_y → 1`) thì `L → 0`; tự tin sai thì `L` rất lớn.
   - **Label smoothing ε** thay nhãn "cứng" (1, 0, 0, 0) bằng `q = (1−ε)·one_hot + ε/K`. Với ε = 0,1 và K = 4 thì
     `q = (0,925; 0,025; 0,025; 0,025)`, và `L = −Σ_i q_i·ln p_i`. Mạng bớt bị ép phải "tự tin tuyệt đối".
   - Mức loss thấp nhất có thể đạt khi có label smoothing là entropy của `q`:
     `−(0,925·ln 0,925 + 3·0,025·ln 0,025) = 0,349`.
   - CE gốc của bài lần 23 là tổng bình phương sai số (`docs/cnn23_reference.md`). Dự án dùng CE vì CE phù hợp với bài phân loại hơn.
4. **Ví dụ.**
   - *Minh hoạ*: với softmax ở mục 10, `L = −ln 0,703 = 0,352`.
   - *Số thật*: ảnh test số 0 của E2 có `L_CE = 0,0533`, `L_label smoothing = 0,4010`.
5. **Trong code.**
   - Công thức gốc: `nn.CrossEntropyLoss()` (`train.py` dòng 104).
   - Công thức opt: `nn.CrossEntropyLoss(label_smoothing=0.1)` (dòng 175, hằng số `config.OPT_LABEL_SMOOTHING` ở `config.py` dòng 40).
   - Khi đánh giá val/test, loss **không** có label smoothing: `nn.CrossEntropyLoss(reduction="sum")` (dòng 39). Vì vậy val loss
     thấp hơn train loss ở các lượt opt.

## 12. Lan truyền xuôi (forward propagation)

1. **Bản chất.** Đưa batch ảnh qua lần lượt các tầng để được logit.
2. **Tại sao cần.** Đây là phép dự đoán. Huấn luyện cần nó để tính loss; suy luận chỉ cần đúng bước này.
3. **Hoạt động.**
   - `model(x)` gọi `forward`, chạy lần lượt `features` (Conv/BN/ReLU/Pool) → flatten → Linear.
   - PyTorch đồng thời ghi lại "đồ thị tính toán" để dùng cho lan truyền ngược.
4. **Ví dụ (số thật).** B: `(64,3,32,32)` → conv1 `(64,16,32,32)` → pool `(64,16,16,16)` → … → `(64,64,4,4)` → `(64,1024)` → `(64,4)`.
5. **Trong code.**
   - `logits = model(x)` ở `train.py` dòng 123.
   - `forward` ở `net.py` dòng 44–47 / 93–94.
   - Phiên bản số nguyên của forward: `golden_int.infer` (dòng 68–84).

## 13. Lan truyền ngược (backpropagation)

1. **Bản chất.** Tính đạo hàm của loss theo **mọi** trọng số bằng quy tắc dây chuyền (chain rule), đi ngược từ loss về đầu mạng.
2. **Tại sao cần.** Gradient descent cần biết mỗi trọng số nên tăng hay giảm, và nhiều hay ít.
3. **Hoạt động.**
   - Với `L = f(g(w))` thì `∂L/∂w = ∂L/∂g · ∂g/∂w`. Mỗi tầng chỉ cần biết đạo hàm cục bộ của mình. Ví dụ ReLU truyền gradient
     nguyên vẹn nếu z > 0, chặn lại nếu z < 0.
   - PyTorch tự làm việc này (autograd). Bài lần 23 viết tay công thức `δ_l = (Σ δ_{l+1}·w)·h'(z_l)` (`docs/cnn23_reference.md`).
4. **Ví dụ (minh hoạ).** CE + softmax có đạo hàm theo logit rất gọn: `∂L/∂z_i = p_i − q_i`. Với softmax ở mục 10 và nhãn cat:
   `∂L/∂z = (0,703 − 1; 0,157; 0,035; 0,105) = (−0,297; 0,157; 0,035; 0,105)`. Nghĩa là: tăng logit cat, giảm 3 logit còn lại.
5. **Trong code.** `loss.backward()` (`train.py` dòng 132). Trước đó phải `opt.zero_grad()` (dòng 131), vì PyTorch **cộng dồn**
   gradient qua các lần gọi.

## 14. Gradient descent

1. **Bản chất.** Đi từng bước nhỏ ngược hướng gradient: `w ← w − η·∂L/∂w`, với η là learning rate.
2. **Tại sao cần.** Không có công thức nghiệm đóng cho trọng số tối ưu của mạng sâu; chỉ có thể tiến dần.
3. **Hoạt động.** Dự án dùng **mini-batch** gradient descent: gradient tính trên một batch (64 hoặc 128 ảnh) chứ không trên toàn bộ dữ
   liệu. Nhờ vậy nhanh hơn, và nhiễu nhẹ của batch còn giúp thoát cực tiểu kém.
4. **Ví dụ (minh hoạ).** w = 0,8, gradient 0,5, η = 0,05 → `w ← 0,8 − 0,05·0,5 = 0,775` (chưa có momentum, weight decay).
5. **Trong code.** `opt.step()` (`train.py` dòng 133).

## 15. Optimizer: SGD momentum/Nesterov, AdamW, weight decay

1. **Bản chất.** Optimizer là quy tắc cụ thể để cập nhật trọng số từ gradient.
2. **Tại sao cần.** Gradient descent thuần dễ dao động và chậm. Momentum và Adam giúp đi nhanh, ổn định hơn. Weight decay phạt trọng số
   lớn để chống overfitting.
3. **Hoạt động** (theo tài liệu PyTorch, kiểm chứng bằng code ở dưới).
   - **SGD + momentum μ + Nesterov** (4 mô hình +cinic):

     ```text
     g ← g + wd·w ;  v ← μ·v + g ;  w ← w − η·(g + μ·v)
     ```

   - **AdamW** (mô hình B):

     ```text
     m ← β₁m + (1−β₁)g ;  s ← β₂s + (1−β₂)g² ;  w ← w − η·( m̂/(√ŝ + ε) + wd·w )
     ```

     Weight decay "tách rời" (decoupled) khỏi gradient. Mặc định β₁ = 0,9, β₂ = 0,999.
4. **Ví dụ (minh hoạ, đã đối chiếu với `torch.optim.SGD`).** Bước đầu tiên với w = 0,8, g = 0,5, η = 0,05, μ = 0,9, wd = 5·10⁻⁴, v = 0:
   - `g' = 0,5 + 0,0005·0,8 = 0,5004`
   - `v = 0,5004`
   - lượng cập nhật `= 0,5004 + 0,9·0,5004 = 0,95076`
   - `w = 0,8 − 0,05·0,95076 = 0,752462`
5. **Trong code.**
   - B: `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)` (`train.py` dòng 106). Weight decay áp cho **mọi** tham số,
     kể cả γ, β của BN.
   - Opt: `torch.optim.SGD(..., momentum=0.9, nesterov=True)` (dòng 176–180). Weight decay **chỉ** áp cho tham số có ≥ 2 chiều (trọng
     số conv/linear), không áp cho γ, β của BN và bias của Linear (dòng 176–177).

## 16. Learning rate và lịch learning rate (warmup, cosine)

1. **Bản chất.** Learning rate η là độ dài mỗi bước cập nhật. Lịch LR là η thay đổi theo epoch.
2. **Tại sao cần.**
   - η lớn: học nhanh nhưng dao động, có thể phân kỳ.
   - η nhỏ: ổn định nhưng chậm.
   - Lịch tốt: đầu tăng dần (**warmup**, tránh "sốc" khi trọng số còn ngẫu nhiên), giữa lớn để khám phá, cuối nhỏ dần (**cosine**) để
     tinh chỉnh vào đáy.
3. **Hoạt động.**
   - Công thức gốc: `η(e) = 0,5·η₀·(1 + cos(π·e/T))`, với e = số epoch đã xong, T = 30.
   - Opt: e < warm thì `η = η₀·(e+1)/warm`; còn lại `η = η₀·0,5·(1 + cos(π·(e−warm)/(E−warm)))`.
   - `sched.step()` được gọi **sau** mỗi epoch.
4. **Ví dụ (số thật).**
   - B (η₀ = 10⁻³, 30 epoch): epoch 1 → 1,0·10⁻³; epoch 16 → 0,5·10⁻³; epoch 30 → 2,7·10⁻⁶.
   - E2 lượt 60 epoch (η₀ = 0,05, warm 5): epoch 1 → 0,01; epoch 5 → 0,05; epoch 6 → 0,05; epoch 33 → 0,0257; epoch 60 → 0,00004.
5. **Trong code.**
   - Gốc: `CosineAnnealingLR(opt, T_max=epochs)` (`train.py` dòng 108).
   - Opt: `LambdaLR(opt, lr_factor)` với `lr_factor` ở dòng 183–188.
   - Hằng số: `config.LR = 1e-3` (dòng 25), `OPT_LR = 0.05` (dòng 36), `OPT_WARMUP_EPOCHS = 5` (dòng 39).
   - Lượt tinh chỉnh dùng `--lr 0.005 --warmup 0` (`kiem_chung.py` dòng 42).

## 17. Batch size

1. **Bản chất.** Batch size là số ảnh dùng cho **một** lần cập nhật trọng số.
2. **Tại sao cần.**
   - Batch lớn: gradient ít nhiễu, tận dụng tính song song, nhưng tốn RAM và ít lần cập nhật hơn mỗi epoch.
   - BatchNorm cũng cần batch đủ lớn để thống kê ổn định.
3. **Hoạt động.**
   - Số batch mỗi epoch = `ceil(N / batch)`.
   - Batch cuối có thể nhỏ hơn: `DataLoader` mặc định `drop_last=False`.
4. **Ví dụ (số thật).**
   - B: 18.000 ảnh / 64 = 281 batch đủ + 1 batch 16 ảnh = **282** batch.
   - Lượt +cinic: 120.000 / 128 = **938** batch (batch cuối 64 ảnh).
   - Lượt tinh chỉnh: 18.000 / 128 = **141** batch (batch cuối 80 ảnh).
5. **Trong code.**
   - `config.BATCH_SIZE = 64` (dòng 23) và `OPT_BATCH_SIZE = 128` (dòng 35).
   - Truyền vào `data.get_loaders` (`train.py` dòng 102, 174).
   - Val/test dùng batch 256 (`data.py` dòng 106–107); batch này chỉ để đánh giá, không ảnh hưởng kết quả.

## 18. Epoch

1. **Bản chất.** Một epoch là một lượt duyệt hết tập train.
2. **Tại sao cần.** Mạng cần nhìn mỗi ảnh nhiều lần, mỗi lần với một biến thể tăng cường khác, để học dần.
3. **Hoạt động.** Sau mỗi epoch, dự án đánh giá trên val và lưu checkpoint nếu val tốt hơn.
4. **Ví dụ (số thật).**
   - B: 30 epoch × 282 batch = 8.460 lần cập nhật.
   - E2 lượt 60 epoch: 60 × 938 = 56.280 lần cập nhật.
5. **Trong code.** `for ep in range(1, epochs + 1)` (`train.py` dòng 116). Số epoch lấy từ `--epochs` hoặc mặc định 30 / 100
   (dòng 257). Cả 5 mô hình đều đặt epoch tường minh hoặc dùng mặc định 30 (B).

## 19. Train / validation / test

1. **Bản chất.**
   - **Train**: tập để học (cập nhật trọng số).
   - **Validation (val)**: tập để **chọn** (chọn epoch tốt nhất, so sánh cấu hình).
   - **Test**: tập chỉ để **báo cáo** cuối cùng.
2. **Tại sao cần.** Nếu vừa chọn vừa báo cáo trên cùng một tập, con số báo cáo sẽ lạc quan hơn thực tế, vì đã "tối ưu theo" chính tập đó.
3. **Hoạt động trong dự án.**
   - 20.000 ảnh train CIFAR-4 được chia **18.000 train / 2.000 val** bằng hoán vị ngẫu nhiên cố định (seed 2027).
   - 4.000 ảnh test CIFAR-4 chỉ dùng một lần ở cuối mỗi lượt huấn luyện.
   - Ảnh CINIC **chỉ** thêm vào train.
4. **Ví dụ (số thật).** E3d lượt 60 epoch: val tốt nhất 90,80%, test 89,65%. Val cao hơn test một chút là bình thường, vì checkpoint
   được *chọn* theo val.
5. **Trong code.**
   - `split_train_val` (`data.py` dòng 27–33).
   - Test cuối lượt huấn luyện: `train.py` dòng 156–157.
   - Lọc ảnh CINIC trùng tập test: `prepare_cinic4.py` dòng 53–57.

## 20. Overfitting và underfitting

1. **Bản chất.**
   - **Overfitting** (học thuộc): train tốt nhưng val/test kém.
   - **Underfitting** (học chưa đủ): cả train và val đều kém.
2. **Tại sao cần quan tâm.** Mục tiêu là đúng trên ảnh **mới** (ảnh ngày thi), không phải trên ảnh đã thấy.
3. **Hoạt động.** Nhận biết qua đường học (`figures/train_<run>.png`): train acc tiếp tục tăng trong khi val acc giảm là overfitting.
   Chống overfitting bằng tăng cường dữ liệu, weight decay, label smoothing, thêm dữ liệu (CINIC), và chọn checkpoint theo val.
4. **Ví dụ (số thật).**
   - Đường học của B: val acc cuối (~0,81) **cao hơn** train acc (~0,80), vì ảnh train bị tăng cường nên khó hơn. Không có dấu hiệu
     overfitting.
   - B có thể đang **underfit** do sức chứa nhỏ: cùng kiến trúc, huấn luyện kỹ hơn trên máy cloud cũng chỉ ~84% (`model/results.md`,
     `B2_16_32_64_opt`).
5. **Trong code.** Vẽ đường học: `save_curves` (`train.py` dòng 192–208). Chọn checkpoint tốt nhất theo val: dòng 148–151.

## 21. Dropout (dự án KHÔNG dùng)

1. **Bản chất.** Dropout tắt ngẫu nhiên một tỉ lệ nơ-ron khi huấn luyện để chống overfitting.
2. **Trong dự án.** Không mô hình nào có `nn.Dropout` (đã kiểm tra `model/net.py`). Việc chống overfitting dựa vào BN, tăng cường dữ liệu,
   weight decay, label smoothing và dữ liệu thêm.
3. **Vì sao hợp lý.** Với mạng nhỏ (≤ 150 nghìn tham số) có BN và tăng cường mạnh, dropout ít cần thiết. Hơn nữa, khi suy luận dropout chỉ
   là phép nhân tỉ lệ, nên không ảnh hưởng gì tới phần cứng.

## 22. Batch Normalization

1. **Bản chất.** BN chuẩn hoá từng kênh về trung bình 0 và phương sai 1, rồi co giãn (γ) và dịch (β) bằng 2 tham số học được.
2. **Tại sao cần.**
   - Ổn định và tăng tốc huấn luyện.
   - Cho phép learning rate lớn.
   - Thay cho việc chuẩn hoá đầu vào: dự án chỉ chia 256, BN ở conv1 tự học phần còn lại.
   - Hiệu quả rõ trong dự án: D (B **không** BN) chỉ đạt 73,88% test, còn B (có BN) đạt 80,23% (`model/results.md`, nhóm chạy tay).
3. **Hoạt động.**
   - Khi **huấn luyện**, μ và σ² lấy từ batch hiện tại và được cập nhật dần vào `running_mean`, `running_var` (momentum 0,1).
   - Khi **suy luận** (`model.eval()`), dùng `running_mean`, `running_var` cố định:

     ```text
     y = γ·(z − μ)/√(σ² + ε) + β        (ε = 1e-5)
     ```

   - Lúc suy luận BN là phép tuyến tính, nên **gộp được vào conv** (BN folding):

     ```text
     k = γ/√(σ² + ε) ;  W' = k·W ;  b' = k·(b − μ) + β
     ```

4. **Ví dụ (minh hoạ).**
   - Một kênh có z = (2, 4, 6, 8) → μ = 5, σ² = 5. Với γ = 1,5, β = 0,2: y = (−1,8125; −0,4708; 0,8708; 2,2125).
   - Gộp BN: k = 1,5/√5 = 0,6708. Trọng số W = 0,4 → W' = 0,2683; bias conv b = 0 → b' = 0,6708·(0 − 5) + 0,2 = −3,1541.
5. **Trong code.**
   - `nn.BatchNorm2d` ở `net.py` dòng 36/84.
   - `model.train()` / `model.eval()` đổi chế độ (`train.py` dòng 117, 38).
   - Gộp BN: `fold_bn` (`quantize.py` dòng 34–65).
   - Kiểm tra sau gộp: sai khác logit lớn nhất chỉ cỡ 10⁻⁶ (ví dụ E2: 3,2·10⁻⁶ – `model/results.md`).

## 23. Tăng cường dữ liệu (data augmentation)

1. **Bản chất.** Biến đổi ngẫu nhiên ảnh train mỗi lần lấy ra, nhưng vẫn giữ nhãn.
2. **Tại sao cần.** Tạo thêm "ảnh mới" miễn phí, dạy mạng rằng vị trí lệch, hướng trái/phải, độ sáng hay bị che một phần không làm đổi lớp.
3. **Hoạt động.**
   - Công thức gốc: đệm 4 pixel 0 rồi cắt ngẫu nhiên 32×32 (lệch tối đa ±4 px, 9×9 = 81 vị trí), sau đó lật ngang với xác suất 50%.
   - Công thức opt thêm: đổi độ sáng/độ tương phản ±20% và Cutout (tô đen ô vuông tối đa 12×12).
4. **Ví dụ (số thật về công thức).**
   - Cutout 12×12 = 144 pixel = 14% ảnh.
   - Đổi sáng/tương phản: `f' = (f − mean)·contrast + mean·bright`, với contrast và bright ngẫu nhiên trong [0,8; 1,2], rồi làm tròn
     và kẹp về 0..255.
5. **Trong code.**
   - `Cifar4.__getitem__` (`data.py` dòng 57–68) và `strong_augment` (dòng 71–88).
   - Val/test **không** tăng cường (dòng 106–107).
   - Chi tiết: `02_DATA_PREPROCESSING.md`.

## 24. Chưng cất tri thức (knowledge distillation)

1. **Bản chất.** Mạng "trò" (student) học thêm từ phân bố xác suất của một mạng "thầy" (teacher) đã huấn luyện.
2. **Tại sao cần.** Xác suất của thầy chứa thông tin mà nhãn 0/1 không có, ví dụ "ảnh mèo này hơi giống chó". Trò nhỏ hoặc nhanh hơn có
   thể học được một phần "kinh nghiệm" đó.
3. **Hoạt động** (Hinton 2015):

   ```text
   L = (1 − α)·L_CE(label smoothing) + α·T²·KL( softmax(z_thầy/T) ‖ softmax(z_trò/T) )
   ```

   - Nhiệt độ T làm "mềm" phân bố.
   - T² bù cho độ lớn gradient bị nhỏ đi khi chia cho T.
   - Dự án dùng α = 0,7, T = 4.
4. **Ví dụ (số thật – logit của ảnh test số 0, thầy E, trò E2; đã đối chiếu với `F.kl_div`).**
   - softmax(z_thầy/4) = (0,572; 0,234; 0,093; 0,101); softmax(z_trò/4) = (0,505; 0,234; 0,129; 0,133).
   - KL = 0,01379 → T²·KL = 0,2206.
   - Phần nhãn cứng (label smoothing) = 0,4010.
   - Tổng `L = 0,3·0,4010 + 0,7·0,2206 = 0,2748`.
5. **Trong code.**
   - `distill_loss` (`train.py` dòng 49–57).
   - Nạp thầy: dòng 92–97. Thầy ở chế độ `eval` và không tính gradient.
   - Dùng cho E2 và E3d ở lượt `_kdft`, với thầy là `E_opt_cnn+cinic_ft` (`kiem_chung.py` dòng 44, 83–84, 92–93).
   - Softmax chỉ dùng khi huấn luyện.

## 25. Tinh chỉnh (fine-tuning) và dữ liệu bổ sung

1. **Bản chất.** Bắt đầu từ trọng số đã huấn luyện, học tiếp một thời gian ngắn với learning rate nhỏ, trên dữ liệu đúng "miền" cần dùng.
2. **Tại sao cần.**
   - Lượt 1 học trên CIFAR ×2 + 84.000 ảnh ImageNet (CINIC-10). Nhiều dữ liệu thì học đặc trưng tốt hơn, nhưng ảnh ImageNet khác
     "phong cách" ảnh CIFAR.
   - Lượt 2 chỉ học trên CIFAR để mạng "quen lại" đúng loại ảnh của tập test.
3. **Hoạt động.** `--init <lượt trước>` nạp trọng số, sau đó chạy 15 epoch với lr 0,005, không warmup, chỉ CIFAR.
4. **Ví dụ (số thật).** B3c: lượt +cinic đạt 86,65% test → sau tinh chỉnh 87,67%. E: 89,20% → 90,45% (`model/results.md`).
5. **Trong code.**
   - `--init` ở `train.py` dòng 82–84; `--extra-data`, `--cifar-repeat` ở `data.py` dòng 99–102.
   - Lệnh tinh chỉnh chung `FT` ở `kiem_chung.py` dòng 42.
   - Chuẩn bị CINIC: `prepare_cinic4.py`.

## 26. Hạt giống ngẫu nhiên (seed) và checkpoint

1. **Bản chất.**
   - **Seed** cố định bộ sinh số ngẫu nhiên, để chạy lại cho cùng kết quả.
   - **Checkpoint** là file lưu trọng số.
2. **Tại sao cần.** Để so sánh công bằng các kiến trúc và để tái lập số liệu báo cáo.
3. **Hoạt động.**
   - `SEED = 2027` cố định PyTorch, NumPy và thứ tự xáo trộn của DataLoader.
   - `torch.use_deterministic_algorithms(True, warn_only=True)` yêu cầu thuật toán tất định, nhưng chỉ *cảnh báo* nếu không có.
   - Dù vậy, hai máy khác CPU hay khác phiên bản thư viện vẫn có thể lệch vài phần mười %, như lệch giữa số trên máy cloud và laptop
     trong `docs/so_sanh_5_mo_hinh.md`.
4. **Ví dụ (số thật).** E2: máy cloud ghi 90,30% số nguyên, laptop ra 89,60%. Cùng kiến trúc, cùng seed, nhưng khác máy; theo `docs/KIEM_CHUNG.md` (mục "Đọc kết quả"), lệnh tinh chỉnh trên laptop được dựng lại từ ghi chú nên có thể không giống hệt lần chạy trên cloud.
5. **Trong code.**
   - `set_seed` (`train.py` dòng 28–32).
   - `split_train_val` dùng seed riêng (`data.py` dòng 29); generator của DataLoader ở dòng 103.
   - Checkpoint `model/checkpoints/<run>.pt` (dòng 111, 150) **không** commit (`.gitignore`).

## 27. MAC – thước đo khối lượng tính

1. **Bản chất.** MAC (multiply–accumulate) là một phép nhân rồi cộng dồn, đúng việc một DSP48E1 làm trong một chu kỳ.
2. **Tại sao cần.** Trên FPGA, tốc độ bị giới hạn bởi số DSP (XC7S25 có 80 DSP – `docs/hardware.md`), nên số MAC mỗi ảnh quyết định thời
   gian suy luận.
3. **Hoạt động.** `MAC_conv = H_out·W_out·C_out·C_in·K²`, `MAC_linear = N_in·N_out`. Thời gian ≈ MAC ÷ (số DSP × tần số).
4. **Ví dụ (số thật).** conv1 của B: 32·32·16·3·9 = 442.368 MAC. Cả mạng B: 2.805.760 MAC → với 16 DSP @ 100 MHz cần
   2.805.760 / 1,6·10⁹ ≈ 1,8 ms.
5. **Trong code.** `net.count_macs` (`net.py` dòng 125–148); thời gian ước lượng ở `kiem_chung.py` dòng 352–353.

## 28. Trường tiếp nhận (receptive field)

1. **Bản chất.** Receptive field là vùng ảnh gốc ảnh hưởng tới một điểm của feature map.
2. **Tại sao cần.** Để nhận ra "con chó", điểm cuối phải "nhìn" được vùng đủ lớn của ảnh. Đây cũng là một cách so sánh kiến trúc.
3. **Hoạt động.** `RF ← RF + (K − 1)·j`, `j ← j·S` qua từng tầng (bắt đầu RF = 1, j = 1).
4. **Ví dụ (số thật, tính từ cấu hình).** B: 22 px; E: 26 px; E2: 30 px; B3c và E3d: 38 px (lớn hơn ảnh 32 px).
5. **Trong code.** Dự án **không** tự tính RF. Đây là phép tính lý thuyết trên cấu hình `net.py` (mục 3.5 của mỗi file kiến trúc).

## 29. Số dấu phẩy tĩnh (fixed-point) và lượng tử hoá

1. **Bản chất.** Biểu diễn số thực `v` bằng số nguyên `q` với `v ≈ q·2^(−f)`, trong đó f là số bit phần lẻ.
2. **Tại sao cần.** FPGA nhân số nguyên 8–16 bit bằng DSP rất rẻ; số thực dấu phẩy động thì đắt. Đề yêu cầu mô hình chạy trên chip
   XC7S25.
3. **Hoạt động.**
   - Trọng số dùng int8; conv1 của 4 mô hình +cinic dùng int16.
   - Activation sau ReLU dùng uint8; bias và tổng tích luỹ dùng int32.
   - Mọi thang là **lũy thừa của 2**, nên đổi thang chỉ là dịch bit: `y = (acc + 2^(s−1)) >> s`, sau đó kẹp về 0..255.
4. **Ví dụ (minh hoạ).** `acc = 1000, s = 4`: (1000 + 8) >> 4 = 63, tức 1000/16 = 62,5 làm tròn thành 63.
5. **Trong code.** `quantize.py` (toàn file) và `golden_int.py`. Chi tiết đầy đủ: [`10_QUANTIZATION_AND_HARDWARE.md`](10_QUANTIZATION_AND_HARDWARE.md).

## 30. Có trong code nhưng 5 mô hình không dùng: EMA, CutMix

- **EMA** (exponential moving average của trọng số – `train.py` dòng 85–91, cờ `--ema`) và **CutMix** (dán một mảng ảnh khác vào,
  trộn nhãn theo diện tích – dòng 60–74, cờ `--cutmix`) được thêm cho thí nghiệm của nhánh `exp/opt-acc-90`.
- `kiem_chung.py` **không** truyền hai cờ này cho 5 mô hình (dòng 57–96), nên với 5 mô hình chúng luôn tắt (`ema_decay = 0`,
  `cutmix_prob = 0`).

---

← [02_DATA_PREPROCESSING.md](02_DATA_PREPROCESSING.md) · [Mục lục](README.md) · [04_ARCHITECTURE_1_B.md](04_ARCHITECTURE_1_B.md) →
