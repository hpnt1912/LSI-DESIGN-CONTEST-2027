"""Ngày 3 – Định nghĩa các kiến trúc CNN.

Mỗi khối (block) = Conv 3x3 (padding 1) -> BatchNorm -> ReLU -> MaxPool 2x2.
Sau 3 khối, ảnh 32x32 thu nhỏ còn 4x4; duỗi phẳng (flatten) rồi qua 1 tầng Linear ra 4 logit.

    ảnh 3x32x32
      -> khối 1 -> C1 x 16 x 16
      -> khối 2 -> C2 x  8 x  8
      -> khối 3 -> C3 x  4 x  4
      -> flatten (C3*16) -> Linear -> 4 logit (cat, dog, deer, horse)

Chạy thử:  python model/net.py   -> in bảng số tham số, số MAC, dung lượng int8 của từng kiến trúc.
"""

import torch
from torch import nn

# Tên kiến trúc -> (số kênh của 3 khối, có BatchNorm hay không)
ARCHS = {
    "A_8_16_32": ((8, 16, 32), True),
    "B_16_32_64": ((16, 32, 64), True),
    "C_32_64_128": ((32, 64, 128), True),
    "D_16_32_64_noBN": ((16, 32, 64), False),
}


class SmallCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64), use_bn=True, n_classes=4):
        super().__init__()
        layers = []
        c_in = 3
        for c_out in channels:
            # bias=False khi có BN, vì BN đã có tham số dịch (beta) làm thay việc của bias
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(c_in * 4 * 4, n_classes)
        self.channels, self.use_bn = tuple(channels), use_bn

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build(name):
    channels, use_bn = ARCHS[name]
    return SmallCNN(channels, use_bn)


def count_params(model):
    """Số tham số sẽ nằm trên phần cứng (sau khi gộp BN: trọng số conv + bias mỗi kênh + FC)."""
    total = 0
    c_in = 3
    for c in model.channels:
        total += c_in * c * 9 + c       # trọng số 3x3 + 1 bias cho mỗi kênh ra
        c_in = c
    total += c_in * 16 * 4 + 4          # tầng Linear
    return total


def count_macs(model):
    """Số phép nhân-cộng (MAC, multiply-accumulate) cho 1 ảnh."""
    macs, c_in, size = 0, 3, 32
    for c in model.channels:
        macs += size * size * c * (c_in * 9)   # mỗi điểm ra cần c_in*9 phép nhân
        c_in, size = c, size // 2
    macs += c_in * 16 * 4
    return macs


def main():
    print(f"{'Kiến trúc':18s} {'Tham số':>9s} {'MAC/ảnh':>11s} {'KB (int8)':>10s}")
    for name in ARCHS:
        m = build(name)
        p = count_params(m)
        print(f"{name:18s} {p:9,d} {count_macs(m):11,d} {p / 1024:10.1f}")
    x = torch.zeros(1, 3, 32, 32)
    print("Kiểm tra kích thước đầu ra:", tuple(build("B_16_32_64")(x).shape))


if __name__ == "__main__":
    main()
