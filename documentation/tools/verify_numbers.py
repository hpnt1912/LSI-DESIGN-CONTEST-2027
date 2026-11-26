"""Kiểm tra lại các con số cốt lõi trong tài liệu bằng chính code của dự án (không sửa file nào của dự án).

Chạy từ thư mục gốc repo, venv đang bật:
    python documentation/tools/verify_numbers.py            # tham số, MAC, kích thước tensor (vài giây)
    python documentation/tools/verify_numbers.py --golden   # + accuracy số nguyên của golden model (vài phút,
                                                            #   cần data/cifar4.npz và model/export*/qparams.npz)

Mọi số "mong đợi" bên dưới là số đã ghi trong tài liệu. Script in OK / SAI cho từng mục.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "model"))          # để import config, net, ... như khi chạy python model/x.py

import torch  # noqa: E402

import config  # noqa: E402
import net  # noqa: E402

# Số liệu đã ghi trong tài liệu (file 04–08, 12)
EXPECTED = {
    "B": dict(run="B_16_32_64", params=27684, torch_params=27796, macs=2805760, flat=1024, golden=80.00),
    "B3c": dict(run="B3c_c4x4_128+cinic_ft", params=105636, torch_params=105876, macs=3989504, flat=2048,
                golden=87.60),
    "E": dict(run="E_opt_cnn+cinic_ft", params=138372, torch_params=138660, macs=19767296, flat=2048,
              golden=90.65),
    "E2": dict(run="E2_lite+cinic60_kdft", params=138372, torch_params=138660, macs=12689408, flat=2048,
               golden=89.60),
    "E3d": dict(run="E3d_deep5+cinic60_kdft", params=147524, torch_params=147836, macs=11582464, flat=1792,
                golden=90.25),
}


def check(label, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 0.005
    print(f"  {'OK ' if ok else 'SAI'} {label:32s} code: {got}   tài liệu: {want}")
    return ok


def shapes(model):
    """In kích thước tensor sau mỗi module của features (giống bảng 3.1 trong file kiến trúc)."""
    x = torch.zeros(1, 3, 32, 32)
    print(f"    vào          {tuple(x.shape[1:])}")
    with torch.no_grad():
        for mod in model.features:
            x = mod(x)
            print(f"    {type(mod).__name__:12s} {tuple(x.shape[1:])}")
    return int(x.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", action="store_true", help="chạy thêm golden model số nguyên trên 4.000 ảnh test")
    args = ap.parse_args()
    all_ok = True
    for key, e in EXPECTED.items():
        print(f"\n== {key} ({e['run']}) ==")
        m = net.build(e["run"]).eval()
        flat = shapes(m)
        torch_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        all_ok &= check("tham số (sau gộp BN, count_params)", net.count_params(m), e["params"])
        all_ok &= check("tham số PyTorch học", torch_params, e["torch_params"])
        all_ok &= check("MAC / ảnh (count_macs)", net.count_macs(m), e["macs"])
        all_ok &= check("đầu vào Linear (flatten)", flat, e["flat"])
        if args.golden:
            import data
            import golden_int
            f = config.export_dir_for(e["run"]) / "qparams.npz"
            if not f.exists():
                print(f"  -- bỏ qua golden: chưa có {f}")
                continue
            _, _, x_te, y_te = data.load_arrays()
            acc, _ = golden_int.accuracy(golden_int.load_qparams(f), x_te, y_te)
            all_ok &= check("accuracy số nguyên (golden, %)", round(float(acc) * 100, 2), e["golden"])
    print("\nTẤT CẢ KHỚP" if all_ok else "\nCÓ MỤC KHÔNG KHỚP – xem dòng SAI ở trên")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
