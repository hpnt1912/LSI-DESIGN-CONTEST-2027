"""Chạy thêm seed cho MỘT mô hình chính để biết accuracy dao động bao nhiêu giữa các lần huấn luyện.

Vì sao cần: mỗi mô hình trong docs/so_sanh_5_mo_hinh.md mới huấn luyện 1 lần (seed 2027). E, E3d, E2 chênh nhau
dưới 1 điểm %, nhỏ hơn sai số đo, nên chưa xếp hạng được. Huấn luyện lại với vài seed khác rồi báo cáo
"trung bình ± độ lệch chuẩn" thì kết luận mới đứng vững.

Cách dùng (từ thư mục gốc repo, venv đang bật):
    python model/nhieu_seed.py E3d                 # thêm seed 2 và 3 (mặc định) -> cùng seed 2027 là 3 lần
    python model/nhieu_seed.py E3d --seeds 2 3 4   # chọn seed khác
    python model/nhieu_seed.py B --thu             # chạy thử 1 epoch (cho người sửa code; nên chạy trong bản sao repo)

Mỗi seed chạy đúng các lệnh huấn luyện của mô hình đó trong kiem_chung.py (cùng công thức, cùng mạng thầy E gốc),
chỉ thêm --seed và đổi tên lượt chạy thêm đuôi _s<seed> (B: +s<seed>) để KHÔNG đè kết quả chính; sau đó lượng
tử hoá và chạy golden model. Bị dừng giữa chừng: chạy lại đúng lệnh cũ, script chạy tiếp từ bước dở.
Kết quả ghi vào mục '## Nhiều seed – <mã>' của model/results.md. Script KHÔNG commit / push.
"""

import argparse
import sys

import numpy as np

import config
import kiem_chung as kc


def seeded(run, s):
    """Tên lượt chạy của seed s: thêm '_s<s>' (hoặc '+s<s>' nếu tên chưa có dấu '+', để net.build vẫn nhận ra)."""
    return f"{run}_s{s}" if "+" in run else f"{run}+s{s}"


def stages_for(key, s):
    """Các lệnh train + quantize của mô hình `key` (lấy từ kiem_chung.MODELS), đổi tên lượt chạy và thêm --seed."""
    m = kc.MODELS[key]
    names = set(m["train_runs"]) | {m["run"]}           # mạng thầy (--teacher) không nằm trong đây nên giữ nguyên
    out = []
    for cmd in m["stages"]:
        if cmd[0] not in ("train.py", "quantize.py"):    # evaluate / golden / export không cần cho việc so seed
            continue
        new = [seeded(t, s) if t in names else t for t in cmd]
        out.append(new + (["--seed", str(s)] if cmd[0] == "train.py" else []))
    return out


def stats(v):
    v = np.array(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def write_section(key, seeds, log):
    """Gom kết quả seed 2027 (lượt chính) + các seed đã xong -> mục '## Nhiều seed – <key>' trong results.md."""
    m = kc.MODELS[key]
    rows_md, fl, it = [], [], []
    rows = kc.results_rows()
    for s in [config.SEED, *seeds]:
        run = m["run"] if s == config.SEED else seeded(m["run"], s)
        train = m["train_runs"] if s == config.SEED else [seeded(t, s) for t in m["train_runs"]]
        g, r = kc.golden_check(run), rows.get(run)
        if g is None or r is None:
            continue
        minutes = sum(rows[t][2] for t in train if t in rows)
        fl.append(r[1])
        it.append(g["int"])
        per = " / ".join(kc.fmt(p, 1) for p in g["per_class"])
        rows_md.append(f"| {s}{' (lượt chính)' if s == config.SEED else ''} | `{run}` | {kc.fmt(r[0])}% | "
                       f"{kc.fmt(r[1])}% | **{kc.fmt(g['int'])}%** | {per} | {kc.fmt(minutes, 1)} phút |")
    if not it:
        return None
    (mf, sf), (mi, si) = stats(fl), stats(it)
    body = (f"> Ghi bởi `python model/nhieu_seed.py {key}` trên máy `{kc.machine()}` ({kc.now()}). Mỗi seed chạy "
            f"đúng công thức của lượt chính (`{m['run']}`), chỉ khác `--seed`"
            + (f"; mạng thầy vẫn là `{kc.TEACHER}` gốc" if m["needs_teacher"] else "") + ".\n\n"
            "| Seed | Lượt chạy cuối | Val float | Test float | Test số nguyên | cat / dog / deer / horse (số nguyên) | "
            "Thời gian huấn luyện |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows_md) + "\n\n"
            f"**{len(it)} lần chạy – test số nguyên: {kc.fmt(mi)}% ± {kc.fmt(si)}** (trung bình ± độ lệch chuẩn), "
            f"thấp nhất {kc.fmt(min(it))}%, cao nhất {kc.fmt(max(it))}%. Test float: {kc.fmt(mf)}% ± {kc.fmt(sf)}.\n\n"
            "Cách dùng khi so sánh: dùng **trung bình** thay cho số của một lần chạy; hai mô hình chênh nhau ít hơn "
            "khoảng độ lệch chuẩn thì coi như tương đương.")
    import report
    report.update_section(f"Nhiều seed – {key}", body)
    return body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ma", choices=kc.ORDER, help="mã mô hình: " + ", ".join(kc.ORDER))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3], help="các seed thêm (mặc định 2 3)")
    ap.add_argument("--thu", action="store_true", help="chạy thử: mọi lượt huấn luyện chỉ 1 epoch, không ghi nhật ký")
    args = ap.parse_args()
    key, m = args.ma, kc.MODELS[args.ma]
    if config.SEED in args.seeds:
        sys.exit(f"✗ Seed {config.SEED} là lượt chính đã có – chọn seed khác.")
    if not config.DATA_FILE.exists():
        sys.exit("✗ Chưa có data/cifar4.npz – chạy: python model/prepare_cifar4.py")
    if m["needs_teacher"] and not (config.CKPT_DIR / f"{kc.TEACHER}.pt").exists():
        sys.exit(f"✗ {key} học từ mạng thầy {kc.TEACHER} nhưng máy chưa có checkpoint của nó.\n"
                 f"  Chạy trước: python model/kiem_chung.py E")
    if m["cinic"]:
        kc.ensure_cinic(args.thu)

    rows = kc.results_rows()
    est = sum(rows[t][2] for t in m["train_runs"] if t in rows)
    print(f"Mô hình {key} ({m['run']}): thêm seed {', '.join(map(str, args.seeds))}"
          + (f" – khoảng {kc.fmt(est, 0)} phút mỗi seed (theo lượt chính)" if est else ""))
    log = kc.load_log()
    ns = log.setdefault("nhieu_seed", {"runs": [], "xong": {}})
    logf = config.CKPT_DIR / f"nhieu_seed_{key}.txt"
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        tag = f"{key}_s{s}"
        done = ns["xong"].get(tag, [])
        print(f"\n{'=' * 78}\n▶ {key}, seed {s}" + (f" (bỏ qua {len(done)} bước đã xong)" if done else "")
              + f"\n{'=' * 78}")
        for i, cmd in enumerate(stages_for(key, s)):
            if i in done and not args.thu:
                continue
            kc.run_step(cmd, logf, args.thu)
            if not args.thu:
                done.append(i)
                ns["xong"][tag] = done
                run = cmd[cmd.index("--arch") + 1]
                if run not in ns["runs"]:
                    ns["runs"].append(run)                 # để kiem_chung.py gắn nhãn "chạy trên máy nhóm"
                kc.save_log(log)
    pre = f"{key}_s"                                      # mọi seed đã xong của mô hình này (cả lần chạy trước)
    seeds = sorted(set(args.seeds) | {int(t[len(pre):]) for t in ns["xong"] if t.startswith(pre)})
    body = write_section(key, seeds, log)
    if body:
        print("\n" + body.split("\n\n", 1)[1])
    print(f"\n✓ Xong. Kết quả: mục '## Nhiều seed – {key}' trong model/results.md. Nhật ký: {logf}")
    if not args.thu:
        exports = " ".join(f'"model/export_{seeded(m["run"], s)}"' for s in args.seeds)
        print("Đẩy kết quả lên GitHub:")
        print(f"  git add model/results.md model/figures model/kiem_chung_log.json {exports}")
        print(f'  git commit -m "Thêm seed {" ".join(map(str, args.seeds))} cho {key}"')
        print("  git push")


if __name__ == "__main__":
    main()
