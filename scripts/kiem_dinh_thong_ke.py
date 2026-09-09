import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.utils import normalize_answer, normalize_yes_no, save_json, setup_logger

logger = setup_logger()


def dung_sai(records):
    ra = []
    for r in records:
        if r.get("answer_type") == "CLOSED":
            ra.append(int(normalize_yes_no(r["pred"]) == normalize_yes_no(r["gt"])))
        else:
            ra.append(int(normalize_answer(r["pred"]) == normalize_answer(r["gt"])))
    return np.array(ra, dtype=np.int8)


def khoang_tin_cay(dung, so_lan=10000, muc=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dung)
    mau = rng.integers(0, n, size=(so_lan, n))
    tb = dung[mau].mean(axis=1)
    duoi = np.percentile(tb, (1 - muc) / 2 * 100)
    tren = np.percentile(tb, (1 + muc) / 2 * 100)
    return float(dung.mean()), float(duoi), float(tren)


def mcnemar(a, b):
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    from math import comb
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return b01, b10, min(1.0, p)


def bootstrap_hieu(a, b, so_lan=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(a)
    mau = rng.integers(0, n, size=(so_lan, n))
    hieu = b[mau].mean(axis=1) - a[mau].mean(axis=1)
    return float(hieu.mean()), float(np.percentile(hieu, 2.5)), float(np.percentile(hieu, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Kiểm định thống kê cho các cấu hình ablation")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--goc", default="abl3_du3", help="cấu hình làm mốc so sánh")
    parser.add_argument("--tags", nargs="*",
                        default=["abl3_du3", "abl3_khong_visual", "abl3_khong_lens",
                                 "abl3_khong_latent", "abl3_khong_prompt"])
    parser.add_argument("--dataset", default="fracatlas")
    parser.add_argument("--out", default="reports/kiem_dinh_thong_ke.json")
    args = parser.parse_args()

    bang = {}
    for t in args.tags:
        p = os.path.join(args.report_dir, f"results_{args.dataset}_{t}.json")
        if not os.path.exists(p):
            logger.warning("Thiếu %s", p)
            continue
        bang[t] = dung_sai(json.load(open(p, encoding="utf-8"))["records"])

    if args.goc not in bang:
        raise SystemExit(f"Không có cấu hình mốc {args.goc}")

    n = len(bang[args.goc])
    logger.info("So sánh %d cấu hình trên cùng %d mẫu test", len(bang), n)
    print()
    print(f"{'Cấu hình':22s}{'Acc':>8}{'KTC 95%':>20}")
    ket_qua = {"n_mau": n, "cau_hinh": {}, "so_sanh": []}
    for t, d in bang.items():
        acc, duoi, tren = khoang_tin_cay(d)
        ket_qua["cau_hinh"][t] = {"acc": round(acc, 4), "ktc_duoi": round(duoi, 4), "ktc_tren": round(tren, 4)}
        print(f"{t:22s}{acc * 100:7.2f}%   [{duoi * 100:5.2f}% – {tren * 100:5.2f}%]")

    print()
    print(f"{'So với ' + args.goc:26s}{'Chênh':>9}{'KTC 95% của chênh':>24}{'b01':>6}{'b10':>6}{'p':>10}")
    goc = bang[args.goc]
    for t, d in bang.items():
        if t == args.goc:
            continue
        hieu, hduoi, htren = bootstrap_hieu(goc, d)
        b01, b10, p = mcnemar(goc, d)
        y_nghia = "CÓ" if p < 0.05 else "không"
        ket_qua["so_sanh"].append({"cau_hinh": t, "chenh": round(hieu, 4), "ktc_duoi": round(hduoi, 4),
                                   "ktc_tren": round(htren, 4), "b01": b01, "b10": b10,
                                   "p_mcnemar": round(p, 5), "y_nghia_thong_ke": p < 0.05})
        print(f"{t:26s}{hieu * 100:+8.2f}%   [{hduoi * 100:+6.2f}% – {htren * 100:+6.2f}%]{b01:6d}{b10:6d}{p:10.4f}  {y_nghia}")

    print()
    co = [c["cau_hinh"] for c in ket_qua["so_sanh"] if c["y_nghia_thong_ke"]]
    if co:
        print("Khác biệt có ý nghĩa thống kê (p < 0.05):", ", ".join(co))
    else:
        print("KHÔNG cấu hình nào khác biệt có ý nghĩa thống kê so với", args.goc)
    save_json(ket_qua, args.out)
    logger.info("Đã lưu %s", args.out)


if __name__ == "__main__":
    main()
