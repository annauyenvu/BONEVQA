import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.utils import bertscore_f1_batch, load_json, rouge_l, save_json, setup_env_dirs, setup_logger, token_f1

setup_env_dirs()
logger = setup_logger()

TAP_KET_QUA_MAC_DINH = {
    "fracatlas_tu_dong": "reports/results_fracatlas_lon_tu_dong.json",
    "fracatlas_ho_tro": "reports/results_fracatlas_lon_ho_tro.json",
    "vqa_rad_nen_cu": "reports/results_vqa_rad_v2_sua_dinh_tuyen.json",
    "vqa_rad_giao_thuc_sach": "reports/results_vqa_rad_v3b_base.json",
    "vqa_rad_hop_nhat": "reports/results_vqa_rad_hop_nhat3.json",
}


def danh_gia_mot_tap(nhan: str, duong_dan: str) -> dict:
    if not os.path.exists(duong_dan):
        logger.warning("Bo qua %s: khong tim thay %s", nhan, duong_dan)
        return None
    du_lieu = load_json(duong_dan)
    ban_ghi_mo = [r for r in du_lieu["records"] if r.get("answer_type") != "CLOSED"]
    if not ban_ghi_mo:
        return None
    preds = [r["pred"] for r in ban_ghi_mo]
    gts = [r["gt"] for r in ban_ghi_mo]
    f1_token = [token_f1(p, g) for p, g in zip(preds, gts)]
    rl = [rouge_l(p, g) for p, g in zip(preds, gts)]
    bs = bertscore_f1_batch(preds, gts)
    return {
        "nhan": nhan,
        "nguon": duong_dan,
        "n_open": len(ban_ghi_mo),
        "token_f1": float(np.mean(f1_token)),
        "rouge_l": float(np.mean(rl)),
        "bertscore_f1": float(np.mean(bs)),
        "overall_acc_goc": du_lieu["metrics"].get("overall_acc"),
        "bleu1_goc": du_lieu["metrics"].get("bleu1"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tinh bo sung token F1, ROUGE-L, BERTScore cho cau hoi mo tu cac ket qua da luu san")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--out", default="reports/metric_mo_rong_cau_mo.json")
    args = parser.parse_args()

    ket_qua = []
    for nhan, ten_file in TAP_KET_QUA_MAC_DINH.items():
        duong_dan = os.path.join(os.path.dirname(args.report_dir) or ".", ten_file) \
            if not os.path.isabs(ten_file) else ten_file
        duong_dan = ten_file if os.path.exists(ten_file) else os.path.join(args.report_dir,
                                                                            os.path.basename(ten_file))
        r = danh_gia_mot_tap(nhan, duong_dan)
        if r:
            ket_qua.append(r)
            print(f"{nhan:<24} n={r['n_open']:>4}  token_f1={r['token_f1']*100:6.2f}%  "
                  f"rouge_l={r['rouge_l']*100:6.2f}%  bertscore_f1={r['bertscore_f1']*100:6.2f}%  "
                  f"(BLEU-1 goc={r['bleu1_goc']*100:6.2f}%)")

    save_json({"ket_qua": ket_qua}, args.out)
    logger.info("Da luu ket qua vao %s", args.out)


if __name__ == "__main__":
    main()
