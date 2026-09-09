import argparse
import gc
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.train import build_split, load_data_api
from src.utils import (compute_metrics, get_device, is_polar_question, normalize_answer, plot_confusion_yes_no,
                       save_json, set_seed, setup_env_dirs, setup_logger)

setup_env_dirs()
logger = setup_logger()


@torch.no_grad()
def du_doan_mot_mo_hinh(checkpoint, loader, device, w_llm, max_samples, amp_dtype=torch.bfloat16):
    override = {"closed_llm_weight": w_llm} if w_llm is not None else None
    model = BoneVQAModel.load(checkpoint, device=str(device), cfg_override=override)
    ket_qua = []
    seen = 0
    for batch in loader:
        nb = normalize_batch(batch)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            preds = model.predict_batch(batch, tra_chi_tiet=True)
        for p, a, t, q, iid in zip(preds, nb["answers"], nb["answer_types"], nb["questions"],
                                   nb.get("image_ids") or [None] * len(preds)):
            ket_qua.append({"image_id": iid, "question": q, "gt": a, "answer_type": t,
                            "pred": p["answer"], "pred_type": p["answer_type"],
                            "p_yes": p.get("p_yes"), "vocab_top1": p.get("vocab_top1"),
                            "vocab_p": p.get("vocab_p"), "confidence": p["confidence"]})
        seen += len(preds)
        if seen >= max_samples:
            break
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ket_qua


def hop_nhat(du_doan_cac_mo_hinh):
    n = min(len(d) for d in du_doan_cac_mo_hinh)
    gop = []
    for i in range(n):
        cac = [d[i] for d in du_doan_cac_mo_hinh]
        goc = cac[0]
        la_dong = sum(1 for c in cac if c["pred_type"] == "CLOSED") > len(cac) / 2
        if la_dong and is_polar_question(goc["question"]):
            p_yes = [c["p_yes"] for c in cac if c["p_yes"] is not None]
            tb = sum(p_yes) / len(p_yes) if p_yes else 0.5
            pred = "yes" if tb >= 0.5 else "no"
            conf = tb if pred == "yes" else 1.0 - tb
            loai = "CLOSED"
        elif la_dong:
            phieu = Counter(c["vocab_top1"] for c in cac if c.get("vocab_top1"))
            pred = phieu.most_common(1)[0][0] if phieu else goc["pred"]
            conf = sum(c.get("vocab_p") or 0.0 for c in cac) / len(cac)
            loai = "CLOSED"
        else:
            phieu = Counter(normalize_answer(c["pred"]) for c in cac if c["pred"])
            if phieu and phieu.most_common(1)[0][1] > 1:
                chuan = phieu.most_common(1)[0][0]
                pred = next(c["pred"] for c in cac if normalize_answer(c["pred"]) == chuan)
            else:
                pred = goc["pred"]
            conf = sum(c["confidence"] for c in cac) / len(cac)
            loai = "OPEN"
        gop.append({"image_id": goc["image_id"], "question": goc["question"], "gt": goc["gt"],
                    "answer_type": goc["answer_type"], "pred": pred, "pred_type": loai,
                    "confidence": round(float(conf), 4)})
    return gop


def main():
    parser = argparse.ArgumentParser(description="Hợp nhất dự đoán của nhiều checkpoint")
    parser.add_argument("--checkpoints", required=True, help="danh sách checkpoint, ngăn cách bằng dấu phẩy")
    parser.add_argument("--dataset", choices=["roco", "pmc_vqa", "vqa_rad", "fracatlas"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--closed_llm_weight", type=float, default=0.3)
    parser.add_argument("--tag", default="hop_nhat")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--figure_dir", default="reports/figures")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    build_dataset, collate_fn, real = load_data_api()
    ds = build_split(build_dataset, args.dataset, args.split, args.max_samples, real)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    gioi_han = args.max_samples or len(ds)
    danh_sach = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    logger.info("Hợp nhất %d checkpoint trên %s (%s), %d mẫu", len(danh_sach), args.dataset, args.split, len(ds))

    tat_ca = []
    for ck in danh_sach:
        logger.info("Đang chạy %s", ck)
        du_doan = du_doan_mot_mo_hinh(ck, loader, device, args.closed_llm_weight, gioi_han)
        m = compute_metrics(du_doan)
        logger.info("  riêng lẻ: closed %.4f open %.4f overall %.4f", m.get("closed_acc", 0.0),
                    m.get("open_acc", 0.0), m.get("overall_acc", 0.0))
        tat_ca.append(du_doan)

    gop = hop_nhat(tat_ca)
    m = compute_metrics(gop)
    logger.info("HỢP NHẤT: closed %.4f open %.4f recall %.4f bleu1 %.4f overall %.4f",
                m.get("closed_acc", 0.0), m.get("open_acc", 0.0), m.get("open_recall", 0.0),
                m.get("bleu1", 0.0), m.get("overall_acc", 0.0))
    out = os.path.join(args.report_dir, f"results_{args.dataset}_{args.tag}.json")
    save_json({"run_name": args.tag, "dataset": args.dataset, "split": args.split,
               "checkpoints": danh_sach, "closed_llm_weight": args.closed_llm_weight,
               "metrics": m, "records": gop}, out)
    logger.info("Đã lưu %s", out)
    plot_confusion_yes_no(gop, os.path.join(args.figure_dir, f"confusion_{args.dataset}_{args.tag}.png"))


if __name__ == "__main__":
    main()
