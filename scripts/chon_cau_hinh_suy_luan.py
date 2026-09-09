import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.train import build_split, load_data_api
from src.utils import compute_metrics, get_device, save_json, set_seed, setup_env_dirs, setup_logger

setup_env_dirs()
logger = setup_logger()

KIEU_TTA = {
    "contour": ["contour"],
    "contour_box": ["contour", "box"],
    "4kieu": ["contour", "box", "circle", "mask"],
}


@torch.no_grad()
def danh_gia(model, loader, device, max_samples, amp_dtype=torch.bfloat16):
    records = []
    seen = 0
    for batch in loader:
        nb = normalize_batch(batch)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            preds = model.predict_batch(batch)
        for p, a, t, q in zip(preds, nb["answers"], nb["answer_types"], nb["questions"]):
            records.append({"pred": p["answer"], "gt": a, "answer_type": t, "question": q})
        seen += len(preds)
        if seen >= max_samples:
            break
    return compute_metrics(records)


def main():
    parser = argparse.ArgumentParser(description="Quét cấu hình suy luận trên tập validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["roco", "pmc_vqa", "vqa_rad", "fracatlas"], required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tta", default="contour,contour_box,4kieu")
    parser.add_argument("--llm_weights", default="0.0,0.3,0.5,0.7")
    parser.add_argument("--select_metric", default="overall_acc")
    parser.add_argument("--out", default="reports/cau_hinh_suy_luan.json")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    model = BoneVQAModel.load(args.checkpoint, device=str(device))
    build_dataset, collate_fn, real = load_data_api()
    ds = build_split(build_dataset, args.dataset, args.split, args.max_samples, real)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info("Quét cấu hình trên %s (%s), %d mẫu", args.dataset, args.split, len(ds))

    ten_tta = [t.strip() for t in args.tta.split(",") if t.strip()]
    trong_so = [float(w) for w in args.llm_weights.split(",") if w.strip()]
    ket_qua = []
    for ten, w in itertools.product(ten_tta, trong_so):
        model.cfg["tta_kinds"] = KIEU_TTA[ten]
        model.cfg["closed_llm_weight"] = w
        m = danh_gia(model, loader, device, args.max_samples)
        m.update({"tta": ten, "closed_llm_weight": w})
        ket_qua.append(m)
        logger.info("tta=%-11s w_llm=%.1f | closed %.4f open %.4f overall %.4f", ten, w,
                    m.get("closed_acc", 0.0), m.get("open_acc", 0.0), m.get("overall_acc", 0.0))

    ket_qua.sort(key=lambda m: -m.get(args.select_metric, 0.0))
    tot_nhat = ket_qua[0]
    logger.info("Tốt nhất theo %s: tta=%s, closed_llm_weight=%.1f (%.4f)", args.select_metric,
                tot_nhat["tta"], tot_nhat["closed_llm_weight"], tot_nhat.get(args.select_metric, 0.0))
    save_json({"dataset": args.dataset, "split": args.split, "checkpoint": args.checkpoint,
               "select_metric": args.select_metric, "tot_nhat": tot_nhat,
               "tta_kinds": KIEU_TTA[tot_nhat["tta"]], "tat_ca": ket_qua}, args.out)
    logger.info("Đã lưu %s", args.out)


if __name__ == "__main__":
    main()
