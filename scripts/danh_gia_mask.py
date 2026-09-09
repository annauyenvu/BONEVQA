import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from src.data import build_dataset
from src.models.segment_prompt import SegmentPromptCreator
from src.utils import get_device, save_json, set_seed, setup_env_dirs, setup_logger

setup_env_dirs()
logger = setup_logger()


def iou_dice(du_doan, that):
    giao = np.logical_and(du_doan, that).sum()
    hop = np.logical_or(du_doan, that).sum()
    tong = du_doan.sum() + that.sum()
    iou = float(giao) / float(hop) if hop else 0.0
    dice = 2.0 * float(giao) / float(tong) if tong else 0.0
    return iou, dice


def gop_mask(masks, h, w):
    if masks is None or len(masks) == 0:
        return np.zeros((h, w), dtype=bool)
    return np.asarray(masks).max(axis=0) > 0


def nap_bo_phat_hien(duong_dan, device):
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    tai = torch.load(duong_dan, map_location="cpu", weights_only=False)
    mo_hinh = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    so_dac_trung = mo_hinh.roi_heads.box_predictor.cls_score.in_features
    mo_hinh.roi_heads.box_predictor = FastRCNNPredictor(so_dac_trung, 2)
    mo_hinh.load_state_dict(tai["state"])
    mo_hinh.to(device).eval()
    return mo_hinh, tai.get("canh_dai", 640)


def du_doan_box(mo_hinh, canh_dai, anh, device, nguong=0.3, toi_da=3):
    import torch

    w, h = anh.size
    ty_le = canh_dai / max(w, h)
    nw, nh = max(1, int(round(w * ty_le))), max(1, int(round(h * ty_le)))
    nho = anh.convert("RGB").resize((nw, nh), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(nho, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
    with torch.no_grad():
        ra = mo_hinh([x.to(device)])[0]
    hop = []
    for b, sc in zip(ra["boxes"].tolist(), ra["scores"].tolist()):
        if sc < nguong or len(hop) >= toi_da:
            continue
        hop.append([v / ty_le for v in b])
    return hop or None


def danh_gia(segmenter, ds, so_anh, dung_box, bo_phat_hien=None, canh_dai=640, device=None):
    ket_qua = []
    da_xet = set()
    for i in range(len(ds)):
        if len(ket_qua) >= so_anh:
            break
        mau = ds[i]
        if mau.get("masks") is None or len(mau["masks"]) == 0:
            continue
        if mau["image_id"] in da_xet:
            continue
        da_xet.add(mau["image_id"])
        anh = mau["image"]
        w, h = anh.size
        that = gop_mask(mau["masks"], h, w)
        if that.sum() == 0:
            continue
        if bo_phat_hien is not None:
            boxes = du_doan_box(bo_phat_hien, canh_dai, anh, device)
        else:
            boxes = mau.get("boxes") if dung_box else None
        du_doan = gop_mask(segmenter(anh, boxes=boxes), h, w)
        iou, dice = iou_dice(du_doan, that)
        ket_qua.append({"image_id": mau["image_id"], "iou": iou, "dice": dice,
                        "dien_tich_that": int(that.sum()), "dien_tich_du_doan": int(du_doan.sum())})
    return ket_qua


def tom_tat(ten, ket_qua):
    if not ket_qua:
        return {"cau_hinh": ten, "so_anh": 0}
    iou = np.array([r["iou"] for r in ket_qua])
    dice = np.array([r["dice"] for r in ket_qua])
    return {"cau_hinh": ten, "so_anh": len(ket_qua),
            "iou_tb": round(float(iou.mean()), 4), "iou_trung_vi": round(float(np.median(iou)), 4),
            "dice_tb": round(float(dice.mean()), 4),
            "ty_le_iou_tren_0_5": round(float((iou > 0.5).mean()), 4),
            "ty_le_iou_duoi_0_1": round(float((iou < 0.1).mean()), 4)}


def main():
    parser = argparse.ArgumentParser(description="Đo chất lượng mask do SAM sinh so với mask thật của FracAtlas")
    parser.add_argument("--so_anh", type=int, default=60)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="reports/danh_gia_mask.json")
    parser.add_argument("--bo_phat_hien", default=None, help="đường dẫn checkpoint bộ phát hiện gãy xương")
    args = parser.parse_args()

    set_seed(42)
    device = str(get_device())
    ds = build_dataset("fracatlas", args.split)
    logger.info("Đánh giá mask trên %s, tối đa %d ảnh có mask thật", args.split, args.so_anh)

    cau_hinh = [
        ("Lưới 2x2 (mặc định)", dict(default_prompt="grid_boxes", grid=2), False),
        ("Lưới 3x3", dict(default_prompt="grid_boxes", grid=3), False),
        ("Lưới 4x4", dict(default_prompt="grid_boxes", grid=4), False),
        ("Điểm lưới 3x3", dict(default_prompt="points", grid=3), False),
        ("Điểm lưới 5x5", dict(default_prompt="points", grid=5), False),
        ("Box thật của tổn thương", dict(default_prompt="grid_boxes", grid=2), True),
    ]
    bo_phat_hien, canh_dai = None, 640
    if args.bo_phat_hien:
        bo_phat_hien, canh_dai = nap_bo_phat_hien(args.bo_phat_hien, device)
        cau_hinh = [("Box do bộ phát hiện đề xuất", dict(default_prompt="grid_boxes", grid=2), False)]
        logger.info("Dùng box do bộ phát hiện đề xuất từ %s", args.bo_phat_hien)

    bang = []
    for ten, tham_so, dung_box in cau_hinh:
        segmenter = SegmentPromptCreator(device=device, cache_dir=None, sam_med2d_ckpt=None, **tham_so)
        kq = danh_gia(segmenter, ds, args.so_anh, dung_box, bo_phat_hien, canh_dai, device)
        t = tom_tat(ten, kq)
        bang.append(t)
        logger.info("%-26s | %d ảnh | IoU TB %.4f | Dice TB %.4f | IoU>0.5: %.1f%% | IoU<0.1: %.1f%%",
                    ten, t["so_anh"], t["iou_tb"], t["dice_tb"],
                    t["ty_le_iou_tren_0_5"] * 100, t["ty_le_iou_duoi_0_1"] * 100)

    save_json({"split": args.split, "so_anh_yeu_cau": args.so_anh, "ket_qua": bang}, args.out)
    logger.info("Đã lưu %s", args.out)


if __name__ == "__main__":
    main()
