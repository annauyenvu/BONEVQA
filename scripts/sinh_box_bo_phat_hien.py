import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from src.data import build_fracatlas_vqa
from src.data.fracatlas_vqa import FRACATLAS_ROOT
from src.utils import get_device, save_json, set_seed, setup_env_dirs, setup_logger

setup_env_dirs()
logger = setup_logger()


def nap_bo_phat_hien(duong_dan, device):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    tai = torch.load(duong_dan, map_location="cpu", weights_only=False)
    mo_hinh = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    so_dac_trung = mo_hinh.roi_heads.box_predictor.cls_score.in_features
    mo_hinh.roi_heads.box_predictor = FastRCNNPredictor(so_dac_trung, 2)
    mo_hinh.load_state_dict(tai["state"])
    mo_hinh.to(device).eval()
    return mo_hinh, tai.get("canh_dai", 640)


@torch.no_grad()
def du_doan(mo_hinh, canh_dai, duong_dan_anh, device, nguong, toi_da):
    with Image.open(duong_dan_anh) as im:
        im = im.convert("RGB")
    w, h = im.size
    ty_le = canh_dai / max(w, h)
    nw, nh = max(1, int(round(w * ty_le))), max(1, int(round(h * ty_le)))
    nho = im.resize((nw, nh), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(nho, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
    ra = mo_hinh([x.to(device)])[0]
    hop = []
    for b, sc in zip(ra["boxes"].tolist(), ra["scores"].tolist()):
        if sc < nguong or len(hop) >= toi_da:
            continue
        hop.append([round(v / ty_le, 2) for v in b])
    return hop


def main():
    parser = argparse.ArgumentParser(description="Sinh sẵn box do bộ phát hiện đề xuất cho toàn bộ FracAtlas")
    parser.add_argument("--checkpoint", default="D:/BaoYen_work/checkpoints/bo_phat_hien_gay.pt")
    parser.add_argument("--nguong", type=float, default=0.3)
    parser.add_argument("--toi_da", type=int, default=3)
    parser.add_argument("--out", default="D:/BaoYen_work/data/fracatlas/box_bo_phat_hien.json")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    mo_hinh, canh_dai = nap_bo_phat_hien(args.checkpoint, device)

    data = build_fracatlas_vqa()
    anh = {}
    for split in ("train", "val", "test"):
        for r in data[split]:
            anh.setdefault(r["image_id"], r["image"])
    logger.info("Sinh box cho %d ảnh FracAtlas", len(anh))

    ket_qua, co_box, t0 = {}, 0, time.time()
    for i, (image_id, rel) in enumerate(sorted(anh.items()), 1):
        hop = du_doan(mo_hinh, canh_dai, os.path.join(FRACATLAS_ROOT, rel), device, args.nguong, args.toi_da)
        ket_qua[image_id] = hop
        co_box += int(bool(hop))
        if i % 500 == 0:
            logger.info("  %d/%d ảnh, %.0fs", i, len(anh), time.time() - t0)
    save_json(ket_qua, args.out)
    logger.info("Đã lưu %s | %d/%d ảnh có box (%.1f%%) | %.0fs",
                args.out, co_box, len(anh), co_box / len(anh) * 100, time.time() - t0)


if __name__ == "__main__":
    main()
