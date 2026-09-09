import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data import build_fracatlas_vqa
from src.data.fracatlas_vqa import FRACATLAS_ROOT
from src.utils import get_device, save_json, set_seed, setup_env_dirs, setup_logger

setup_env_dirs()
logger = setup_logger()

CANH_DAI = 640


class BoPhatHienDataset(Dataset):
    def __init__(self, split, root=FRACATLAS_ROOT, canh_dai=CANH_DAI, tang_cuong=False):
        data = build_fracatlas_vqa(root)
        anh = {}
        for r in data[split]:
            if r.get("boxes") and r["image_id"] not in anh:
                anh[r["image_id"]] = (r["image"], r["boxes"])
        self.muc = sorted(anh.items())
        self.root = root
        self.canh_dai = canh_dai
        self.tang_cuong = tang_cuong

    def __len__(self):
        return len(self.muc)

    def __getitem__(self, i):
        image_id, (rel, boxes) = self.muc[i]
        with Image.open(os.path.join(self.root, rel)) as im:
            im = im.convert("RGB")
        w, h = im.size
        ty_le = self.canh_dai / max(w, h)
        nw, nh = max(1, int(round(w * ty_le))), max(1, int(round(h * ty_le)))
        im = im.resize((nw, nh), Image.BILINEAR)
        hop = []
        for b in boxes:
            x1, y1, x2, y2 = [v * ty_le for v in b[:4]]
            x1, x2 = sorted((max(0.0, x1), min(float(nw), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(nh), y2)))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                hop.append([x1, y1, x2, y2])
        if not hop:
            hop = [[0.0, 0.0, float(nw), float(nh)]]
        anh = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        if self.tang_cuong and np.random.rand() < 0.5:
            anh = torch.flip(anh, dims=[2])
            hop = [[nw - b[2], b[1], nw - b[0], b[3]] for b in hop]
        muc_tieu = {"boxes": torch.tensor(hop, dtype=torch.float32),
                    "labels": torch.ones(len(hop), dtype=torch.int64),
                    "image_id": torch.tensor([i])}
        return anh, muc_tieu, image_id


def collate(batch):
    return tuple(zip(*batch))


def tao_mo_hinh():
    mo_hinh = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    so_dac_trung = mo_hinh.roi_heads.box_predictor.cls_score.in_features
    mo_hinh.roi_heads.box_predictor = FastRCNNPredictor(so_dac_trung, 2)
    return mo_hinh


def iou_hop(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    giao = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    dt_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    dt_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    hop_ = dt_a + dt_b - giao
    return giao / hop_ if hop_ > 0 else 0.0


@torch.no_grad()
def danh_gia(mo_hinh, loader, device, nguong=0.05):
    mo_hinh.eval()
    iou_tot, so_anh, trung = [], 0, 0
    for anhs, muc_tieus, _ in loader:
        anhs = [a.to(device) for a in anhs]
        ra = mo_hinh(anhs)
        for r, mt in zip(ra, muc_tieus):
            that = mt["boxes"].tolist()
            du_doan = [b for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()) if s >= nguong]
            so_anh += 1
            if not du_doan:
                iou_tot.append(0.0)
                continue
            tot = max(iou_hop(d, t) for d in du_doan[:5] for t in that)
            iou_tot.append(tot)
            trung += int(tot >= 0.5)
    return {"so_anh": so_anh, "iou_tot_nhat_tb": round(float(np.mean(iou_tot)), 4),
            "ty_le_iou_tren_0_5": round(trung / max(so_anh, 1), 4),
            "ty_le_khong_tim_thay": round(float(np.mean([i == 0.0 for i in iou_tot])), 4)}


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện bộ phát hiện gãy xương trên FracAtlas")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="D:/BaoYen_work/checkpoints/bo_phat_hien_gay.pt")
    parser.add_argument("--bao_cao", default="reports/bo_phat_hien_gay.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_ds = BoPhatHienDataset("train", tang_cuong=True)
    val_ds = BoPhatHienDataset("val")
    test_ds = BoPhatHienDataset("test")
    logger.info("Bộ phát hiện: train %d ảnh, val %d, test %d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    mo_hinh = tao_mo_hinh().to(device)
    tham_so = [p for p in mo_hinh.parameters() if p.requires_grad]
    toi_uu = torch.optim.SGD(tham_so, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lich = torch.optim.lr_scheduler.CosineAnnealingLR(toi_uu, T_max=args.epochs)

    tot_nhat, lich_su = -1.0, []
    for epoch in range(args.epochs):
        mo_hinh.train()
        t0, tong, n = time.time(), 0.0, 0
        for anhs, muc_tieus, _ in train_loader:
            anhs = [a.to(device) for a in anhs]
            muc_tieus = [{k: v.to(device) for k, v in t.items()} for t in muc_tieus]
            mat_mat = sum(mo_hinh(anhs, muc_tieus).values())
            toi_uu.zero_grad(set_to_none=True)
            mat_mat.backward()
            torch.nn.utils.clip_grad_norm_(tham_so, 5.0)
            toi_uu.step()
            tong += float(mat_mat.detach())
            n += 1
        lich.step()
        m = danh_gia(mo_hinh, val_loader, device)
        m.update({"epoch": epoch + 1, "loss": round(tong / max(n, 1), 4), "time_s": round(time.time() - t0, 1)})
        lich_su.append(m)
        logger.info("Epoch %d | loss %.4f | val IoU %.4f | IoU>0.5 %.1f%% | %.0fs",
                    epoch + 1, m["loss"], m["iou_tot_nhat_tb"], m["ty_le_iou_tren_0_5"] * 100, m["time_s"])
        if m["iou_tot_nhat_tb"] > tot_nhat:
            tot_nhat = m["iou_tot_nhat_tb"]
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state": mo_hinh.state_dict(), "canh_dai": CANH_DAI, "val": m}, args.out)
            logger.info("  đã lưu checkpoint tốt nhất")

    tai = torch.load(args.out, map_location=device, weights_only=False)
    mo_hinh.load_state_dict(tai["state"])
    kq_test = danh_gia(mo_hinh, test_loader, device)
    logger.info("TEST: IoU tốt nhất TB %.4f | IoU>0.5 %.1f%% | không tìm thấy %.1f%%",
                kq_test["iou_tot_nhat_tb"], kq_test["ty_le_iou_tren_0_5"] * 100,
                kq_test["ty_le_khong_tim_thay"] * 100)
    save_json({"lich_su": lich_su, "test": kq_test, "checkpoint": args.out,
               "so_anh_train": len(train_ds)}, args.bao_cao)
    logger.info("Đã lưu %s", args.bao_cao)


if __name__ == "__main__":
    main()
