import os
import sys
import time

os.environ.setdefault("HF_HOME", "D:/BaoYen_work/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from src.models.bonevqa import BoneVQAModel
from src.utils import count_parameters, format_params, gpu_memory_str, setup_logger

logger = setup_logger()


def make_batch(n=2, size=224):
    rng = np.random.default_rng(0)
    images, masks, boxes = [], [], []
    for _ in range(n):
        images.append(Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)))
        m = np.zeros((1, size, size), dtype=np.uint8)
        m[0, 50:150, 60:170] = 1
        masks.append(m)
        boxes.append([[60, 50, 170, 150]])
    return {"images": images, "questions": ["Is there a fracture in this image?", "Which body part is shown?"],
            "answers": ["yes", "hand"], "answer_types": ["CLOSED", "OPEN"], "regions": ["hand", "hand"],
            "masks": masks, "boxes": boxes, "image_ids": ["s1", "s2"]}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"sam": {"cache_dir": None, "sam_med2d_ckpt": None}}
    model = BoneVQAModel(cfg, ["yes", "no", "hand", "leg", "hip", "shoulder"], device=device).to(device)
    logger.info("Tham số: %s", format_params(count_parameters(model)))
    model.train()
    batch = make_batch()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(batch, stage=2)
    out["loss"].backward()
    logger.info("Forward+backward %.1fs | loss=%.4f cls=%.4f lm=%.4f cs=%.4f con=%.4f type=%.4f", time.time() - t0,
                out["loss"].item(), out["loss_cls"].item(), out["loss_lm"].item(), out["loss_cs"].item(),
                out["loss_con"].item(), out["loss_type"].item())
    logger.info(gpu_memory_str())
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    logger.info("Số tensor có gradient: %d", grads)
    model.zero_grad(set_to_none=True)
    model.eval()
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        r1 = model.answer(img, "Is there a fracture?", boxes=[[40, 40, 200, 220]])
        r2 = model.answer(img, "What abnormality is seen in the image?")
    logger.info("answer() đóng: %s | %s | conf=%.3f | masks=%d", r1["answer"], r1["answer_type"], r1["confidence"], len(r1["masks"]))
    logger.info("answer() mở: '%s' | %s | conf=%.3f | masks=%d", r2["answer"], r2["answer_type"], r2["confidence"], len(r2["masks"]))
    os.makedirs("reports/figures", exist_ok=True)
    r1["prompt_image"].save("reports/figures/smoke_prompt.png")
    r1["lens_image"].save("reports/figures/smoke_lens.png")
    ck = "D:/BaoYen_work/checkpoints/smoke/best.pt"
    model.save(ck)
    m2 = BoneVQAModel.load(ck, device=device)
    logger.info("Nạp lại checkpoint OK, vocab=%d", len(m2.answer_vocab))
    logger.info("Kết thúc smoke test. %s", gpu_memory_str())


if __name__ == "__main__":
    main()
