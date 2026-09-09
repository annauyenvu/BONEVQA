import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pmc_vqa import PMC_ROOT, VAL_FRACTION
from src.utils import setup_logger

logger = setup_logger()


def main():
    train = json.load(open(PMC_ROOT / "train.json", encoding="utf-8"))
    test = json.load(open(PMC_ROOT / "test.json", encoding="utf-8"))

    anh = sorted({r["image"] for r in train})
    rng = random.Random(42)
    rng.shuffle(anh)
    anh_val = set(anh[: max(1, int(len(anh) * VAL_FRACTION))])

    train_khong_val = [r for r in train if r["image"] not in anh_val]
    test_loc = [r for r in test if r["image"] not in anh_val]
    gop = train_khong_val + test_loc

    thieu = sum(1 for r in gop[:3000] if not os.path.exists(PMC_ROOT / r["image"]))
    logger.info("train %d (bỏ val còn %d) + test %d = %d QA | thiếu ảnh trong 3000 mẫu đầu: %d",
                len(train), len(train_khong_val), len(test_loc), len(gop), thieu)

    with open(PMC_ROOT / "train_lon.json", "w", encoding="utf-8") as f:
        json.dump(gop, f, ensure_ascii=False)
    logger.info("Đã lưu %s", PMC_ROOT / "train_lon.json")


if __name__ == "__main__":
    main()
