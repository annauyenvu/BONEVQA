import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATA_ROOT, DATASET_NAMES, build_answer_vocab
from src.data.fracatlas_vqa import FRACATLAS_ROOT, build_fracatlas_vqa, summarize
from src.data.pmc_vqa import load_pmc_vqa_records
from src.data.roco import load_roco_records
from src.data.vqa_rad import load_vqa_rad_records

LOADERS = {
    "vqa_rad": lambda split: load_vqa_rad_records(split),
    "roco": lambda split: load_roco_records(split, allow_export=False),
    "pmc_vqa": lambda split: load_pmc_vqa_records(split, allow_export=False),
}


def describe(records):
    c = Counter(r["answer_type"] for r in records)
    yes_no = Counter(r["answer"].lower() for r in records if r["answer"].lower() in ("yes", "no"))
    regions = Counter(r.get("region") or "không rõ" for r in records)
    return {
        "số QA": len(records),
        "số ảnh": len({r["image"] for r in records}),
        "answer_type": dict(c),
        "yes/no": dict(yes_no),
        "vùng (top 6)": dict(regions.most_common(6)),
        "top answer": dict(Counter(r["answer"].lower() for r in records).most_common(8)),
    }


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu BoneVQA-Prompt: sinh FracAtlas-VQA, answer vocab, thống kê")
    parser.add_argument("--force", action="store_true", help="Sinh lại FracAtlas-VQA và answer vocab")
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_NAMES))
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    stats = {}
    if "fracatlas" in args.datasets:
        if (FRACATLAS_ROOT / "dataset.csv").exists():
            print("== Sinh FracAtlas-VQA ==")
            data = build_fracatlas_vqa(force=args.force)
            stats["fracatlas"] = {s: summarize(r) for s, r in data.items()}
            for s, r in data.items():
                print(f"  {s}: {summarize(r)}")
        else:
            print(f"!! Không tìm thấy FracAtlas tại {FRACATLAS_ROOT}")

    for name in ("vqa_rad", "roco", "pmc_vqa"):
        if name not in args.datasets:
            continue
        print(f"== {name} ==")
        stats[name] = {}
        for split in ("train", "val", "test"):
            try:
                recs = LOADERS[name](split)
            except FileNotFoundError as e:
                print(f"  {split}: bỏ qua ({e})")
                continue
            info = describe(recs)
            stats[name][split] = info
            print(f"  {split}: {json.dumps(info, ensure_ascii=False)}")

    print("== Answer vocab ==")
    for name in args.datasets:
        if name in stats and stats[name]:
            try:
                vocab = build_answer_vocab([name], top_k=args.top_k, force=args.force)
                closed = build_answer_vocab([name], top_k=args.top_k, closed_only=True, force=args.force)
                print(f"  {name}: {len(vocab)} answers (closed-only {len(closed)}), 10 đầu: {vocab[:10]}")
            except FileNotFoundError as e:
                print(f"  {name}: bỏ qua ({e})")
    available = [n for n in ("fracatlas", "vqa_rad") if n in stats and stats[n]]
    if len(available) > 1:
        vocab = build_answer_vocab(available, top_k=args.top_k, force=args.force)
        print(f"  {'+'.join(available)}: {len(vocab)} answers")

    out = DATA_ROOT / "data_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"Đã lưu thống kê vào {out}")


if __name__ == "__main__":
    main()
