import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataset, build_lens, build_visual_prompts, overlay_masks
from src.data.fracatlas_vqa import FRACATLAS_ROOT, REGIONS, build_fracatlas_vqa
from src.data.pmc_vqa import load_pmc_vqa_records
from src.data.roco import load_roco_records
from src.data.vqa_rad import load_vqa_rad_records

FIG_DIR = ROOT / "reports" / "figures"
COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "gray": "#9a9994"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})
REGION_VI = {"hand": "Bàn tay", "leg": "Chân", "hip": "Hông", "shoulder": "Vai"}


def _bar(ax, labels, values, color, title, xlabel="", ylabel="Số lượng", horizontal=False):
    if horizontal:
        bars = ax.barh(labels, values, color=color, height=0.6)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        for b, v in zip(bars, values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v}", va="center", fontsize=8)
    else:
        bars = ax.bar(labels, values, color=color, width=0.6)
        ax.set_ylabel(ylabel)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)


def fig_fracatlas():
    data = build_fracatlas_vqa()
    all_records = sum(data.values(), [])
    per_image = {}
    for r in all_records:
        per_image[r["image_id"]] = r
    fractured = Counter("Gãy xương" if r["fractured"] else "Không gãy" for r in per_image.values())
    region_counter = Counter()
    for r in per_image.values():
        parts = [x.strip() for x in (r["region"] or "").replace(" and ", ", ").split(",") if x.strip()]
        for reg in REGIONS:
            if reg in parts:
                region_counter[REGION_VI[reg]] += 1
    split_counts = {s: (len({r["image_id"] for r in recs}), len(recs)) for s, recs in data.items()}
    qtype = Counter(r["answer_type"] for r in all_records)
    yes_no = Counter(r["answer"] for r in all_records if r["answer_type"] == "CLOSED")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _bar(axes[0, 0], list(fractured.keys()), list(fractured.values()), [COLORS["orange"], COLORS["blue"]], "Phân bố nhãn gãy / không gãy (theo ảnh)")
    _bar(axes[0, 1], list(region_counter.keys()), list(region_counter.values()), COLORS["aqua"], "Phân bố vùng cơ thể (theo ảnh)")
    labels = ["Huấn luyện", "Kiểm định", "Kiểm tra"]
    x = np.arange(3)
    w = 0.35
    axes[1, 0].bar(x - w / 2, [split_counts[s][0] for s in ("train", "val", "test")], w, color=COLORS["blue"], label="Số ảnh")
    axes[1, 0].bar(x + w / 2, [split_counts[s][1] for s in ("train", "val", "test")], w, color=COLORS["orange"], label="Số cặp hỏi–đáp")
    for i, s in enumerate(("train", "val", "test")):
        axes[1, 0].text(i - w / 2, split_counts[s][0], str(split_counts[s][0]), ha="center", va="bottom", fontsize=8)
        axes[1, 0].text(i + w / 2, split_counts[s][1], str(split_counts[s][1]), ha="center", va="bottom", fontsize=8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title("Chia tập FracAtlas-VQA (70/15/15 theo ảnh)", fontsize=11, loc="left")
    axes[1, 0].legend(frameon=False)
    _bar(axes[1, 1], ["Câu hỏi đóng", "Câu hỏi mở", "Đáp án 'yes'", "Đáp án 'no'"],
         [qtype["CLOSED"], qtype["OPEN"], yes_no["yes"], yes_no["no"]],
         [COLORS["blue"], COLORS["orange"], COLORS["aqua"], COLORS["yellow"]], "Loại câu hỏi và cân bằng yes/no")
    fig.tight_layout()
    out = FIG_DIR / "data_fracatlas_phan_bo.png"
    fig.savefig(out)
    plt.close(fig)
    print("Đã lưu", out)


def fig_answers(name, records, title):
    qtype = Counter(r["answer_type"] for r in records)
    closed = Counter(r["answer"].lower() for r in records if r["answer_type"] == "CLOSED").most_common(10)
    opened = Counter(r["answer"].lower() for r in records if r["answer_type"] == "OPEN").most_common(12)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.4, 1.8]})
    _bar(axes[0], ["Đóng (CLOSED)", "Mở (OPEN)"], [qtype["CLOSED"], qtype["OPEN"]], [COLORS["blue"], COLORS["orange"]], "Phân bố loại câu trả lời")
    if closed:
        _bar(axes[1], [a[:28] for a, _ in closed], [c for _, c in closed], COLORS["blue"], "Top câu trả lời đóng", horizontal=True)
    if opened:
        _bar(axes[2], [a[:40] + ("…" if len(a) > 40 else "") for a, _ in opened], [c for _, c in opened], COLORS["orange"], "Top câu trả lời mở", horizontal=True)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    out = FIG_DIR / f"data_{name}_answers.png"
    fig.savefig(out)
    plt.close(fig)
    print("Đã lưu", out)


def fig_prompt_examples(n_rows=3, seed=42):
    ds = build_dataset("fracatlas", "train", max_side=768)
    idx = [i for i, r in enumerate(ds.records) if r["boxes"]]
    rng = np.random.default_rng(seed)
    multi = [i for i in idx if len(ds.records[i]["boxes"]) > 1]
    chosen = list(rng.choice(multi, size=min(1, len(multi)), replace=False)) + list(rng.choice(idx, size=n_rows - 1, replace=False))
    cols = ["Ảnh gốc", "Mask overlay", "Prompt: contour", "Prompt: box", "Prompt: circle", "Lens: unicolor", "Lens: multicolor", "Lens: masked"]
    fig, axes = plt.subplots(len(chosen), len(cols), figsize=(2.3 * len(cols), 2.9 * len(chosen)))
    axes = np.atleast_2d(axes)
    lens_rng = np.random.default_rng(seed)
    for row, i in enumerate(chosen):
        s = ds[i]
        img, masks = s["image"], s["masks"]
        panels = [
            img,
            overlay_masks(img, masks),
            build_visual_prompts(img, masks, "contour")[0] if masks.shape[0] == 1 else _merge_prompts(img, masks, "contour"),
            _merge_prompts(img, masks, "box"),
            _merge_prompts(img, masks, "circle"),
            build_lens(img, masks, "unicolor", rng=lens_rng),
            build_lens(img, masks, "multicolor", rng=lens_rng),
            build_lens(img, masks, "masked"),
        ]
        for col, (ax, p) in enumerate(zip(axes[row], panels)):
            ax.imshow(p)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
        axes[row, 0].set_ylabel(f"{s['image_id']}\nvùng: {s['region']}\n{masks.shape[0]} mask", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "data_vi_du_prompt.png"
    fig.savefig(out)
    plt.close(fig)
    print("Đã lưu", out)


def _merge_prompts(img, masks, kind):
    from src.data.prompts import draw_boxes, draw_circles, draw_contours, masks_to_boxes

    if kind == "contour":
        return draw_contours(img, masks)
    if kind == "box":
        return draw_boxes(img, masks_to_boxes(masks))
    return draw_circles(img, masks)


def fig_sample_grid(name, split="train", n=8, seed=42, max_side=512):
    ds = build_dataset(name, split, max_side=max_side)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    for ax, i in zip(axes.ravel(), idx):
        s = ds[int(i)]
        ax.imshow(s["image"], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        q = s["question"]
        a = s["answer"]
        ax.set_title(f"H: {q[:60]}{'…' if len(q) > 60 else ''}\nĐ: {a[:60]}{'…' if len(a) > 60 else ''} [{s['answer_type']}]", fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / f"data_{name}_mau.png"
    fig.savefig(out)
    plt.close(fig)
    print("Đã lưu", out)


def main():
    parser = argparse.ArgumentParser(description="EDA và vẽ biểu đồ dữ liệu BoneVQA-Prompt")
    parser.add_argument("--skip-samples", action="store_true")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if (FRACATLAS_ROOT / "dataset.csv").exists():
        fig_fracatlas()
        fig_answers("fracatlas", build_fracatlas_vqa()["train"], "FracAtlas-VQA")
        fig_prompt_examples()
        if not args.skip_samples:
            fig_sample_grid("fracatlas")
    else:
        print("!! Bỏ qua FracAtlas: chưa có dữ liệu")

    for name, loader, title in (
        ("vqa_rad", lambda: load_vqa_rad_records("train"), "VQA-RAD"),
        ("roco", lambda: load_roco_records("train", allow_export=False), "ROCO (caption)"),
        ("pmc_vqa", lambda: load_pmc_vqa_records("train", allow_export=False), "PMC-VQA"),
    ):
        try:
            recs = loader()
        except FileNotFoundError as e:
            print(f"!! Bỏ qua {name}: {e}")
            continue
        fig_answers(name, recs, title)
        if not args.skip_samples:
            fig_sample_grid(name)


if __name__ == "__main__":
    main()
