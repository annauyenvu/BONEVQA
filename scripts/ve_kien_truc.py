import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "reports" / "figures" / "kien_truc_model.png"

MAU_KHOI = {
    "tien_xu_ly": "#E8F1FB",
    "text": "#FFF4DE",
    "visual": "#E6F6EA",
    "fusion": "#F3E8FF",
    "output": "#FFE9E6",
}
MAU_VIEN = {
    "tien_xu_ly": "#4A78B0",
    "text": "#C88A1F",
    "visual": "#3E8E55",
    "fusion": "#7E4FB3",
    "output": "#C64B3C",
}
MAU_HOP = "#FFFFFF"
MAU_CHU = "#1F2A37"
MAU_MUI_TEN = "#3A4756"


def ve_khung(ax, x, y, w, h, ten, mau_nen, mau_vien):
    khung = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                           linewidth=1.8, edgecolor=mau_vien, facecolor=mau_nen, zorder=1)
    ax.add_patch(khung)
    ax.text(x + 0.18, y + h - 0.22, ten, fontsize=11.5, fontweight="bold", color=mau_vien,
            ha="left", va="top", zorder=3)


def ve_hop(ax, x, y, w, h, noi_dung, mau_vien, fontsize=9.2, mau_nen=MAU_HOP, dam=False):
    hop = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=1.4, edgecolor=mau_vien, facecolor=mau_nen, zorder=2)
    ax.add_patch(hop)
    ax.text(x + w / 2, y + h / 2, noi_dung, fontsize=fontsize, color=MAU_CHU, ha="center", va="center",
            fontweight="bold" if dam else "normal", zorder=4, linespacing=1.35)
    return (x, y, w, h)


def tam(hop, phia):
    x, y, w, h = hop
    if phia == "trai":
        return (x, y + h / 2)
    if phia == "phai":
        return (x + w, y + h / 2)
    if phia == "tren":
        return (x + w / 2, y + h)
    return (x + w / 2, y)


def mui_ten(ax, p1, p2, nhan=None, style="-|>", mau=MAU_MUI_TEN, cong=0.0, lw=1.5, vi_tri_nhan=0.5, dich_nhan=(0, 0.14), fontsize=8.2):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, linewidth=lw, color=mau,
                        connectionstyle=f"arc3,rad={cong}", zorder=5, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if nhan:
        mx = p1[0] + (p2[0] - p1[0]) * vi_tri_nhan + dich_nhan[0]
        my = p1[1] + (p2[1] - p1[1]) * vi_tri_nhan + dich_nhan[1]
        ax.text(mx, my, nhan, fontsize=fontsize, color=mau, ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))


def duong_gap_khuc(ax, diem, nhan=None, mau=MAU_MUI_TEN, lw=1.5, chi_so_nhan=0, dich_nhan=(0, 0.14), fontsize=8.2):
    for k in range(len(diem) - 1):
        p1, p2 = diem[k], diem[k + 1]
        if k == len(diem) - 2:
            mui_ten(ax, p1, p2, mau=mau, lw=lw)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=mau, linewidth=lw, zorder=5, solid_capstyle="round")
    if nhan:
        p1, p2 = diem[chi_so_nhan], diem[chi_so_nhan + 1]
        mx = (p1[0] + p2[0]) / 2 + dich_nhan[0]
        my = (p1[1] + p2[1]) / 2 + dich_nhan[1]
        ax.text(mx, my, nhan, fontsize=fontsize, color=mau, ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    fig = plt.figure(figsize=(19, 12.6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 12.6)
    ax.axis("off")

    ax.text(9.5, 12.25, "Kiến trúc BoneVQA-Prompt: hội thoại trực quan ảnh X-quang xương với Visual / Textual / Latent Prompt",
            fontsize=15, fontweight="bold", color=MAU_CHU, ha="center", va="center")

    y_top_row_bottom = 6.75
    y_bottom_row_top = 6.0

    ve_khung(ax, 0.3, y_top_row_bottom, 5.6, 11.9 - y_top_row_bottom, "1. Tiền xử lý & sinh Prompt (SAM)",
             MAU_KHOI["tien_xu_ly"], MAU_VIEN["tien_xu_ly"])
    hop_box = ve_hop(ax, 0.55, 10.35, 2.4, 0.95, "Box / point / grid\n(người dùng vẽ hoặc tự động)", MAU_VIEN["tien_xu_ly"])
    hop_anh = ve_hop(ax, 3.2, 10.35, 2.45, 0.95, "Ảnh X-quang\n(ảnh global)", MAU_VIEN["tien_xu_ly"], dam=True)
    hop_sam = ve_hop(ax, 0.55, 7.9, 2.4, 1.6, "SegmentPromptCreator\nMedSAM / SAM-Med2D\n(box / point / grid)\n→ masks", MAU_VIEN["tien_xu_ly"], dam=True)
    hop_vp = ve_hop(ax, 3.2, 8.55, 2.45, 1.2, "Visual prompt (FAVP)\nảnh local vẽ contour /\nbox / circle / mask", MAU_VIEN["tien_xu_ly"])
    hop_lens = ve_hop(ax, 3.2, 6.95, 2.45, 1.2, "Lens (Localization Lens)\nunicolor / multicolor /\nmasked", MAU_VIEN["tien_xu_ly"])
    mui_ten(ax, tam(hop_box, "duoi"), tam(hop_sam, "tren"))
    mui_ten(ax, tam(hop_anh, "duoi"), (2.5, 9.5), cong=0.2)
    mui_ten(ax, (2.95, 9.05), (3.2, 9.15), nhan="masks", dich_nhan=(0.0, 0.2))
    mui_ten(ax, (2.95, 8.25), (3.2, 7.55), nhan="masks", cong=0.15, dich_nhan=(-0.2, -0.3))

    ve_khung(ax, 6.4, y_top_row_bottom, 6.0, 11.9 - y_top_row_bottom, "2. Visual encoder + LensFusion / PixelShuffle",
             MAU_KHOI["visual"], MAU_VIEN["visual"])
    hop_ve = ve_hop(ax, 6.7, 9.3, 5.4, 1.5,
                    "VisualEncoder dùng chung\nBiomedCLIP ViT-B/16 (hoặc CLIP ViT-B/16) + LoRA r=4\nvào: ảnh global + ảnh local (visual prompt) + 3 view lens",
                    MAU_VIEN["visual"], dam=True)
    hop_qf = ve_hop(ax, 7.3, 7.1, 1.95, 1.35, "Q-Former lite\n32 query × d\n→ prefix_vis (LLM)", MAU_VIEN["visual"], fontsize=8.8)
    hop_lf = ve_hop(ax, 9.45, 7.1, 2.65, 1.35, "LensFusion\nattention 2 lớp +\nPixelShuffle r=2 → F_I", MAU_VIEN["visual"])
    mui_ten(ax, (8.9, 9.3), tam(hop_qf, "tren"), nhan="tokens\nglobal+local", dich_nhan=(-0.55, 0.0), vi_tri_nhan=0.5)
    mui_ten(ax, (10.0, 9.3), tam(hop_lf, "tren"), nhan="tokens 3 view", dich_nhan=(0.55, 0.0), vi_tri_nhan=0.5)
    mui_ten(ax, tam(hop_anh, "phai"), (6.7, 10.45), nhan="ảnh global", cong=0.0, dich_nhan=(0.0, 0.16))
    mui_ten(ax, tam(hop_vp, "phai"), (6.7, 9.6), nhan="ảnh local", cong=0.0, dich_nhan=(0.0, 0.16))
    duong_gap_khuc(ax, [tam(hop_lens, "phai"), (7.0, 7.55), (7.0, 9.3)], nhan="lens", chi_so_nhan=0, dich_nhan=(0.0, 0.16))

    ve_khung(ax, 12.9, y_top_row_bottom, 5.8, 11.9 - y_top_row_bottom, "3. Text encoder + textual prompt", MAU_KHOI["text"], MAU_VIEN["text"])
    hop_q = ve_hop(ax, 13.2, 10.35, 2.4, 0.95, "Câu hỏi\n(người dùng nhập)", MAU_VIEN["text"], dam=True)
    hop_reg = ve_hop(ax, 15.95, 10.35, 2.45, 0.95, "Vùng giải phẫu\n(hand / leg / hip / shoulder)", MAU_VIEN["text"])
    hop_tp = ve_hop(ax, 13.2, 8.7, 5.2, 1.05,
                    "Textual prompt (template)\n\"This is a bone X-ray image of the {region}.\nAnswer the question briefly. Question: {q}\"",
                    MAU_VIEN["text"], fontsize=8.6)
    hop_te = ve_hop(ax, 13.2, 7.0, 5.2, 1.3, "TextEncoder\nPubMedBERT (BiomedCLIP) hoặc BERT → F_L\n(cũng mã hoá answer bank & target answer, đóng băng)",
                    MAU_VIEN["text"], dam=True)
    mui_ten(ax, tam(hop_q, "duoi"), (14.4, 9.75))
    mui_ten(ax, tam(hop_reg, "duoi"), (17.2, 9.75))
    mui_ten(ax, tam(hop_tp, "duoi"), tam(hop_te, "tren"))

    ve_khung(ax, 0.3, 2.95, 12.1, y_bottom_row_top - 2.95, "4. Multimodal fusion + Latent prompt (LaPA)", MAU_KHOI["fusion"], MAU_VIEN["fusion"])
    hop_lat = ve_hop(ax, 0.55, 3.2, 3.5, 2.35,
                     "LatentPromptGeneration\n32 latent tokens học được\ncross-attn với embedding\ntoàn bộ answer vocab\n(+ prior nhãn vùng/bệnh) → X_LP\nL_CS = 1 − cos(X_LP, answer)",
                     MAU_VIEN["fusion"], fontsize=8.8)
    hop_co = ve_hop(ax, 4.5, 4.75, 7.6, 0.8, "Co-attention blocks:  F_I ⇄ F_L  →  F_FI, F_FL, F_MM = [F_FI; F_FL]", MAU_VIEN["fusion"])
    hop_lpf = ve_hop(ax, 4.5, 3.2, 3.5, 1.3, "LatentPromptFusion\nX_LP → cross-attn L → I → MM\n→ X_II", MAU_VIEN["fusion"])
    hop_xf = ve_hop(ax, 8.6, 3.2, 3.5, 1.3,
                    "X_F = α·X_II + θ·F_FI + β·F_FL\n(α=1, θ=0.1, β=0.1)\nL_con: DCL (pha 1) / InfoNCE (pha 2+)",
                    MAU_VIEN["fusion"], fontsize=8.6)
    mui_ten(ax, tam(hop_lat, "phai"), (4.5, 3.85), nhan="X_LP", dich_nhan=(0.0, 0.16))
    mui_ten(ax, (6.25, 4.75), (6.25, 4.5), nhan="F_L, F_I, F_MM", dich_nhan=(0.95, 0.0))
    mui_ten(ax, (10.35, 4.75), (10.35, 4.5), nhan="F_FI, F_FL", dich_nhan=(0.8, 0.0))
    mui_ten(ax, tam(hop_lpf, "phai"), tam(hop_xf, "trai"), nhan="X_II", dich_nhan=(0.0, 0.16))
    mui_ten(ax, tam(hop_lf, "duoi"), (10.5, 5.55), nhan="F_I", dich_nhan=(0.25, 0.0))
    duong_gap_khuc(ax, [(15.8, 7.0), (15.8, 6.5), (11.4, 6.5), (11.4, 5.55)], nhan="F_L", chi_so_nhan=1, dich_nhan=(0.0, 0.16))

    ve_khung(ax, 12.9, 0.35, 5.8, y_bottom_row_top - 0.35, "5. Output (Closed head / LLM Open head)", MAU_KHOI["output"], MAU_VIEN["output"])
    hop_type = ve_hop(ax, 13.2, 4.75, 5.2, 0.8, "AnswerTypeHead (từ X_F): chọn nhánh ĐÓNG hay MỞ", MAU_VIEN["output"], fontsize=9.0)
    hop_open = ve_hop(ax, 13.2, 2.9, 2.45, 1.5, "OpenHead\nQwen2.5-0.5B-Instruct\n+ LoRA r=8, prefix =\nprefix_vis + X_II → L_LM", MAU_VIEN["output"], dam=True, fontsize=8.6)
    hop_closed = ve_hop(ax, 15.95, 2.9, 2.45, 1.5, "ClosedHead\nphân lớp answer\nvocab (yes/no, ...)\nL_BCE", MAU_VIEN["output"], dam=True)
    hop_ans = ve_hop(ax, 13.2, 0.6, 5.2, 1.4,
                     "Câu trả lời + độ tin cậy + ảnh prompt / lens\nhiển thị trong web app hội thoại (FastAPI)\nL = L_BCE + L_LM + η·L_CS (η=0.01) + λ·L_con",
                     MAU_VIEN["output"], mau_nen="#FFF7F5", fontsize=9.0)
    duong_gap_khuc(ax, [(12.1, 3.85), (12.6, 3.85), (12.6, 5.15), (13.2, 5.15)], nhan="X_F", chi_so_nhan=1, dich_nhan=(0.0, 0.0), fontsize=8.0)
    duong_gap_khuc(ax, [(6.25, 3.2), (6.25, 2.65), (14.0, 2.65), (14.0, 2.9)], nhan="prefix LLM = 32 query (Q-Former) + X_II", chi_so_nhan=1, dich_nhan=(0.6, -0.17))
    mui_ten(ax, (14.4, 4.75), (14.4, 4.4), nhan="nhánh mở", dich_nhan=(0.7, 0.0), lw=1.2)
    mui_ten(ax, (17.2, 4.75), (17.2, 4.4), nhan="X_F", dich_nhan=(0.4, 0.0), lw=1.2)
    mui_ten(ax, (15.2, 2.9), (15.2, 2.0), nhan="mở", dich_nhan=(0.3, 0.0))
    mui_ten(ax, (17.6, 2.9), (17.6, 2.0), nhan="đóng", dich_nhan=(0.35, 0.0))

    ghi_chu = [
        "Huấn luyện 3 giai đoạn (theo FAVP): (1) ROCO caption alignment với DCL → (2) PMC-VQA pretraining",
        "→ (3) fine-tune VQA-RAD & FracAtlas-VQA (xương, có mask/bbox).",
        "Backbone chạy fp16/bf16 + LoRA trên GPU 8GB; encoder đóng băng, chỉ huấn luyện LoRA, Q-Former,",
        "LensFusion, latent prompt, fusion, các head và LoRA của LLM.",
    ]
    for i, dong in enumerate(ghi_chu):
        ax.text(0.35, 2.2 - 0.3 * i, dong, fontsize=9.2, color=MAU_CHU, ha="left", va="center")

    chu_thich = [
        ("Visual prompt", "FAVP (AAAI-25)", MAU_VIEN["tien_xu_ly"]),
        ("Latent prompt", "LaPA (CVPRW-24)", MAU_VIEN["fusion"]),
        ("Textual / Localization prompt", "Localization Lens (2025)", MAU_VIEN["text"]),
    ]
    ax.text(0.35, 1.0, "Ánh xạ tới 3 bài báo:", fontsize=9.4, fontweight="bold", color=MAU_CHU, ha="left", va="center")
    for i, (ten, bai, mau) in enumerate(chu_thich):
        x = 0.4
        y = 0.7 - 0.28 * i
        ax.add_patch(FancyBboxPatch((x, y - 0.1), 0.32, 0.2, boxstyle="round,pad=0.01", facecolor=mau, edgecolor=mau))
        ax.text(x + 0.45, y, f"{ten} ← {bai}", fontsize=9.0, color=MAU_CHU, ha="left", va="center")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Đã lưu sơ đồ kiến trúc: {OUT_PATH}")


if __name__ == "__main__":
    main()
