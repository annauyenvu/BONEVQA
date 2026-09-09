import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import TAG_LABELS

MARKER_BAT_DAU = "<!-- KET_QUA_BAT_DAU -->"
MARKER_KET_THUC = "<!-- KET_QUA_KET_THUC -->"

TEN_DATASET = {"fracatlas": "FracAtlas-VQA", "vqa_rad": "VQA-RAD", "roco": "ROCO", "pmc_vqa": "PMC-VQA"}

THU_TU_ABLATION = ["abl3_du3", "abl3_khong_visual", "abl3_khong_lens", "abl3_khong_latent", "abl3_khong_prompt"]

THU_TU_ABLATION_HO_TRO = ["abl5_du3", "abl5_khong_visual", "abl5_khong_lens", "abl5_khong_latent",
                          "abl5_khong_prompt"]

NHAN_ABLATION = {
    "du3": "Đủ 3 prompt", "khong_visual": "Bỏ visual prompt", "khong_lens": "Bỏ lens",
    "khong_latent": "Bỏ latent prompt", "khong_prompt": "Bỏ cả 3 prompt",
}

TAG_CHINH = {"fracatlas": ["lon_tu_dong", "full_tu_dong", "full"],
             "vqa_rad": ["v2_sua_dinh_tuyen", "v3b_vote", "full"]}

THU_TU_CHE_DO = ["lon_tu_dong", "lon_ho_tro"]

NHAN_CHE_DO = {"lon_tu_dong": "Tự động hoàn toàn", "lon_ho_tro": "Có bác sĩ khoanh vùng"}

THU_TU_DIEU_KIEN = ["lon_tu_dong", "lon_ho_tro", "full_tu_dong", "full_khong_ro_ri",
                    "full_khong_mask_that", "full"]

NHAN_DIEU_KIEN = {
    "lon_tu_dong": "MÔ HÌNH CÔNG BỐ — tự động hoàn toàn, huấn luyện khớp điều kiện",
    "lon_ho_tro": "MÔ HÌNH CÔNG BỐ — có bác sĩ khoanh vùng",
    "full_tu_dong": "Mô hình cũ (huấn luyện có rò rỉ) — tự động hoàn toàn",
    "full_khong_ro_ri": "Có box người dùng khoanh — không mask, không tên vùng",
    "full_khong_mask_that": "Bỏ mask thật, giữ box và tên vùng",
    "full": "Được cấp mask thật, box và tên vùng (số đã báo cáo trước đây)",
}

THU_TU_VQA_RAD = ["v2_sua_dinh_tuyen", "hop_nhat3", "v3b_vote", "full", "v3b_base", "v3c_vote", "v3c_base"]

NHAN_VQA_RAD = {
    "v2_sua_dinh_tuyen": "Mô hình chính (sau khi sửa định tuyến)",
    "hop_nhat3": "Hợp nhất 3 checkpoint",
    "v3b_vote": "Giao thức sạch v3b + phiếu LLM 0,3",
    "full": "Bản đầu (chưa sửa định tuyến)",
    "v3b_base": "Giao thức sạch v3b",
    "v3c_vote": "v3c (train đủ) + phiếu LLM 0,3",
    "v3c_base": "v3c (train đủ)",
}

DUONG_CONG = [
    ("1", "ROCO", "stage1_roco_full"),
    ("2", "PMC-VQA", "stage2_pmcvqa_full"),
    ("3", "FracAtlas-VQA", "stage3_fracatlas_full"),
    ("3", "VQA-RAD", "stage3_vqa_rad_full"),
]


def doc_ket_qua(report_dir):
    ket_qua = {}
    for path in sorted(glob.glob(os.path.join(report_dir, "results_*.json"))):
        ten = os.path.splitext(os.path.basename(path))[0][len("results_"):]
        for ds in TEN_DATASET:
            if ten.startswith(ds + "_"):
                ket_qua.setdefault(ds, {})[ten[len(ds) + 1:]] = json.load(open(path, encoding="utf-8"))["metrics"]
                break
    return ket_qua


def _pt(m, key):
    return f"{m.get(key, 0.0) * 100:.2f}%"


def bang_chinh(ket_qua):
    dong = ["| Dataset | Closed acc | Open acc | Open recall | BLEU-1 | Overall acc | Số mẫu |",
            "|---|---|---|---|---|---|---|"]
    co_du_lieu = False
    for ds in ("vqa_rad", "fracatlas"):
        m = None
        for tag in TAG_CHINH.get(ds, ["full"]):
            m = ket_qua.get(ds, {}).get(tag)
            if m:
                break
        if not m:
            continue
        co_du_lieu = True
        dong.append(f"| {TEN_DATASET[ds]} | {_pt(m, 'closed_acc')} | {_pt(m, 'open_acc')} | {_pt(m, 'open_recall')} "
                    f"| {_pt(m, 'bleu1')} | {_pt(m, 'overall_acc')} | {m.get('n', 0)} |")
    if not co_du_lieu:
        return "_Chưa có file `reports/results_*.json`; chạy `src.evaluate` trước._"
    return "\n".join(dong)


def bang_dieu_kien_fracatlas(ket_qua):
    fra = ket_qua.get("fracatlas", {})
    tags = [t for t in THU_TU_DIEU_KIEN if t in fra]
    if len(tags) < 2:
        return None
    dong = ["| Điều kiện suy luận | Closed acc | Open acc | Open recall | BLEU-1 | Overall acc |",
            "|---|---|---|---|---|---|"]
    for t in tags:
        m = fra[t]
        dong.append(f"| {NHAN_DIEU_KIEN.get(t, t)} | {_pt(m, 'closed_acc')} | {_pt(m, 'open_acc')} "
                    f"| {_pt(m, 'open_recall')} | {_pt(m, 'bleu1')} | {_pt(m, 'overall_acc')} |")
    return chr(10).join(dong)


def bang_bien_the_vqa_rad(ket_qua):
    vr = ket_qua.get("vqa_rad", {})
    tags = [t for t in THU_TU_VQA_RAD if t in vr]
    if len(tags) < 2:
        return None
    dong = ["| Cấu hình | Closed acc | Open acc | Open recall | BLEU-1 | Overall acc |", "|---|---|---|---|---|---|"]
    for t in tags:
        m = vr[t]
        dong.append(f"| {NHAN_VQA_RAD.get(t, t)} | {_pt(m, 'closed_acc')} | {_pt(m, 'open_acc')} "
                    f"| {_pt(m, 'open_recall')} | {_pt(m, 'bleu1')} | {_pt(m, 'overall_acc')} |")
    return chr(10).join(dong)


def bang_ablation_hai_che_do(ket_qua):
    fra = ket_qua.get("fracatlas", {})
    dong = ["| Cấu hình | Có box bác sĩ | Chênh | Tự động hoàn toàn | Chênh |", "|---|---|---|---|---|"]
    goc_ht = fra.get("abl5_du3", {}).get("overall_acc")
    goc_td = fra.get("abl3_du3", {}).get("overall_acc")
    if goc_ht is None or goc_td is None:
        return None
    for hau in ("du3", "khong_visual", "khong_lens", "khong_latent", "khong_prompt"):
        ht = fra.get(f"abl5_{hau}", {}).get("overall_acc")
        td = fra.get(f"abl3_{hau}", {}).get("overall_acc")
        if ht is None or td is None:
            continue
        c_ht = "—" if hau == "du3" else f"{(ht - goc_ht) * 100:+.2f}"
        c_td = "—" if hau == "du3" else f"{(td - goc_td) * 100:+.2f}"
        dong.append(f"| {NHAN_ABLATION[hau]} | {ht * 100:.2f}% | {c_ht} | {td * 100:.2f}% | {c_td} |")
    return chr(10).join(dong)


def bang_ablation(ket_qua):
    fra = ket_qua.get("fracatlas", {})
    tags = [t for t in THU_TU_ABLATION if t in fra]
    if len(tags) < 2:
        return "_Chưa đủ kết quả ablation (cần từ 2 cấu hình)._"
    dong = ["| Cấu hình | Closed acc | Open acc | Open recall | BLEU-1 | Overall acc |", "|---|---|---|---|---|---|"]
    for t in tags:
        m = fra[t]
        dong.append(f"| {TAG_LABELS.get(t, t)} | {_pt(m, 'closed_acc')} | {_pt(m, 'open_acc')} | {_pt(m, 'open_recall')} "
                    f"| {_pt(m, 'bleu1')} | {_pt(m, 'overall_acc')} |")
    return "\n".join(dong)


def bang_duong_cong(fig_dir):
    dong = ["| Stage | Dataset | Hình |", "|---|---|---|"]
    for stage, ds, run in DUONG_CONG:
        path = os.path.join(fig_dir, f"curve_{run}.png").replace("\\", "/")
        trang_thai = f"`{path}`" if os.path.exists(path) else "_chưa có_"
        dong.append(f"| {stage} | {ds} | {trang_thai} |")
    return "\n".join(dong)


def khoi_ket_qua(report_dir, fig_dir):
    ket_qua = doc_ket_qua(report_dir)
    phan = [MARKER_BAT_DAU, "",
            "### 12.1 Đường cong huấn luyện", "", bang_duong_cong(fig_dir), "",
            "### 12.2 Kết quả trên tập test", "", bang_chinh(ket_qua), "",
            "### 12.3 Ảnh hưởng của rò rỉ ground-truth trên FracAtlas-VQA", "",
            "Con số 96.75% từng công bố phụ thuộc vào ba thông tin ground-truth không có khi triển khai thật: "
            "mask tổn thương (chỉ ảnh có gãy mới có), bounding box ổ gãy, và tên vùng giải phẫu ghi sẵn trong "
            "textual prompt trong khi có câu hỏi hỏi đúng vùng đó. Bảng dưới đo lại trên cùng 1 200 QA test.", "",
            bang_dieu_kien_fracatlas(ket_qua) or "_Chưa đủ kết quả._", "",
            "**Con số công bố của đề tài là 90.42% ở chế độ tự động hoàn toàn và 93.67% khi có bác sĩ khoanh "
            "vùng.** Hai dòng cuối là mô hình cũ huấn luyện khi còn rò rỉ, giữ lại để đối chiếu: chỉ riêng việc "
            "huấn luyện khớp điều kiện suy luận đã nâng kết quả 15.09 điểm ở chế độ tự động. Chi tiết mục 7.4.", "",
            "### 12.4 Các biến thể trên VQA-RAD", "",
            f"{len([t for t in THU_TU_VQA_RAD if t in ket_qua.get('vqa_rad', {})])} cấu hình dưới đây "
            "đánh giá trên cùng 451 mẫu test của VQA-RAD.", "",
            bang_bien_the_vqa_rad(ket_qua) or "_Chưa đủ kết quả._", "",
            "### 12.5 Ablation: hai chế độ định vị — kết luận của đề tài", "",
            "Cùng kiến trúc, cùng dữ liệu, cùng số epoch, cùng tập test. Biến duy nhất thay đổi là nguồn thông "
            "tin định vị: box do bác sĩ khoanh (SAM đạt IoU 0.3207) so với lưới 2×2 tự động (IoU 0.0205).", "",
            bang_ablation_hai_che_do(ket_qua) or "_Chưa đủ kết quả._", "",
            "Ở chế độ có box, **cả bốn so sánh đều có ý nghĩa thống kê** (p < 0.001, McNemar); bỏ cả ba prompt "
            "mất 4.17 điểm. Ở chế độ tự động thì ngược lại: bỏ cả ba prompt lại tốt hơn 1.67 điểm (p = 0.0078). "
            "Kết luận: ba loại prompt có tác dụng thật nhưng **chỉ khi vùng quan tâm được định vị chính xác**.", "",
            "### 12.6 Chi tiết ablation ở chế độ tự động hoàn toàn", "",
            "Năm cấu hình được **huấn luyện lại từ đầu** trong điều kiện sạch (không mask, không box, không tên "
            "vùng ground-truth; 6 000 QA, 2 epoch, cùng init từ stage 2) và đánh giá trên 1 200 QA test cũng ở "
            "chế độ tự động hoàn toàn. Con số trong ngoặc là bản cũ đo khi còn rò rỉ.", "",
            bang_ablation(ket_qua), "",
            f"Biểu đồ so sánh: `{os.path.join(fig_dir, 'ablation_fracatlas.png').replace(chr(92), '/')}`", "",
            MARKER_KET_THUC]
    return "\n".join(phan)


def main():
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả đánh giá vào README")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--figure_dir", default="reports/figures")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    khoi = khoi_ket_qua(args.report_dir, args.figure_dir)
    with open(args.readme, "r", encoding="utf-8") as f:
        noi_dung = f.read()
    if MARKER_BAT_DAU not in noi_dung or MARKER_KET_THUC not in noi_dung:
        raise SystemExit(f"README thiếu marker {MARKER_BAT_DAU} / {MARKER_KET_THUC}")
    dau = noi_dung.index(MARKER_BAT_DAU)
    cuoi = noi_dung.index(MARKER_KET_THUC) + len(MARKER_KET_THUC)
    with open(args.readme, "w", encoding="utf-8") as f:
        f.write(noi_dung[:dau] + khoi + noi_dung[cuoi:])
    print(khoi)
    print(f"\nĐã cập nhật {args.readme}")


if __name__ == "__main__":
    main()
