import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_json

DIEU_KIEN = [
    ("full", "FracAtlas", "Có mask + box + tên vùng (RÒ RỈ)", "mô hình chính"),
    ("full_khong_mask_that", "FracAtlas", "Bỏ mask, giữ box + tên vùng", "mô hình chính"),
    ("full_khong_ro_ri", "FracAtlas", "Có box, bỏ mask + tên vùng", "mô hình chính"),
    ("full_tu_dong", "FracAtlas", "Tự động hoàn toàn", "mô hình chính"),
    ("lon_tu_dong", "FracAtlas", "Tự động hoàn toàn", "quy mô lớn"),
    ("lon_ho_tro", "FracAtlas", "Có box bác sĩ", "quy mô lớn"),
    ("abl3_du3", "FracAtlas", "Tự động hoàn toàn", "ablation, đủ 3 prompt"),
    ("abl3_khong_prompt", "FracAtlas", "Tự động hoàn toàn", "ablation, bỏ cả 3"),
    ("abl5_du3", "FracAtlas", "Có box bác sĩ", "ablation, đủ 3 prompt"),
    ("abl5_khong_prompt", "FracAtlas", "Có box bác sĩ", "ablation, bỏ cả 3"),
    ("abl7_du3", "FracAtlas", "Box bộ phát hiện", "ablation, đủ 3 prompt"),
    ("abl7_khong_prompt", "FracAtlas", "Box bộ phát hiện", "ablation, bỏ cả 3"),
    ("full", "VQA-RAD", "Bản đầu", "mô hình chính"),
    ("v2_sua_dinh_tuyen", "VQA-RAD", "Sau khi sửa định tuyến", "mô hình chính"),
    ("v3b_vote", "VQA-RAD", "Giao thức sạch + phiếu LLM", "giao thức sạch"),
    ("lon_vqa_rad", "VQA-RAD", "Quy mô lớn", "quy mô lớn"),
    ("lon_vqa_rad_vote", "VQA-RAD", "Quy mô lớn + phiếu LLM", "quy mô lớn"),
]

TEN_FILE = {"FracAtlas": "fracatlas", "VQA-RAD": "vqa_rad"}


def main():
    parser = argparse.ArgumentParser(description="Bảng tổng hợp toàn bộ thí nghiệm")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--out", default="reports/bang_toan_bo.json")
    args = parser.parse_args()

    dong, thieu = [], []
    print(f"{'Dataset':10s}{'Điều kiện suy luận':34s}{'Vai trò':26s}{'Closed':>8}{'Open':>8}{'Overall':>9}")
    print("-" * 95)
    for tag, ds, dieu_kien, vai_tro in DIEU_KIEN:
        p = os.path.join(args.report_dir, f"results_{TEN_FILE[ds]}_{tag}.json")
        if not os.path.exists(p):
            thieu.append(f"{ds}/{tag}")
            continue
        m = json.load(open(p, encoding="utf-8"))["metrics"]
        dong.append({"dataset": ds, "tag": tag, "dieu_kien": dieu_kien, "vai_tro": vai_tro,
                     "closed_acc": round(m["closed_acc"], 4), "open_acc": round(m["open_acc"], 4),
                     "overall_acc": round(m["overall_acc"], 4), "n": m.get("n")})
        print(f"{ds:10s}{dieu_kien:34s}{vai_tro:26s}{m['closed_acc']*100:7.2f}%{m['open_acc']*100:7.2f}%"
              f"{m['overall_acc']*100:8.2f}%")

    con_lai = sorted(set(os.path.basename(f)[len('results_'):-5]
                         for f in glob.glob(os.path.join(args.report_dir, "results_*.json")))
                     - {f"{TEN_FILE[d]}_{t}" for t, d, _, _ in DIEU_KIEN})
    print()
    if thieu:
        print("Chưa có kết quả:", ", ".join(thieu))
    if con_lai:
        print(f"Còn {len(con_lai)} tệp kết quả khác chưa liệt kê (ablation chi tiết, đa seed)")
    save_json({"bang": dong, "thieu": thieu, "khac": con_lai}, args.out)
    print("Đã lưu", args.out)


if __name__ == "__main__":
    main()
