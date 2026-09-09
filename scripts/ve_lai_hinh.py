import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import plot_confusion_yes_no, plot_training_curves, setup_logger

logger = setup_logger()

THU_MUC_CHECKPOINT = "D:/BaoYen_work/checkpoints"
THU_MUC_KET_QUA = "reports"
THU_MUC_HINH = "reports/figures"


def ve_lai_duong_cong(thu_muc_checkpoint, thu_muc_hinh):
    so = 0
    for duong_dan in sorted(glob.glob(os.path.join(thu_muc_checkpoint, "*", "history.json"))):
        ten_lan_chay = os.path.basename(os.path.dirname(duong_dan))
        with open(duong_dan, encoding="utf-8") as f:
            lich_su = json.load(f)
        if not lich_su:
            continue
        plot_training_curves(lich_su, thu_muc_hinh, ten_lan_chay)
        so += 1
    logger.info("Đã vẽ lại %d đường cong huấn luyện", so)
    return so


def ve_lai_ma_tran_nham_lan(thu_muc_ket_qua, thu_muc_hinh):
    so = 0
    for duong_dan in sorted(glob.glob(os.path.join(thu_muc_ket_qua, "results_*.json"))):
        with open(duong_dan, encoding="utf-8") as f:
            du_lieu = json.load(f)
        ban_ghi = du_lieu.get("records")
        if not ban_ghi:
            continue
        nhan = os.path.basename(duong_dan)[len("results_"):-len(".json")]
        plot_confusion_yes_no(ban_ghi, os.path.join(thu_muc_hinh, f"confusion_{nhan}.png"))
        so += 1
    logger.info("Đã vẽ lại %d ma trận nhầm lẫn", so)
    return so


def ve_lai_hinh_du_lieu():
    import scripts.eda as eda

    eda.main()
    logger.info("Đã vẽ lại các hình thống kê dữ liệu")


def main():
    parser = argparse.ArgumentParser(
        description="Vẽ lại toàn bộ hình của báo cáo sau khi bỏ tiêu đề nướng sẵn trong ảnh")
    parser.add_argument("--checkpoint_dir", default=THU_MUC_CHECKPOINT)
    parser.add_argument("--report_dir", default=THU_MUC_KET_QUA)
    parser.add_argument("--figure_dir", default=THU_MUC_HINH)
    parser.add_argument("--bo_qua_du_lieu", action="store_true",
                        help="Không vẽ lại nhóm hình thống kê dữ liệu (cần nạp dataset, chậm)")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)
    ve_lai_duong_cong(args.checkpoint_dir, args.figure_dir)
    ve_lai_ma_tran_nham_lan(args.report_dir, args.figure_dir)
    if not args.bo_qua_du_lieu:
        ve_lai_hinh_du_lieu()


if __name__ == "__main__":
    main()
