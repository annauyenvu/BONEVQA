import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("bonevqa.inference")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CKPT = r"D:\BaoYen_work\checkpoints\best.pt"

MAU_VUNG = [
    (239, 68, 68),
    (59, 130, 246),
    (16, 185, 129),
    (245, 158, 11),
    (139, 92, 246),
    (236, 72, 153),
]


def _mask_elip(h, w, box=None):
    yy, xx = np.mgrid[0:h, 0:w]
    if box is None:
        cx, cy = w / 2, h / 2
        rx, ry = w * 0.22, h * 0.32
    else:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = max((x2 - x1) / 2, 2), max((y2 - y1) / 2, 2)
    mask = ((xx - cx) ** 2 / rx**2 + (yy - cy) ** 2 / ry**2) <= 1.0
    return mask.astype(np.uint8)


def _mask_tron(h, w, point, r=None):
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = point
    r = r or max(min(h, w) * 0.1, 8)
    return (((xx - px) ** 2 + (yy - py) ** 2) <= r * r).astype(np.uint8)


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _contours_from_mask(mask):
    m = Image.fromarray((mask * 255).astype(np.uint8))
    edge = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    return np.array(edge) > 0


def build_visual_prompts(image, masks, kind="contour"):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for i, mask in enumerate(masks):
        color = MAU_VUNG[i % len(MAU_VUNG)]
        x1, y1, x2, y2 = mask_to_bbox(mask)
        if kind == "box":
            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
        elif kind == "circle":
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            r = max(x2 - x1, y2 - y1) / 2 + 4
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (255,), width=3)
        elif kind == "mask":
            overlay = Image.new("RGBA", out.size, color + (0,))
            overlay.putalpha(Image.fromarray((mask * 110).astype(np.uint8)))
            out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(out, "RGBA")
        else:
            edges = _contours_from_mask(mask)
            layer = np.array(out).copy()
            layer[edges] = color
            out = Image.fromarray(layer)
            draw = ImageDraw.Draw(out, "RGBA")
        draw.text((x1 + 4, max(y1 - 14, 2)), f"#{i + 1}", fill=color + (255,))
    return out


def build_lens(image, masks, mode="multicolor"):
    base = image.convert("RGB")
    arr = np.array(base).astype(np.float32)
    if len(masks) == 0:
        return base
    if mode == "masked":
        union = np.clip(np.sum(masks, axis=0), 0, 1)
        arr = arr * union[..., None]
        return Image.fromarray(arr.astype(np.uint8))
    for i, mask in enumerate(masks):
        color = np.array(MAU_VUNG[i % len(MAU_VUNG)] if mode == "multicolor" else MAU_VUNG[0], dtype=np.float32)
        m = mask.astype(bool)
        arr[m] = arr[m] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build_overlay(image, masks):
    out = build_lens(image, masks, mode="multicolor")
    draw = ImageDraw.Draw(out)
    for i, mask in enumerate(masks):
        x1, y1, x2, y2 = mask_to_bbox(mask)
        draw.rectangle([x1, y1, x2, y2], outline=MAU_VUNG[i % len(MAU_VUNG)], width=2)
    return out


TU_KHOA_DONG = [
    "có ", "không", "phải", "is there", "are there", "does", "do ", "is ", "are ",
    "has ", "have ", "can ", "was ", "were ",
]

VUNG_CO_THE = [
    (["hand", "bàn tay", "ngón tay", "finger", "wrist", "cổ tay"], "hand"),
    (["leg", "chân", "tibia", "fibula", "xương chày"], "leg"),
    (["hip", "háng", "pelvis", "xương chậu", "khung chậu"], "hip"),
    (["shoulder", "vai", "clavicle", "xương đòn"], "shoulder"),
    (["skull", "sọ", "đầu"], "skull"),
    (["chest", "ngực", "rib", "sườn", "lồng ngực"], "chest"),
    (["spine", "cột sống", "vertebra", "đốt sống"], "spine"),
    (["knee", "gối", "đầu gối", "patella"], "knee"),
    (["arm", "cánh tay", "humerus", "elbow", "khuỷu", "cẳng tay", "radius", "ulna"], "arm"),
    (["foot", "bàn chân", "ankle", "mắt cá"], "foot"),
]


class MockEngine:
    name = "MockEngine"
    device = "cpu"
    model_loaded = False

    def segment(self, image, boxes=None, points=None):
        w, h = image.size
        masks = []
        if boxes:
            for b in boxes:
                x1, y1, x2, y2 = [float(v) for v in b]
                x1, x2 = sorted([max(0, x1), min(w, x2)])
                y1, y2 = sorted([max(0, y1), min(h, y2)])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                masks.append(_mask_elip(h, w, (x1, y1, x2, y2)))
        if points:
            for p in points:
                masks.append(_mask_tron(h, w, (float(p[0]), float(p[1]))))
        if not masks:
            masks.append(_mask_elip(h, w))
        return np.stack(masks).astype(np.uint8)

    def _phan_loai(self, question):
        q = question.strip().lower()
        for k in TU_KHOA_DONG:
            if q.startswith(k) or f" {k}" in q[:25]:
                return "CLOSED"
        if "bao nhiêu" in q or "how many" in q:
            return "CLOSED"
        return "OPEN"

    def _tra_loi(self, question, masks, image):
        q = question.lower()
        n = int(len(masks))
        arr = np.array(image.convert("L"), dtype=np.float32)
        do_sang = float(arr.mean()) / 255.0
        ty_le_mask = float(np.clip(np.sum(masks, axis=0), 0, 1).mean()) if n else 0.0
        answer_type = self._phan_loai(question)
        conf = 0.55 + 0.35 * min(1.0, ty_le_mask * 4)
        vn = any(ch in q for ch in "ăâđêôơưàáảãạ")

        if any(k in q for k in ["gãy", "gay", "fracture", "broken", "nứt", "crack"]):
            if "bao nhiêu" in q or "how many" in q:
                return str(n), "CLOSED", conf
            co_gay = ty_le_mask > 0.03 and do_sang > 0.15
            return ("yes" if co_gay else "no"), "CLOSED", conf
        if any(k in q for k in ["kim loại", "nẹp", "vít", "metal", "implant", "screw", "plate", "hardware", "đinh"]):
            sang_manh = float((arr > 240).mean())
            return ("yes" if sang_manh > 0.01 else "no"), "CLOSED", 0.5 + min(0.4, sang_manh * 10)
        if any(k in q for k in ["bao nhiêu", "how many", "số lượng", "count"]):
            return str(n), "CLOSED", conf
        if any(k in q for k in ["vùng", "region", "bộ phận", "body part", "cơ thể", "part of the body", "đâu", "where", "anatom"]):
            for tu_khoa, ten in VUNG_CO_THE:
                if any(t in q for t in tu_khoa):
                    if answer_type == "CLOSED":
                        return "yes", "CLOSED", conf
                    return ten, "OPEN", conf
            ratio = image.size[0] / max(image.size[1], 1)
            ten = "hand" if ratio < 0.9 else ("leg" if ratio < 1.15 else "chest")
            return ten, "OPEN", conf * 0.9
        if any(k in q for k in ["x-quang", "x-ray", "xray", "modality", "loại ảnh", "phương thức"]):
            return ("yes" if answer_type == "CLOSED" else "x-ray"), answer_type, 0.9
        if any(k in q for k in ["bất thường", "abnormal", "tổn thương", "lesion", "bệnh lý"]):
            return ("yes" if ty_le_mask > 0.03 else "no"), "CLOSED", conf
        if any(k in q for k in ["mô tả", "describe", "thấy gì", "what do you see", "nhận xét"]):
            if vn:
                return (
                    f"Ảnh X-quang xương với {n} vùng quan tâm được phân đoạn; mật độ xương phân bố tương đối đồng đều, "
                    "chưa thấy đường gãy rõ rệt trong vùng đánh dấu.",
                    "OPEN",
                    conf * 0.8,
                )
            return (
                f"Bone radiograph with {n} segmented region(s); bone density is fairly uniform and no clear fracture "
                "line is visible in the marked region.",
                "OPEN",
                conf * 0.8,
            )
        if answer_type == "CLOSED":
            return ("yes" if ty_le_mask > 0.05 else "no"), "CLOSED", conf * 0.8
        if vn:
            return "Vùng đánh dấu cho thấy cấu trúc xương; cần thêm ngữ cảnh lâm sàng để kết luận.", "OPEN", conf * 0.7
        return "The marked region shows bone structure; further clinical context is needed.", "OPEN", conf * 0.7

    def answer(self, image, question, boxes=None, prompt_kind="contour", masks=None):
        if masks is None:
            masks = self.segment(image, boxes=boxes)
        text, answer_type, conf = self._tra_loi(question, masks, image)
        return dict(
            answer=text,
            answer_type=answer_type,
            confidence=float(round(min(max(conf, 0.0), 0.99), 3)),
            masks=masks,
            prompt_image=build_visual_prompts(image, masks, kind=prompt_kind),
            lens_image=build_lens(image, masks, mode="multicolor"),
        )


class RealEngine:
    name = "BoneVQAModel"
    model_loaded = True

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mock = MockEngine()

    def segment(self, image, boxes=None, points=None):
        seg = getattr(self.model, "segmenter", None) or getattr(self.model, "segment_prompt", None)
        if seg is not None:
            return np.asarray(seg(image, boxes=boxes, points=points)).astype(np.uint8)
        return self.mock.segment(image, boxes=boxes, points=points)

    def answer(self, image, question, boxes=None, prompt_kind="contour", masks=None):
        try:
            out = self.model.answer(image, question, boxes=boxes, prompt_kind=prompt_kind, masks=masks)
        except TypeError:
            out = self.model.answer(image, question)
        masks_out = out.get("masks")
        if masks_out is None:
            masks_out = masks if masks is not None else self.segment(image, boxes=boxes)
        masks_out = np.asarray(masks_out).astype(np.uint8)
        if masks_out.ndim == 2:
            masks_out = masks_out[None]
        prompt_image = out.get("prompt_image") or build_visual_prompts(image, masks_out, kind=prompt_kind)
        lens_image = out.get("lens_image") or build_lens(image, masks_out, mode="multicolor")
        return dict(
            answer=str(out.get("answer", "")),
            answer_type=str(out.get("answer_type", "OPEN")).upper(),
            confidence=float(out.get("confidence", 0.0)),
            masks=masks_out,
            prompt_image=prompt_image,
            lens_image=lens_image,
        )


_ENGINE = None


def _thu_tai_model_that():
    ckpt = os.environ.get("BONEVQA_CKPT", DEFAULT_CKPT)
    if not os.path.exists(ckpt):
        logger.warning("Không tìm thấy checkpoint %s → dùng MockEngine.", ckpt)
        return None
    try:
        import torch
        from src.models.bonevqa import BoneVQAModel
    except Exception as exc:
        logger.warning("Không import được BoneVQAModel/torch (%s) → dùng MockEngine.", exc)
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BoneVQAModel.load(ckpt, device=device)
        model.refresh_answer_bank()
        logger.info("Đã nạp BoneVQAModel từ %s trên %s (vocab %d đáp án).", ckpt, device, len(model.answer_vocab))
        return RealEngine(model, device)
    except Exception as exc:
        logger.warning("Nạp checkpoint thất bại (%s) → dùng MockEngine.", exc, exc_info=True)
        return None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        t0 = time.time()
        _ENGINE = _thu_tai_model_that() or MockEngine()
        logger.info("Engine đang dùng: %s (%.1fs).", _ENGINE.name, time.time() - t0)
    return _ENGINE
