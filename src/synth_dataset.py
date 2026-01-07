import os, json
from pathlib import Path
import numpy as np
import cv2
from .control_utils import make_ellipse_mask, build_control_map, to_3ch

def poisson_blend(dst_gray_u8, src_gray_u8, mask_u8, cx, cy):
    dst3 = to_3ch(dst_gray_u8)
    src3 = to_3ch(src_gray_u8)
    center = (int(cx), int(cy))
    blended = cv2.seamlessClone(src3, dst3, mask_u8, center, cv2.NORMAL_CLONE)
    return blended[:, :, 0]  # back to gray

def synth_one(i_clean, i_defect, defect_mask, class_id, rng,
              area_px, visibility, out_dir: Path, name: str):
    """
    i_clean/i_defect: uint8 gray (H,W)
    defect_mask: uint8 0/255 (H,W) 原始缺陷区域（用于扣残差）
    area_px: 目标合成缺陷面积（像素）
    visibility: 0..1 合成缺陷可见度
    """
    h, w = i_clean.shape

    # 1) 扣残差（只在原缺陷mask内取）
    r = (i_defect.astype(np.float32) - i_clean.astype(np.float32))
    r = r * (defect_mask.astype(np.float32) / 255.0)

    # 2) 采样一个目标 mask（用椭圆做最小可跑版本；你后续可换成 blob/多边形）
    #    位置点：这里用原缺陷mask的质心（也可随机）
    ys, xs = np.where(defect_mask > 0)
    if len(xs) == 0:
        return False
    cx = float(xs.mean())
    cy = float(ys.mean())
    mask_u8 = make_ellipse_mask(h, w, cx, cy, area_px=area_px, rng=rng)

    # 3) 强弱控制：缩放残差
    r_scaled = r * float(visibility)

    # 4) 构造 src（只在 mask 内加残差）
    m = (mask_u8.astype(np.float32) / 255.0)
    src = i_clean.astype(np.float32) + r_scaled * m
    src = np.clip(src, 0, 255).astype(np.uint8)

    # 5) 泊松融合
    i_synth = poisson_blend(i_clean, src, mask_u8, cx, cy)

    # 6) control map (6ch)
    area_norm = min(1.0, float(area_px) / float(h * w))
    vis_norm = float(np.clip(visibility, 0.0, 1.0))
    ctrl = build_control_map(mask_u8, class_id=class_id, area_norm=area_norm, vis_norm=vis_norm)

    # 7) 保存
    sample_dir = out_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample_dir / "clean.png"), i_clean)
    cv2.imwrite(str(sample_dir / "defect.png"), i_synth)
    cv2.imwrite(str(sample_dir / "mask.png"), mask_u8)
    np.save(str(sample_dir / "control.npy"), ctrl)

    meta = dict(class_id=int(class_id), area_px=int(area_px), visibility=float(visibility),
                cx=float(cx), cy=float(cy), area_norm=area_norm, vis_norm=vis_norm)
    (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
