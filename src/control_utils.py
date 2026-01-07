import math
import numpy as np
import cv2
import torch

def make_ellipse_mask(h, w, cx, cy, area_px, rng, min_ar=0.3, max_ar=3.0):
    """
    生成一个面积约等于 area_px 的椭圆 mask（uint8: 0/255）
    """
    area_px = max(10, int(area_px))
    ar = float(np.exp(rng.uniform(np.log(min_ar), np.log(max_ar))))  # aspect ratio
    # area = pi * a * b, with a = ar*b
    b = math.sqrt(area_px / (math.pi * ar))
    a = ar * b
    ax = max(1, int(round(a)))
    by = max(1, int(round(b)))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (ax, by), rng.uniform(0, 180), 0, 360, 255, -1)
    # soft edge
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)
    mask = (mask > 32).astype(np.uint8) * 255
    return mask

def build_control_map(mask_u8, class_id, area_norm, vis_norm, num_classes=4):
    """
    返回 control: (6, H, W), float32 in [0,1]
    """
    h, w = mask_u8.shape
    m = (mask_u8.astype(np.float32) / 255.0)[None, :, :]  # (1,H,W)

    ctrl = np.zeros((num_classes + 2, h, w), dtype=np.float32)
    ctrl[class_id, :, :] = m[0]  # one-hot only inside mask
    ctrl[num_classes + 0, :, :] = m[0] * float(area_norm)
    ctrl[num_classes + 1, :, :] = m[0] * float(vis_norm)
    return ctrl

def to_3ch(img_gray_u8):
    if img_gray_u8.ndim == 2:
        return np.stack([img_gray_u8]*3, axis=-1)
    return img_gray_u8

def user_to_mask_and_control(h, w, x, y, area_px, visibility, class_id, seed=0):
    rng = np.random.default_rng(seed)
    mask_u8 = make_ellipse_mask(h, w, x, y, area_px=area_px, rng=rng)
    area_norm = min(1.0, float(area_px) / float(h * w))
    vis_norm = float(np.clip(visibility, 0.0, 1.0))
    ctrl = build_control_map(mask_u8, class_id=int(class_id), area_norm=area_norm, vis_norm=vis_norm)
    # torch
    mask = torch.from_numpy((mask_u8 > 127).astype(np.float32))[None, :, :]   # (1,H,W)
    ctrl = torch.from_numpy(ctrl)                                            # (6,H,W)
    return mask, ctrl