import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.control_utils import build_control_map, to_3ch


def _bbox_from_mask(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _centroid_from_mask(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _largest_component(mask_u8: np.ndarray):
    mask_bin = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return (mask_bin * 255).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    return (labels == k).astype(np.uint8) * 255


def _paste_center(canvas: np.ndarray, patch: np.ndarray, cx: float, cy: float):
    H, W = canvas.shape[:2]
    h, w = patch.shape[:2]

    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    x1 = x0 + w
    y1 = y0 + h

    px0 = 0
    py0 = 0
    px1 = w
    py1 = h

    if x0 < 0:
        px0 = -x0
        x0 = 0
    if y0 < 0:
        py0 = -y0
        y0 = 0
    if x1 > W:
        px1 = w - (x1 - W)
        x1 = W
    if y1 > H:
        py1 = h - (y1 - H)
        y1 = H

    if x0 >= x1 or y0 >= y1 or px0 >= px1 or py0 >= py1:
        return canvas

    canvas[y0:y1, x0:x1] = patch[py0:py1, px0:px1]
    return canvas


def _poisson_blend(dst_gray_u8, src_gray_u8, mask_u8):
    dst_gray_u8 = dst_gray_u8.astype(np.uint8)
    src_gray_u8 = src_gray_u8.astype(np.uint8)
    mask = (mask_u8 > 0).astype(np.uint8) * 255

    x, y, w, h = cv2.boundingRect(mask)
    if w <= 0 or h <= 0:
        return dst_gray_u8

    dst3 = to_3ch(dst_gray_u8)
    src3 = to_3ch(src_gray_u8)

    src_crop = src3[y:y + h, x:x + w]
    mask_crop = mask[y:y + h, x:x + w]
    center = (x + w // 2, y + h // 2)

    blended = cv2.seamlessClone(src_crop, dst3, mask_crop, center, cv2.NORMAL_CLONE)
    return blended[:, :, 0]


def _scale_shape_and_residual(
    i_clean: np.ndarray,
    i_defect: np.ndarray,
    defect_mask_u8: np.ndarray,
    target_area_px: int,
    max_iters: int = 2,
):
    H, W = i_clean.shape[:2]
    target_area_px = int(np.clip(target_area_px, 10, H * W - 1))

    m0 = _largest_component(defect_mask_u8)
    orig_area = int((m0 > 0).sum())
    if orig_area <= 0:
        return None, None, None

    c0 = _centroid_from_mask(m0)
    if c0 is None:
        return None, None, None
    cx0, cy0 = c0

    r_full = (i_defect.astype(np.float32) - i_clean.astype(np.float32))
    r_full *= (m0.astype(np.float32) / 255.0)

    bb0 = _bbox_from_mask(m0)
    if bb0 is None:
        return None, None, None
    x0, y0, x1, y1 = bb0
    m_patch = (m0[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    r_patch = r_full[y0:y1, x0:x1].astype(np.float32)

    s = float(np.sqrt(target_area_px / float(orig_area)))

    target_mask = np.zeros((H, W), dtype=np.uint8)
    residual_canvas = np.zeros((H, W), dtype=np.float32)

    for _ in range(max_iters):
        new_w = max(1, int(round(m_patch.shape[1] * s)))
        new_h = max(1, int(round(m_patch.shape[0] * s)))

        m_rs = cv2.resize(m_patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        r_rs = cv2.resize(r_patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        r_rs *= (m_rs > 0).astype(np.float32)

        target_mask.fill(0)
        residual_canvas.fill(0.0)
        _paste_center(target_mask, m_rs, cx0, cy0)
        _paste_center(residual_canvas, r_rs, cx0, cy0)

        actual_area = int((target_mask > 0).sum())
        if actual_area <= 0:
            break
        s = s * float(np.sqrt(target_area_px / float(actual_area)))

    c1 = _centroid_from_mask(target_mask)
    if c1 is None:
        c1 = (cx0, cy0)

    return target_mask, residual_canvas, c1


def synth_one(
    i_clean: np.ndarray,
    i_defect: np.ndarray,
    defect_mask_u8: np.ndarray,
    class_id: int,
    area_px: int,
    visibility: float,
    out_dir: Path,
    name: str,
    vis_ref: float = 3.0,
):
    if i_clean is None or i_defect is None or defect_mask_u8 is None:
        return False
    if i_clean.shape != i_defect.shape or i_clean.shape != defect_mask_u8.shape:
        return False

    H, W = i_clean.shape[:2]
    scaled = _scale_shape_and_residual(i_clean, i_defect, defect_mask_u8, int(area_px), max_iters=2)
    if scaled[0] is None:
        return False

    target_mask_u8, residual_canvas, (cx, cy) = scaled

    src = i_clean.astype(np.float32) + float(visibility) * residual_canvas
    src_u8 = np.clip(src, 0, 255).astype(np.uint8)
    i_synth = _poisson_blend(i_clean, src_u8, target_mask_u8)

    area_norm = float((target_mask_u8 > 0).sum() / (H * W))
    vis_norm = float(np.clip(float(visibility) / float(vis_ref), 0.0, 1.0))
    ctrl = build_control_map(target_mask_u8, int(class_id), area_norm, vis_norm).astype(np.float32)

    sample_dir = Path(out_dir) / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample_dir / "clean.png"), i_clean)
    cv2.imwrite(str(sample_dir / "defect.png"), i_synth)
    cv2.imwrite(str(sample_dir / "mask.png"), target_mask_u8)
    np.save(str(sample_dir / "control.npy"), ctrl)

    meta = dict(
        class_id=int(class_id),
        area_px=int((target_mask_u8 > 0).sum()),
        target_area_px=int(area_px),
        visibility=float(visibility),
        cx=float(cx),
        cy=float(cy),
        area_norm=float(area_norm),
        vis_norm=float(vis_norm),
        vis_ref=float(vis_ref),
    )
    (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _parse_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _process_one(item: dict, out_dir: str, ratios: list[float], vis: list[float], vis_ref: float) -> int:
    i_def = cv2.imread(item["defect_path"], cv2.IMREAD_GRAYSCALE)
    i_cln = cv2.imread(item["clean_path"], cv2.IMREAD_GRAYSCALE)
    m_def = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
    if i_def is None or i_cln is None or m_def is None:
        return 0

    H, W = i_cln.shape[:2]
    class_id = int(item["class_id"])
    base_id = str(item["id"])

    total = 0
    for ri, area_ratio in enumerate(ratios):
        area_px = int(round(float(area_ratio) * H * W))
        area_px = int(np.clip(area_px, 10, H * W - 1))
        for vj, visibility in enumerate(vis):
            name = f"{base_id}_c{class_id}_r{ri:02d}_v{vj:02d}"
            ok = synth_one(i_cln, i_def, m_def, class_id, area_px, float(visibility), Path(out_dir), name, vis_ref)
            total += int(ok)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=str, default="data/raw/annotations.jsonl")
    ap.add_argument("--out", type=str, default="data/synth")
    ap.add_argument("--ratios", type=str, default="0.005,0.01,0.02,0.03,0.04,0.05")
    ap.add_argument("--vis", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--vis_ref", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunksize", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios = _parse_list(args.ratios)
    vis = _parse_list(args.vis)

    items = []
    for line in Path(args.ann).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))

    total = 0
    with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_worker_init) as ex:
        it = ex.map(
            _process_one,
            items,
            [str(out_dir)] * len(items),
            [ratios] * len(items),
            [vis] * len(items),
            [float(args.vis_ref)] * len(items),
            chunksize=max(1, int(args.chunksize)),
        )
        for n in tqdm(it, total=len(items)):
            total += int(n)

    print("done, total:", total)


if __name__ == "__main__":
    main()
