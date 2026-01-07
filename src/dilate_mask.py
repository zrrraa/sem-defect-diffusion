# dilate_mask.py
import argparse
from pathlib import Path
import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=str, help="input mask.png")
    ap.add_argument("out", type=str, help="output dilated_mask.png")
    ap.add_argument("--radius", type=int, default=5, help="dilation radius in pixels (approx rings), default=5")
    ap.add_argument("--threshold", type=int, default=127, help="binarize threshold, default=127")
    ap.add_argument("--kernel", choices=["ellipse", "rect", "cross"], default="ellipse",
                    help="structuring element shape, default=ellipse")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    r = max(0, int(args.radius))

    img = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {inp}")

    # 1) 二值化：确保 mask 是 0/255
    mask = (img > args.threshold).astype(np.uint8) * 255

    # 2) 构造结构元素（核）
    ksize = 2 * r + 1
    if r == 0:
        dil = mask
    else:
        shape_map = {
            "ellipse": cv2.MORPH_ELLIPSE,
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
        }
        kernel = cv2.getStructuringElement(shape_map[args.kernel], (ksize, ksize))
        # 用半径 r 的核做一次膨胀，效果≈往外扩 r 像素
        dil = cv2.dilate(mask, kernel, iterations=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), dil)
    print(f"Saved: {out}  (radius={r}, kernel={args.kernel})")

if __name__ == "__main__":
    main()
