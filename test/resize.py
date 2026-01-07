# resize.py
import argparse
from pathlib import Path
from PIL import Image, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def get_resample():
    # 兼容新旧 Pillow：新版本用 Image.Resampling.LANCZOS，旧版本用 Image.LANCZOS
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS

def resize_one(img: Image.Image, size=(480, 480), mode="stretch"):
    resample = get_resample()
    if mode == "stretch":
        return img.resize(size, resample=resample)
    elif mode == "fit":
        # 保持长宽比 + 居中裁剪成目标尺寸
        return ImageOps.fit(img, size, method=resample, centering=(0.5, 0.5))
    elif mode == "contain":
        # 保持长宽比，缩放到能放进目标框（输出尺寸可能小于目标）
        return ImageOps.contain(img, size, method=resample)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="输入文件夹")
    ap.add_argument("output_dir", help="输出文件夹（建议不要和输入同一个）")
    ap.add_argument("--size", type=int, default=480, help="目标尺寸（正方形），默认 480")
    ap.add_argument("--mode", choices=["stretch", "fit", "contain"], default="stretch",
                    help="stretch=拉伸；fit=保持比例并裁剪；contain=保持比例但不补边")
    ap.add_argument("--recursive", action="store_true", help="递归处理子文件夹")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if args.recursive else "*"
    files = [p for p in in_dir.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS]

    if not files:
        print(f"[WARN] No images found in: {in_dir}")
        return

    target = (args.size, args.size)

    for p in files:
        rel = p.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(p)
            out_img = resize_one(img, size=target, mode=args.mode)
            out_img.save(out_path)  # 格式由扩展名决定 :contentReference[oaicite:3]{index=3}
            print(f"[OK] {p} -> {out_path}")
        except Exception as e:
            print(f"[ERR] {p}: {e}")

if __name__ == "__main__":
    main()
