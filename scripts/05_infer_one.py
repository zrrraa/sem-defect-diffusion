import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import ControlNetModel, UniPCMultistepScheduler
from src.pipeline_6ch import StableDiffusionControlNetInpaintPipeline6Ch
from src.control_utils import user_to_mask_and_control


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_inpaint_model", type=str,
                   default="/home/share/rzhong/model/stable-diffusion-inpainting")
    p.add_argument("--controlnet_dir", type=str, default="outputs/controlnet_6ch")
    p.add_argument("--lora_unet_dir", type=str, default="outputs/lora_unet_sem")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out", type=str, default="out.png")
    p.add_argument("--x", type=int, required=True)
    p.add_argument("--y", type=int, required=True)
    p.add_argument("--area_ratio", type=float, default=0.0009,
                   help="defect area ratio in (0,1), e.g. 0.001 means 0.1%% of image pixels")
    p.add_argument("--visibility", type=float, default=1.0)
    p.add_argument("--class_id", type=int, default=0)
    p.add_argument("--strength", type=float, default=0.3)
    p.add_argument("--control_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--dilate_kernel", type=int, default=15)
    p.add_argument("--dilate_iters", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline6Ch.from_pretrained(
        args.pretrained_inpaint_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    print("UNet in_channels:", pipe.unet.config.in_channels)
    print("UNet conv_in weight in:", pipe.unet.conv_in.weight.shape[1])
    print("ControlNet in_channels:", pipe.controlnet.config.in_channels)
    print("ControlNet conv_in weight in:", pipe.controlnet.conv_in.weight.shape[1])

    vae = pipe.vae
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    img3 = np.stack([img] * 3, -1)
    x = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    x = x.to(device=pipe.device, dtype=pipe.vae.dtype)

    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mean
        rec = vae.decode(lat).sample
    rec = (rec.clamp(-1, 1) + 1) * 127.5
    rec = rec[0].permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)
    cv2.imwrite("vae_recon.png", rec[:, :, 0])

    from pathlib import Path

    lora_dir = Path(args.lora_unet_dir)
    weight_name = "pytorch_lora_weights.safetensors"
    if not (lora_dir / weight_name).exists():
        safes = sorted(lora_dir.glob("*.safetensors"))
        if not safes:
            raise FileNotFoundError(f"No .safetensors found in {lora_dir}")
        weight_name = safes[0].name

    pipe.unet.load_lora_adapter(
        str(lora_dir),
        weight_name=weight_name,
        adapter_name="sem",
        prefix=None,
    )
    pipe.unet.set_adapters("sem")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()

    pipe.unet.to(device)
    pipe.unet.eval()

    print("UNet in_channels:", pipe.unet.config.in_channels)
    print("UNet conv_in:", pipe.unet.conv_in.weight.shape)
    print("ControlNet in_channels:", pipe.controlnet.config.in_channels)
    print("ControlNet conv_in:", pipe.controlnet.conv_in.weight.shape)

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.image)

    h, w = img.shape

    area_ratio = float(args.area_ratio)
    if not (0.0 < area_ratio < 1.0):
        raise ValueError(f"--area_ratio must be in (0,1), got {area_ratio}")

    area_px = int(round(area_ratio * h * w))
    area_px = max(area_px, 10)
    area_px = min(area_px, h * w - 1)

    mask, ctrl = user_to_mask_and_control(
        h, w,
        args.x, args.y,
        area_px,
        args.visibility,
        args.class_id,
        seed=args.seed
    )

    img_rgb = np.stack([img] * 3, axis=-1)

    mask_u8 = (mask[0].numpy() * 255).astype(np.uint8)
    print("mask coverage:", (mask_u8 > 0).mean())
    print("mask min/max/mean:", mask_u8.min(), mask_u8.max(), mask_u8.mean())
    cv2.imwrite("mask_debug.png", mask_u8)

    k = int(args.dilate_kernel)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_dil_u8 = cv2.dilate(mask_u8, kernel, iterations=int(args.dilate_iters))

    ctrl_cpu = ctrl.detach().cpu()
    print("ctrl range:", float(ctrl_cpu.min()), float(ctrl_cpu.max()), float(ctrl_cpu.mean()))

    img_pil = Image.fromarray(img_rgb).convert("RGB")
    mask_pil = Image.fromarray(mask_u8).convert("L")

    out = pipe(
        prompt="",
        image=img_pil,
        mask_image=mask_pil,
        control_image=ctrl.unsqueeze(0).to(device=device, dtype=torch.float16),
        height=int(h),
        width=int(w),
        strength=float(args.strength),
        controlnet_conditioning_scale=float(args.control_scale),
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
    ).images[0]

    gen = np.array(out)[:, :, 0].astype(np.uint8)
    orig = img.astype(np.uint8)

    m = mask_dil_u8 > 0
    final = orig.copy()
    final[m] = gen[m]

    cv2.imwrite(args.out, final)

    print(f"saved {args.out} | HxW={h}x{w} area_ratio={area_ratio:.6f} area_px={area_px} "
          f"vis={args.visibility} class_id={args.class_id} dilate_k={k} dilate_iters={int(args.dilate_iters)}")


if __name__ == "__main__":
    main()
