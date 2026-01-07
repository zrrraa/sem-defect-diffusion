import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from diffusers import ControlNetModel, UniPCMultistepScheduler
from src.pipeline_6ch import StableDiffusionControlNetInpaintPipeline6Ch
from src.control_utils import build_control_map


VIS_REF = 1.0
DILATE_KERNEL = 1
DILATE_ITERS = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_inpaint_model", type=str, default="/home/share/rzhong/model/stable-diffusion-inpainting")
    p.add_argument("--controlnet_dir", type=str, default="outputs/controlnet_6ch")
    p.add_argument("--lora_unet_dir", type=str, default="outputs/lora_unet_sem")
    p.add_argument("--lora_weight_name", type=str, default="pytorch_lora_weights.safetensors")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def _to_gray_u8(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img is None:
        return None
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _extract_bg_and_mask(editor_value):
    if editor_value is None:
        return None, None
    bg = editor_value.get("background", None)
    layers = editor_value.get("layers", []) or []

    layer = None
    for x in reversed(layers):
        if x is not None:
            layer = x
            break

    bg_u8 = _to_gray_u8(bg)
    if layer is None:
        return bg_u8, None

    if isinstance(layer, Image.Image):
        layer = np.array(layer)

    if layer.ndim == 2:
        m = layer > 0
    else:
        if layer.shape[2] == 4:
            m = layer[:, :, 3] > 0
        else:
            m = layer.mean(axis=2) > 0

    mask_u8 = (m.astype(np.uint8) * 255)
    return bg_u8, mask_u8


def _load_pipe(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=dtype)
    pipe = StableDiffusionControlNetInpaintPipeline6Ch.from_pretrained(
        args.pretrained_inpaint_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    lora_dir = Path(args.lora_unet_dir)
    weight_name = args.lora_weight_name
    if not (lora_dir / weight_name).exists():
        safes = sorted(lora_dir.glob("*.safetensors"))
        if not safes:
            raise FileNotFoundError(f"No .safetensors found in {lora_dir}")
        weight_name = safes[0].name

    pipe.unet.load_lora_adapter(str(lora_dir), weight_name=weight_name, adapter_name="sem", prefix=None)
    pipe.unet.set_adapters("sem")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()

    pipe.unet.eval()
    pipe.controlnet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    return pipe, device, dtype


def make_app(pipe, device, dtype):
    class_map = {"hole": 0, "particle": 1, "scratch": 2}

    def _dilate(mask_u8: np.ndarray):
        k = int(DILATE_KERNEL)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask_u8, kernel, iterations=int(DILATE_ITERS))

    @torch.inference_mode()
    def run(editor_value, defect_class, visibility, seed):
        orig_u8, mask_u8 = _extract_bg_and_mask(editor_value)
        if orig_u8 is None:
            return None, None, "请先上传图片。"
        if mask_u8 is None or int((mask_u8 > 0).sum()) == 0:
            return None, None, "请在图上涂抹生成 mask。"

        h, w = orig_u8.shape
        area_norm = float((mask_u8 > 0).sum() / (h * w))
        vis_norm = float(np.clip(float(visibility) / float(VIS_REF), 0.0, 1.0))
        class_id = int(class_map[defect_class])

        ctrl = build_control_map(mask_u8, class_id=class_id, area_norm=area_norm, vis_norm=vis_norm).astype(np.float32)
        if ctrl.ndim == 3 and ctrl.shape[0] != 6 and ctrl.shape[-1] == 6:
            ctrl = np.transpose(ctrl, (2, 0, 1))
        if ctrl.shape[0] != 6:
            raise ValueError(f"control map shape must be (6,H,W), got {ctrl.shape}")

        ctrl_t = torch.from_numpy(ctrl).unsqueeze(0).to(device=device, dtype=dtype)

        img_rgb = np.stack([orig_u8] * 3, axis=-1)
        img_pil = Image.fromarray(img_rgb).convert("RGB")
        mask_pil = Image.fromarray(mask_u8).convert("L")

        gen_torch = torch.Generator(device=device).manual_seed(int(seed)) if device != "cpu" else None
        out = pipe(
            prompt="",
            image=img_pil,
            mask_image=mask_pil,
            control_image=ctrl_t,
            height=int(h),
            width=int(w),
            strength=1.0,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=100,
            guidance_scale=1.0,
            generator=gen_torch,
        ).images[0]

        gen_u8 = np.array(out)[:, :, 0].astype(np.uint8)

        mask_dil_u8 = _dilate(mask_u8)
        m = mask_dil_u8 > 0
        final_u8 = orig_u8.copy()
        final_u8[m] = gen_u8[m]

        final_pil = Image.fromarray(final_u8).convert("L")
        mask_show = Image.fromarray(mask_dil_u8).convert("L")

        info = (
            f"HxW={h}x{w} class_id={class_id} vis={float(visibility):.3f} "
            f"area_norm={area_norm:.6f} vis_norm={vis_norm:.3f} "
            f"dilate_k={int(DILATE_KERNEL)} iters={int(DILATE_ITERS)}"
        )
        return final_pil, mask_show, info

    with gr.Blocks() as demo:
        gr.Markdown("## SEM Defect Diffusion Demo")

        editor = gr.ImageEditor(
            type="numpy",
            image_mode="RGBA",
            label="上传原图并涂抹 mask（涂抹决定位置和面积）",
            height=560,
        )

        with gr.Row():
            defect_class = gr.Dropdown(choices=["hole", "particle", "scratch"], value="hole", label="class")
            visibility = gr.Number(value=1.0, label="visibility")
            seed = gr.Number(value=0, precision=0, label="seed")
            btn = gr.Button("生成")

        with gr.Row():
            out_img = gr.Image(type="pil", label="输出")
            mask_img = gr.Image(type="pil", label="mask")

        info = gr.Textbox(label="info", interactive=False)

        btn.click(run, inputs=[editor, defect_class, visibility, seed], outputs=[out_img, mask_img, info])

    return demo


def main():
    args = parse_args()
    pipe, device, dtype = _load_pipe(args)
    demo = make_app(pipe, device, dtype)
    demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
