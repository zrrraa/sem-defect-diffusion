import argparse
import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from peft import LoraConfig

from src.sem_dataset import SEMSynthDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_inpaint_model", type=str, default="/home/share/rzhong/model/stable-diffusion-inpainting")
    p.add_argument("--controlnet_dir", type=str, default="outputs/controlnet_6ch")
    p.add_argument("--data_root", type=str, default="data/synth")
    p.add_argument("--output_dir", type=str, default="outputs/lora_unet_sem")

    p.add_argument("--resolution", type=int, default=480)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--adapter_name", type=str, default="default")
    p.add_argument("--weight_name", type=str, default="pytorch_lora_weights.safetensors")

    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true")

    p.add_argument("--use_safetensors", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp(device: torch.device, mixed_precision: str):
    if mixed_precision == "no":
        return contextlib.nullcontext, None, torch.float32

    if mixed_precision == "fp16":
        if device.type != "cuda":
            return contextlib.nullcontext, None, torch.float32
        return (lambda: torch.autocast("cuda", dtype=torch.float16)), torch.cuda.amp.GradScaler(enabled=True), torch.float16

    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and (not torch.cuda.is_bf16_supported()):
        return (lambda: torch.autocast("cuda", dtype=torch.float16)), torch.cuda.amp.GradScaler(enabled=True), torch.float16
    return (lambda: torch.autocast(device.type, dtype=torch.bfloat16)), None, torch.bfloat16


class PromptEncoder:
    def __init__(self, tokenizer, text_encoder, device):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.cache = {}

    @torch.no_grad()
    def __call__(self, prompts):
        out = []
        for p in prompts:
            if p not in self.cache:
                inputs = self.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                h = self.text_encoder(inputs.input_ids.to(self.device))[0]
                self.cache[p] = h.detach()
            out.append(self.cache[p])
        return torch.cat(out, dim=0)


def trainable_param_count(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    autocast_ctx, scaler, weight_dtype = get_amp(device, args.mixed_precision)
    use_safetensors = True if args.use_safetensors else False

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_inpaint_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_inpaint_model, subfolder="text_encoder", use_safetensors=use_safetensors
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_inpaint_model, subfolder="vae", use_safetensors=use_safetensors
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_inpaint_model, subfolder="unet", use_safetensors=use_safetensors
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_inpaint_model, subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir)

    text_encoder.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)
    unet.to(device=device, dtype=weight_dtype)
    controlnet.to(device=device, dtype=weight_dtype)

    text_encoder.eval()
    vae.eval()
    controlnet.eval()

    for m in (text_encoder, vae, controlnet):
        m.requires_grad_(False)

    unet.requires_grad_(False)
    lora_cfg = LoraConfig(
        r=int(args.rank),
        lora_alpha=int(args.lora_alpha) if args.lora_alpha is not None else int(args.rank),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg, adapter_name=args.adapter_name)
    unet.set_adapter(args.adapter_name)
    unet.enable_lora()
    unet.train()

    for p in unet.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()

    trainable, total = trainable_param_count(unet)
    print(f"Trainable params: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")

    optimizer = torch.optim.AdamW((p for p in unet.parameters() if p.requires_grad), lr=args.lr)

    ds = SEMSynthDataset(root=args.data_root, size=args.resolution)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    prompt_encoder = PromptEncoder(tokenizer, text_encoder, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        for batch in dl:
            micro_step += 1

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)   # (B,3,H,W) in [-1,1]
            masked_init = batch["masked_init"].to(device, non_blocking=True)     # (B,3,H,W) in [-1,1]
            mask = batch["mask"].to(device, non_blocking=True)                   # (B,1,H,W) in {0,1}
            control = batch["control"].to(device, non_blocking=True)             # (B,6,H,W) in [0,1]
            prompts = batch["prompt"]

            with autocast_ctx():
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    masked_latents = vae.encode(masked_init).latent_dist.sample() * 0.18215
                    encoder_hidden_states = prompt_encoder(prompts)

                mask_latents = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)

                with torch.no_grad():
                    down_samples, mid_sample = controlnet(
                        model_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=control,
                        conditioning_scale=1.0,
                        return_dict=False,
                    )

                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / max(1, args.grad_accum_steps)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if micro_step % max(1, args.grad_accum_steps) == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in unet.parameters() if p.requires_grad), float(args.max_grad_norm)
                    )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if global_step % int(args.log_every) == 0:
                    print(f"step={global_step} loss={(loss.item()*max(1,args.grad_accum_steps)):.6f}")

                global_step += 1
                if global_step >= args.max_steps:
                    break

    unet.save_lora_adapter(
        save_directory=str(out_dir),
        adapter_name=args.adapter_name,
        safe_serialization=True,
        weight_name=args.weight_name,
        upcast_before_saving=True,
    )
    print("saved LoRA to", str(out_dir))


if __name__ == "__main__":
    main()
