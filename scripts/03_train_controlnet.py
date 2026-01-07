import argparse
import contextlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel

from src.sem_dataset import SEMSynthDataset


@dataclass(frozen=True)
class TrainCfg:
    pretrained_inpaint_model: str
    data_root: str
    output_dir: str
    resolution: int
    batch_size: int
    lr: float
    max_steps: int
    grad_accum_steps: int
    mixed_precision: str
    num_workers: int
    pin_memory: bool
    seed: int
    use_safetensors: bool
    log_every: int
    save_every: int
    max_grad_norm: float
    snr_gamma: float
    tf32: bool


def parse_args() -> TrainCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_inpaint_model", type=str, default="/home/share/rzhong/model/stable-diffusion-inpainting")
    p.add_argument("--data_root", type=str, default="data/synth")
    p.add_argument("--output_dir", type=str, default="outputs/controlnet_6ch")
    p.add_argument("--resolution", type=int, default=480)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_steps", type=int, default=12000)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_safetensors", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--snr_gamma", type=float, default=0.0)
    p.add_argument("--tf32", action="store_true")
    a = p.parse_args()
    return TrainCfg(**vars(a))


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % 2**32
    random.seed(base_seed + worker_id)


def resolve_device_and_dtype(cfg: TrainCfg) -> Tuple[torch.device, torch.dtype, contextlib.AbstractContextManager, torch.cuda.amp.GradScaler]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if device.type != "cuda" and cfg.mixed_precision in ("fp16", "bf16"):
        return device, torch.float32, contextlib.nullcontext(), None

    if cfg.mixed_precision == "no":
        return device, torch.float32, contextlib.nullcontext(), None

    if cfg.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        return device, torch.float16, torch.autocast(device_type="cuda", dtype=torch.float16), scaler

    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        return device, torch.float16, torch.autocast(device_type="cuda", dtype=torch.float16), scaler

    return device, torch.bfloat16, torch.autocast(device_type=device.type, dtype=torch.bfloat16), None


class PromptCache:
    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def encode(self, prompts: List[str]) -> torch.Tensor:
        uniq = list(dict.fromkeys(prompts))
        missing = [p for p in uniq if p not in self._cache]
        if missing:
            inputs = self.tokenizer(
                missing,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            ids = inputs.input_ids.to(self.device)
            embeds = self.text_encoder(ids)[0]
            for p, e in zip(missing, embeds):
                self._cache[p] = e.detach()
        out = torch.stack([self._cache[p] for p in prompts], dim=0)
        return out


def snr_weights(scheduler: DDPMScheduler, timesteps: torch.Tensor, gamma: float, prediction_type: str) -> torch.Tensor:
    if gamma <= 0:
        return torch.ones_like(timesteps, dtype=torch.float32)

    alphas_cumprod = scheduler.alphas_cumprod.to(device=timesteps.device)
    a = alphas_cumprod[timesteps].float()
    snr = a / (1.0 - a)

    g = torch.full_like(snr, float(gamma))
    if prediction_type == "v_prediction":
        w = torch.minimum(snr, g) / (snr + 1.0)
    else:
        w = torch.minimum(snr, g) / snr
    return w


def save_model(controlnet: ControlNetModel, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(out_dir)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device, weight_dtype, autocast_ctx, scaler = resolve_device_and_dtype(cfg)
    use_safetensors = bool(cfg.use_safetensors)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_inpaint_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_inpaint_model, subfolder="text_encoder", torch_dtype=weight_dtype, use_safetensors=use_safetensors
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_inpaint_model, subfolder="vae", torch_dtype=weight_dtype, use_safetensors=use_safetensors
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_inpaint_model, subfolder="unet", torch_dtype=weight_dtype, use_safetensors=use_safetensors
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_inpaint_model, subfolder="scheduler")

    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=6).to(device)

    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()
    controlnet.train()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=cfg.lr)

    ds = SEMSynthDataset(root=cfg.data_root, size=cfg.resolution)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    prompt_cache = PromptCache(tokenizer, text_encoder, device)

    accum = max(1, cfg.grad_accum_steps)
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")

    while global_step < cfg.max_steps:
        for batch in dl:
            micro_step += 1

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            masked_init = batch["masked_init"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            control = batch["control"].to(device, non_blocking=True)

            prompts = batch["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts] * pixel_values.shape[0]

            with autocast_ctx:
                with torch.inference_mode():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    masked_latents = vae.encode(masked_init).latent_dist.sample() * 0.18215

                mask_latents = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)

                encoder_hidden_states = prompt_cache.encode(list(prompts)).to(dtype=model_input.dtype)

                down_samples, mid_sample = controlnet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control.to(dtype=model_input.dtype),
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

                per = (noise_pred.float() - noise.float()).pow(2).mean(dim=(1, 2, 3))
                w = snr_weights(noise_scheduler, timesteps, cfg.snr_gamma, prediction_type)
                loss = (per * w).mean() / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if micro_step % accum == 0:
                if cfg.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(controlnet.parameters(), cfg.max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if global_step % cfg.log_every == 0:
                    print(f"step={global_step} loss={loss.item():.6f}")

                if cfg.save_every > 0 and (global_step > 0) and (global_step % cfg.save_every == 0):
                    save_model(controlnet, cfg.output_dir)

                global_step += 1
                if global_step >= cfg.max_steps:
                    break

    save_model(controlnet, cfg.output_dir)
    print("saved to", cfg.output_dir)


if __name__ == "__main__":
    main()
