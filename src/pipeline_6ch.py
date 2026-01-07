import torch
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetInpaintPipeline


def _match_batch(t: torch.Tensor, target_b: int) -> torch.Tensor:
    """
    让 t 的 batch 维度对齐到 target_b：
    - 常见情况：t 是 2B（CFG 已拼），controlnet 在 guess_mode 可能用 B -> 取前 B
    - 或 t 是 1 -> repeat 到 target_b
    - 或 target_b 是 t 的整数倍 -> repeat
    """
    if t is None:
        return None
    b = int(t.shape[0])
    if b == target_b:
        return t
    if b == 1:
        return t.repeat(target_b, 1, 1, 1)
    if b == 2 * target_b:
        # CFG 拼出来的两份通常相同，取前半即可
        return t[:target_b]
    if target_b % b == 0:
        return t.repeat(target_b // b, 1, 1, 1)
    # 兜底：裁剪
    return t[:target_b]


class StableDiffusionControlNetInpaintPipeline6Ch(StableDiffusionControlNetInpaintPipeline):
    """
    改动点：
    1) 支持 control_image 直接传 torch.Tensor (B,6,H,W)，跳过 RGB preprocess
    2) 修复 inpaint-controlnet 的 9/4 通道不匹配：若 controlnet 收到 4ch sample，自动拼 mask+masked_latents -> 9ch
    """

    # ---------- 1) 6ch control image ----------
    @torch.no_grad()
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        crops_coords,
        resize_mode,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        # 如果传进来的是 tensor（B,6,H,W），跳过默认的 RGB preprocess
        if isinstance(image, torch.Tensor):
            ctrl = image

            if ctrl.dim() == 3:
                ctrl = ctrl.unsqueeze(0)  # (1,C,H,W)

            if ctrl.dim() != 4:
                raise ValueError(f"control_image tensor must be (B,C,H,W), got {tuple(ctrl.shape)}")

            # 先推断 height/width（避免 None）
            if height is None:
                height = int(ctrl.shape[-2])
            if width is None:
                width = int(ctrl.shape[-1])

            # 对齐到 8 的倍数（SD 常规做法）
            height = (int(height) // 8) * 8
            width = (int(width) // 8) * 8

            ctrl = ctrl.to(dtype=torch.float32)
            if ctrl.shape[-2:] != (height, width):
                ctrl = F.interpolate(ctrl, size=(height, width), mode="bilinear", align_corners=False)

            # batch 对齐
            image_batch_size = int(ctrl.shape[0])
            repeat_by = batch_size if image_batch_size == 1 else num_images_per_prompt
            ctrl = ctrl.repeat_interleave(repeat_by, dim=0)

            ctrl = ctrl.to(device=device, dtype=dtype)

            # CFG: 非 guess_mode 时才双倍
            if do_classifier_free_guidance and not guess_mode:
                ctrl = torch.cat([ctrl, ctrl], dim=0)

            return ctrl

        # 否则走原版（PIL/np/RGB）
        return super().prepare_control_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=dtype,
            crops_coords=crops_coords,
            resize_mode=resize_mode,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )

    # ---------- 2) cache mask_latents & masked_image_latents ----------
    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        mask_latents, masked_image_latents = super().prepare_mask_latents(
            mask=mask,
            masked_image=masked_image,
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # 缓存下来，给 controlnet.forward 的补丁用
        self._cached_mask_latents = mask_latents
        self._cached_masked_image_latents = masked_image_latents
        return mask_latents, masked_image_latents

    # ---------- 3) patch controlnet forward during __call__ ----------
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        # 只在本次调用期间 patch，避免影响别的推理
        orig_forward = self.controlnet.forward

        def patched_forward(sample, *f_args, **f_kwargs):
            # 有些版本里，inpaint pipeline 会错误地给 controlnet 传 4ch latent
            # 但 inpaint controlnet conv_in 需要 9ch（4+1+4）
            in_ch = getattr(self.controlnet.config, "in_channels", None)
            if isinstance(sample, torch.Tensor) and in_ch == 9 and sample.shape[1] == 4:
                mask_latents = getattr(self, "_cached_mask_latents", None)
                masked_image_latents = getattr(self, "_cached_masked_image_latents", None)
                if mask_latents is None or masked_image_latents is None:
                    raise RuntimeError(
                        "ControlNet got 4ch sample but mask_latents cache is missing. "
                        "Did prepare_mask_latents run?"
                    )

                b = int(sample.shape[0])
                m = _match_batch(mask_latents, b).to(device=sample.device, dtype=sample.dtype)
                mi = _match_batch(masked_image_latents, b).to(device=sample.device, dtype=sample.dtype)

                sample = torch.cat([sample, m, mi], dim=1)  # -> (B,9,H,W)

            return orig_forward(sample, *f_args, **f_kwargs)

        self.controlnet.forward = patched_forward

        try:
            return super().__call__(*args, **kwargs)
        finally:
            self.controlnet.forward = orig_forward
            # 清理缓存
            if hasattr(self, "_cached_mask_latents"):
                delattr(self, "_cached_mask_latents")
            if hasattr(self, "_cached_masked_image_latents"):
                delattr(self, "_cached_masked_image_latents")
