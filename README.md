# sem-defect-diffusion

A controllable defect synthesis pipeline for SEM images using **Stable Diffusion Inpainting** + **ControlNet** (6-channel control map) and optional **UNet LoRA** domain adaptation.

![video](https://github.com/zrrraa/sem-defect-diffusion/blob/master/video.gif)

This repository provides scripts to:
- build paired **(defect, clean, mask)** data using **LaMa** inpainting,
- generate a controllable synthetic dataset with **area/visibility/class** control,
- train a **6-channel ControlNet** and a **UNet LoRA** adapter,
- run single-image inference and an interactive **Gradio** demo.

## Key Dependencies (External)
- LaMa (image inpainting): https://github.com/advimman/lama
- Diffusers (inpainting, ControlNet, LoRA adapters):  
  - Inpainting guide: https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint  
  - ControlNet API: https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet  
  - Loading adapters / LoRA: https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters
- Gradio ImageEditor (paintable mask UI): https://www.gradio.app/docs/gradio/imageeditor

---

## 1 Setup

### 1.1 Pull LaMa into `third_party/`
Clone LaMa into `third_party/lama`:
```bash
mkdir -p third_party
cd third_party
git clone https://github.com/advimman/lama.git
```

### 1.2 Create two Python environments

You will need **two virtual environments**:

1. **LaMa environment**: follow the official LaMa instructions (often pinned/older dependencies).
2. **Diffusion environment**: for training/inference with Diffusers (newer PyTorch/Python).

Rationale: LaMa and Diffusers stacks may have incompatible Python/dependency requirements.

## 2 Data Preparation

### 2.1 Mask dilation and LaMa inpainting

Currently, `scripts/01_run_lama.sh` is used for **mask dilation** only. LaMa inpainting is executed manually afterwards.

```bash
bash scripts/01_run_lama.sh

cd third_party/lama
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)
python3 bin/predict.py \
  model.path=/home/share/rzhong/model/big-lama \
  indir=/home/rzhong/project/sem-defect-diffusion/data/lama_inpaint/hole \
  outdir=$(pwd)/output
```

### 2.2 Organize files and build annotations

Place defect images, masks, and LaMa-inpainted clean images under `data/`, then run `test/annotations_make.sh` to generate `annotations.jsonl`.

Recommended directory structure: (I did not add the "pattern_deform" category)

```text
data/
  lama_inpaint/
    hole/
    particle/
    scratch/
  raw/
    defect/
      hole/
      particle/
      scratch/
    mask/
      hole/
      particle/
      scratch/
  annotations.jsonl
  synth/
```

## 3 Synthetic Dataset Generation

Generate controllable synthetic samples into `data/synth`:

```bash
PYTHONPATH=. python scripts/02_make_synth_dataset.py
```

## 4 Training

Training is two-stage:

* **Stage 1**: train **ControlNet** on the 6-channel control map.
* **Stage 2**: freeze ControlNet; train **UNet LoRA** for domain adaptation.

### 4.1 Train ControlNet

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/03_train_controlnet.py \
  --pretrained_inpaint_model /home/share/rzhong/model/stable-diffusion-inpainting \
  --data_root data/synth \
  --output_dir outputs/controlnet_6ch \
  --resolution 480 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 2e-5 \
  --max_steps 12000 \
  --mixed_precision fp16 \
  --num_workers 4 \
  --pin_memory \
  --save_every 2000 \
  --log_every 50 \
  --max_grad_norm 1.0 \
  --snr_gamma 5.0 \
  --seed 0
```

### 4.2 Train UNet LoRA (with frozen ControlNet)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/04_train_lora.py \
  --pretrained_inpaint_model /home/share/rzhong/model/stable-diffusion-inpainting \
  --controlnet_dir outputs/controlnet_6ch \
  --data_root data/synth \
  --output_dir outputs/lora_unet_sem \
  --resolution 480 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 1e-4 \
  --max_steps 7000 \
  --rank 8 \
  --lora_alpha 8 \
  --adapter_name default \
  --weight_name pytorch_lora_weights.safetensors \
  --mixed_precision fp16 \
  --max_grad_norm 1.0 \
  --num_workers 8 \
  --pin_memory \
  --tf32 \
  --seed 0
```

## 5 Inference

Single-image inference:

```bash
PYTHONPATH=. python scripts/05_infer_one.py \
  --image data/lama_inpaint/hole/000.png \
  --out out.png \
  --x 200 --y 180 \
  --area_ratio 0.01 \
  --visibility 1.0 \
  --class_id 0 \
  --strength 1.0 \
  --control_scale 1.0 \
  --steps 100
```

Class mapping:

* `class_id=0` → hole
* `class_id=1` → particle
* `class_id=2` → scratch

## 6 Interactive Demo (Gradio)

Launch the interactive demo:

```bash
PYTHONPATH=. python scripts/06_demo_gradio.py \
  --pretrained_inpaint_model /home/share/rzhong/model/stable-diffusion-inpainting \
  --controlnet_dir outputs/controlnet_6ch \
  --lora_unet_dir outputs/lora_unet_sem \
  --device cuda \
  --port 7860
```

The demo uses Gradio **ImageEditor** so you can paint a mask directly on the input image, specify class/visibility, and generate a defect sample.

## Notes

* `data/` and `outputs/` can be large; they are typically excluded from version control (see `.gitignore`).
* The repository assumes local paths for pretrained checkpoints; adjust paths to match your environment.
