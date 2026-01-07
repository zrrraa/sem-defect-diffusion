from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class SEMSynthDataset(Dataset):
    def __init__(self, root="data/synth", size=512):
        self.root = Path(root)
        self.size = int(size)
        self.items = sorted([p for p in self.root.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.items)

    def _rand_crop_params(self, h, w):
        if h == self.size and w == self.size:
            return 0, 0
        top = np.random.randint(0, max(1, h - self.size + 1))
        left = np.random.randint(0, max(1, w - self.size + 1))
        return top, left

    def __getitem__(self, i):
        p = self.items[i]
        clean = cv2.imread(str(p / "clean.png"), cv2.IMREAD_GRAYSCALE)
        defect = cv2.imread(str(p / "defect.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(p / "mask.png"), cv2.IMREAD_GRAYSCALE)
        ctrl = np.load(str(p / "control.npy")).astype(np.float32)  # (6,H,W)

        h, w = clean.shape
        top, left = self._rand_crop_params(h, w)

        clean = clean[top:top+self.size, left:left+self.size]
        defect = defect[top:top+self.size, left:left+self.size]
        mask = mask[top:top+self.size, left:left+self.size]
        ctrl = ctrl[:, top:top+self.size, left:left+self.size]

        # 3ch：把灰度复制成 RGB（SD 期望 3 通道）
        clean = np.stack([clean]*3, axis=0)  # (3,H,W)
        defect = np.stack([defect]*3, axis=0)

        clean = torch.from_numpy(clean).float() / 127.5 - 1.0   # [-1,1]
        defect = torch.from_numpy(defect).float() / 127.5 - 1.0
        mask = torch.from_numpy((mask > 127).astype(np.float32))[None, :, :]  # (1,H,W) in {0,1}
        ctrl = torch.from_numpy(ctrl)  # (6,H,W) in [0,1]

        # masked_image：inpainting 常用做法是把 mask 区域抹掉（保持区域乘 1-mask）
        masked_clean = clean * (1.0 - mask)

        return {
            "pixel_values": defect,          # target
            "init_pixel_values": clean,      # init image
            "mask": mask,                    # 1 means "edit here" :contentReference[oaicite:6]{index=6}
            "masked_init": masked_clean,
            "control": ctrl,
            "prompt": "",                    # 你也可放 "sem" 或 类别名
        }
