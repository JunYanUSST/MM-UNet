import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from collections import defaultdict

class BUSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.image_to_masks = defaultdict(list)
        self.transform = transform
        self.target_size = target_size
        # 构建 image_name → [mask_path1, mask_path2, ...] 映射
        for mask_path in glob(os.path.join(mask_dir, "*_mask*.png")):
            base_name = os.path.basename(mask_path)
            image_base = base_name.split("_mask")[0] + ".png"
            self.image_to_masks[image_base].append(mask_path)

        # 仅保留那些在 image_to_masks 中存在的图像
        self.valid_image_paths = [
            path for path in self.image_paths if os.path.basename(path) in self.image_to_masks
        ]

        print(f"[BUSIDataset] Matched {len(self.valid_image_paths)} images with masks.")

    def __len__(self):
        return len(self.valid_image_paths)

    def __getitem__(self, idx):
        image_path = self.valid_image_paths[idx]
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        # 合并多个 mask（取最大值）
        merged_mask = None
        for mask_path in self.image_to_masks[image_name]:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size)
            mask = (mask > 127).astype(np.uint8)
            if merged_mask is None:
                merged_mask = mask
            else:
                merged_mask = np.maximum(merged_mask, mask)

        merged_mask = merged_mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=merged_mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(merged_mask).unsqueeze(0)

        return image, mask
