import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob

class TNSCUI2020Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.bmp")))
        self.mask_dir = mask_dir
        self.transform = transform

        # 只保留那些有对应掩膜的图像
        self.valid_image_paths = [
            img_path for img_path in self.image_paths
            if os.path.exists(os.path.join(mask_dir, os.path.basename(img_path)))
        ]

        print(f"[TNSCUI2020Dataset] Matched {len(self.valid_image_paths)} images with masks.")

    def __len__(self):
        return len(self.valid_image_paths)

    def __getitem__(self, idx):
        image_path = self.valid_image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
