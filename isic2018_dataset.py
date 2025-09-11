import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from glob import glob

class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = []
        self.valid_image_paths = []
        self.mask_dir = mask_dir
        self.transform = transform

        for img_path in self.image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, base + "_segmentation.png")
            if os.path.exists(mask_path):
                self.valid_image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        print(f" Matched {len(self.valid_image_paths)} images with masks.")

    def __len__(self):
        return len(self.valid_image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.valid_image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32)  # ✅ 转为 0/1 float32 掩膜

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]                   # [3, H, W]
            mask = augmented["mask"].unsqueeze(0)        # [1, H, W]
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
