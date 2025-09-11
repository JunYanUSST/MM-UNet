import os
from torch.utils.data import DataLoader, random_split
from data.isic2016_dataset import ISIC2016Dataset
from data.isic2018_dataset import ISIC2018Dataset
from data.busidataset import BUSIDataset
from data.carpladataset import CPDataset
from data.DDTIdataset import DDTIDataset
from data.TNSCUIdataset import TNSCUI2020Dataset
from data.tg3kdataset import tgDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from albumentations.core.transforms_interface import DualTransform

# ------------------------- ISIC2016 Dataset Loader -------------------------
def get_dataloaders_isic2016(data_root, batch_size=4, test_ratio=0.2):
    image_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset = ISIC2016Dataset(image_dir, mask_dir, transform=transform)

    print(f" Loaded {len(dataset)} image-mask pairs.")
    assert len(dataset) > 0, " Dataset is empty!"

    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_test_loader_from_isic2016(test_root, batch_size=1):
    image_dir = os.path.join(test_root, "images")
    mask_dir = os.path.join(test_root, "masks")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset = ISIC2016Dataset(image_dir, mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ------------------------- ISIC2018 Dataset Loader -------------------------
def get_dataloaders_isic2018(root, batch_size=4, shuffle=True):
    train_img_dir = os.path.join(root, 'ISIC2018_Task1-2_Training_Input')
    train_mask_dir = os.path.join(root, 'ISIC2018_Task1_Training_GroundTruth')

    val_img_dir = os.path.join(root, 'ISIC2018_Task1-2_Validation_Input')
    val_mask_dir = os.path.join(root, 'ISIC2018_Task1_Validation_GroundTruth')

    # ✅ 数据增强 transform（只用于训练集）
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # ✅ 验证集 transform（不做增强）
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    train_dataset = ISIC2018Dataset(train_img_dir, train_mask_dir, transform=train_transform)
    val_dataset = ISIC2018Dataset(val_img_dir, val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader_isic2018(root):
    test_img_dir = os.path.join(root, 'ISIC2018_Task1-2_Test_Input')
    test_mask_dir = os.path.join(root, 'ISIC2018_Task1_Test_GroundTruth')

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test_dataset = ISIC2018Dataset(test_img_dir, test_mask_dir, transform=transform)
    return DataLoader(test_dataset, batch_size=4, shuffle=False)

# ------------------------- BUSI Dataset Loader -------------------------

def get_busi_train_val_loaders(data_root, batch_size=4, val_ratio=0.2, shuffle=True,drop_last=True):
    img_dir = os.path.join(data_root, "train", "images")
    mask_dir = os.path.join(data_root, "train", "masks")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    full_dataset = BUSIDataset(img_dir, mask_dir, transform=transform,target_size=(256, 256))
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=2, pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,drop_last=False, num_workers=2)

    return train_loader, val_loader

def get_busi_test_loader(data_root, batch_size=1):
    img_dir = os.path.join(data_root, "test", "images")
    mask_dir = os.path.join(data_root, "test", "masks")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test_dataset = BUSIDataset(img_dir, mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader

# ------------------------- CAR PLA  Dataset Loader -------------------------
def get_dataloaders_carpla(data_root, batch_size=4, test_ratio=0.2):
    image_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "mask")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset = CPDataset(image_dir, mask_dir, transform=transform)
    print(f"[get_dataloaders_carpla] Loaded {len(dataset)} image-mask pairs.")
    assert len(dataset) > 0, "Dataset is empty!"

    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_test_loader_carpla(test_root, batch_size=1):
    image_dir = os.path.join(test_root, "images")
    mask_dir = os.path.join(test_root, "mask")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset = CPDataset(image_dir, mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
# ------------------------- DDTI  Dataset Loader -------------------------
def get_ddti_train_val_loaders(data_root, batch_size=4, val_ratio=0.2):
    img_dir = os.path.join(data_root, "train", "images")
    mask_dir = os.path.join(data_root, "train", "masks")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    full_dataset = DDTIDataset(img_dir, mask_dir, transform=transform)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def get_ddti_test_loader(data_root, batch_size=1):
    img_dir = os.path.join(data_root, "test", "images")
    mask_dir = os.path.join(data_root, "test", "masks")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test_dataset = DDTIDataset(img_dir, mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader

# ------------------------- TN-SCUI2020 Dataset Loader -------------------------

def get_tnscui_train_val_loaders(data_root, batch_size=4, val_ratio=0.2):
    img_dir = os.path.join(data_root, "train", "images")
    mask_dir = os.path.join(data_root, "train", "masks")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    full_dataset = TNSCUI2020Dataset(img_dir, mask_dir, transform=transform)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def get_tnscui_test_loader(data_root, batch_size=1):
    img_dir = os.path.join(data_root, "test", "images")
    mask_dir = os.path.join(data_root, "test", "masks")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test_dataset = TNSCUI2020Dataset(img_dir, mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader

# ------------------------- tg Dataset Loader -------------------------

def get_tg_train_val_loaders(data_root, batch_size=4, val_ratio=0.2):
    img_dir = os.path.join(data_root, "train", "images")
    mask_dir = os.path.join(data_root, "train", "masks")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    full_dataset = tgDataset(img_dir, mask_dir, transform=transform)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def get_tg_test_loader(data_root, batch_size=1):
    img_dir = os.path.join(data_root, "test", "images")
    mask_dir = os.path.join(data_root, "test", "masks")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test_dataset = tgDataset(img_dir, mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader

