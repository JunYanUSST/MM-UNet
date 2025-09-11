import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

# ---------- 自定义可视化函数 ----------

def tensor_to_image(tensor, normalize=True):
    """将Tensor转换为可显示的图像（通道平均）"""
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
    tensor = tensor.mean(dim=1, keepdim=True)  # 平均通道
    return tensor.squeeze(0).squeeze(0).detach().cpu().numpy()

def save_feature_map(feature, save_path, title=None):
    """保存单张特征图"""
    image = tensor_to_image(feature)
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='viridis')
    if title:
        plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------- 可视化主流程 ----------

def visualize_MRI(mri_module, skip_tensor, up_tensor, save_dir='vis/MRI'):
    """
    输入跳跃连接特征与上采样特征，展示 MRI 模块的特征交互过程
    """
    os.makedirs(save_dir, exist_ok=True)
    # 尺寸对齐
    if skip_tensor.shape[-2:] != up_tensor.shape[-2:]:
        up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:], mode='bilinear', align_corners=False)
    # 1. 保存 skip 和 up 特征图
    save_feature_map(skip_tensor, os.path.join(save_dir, 'skip.png'), title='Skip Feature')
    save_feature_map(up_tensor, os.path.join(save_dir, 'up.png'), title='Up-sampled Feature')

    # 2. 保存乘性交互图
    multiplicative = skip_tensor * up_tensor
    save_feature_map(multiplicative, os.path.join(save_dir, 'multiplicative.png'), title='Multiplicative Interaction')

    # 3. MRI 模块输出
    with torch.no_grad():
        mri_out = mri_module(skip_tensor, up_tensor)
    save_feature_map(mri_out, os.path.join(save_dir, 'mri_output.png'), title='MRI Output')

    print(f"✅ MRI 可视化结果已保存至：{save_dir}")

# ---------- 示例调用方式 ----------
if __name__ == "__main__":
    from model.unet_starfusion import MRIBlock  # 替换为你的 MRI 模块路径
    import cv2
    from torchvision import transforms

    # 示例数据读取（假设输入为 BUSI 图像和 mask）
    img_path = r"D:\UFKSNet\data\BUSI\train\images\benign (300).png"   # 替换为实际路径
    mask_path = r"D:\UFKSNet\data\BUSI\train\masks\benign (300)_mask.png"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img_tensor = torch.tensor(img / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    # 模拟 skip 和 up 特征（可改为来自真实模型的中间输出）
    dummy_conv = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1).cuda()
    skip_feat = dummy_conv(img_tensor)
    up_feat = F.interpolate(dummy_conv(img_tensor), scale_factor=2, mode='bilinear', align_corners=False)

    # MRI 模块实例
    mri = MRIBlock(use_residual=True, activate='relu').cuda()
    visualize_MRI(mri, skip_feat, up_feat)