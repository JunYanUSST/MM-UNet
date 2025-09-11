# test.py
import torch
import os
import numpy as np
from tqdm import tqdm
from data.loader import get_busi_test_loader,get_ddti_test_loader
from data.loader import get_test_loader_from_isic2016
from core.metrics.metrics import dice_coef, iou_coef, precision, recall, f1_score,compute_hd95
from tools.vis_prediction import visualize_sample
from data.loader import get_test_loader_isic2018,get_busi_test_loader,get_tnscui_test_loader,get_tg_test_loader
from model.unet_starfusion import UNetStarFusion
# from model.ukan import UKAN

from model.hwdukan import UKAN
@torch.no_grad()
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UKAN(num_classes=1).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device), strict=False)
    model.eval()

    test_root = r"D:/UFKSNet/data/BUSI"
    test_loader = get_busi_test_loader(test_root)

    # 初始化指标列表（用于均值 ± 标准差）
    dice_total = iou_total = prec_total = rec_total = f1_total = 0
    hd95_total = 0

    os.makedirs("test_results_hwdukan_busi", exist_ok=True)

    print(" Running test evaluation...")
    for idx, (img, mask) in enumerate(tqdm(test_loader)):
        img, mask = img.to(device), mask.to(device)
        preds = model(img)
        dice_total += dice_coef(preds, mask)
        iou_total += iou_coef(preds, mask)
        prec_total += precision(preds, mask)
        rec_total += recall(preds, mask)
        f1_total += f1_score(preds, mask)
        hd95_total += compute_hd95(preds, mask)


        save_path = f"test_results_hwdukan_busi/{idx:03d}.png"
        visualize_sample(img[0], mask[0], preds[0], save_path)


    N = len(test_loader)
    print(f"\n Final Test Results  ({N} samples):")
    print(f"Dice     : {dice_total / N:.4f}")
    print(f"IoU      : {iou_total / N:.4f}")
    print(f"Precision: {prec_total / N:.4f}")
    print(f"Recall   : {rec_total / N:.4f}")
    print(f"F1-Score : {f1_total / N:.4f}")
    print(f"HD95     : {hd95_total / N:.4f}")


if __name__ == "__main__":
    test()
