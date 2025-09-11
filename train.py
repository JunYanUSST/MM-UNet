# train.py
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torchvision
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.optim import AdamW
from core.losses.dice_bce_loss import DiceLoss,FocalTverskyLoss,ComboLoss,compute_sid_losses,DiceBCELoss,MultiOutputLoss
from core.metrics.metrics import dice_coef, iou_coef, precision, recall, f1_score,compute_hd95, jaccard_index
from data.loader import get_ddti_train_val_loaders,get_busi_train_val_loaders,get_tnscui_train_val_loaders,get_tg_train_val_loaders
from tools.plot_curve import plot_metrics
from tools.model_stats import print_model_stats
from utils.edge_region import compute_edge_map, compute_region_weight

from model.blindwaveukan import UKAN
def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train():

    epochs = 200
    lr = 1e-4
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UKAN(num_classes=1).to(device)
    model.apply(weights_init_kaiming)

    criterion = ComboLoss(weight_focal=0.5, weight_dice=0.5)
    # criterion = DiceBCELoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    print_model_stats(model, input_size=(1, 3, 256, 256), device=device)
    writer = SummaryWriter(log_dir="logs/tensorboard")

    data_root = r"D:\UFKSNet\data\BUSI"

    train_loader, val_loader = get_busi_train_val_loaders(data_root, batch_size=batch_size, shuffle=True, drop_last=True)

    train_losses = []
    train_dices = []
    test_dices = []
    best_dice = 0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0


        for img, mask in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]"):
            img, mask = img.to(device), mask.to(device)

            if torch.isnan(img).any() or torch.isinf(img).any():
                print("Invalid input detected, skipping batch")
                continue
            preds = model(img)


            edge_map = compute_edge_map(mask)
            region_weight = compute_region_weight(mask)

            region_weight = (region_weight / region_weight.max()).clamp(min=1e-6)
            pixel_weight = (1 + edge_map * region_weight).detach()
            pixel_weight = pixel_weight.clamp(min=1.0, max=5.0)  # 再次限制范围，防止过大
            preds = F.interpolate(preds, size=mask.shape[2:], mode='bilinear', align_corners=True)
            bce_loss = F.binary_cross_entropy_with_logits(preds, mask,pixel_weight)
            dice_loss = criterion(preds, mask)
            loss = 0.5 * dice_loss + 0.5 * bce_loss

            if not isinstance(preds, tuple):
                preds = (preds,)
            # 计算损失
            # loss = criterion(preds, mask)

            if torch.isnan(loss).any():
                print("NaN loss detected, skipping backward")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()

            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)


        model.eval()
        # 评估训练集 Dice
        train_dice_total = 0
        with torch.no_grad():
            for img, mask in train_loader:
                img, mask = img.to(device), mask.to(device)

                preds = model(img)
                train_dice_total += dice_coef(preds, mask)
        train_dices.append(train_dice_total / len(train_loader))

        # 评估测试集
        dice_total, iou_total = 0, 0
        prec_total, rec_total, f1_total = 0, 0, 0

        with torch.no_grad():
            for idx, (img, mask) in enumerate(val_loader):  # ✅ 修复后
                img, mask = img.to(device), mask.to(device)
                preds = model(img)



                dice_total += dice_coef(preds, mask)
                iou_total += iou_coef(preds, mask)
                prec_total += precision(preds, mask)
                rec_total += recall(preds, mask)
                f1_total += f1_score(preds, mask)

                if idx == 0 and ((epoch + 1) % 5 == 0 or epoch == epochs - 1):
                    img_vis = img[0].detach().cpu()
                    mask_vis = mask[0].detach().cpu()
                    pred_vis = torch.sigmoid(preds[0]).detach().cpu()
                    pred_bin = (pred_vis > 0.5).float()

                    grid = torchvision.utils.make_grid(
                        [img_vis, mask_vis.expand_as(img_vis), pred_bin.expand_as(img_vis)],
                        nrow=3, normalize=True
                    )
                    writer.add_image(f"Prediction/sample_epoch{epoch + 1}", grid, epoch)

            avg_dice = dice_total / len(val_loader)
            avg_iou = iou_total / len(val_loader)
            avg_prec = prec_total / len(val_loader)
            avg_rec = rec_total / len(val_loader)
            avg_f1 = f1_total / len(val_loader)
            avg_hd95 = compute_hd95(preds, mask)
            avg_jaccard = jaccard_index(preds, mask)

        test_dices.append(avg_dice)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Dice/test", avg_dice, epoch)
        writer.add_scalar("IoU/test", avg_iou, epoch)
        writer.add_scalar("Precision/test", avg_prec, epoch)
        writer.add_scalar("Recall/test", avg_rec, epoch)
        writer.add_scalar("F1/test", avg_f1, epoch)
        writer.add_scalar("HD95/test", avg_hd95, epoch)


        print(f"\n Epoch {epoch+1}: TrainLoss={avg_loss:.4f} | TestDice={avg_dice:.4f} | TestIoU={avg_iou:.4f}")
        print(f" Precision={avg_prec:.4f} | Recall={avg_rec:.4f} | F1={avg_f1:.4f}")


        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Best model saved!")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            os.makedirs("logs", exist_ok=True)
            epochs_range = list(range(1, len(train_losses) + 1))

            plot_metrics(
                x=epochs_range,
                y_list=[train_losses],
                labels=["Train Loss"],
                title="Training Loss Curve",
                ylabel="Loss",
                save_path="logs/loss_curve.png"
            )

            plot_metrics(
                x=epochs_range,
                y_list=[train_dices, test_dices],
                labels=["Train Dice", "Test Dice"],
                title="Dice Score Curve",
                ylabel="Dice",
                save_path="logs/dice_curve.png"
            )

        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    train()