import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- SimAM ----------------
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - x_mean).pow(2)
        var = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        # y = x_minus_mu_square / (4 * var) + 0.5
        if not torch.isfinite(var).all():
            print("[SIMAM ERROR] var has NaN or Inf", var)

        y = x_minus_mu_square / (4 * var) + 0.5
        if not torch.isfinite(y).all():
            print("[SIMAM ERROR] y has NaN or Inf", y)

        out = x * self.activation(y)
        if not torch.isfinite(out).all():
            print("[SIMAM ERROR] output has NaN or Inf", out)


        return x * self.activation(y)


# ---------------- EdgeEnhance ----------------
class EdgeEnhance(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 使用 Laplacian 构造简单边缘提示（或 Sobel 可选）
        edge = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        return self.conv(x + edge)

# ---------------- ASPP + SimAM Block ----------------
class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[2,4,8,16]):
        super(ASPPBlock, self).__init__()

        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            branch = [
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            # 仅在大感受野分支加 SimAM
            if rate >= 4:
                branch.append(SimAM())
            self.branches.append(nn.Sequential(*branch))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d((len(dilation_rates) + 1) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.edge = EdgeEnhance(out_channels)

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=False)
        feats.append(gp)
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.edge(x)
        return x







