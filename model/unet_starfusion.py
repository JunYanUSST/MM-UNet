import torch
import torch.nn as nn
import torch.nn.functional as F
from model.act import TeLU,P_TeLU



# ---------------- SimAM ----------------
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_mean = x.mean(dim=[2, 3], keepdim=True).detach()
        x_minus_mu_square = (x - x_mean).pow(2)
        # var = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        # y = x_minus_mu_square / (4 * var) + 0.5
        # 防止除零和负值
        var = (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / max(n, 1)) + self.e_lambda
        var = var.clamp(min=1e-8)  # 确保方差不为零
        # 限制计算范围
        y = (x_minus_mu_square / (4 * var + 1e-8)) + 0.5
        y = y.clamp(0, 1)  # 限制在 [0, 1] 范围内

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

# ---------------- MSAB Block ----------------
class MSABBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[2,4,8,12]):
        super(MSABBlock, self).__init__()

        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            branch = [
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)

            ]
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


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,activate='telu'):
        super(DoubleConv, self).__init__()
        if activate == 'relu':
            act_layer = nn.ReLU(inplace=False)
        elif activate == 'telu':
            act_layer = TeLU()
        elif activate == 'ptelu':
            act_layer = P_TeLU(alpha=0.8, beta=1.0, learnable_alpha=True, learnable_beta=True)
        else:
            raise ValueError("Unsupported activation type")

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=False),
            act_layer,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=False)
            act_layer
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, activate='relu'):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch,activate='relu')

    def forward(self, x):
        return self.conv(self.down(x))

class MRIBlock(nn.Module):
    def __init__(self, channels,norm='instance',use_residual=True, activate='relu'):
        super().__init__()
        self.use_residual = use_residual

        if activate == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activate == 'gelu':
            self.act = nn.GELU()
        elif activate == 'telu':
            self.act = TeLU()
        elif activate == 'ptelu':
            self.act = P_TeLU(alpha=0.8, beta=1.0, learnable_alpha=True, learnable_beta=True)
        else:
            raise ValueError("Unsupported activation type")
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(channels)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(channels)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout2d(0.1)
    def forward(self, x, skip):
        x_fused = self.act(self.norm(skip)) * x
        x_fused = self.dropout(x_fused + x)

        return x_fused

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=False, use_cbam=True,norm_layer='instance', channels=None):
        super(Up, self).__init__()
        if norm_layer == 'batch':
            self.norm = nn.BatchNorm2d(channels) if channels else nn.Identity()
        elif norm_layer == 'instance':
            self.norm = nn.InstanceNorm2d(channels) if channels else nn.Identity()
        else:
            self.norm = nn.Identity()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.fusion = MRIBlock(channels=out_ch, norm='instance', activate='relu')
        self.cbam = CBAMLayer(out_ch) if use_cbam else nn.Identity()
        self.conv = nn.Sequential(DoubleConv(out_ch+skip_ch, out_ch,activate='relu'),nn.Dropout2d(0.2))

    def forward(self, x, skip):
        x_up = self.up(x)
        x_skip = self.norm(skip)
        fused = self.cbam(self.fusion(x_up, x_skip))
        x = torch.cat([fused, skip], dim=1)
        return self.conv(x)




class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetStarFusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32, bilinear=False):
        super(UNetStarFusion, self).__init__()


        self.inc = DoubleConv(in_channels, base_ch, activate='telu')
        self.down1 = Down(base_ch, base_ch * 2,activate='telu')
        self.down2 = Down(base_ch * 2, base_ch * 4,activate='telu')
        self.down3 = Down(base_ch * 4, base_ch * 8,activate='telu')
        self.down4 = Down(base_ch * 8, base_ch * 8,activate='telu')

        self.bridge = MSABBlock(base_ch * 8, base_ch * 16)  # [B, base_c*16, H/16, W/16]


        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8, bilinear)
        self.up2 = Up(base_ch*8, base_ch*4, base_ch*4, bilinear)
        self.up3 = Up(base_ch*4, base_ch*2, base_ch*2, bilinear)
        self.up4 = Up(base_ch*2, base_ch, base_ch, bilinear)

        self.outc = OutConv(base_ch, out_channels)



    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        b = self.bridge(x5)

        x = self.up1(b, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x