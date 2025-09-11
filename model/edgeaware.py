import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EdgeAwareDown, self).__init__()
        # 下采样卷积
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # Sobel edge detector（固定权重）
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_kernel_y = sobel_kernel_x.transpose(-1, -2)
        self.register_buffer('sobel_x', sobel_kernel_x)
        self.register_buffer('sobel_y', sobel_kernel_y)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + 1, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down = self.down_conv(x)
        # 提取结构边缘图
        edge_x = F.conv2d(x, self.sobel_x.expand(x.size(1), -1, -1, -1), groups=x.size(1), padding=1)
        edge_y = F.conv2d(x, self.sobel_y.expand(x.size(1), -1, -1, -1), groups=x.size(1), padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6).mean(dim=1, keepdim=True)
        edge = F.interpolate(edge, size=down.shape[2:], mode='bilinear', align_corners=True)
        # 融合结构感知图
        out = torch.cat([down, edge], dim=1)
        return self.fuse(out)