import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    def __init__(self, in_channels, scale_factor=2, up_kernel=5, compressed_channels=64, encoder_kernel=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.kernel_area = up_kernel * up_kernel

        # 通道压缩
        self.channel_compress = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)

        # 内容编码器，输出采样核权重（未归一化）
        self.encoder = nn.Conv2d(
            compressed_channels,
            (scale_factor ** 2) * self.kernel_area,
            kernel_size=encoder_kernel,
            padding=encoder_kernel // 2
        )

    def forward(self, x):
        B, C, H, W = x.size()
        sf = self.scale_factor
        up_H, up_W = H * sf, W * sf

        # Step 1: 压缩通道
        x_compressed = self.channel_compress(x)  # [B, Cc, H, W]

        # Step 2: 预测动态上采样权重（未归一化）
        mask = self.encoder(x_compressed)  # [B, sf^2 * K^2, H, W]

        # Step 3: 展开为 [B, sf^2, K^2, H, W] 并 softmax 归一化
        mask = mask.view(B, sf * sf, self.kernel_area, H, W)
        mask = F.softmax(mask, dim=2)

        # Step 4: 展开输入特征为 im2col 形式
        unfold = F.unfold(x, kernel_size=self.up_kernel, padding=self.up_kernel // 2)  # [B, C * K^2, H*W]
        unfold = unfold.view(B, C, self.kernel_area, H, W)  # [B, C, K^2, H, W]

        # Step 5: 对每个位置用 mask 加权求和（模拟 CARAFE 重组）
        out = torch.einsum("bckhw, bqkhw -> bqchw", unfold, mask)  # [B, sf^2, C, H, W]

        # Step 6: reshape 到空间维度（pixel shuffle 反操作）
        out = out.view(B, sf, sf, C, H, W).permute(0, 3, 4, 1, 5, 2)  # [B, C, H, sf, W, sf]
        out = out.reshape(B, C, up_H, up_W)

        return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Function
# from torch.nn.modules.module import Module
#
# import carafe_ext, carafe_naive_ext
#
#
# def xavier_init(module, gain=1, bias=0, distribution='normal'):
#     assert distribution in ['uniform', 'normal']
#     if distribution == 'uniform':
#         nn.init.xavier_uniform_(module.weight, gain=gain)
#     else:
#         nn.init.xavier_normal_(module.weight, gain=gain)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)
#
#
# def normal_init(module, mean=0, std=1, bias=0):
#     nn.init.normal_(module.weight, mean, std)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)
#
#
# class CARAFENaiveFunction(Function):
#
#     @staticmethod
#     def forward(ctx, features, masks, kernel_size, group_size, scale_factor):
#         assert scale_factor >= 1
#         assert masks.size(1) == kernel_size * kernel_size * group_size
#         assert masks.size(-1) == features.size(-1) * scale_factor
#         assert masks.size(-2) == features.size(-2) * scale_factor
#         assert features.size(1) % group_size == 0
#         assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
#         ctx.kernel_size = kernel_size
#         ctx.group_size = group_size
#         ctx.scale_factor = scale_factor
#         ctx.feature_size = features.size()
#         ctx.mask_size = masks.size()
#
#         n, c, h, w = features.size()
#         output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
#         if features.is_cuda:
#             carafe_naive_ext.forward(features, masks, kernel_size, group_size,
#                                      scale_factor, output)
#         else:
#             raise NotImplementedError
#
#         if features.requires_grad or masks.requires_grad:
#             ctx.save_for_backward(features, masks)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         assert grad_output.is_cuda
#
#         features, masks = ctx.saved_tensors
#         kernel_size = ctx.kernel_size
#         group_size = ctx.group_size
#         scale_factor = ctx.scale_factor
#
#         grad_input = torch.zeros_like(features)
#         grad_masks = torch.zeros_like(masks)
#         carafe_naive_ext.backward(grad_output.contiguous(), features, masks,
#                                   kernel_size, group_size, scale_factor,
#                                   grad_input, grad_masks)
#
#         return grad_input, grad_masks, None, None, None
#
#
# carafe_naive = CARAFENaiveFunction.apply
#
#
# class CARAFENaive(Module):
#
#     def __init__(self, kernel_size, group_size, scale_factor):
#         super(CARAFENaive, self).__init__()
#
#         assert isinstance(kernel_size, int) and isinstance(
#             group_size, int) and isinstance(scale_factor, int)
#         self.kernel_size = kernel_size
#         self.group_size = group_size
#         self.scale_factor = scale_factor
#
#     def forward(self, features, masks):
#         return CARAFENaiveFunction.apply(features, masks, self.kernel_size,
#                                          self.group_size, self.scale_factor)
#
#
# class CARAFEFunction(Function):
#
#     @staticmethod
#     def forward(ctx, features, masks, kernel_size, group_size, scale_factor):
#         assert scale_factor >= 1
#         assert masks.size(1) == kernel_size * kernel_size * group_size
#         assert masks.size(-1) == features.size(-1) * scale_factor
#         assert masks.size(-2) == features.size(-2) * scale_factor
#         assert features.size(1) % group_size == 0
#         assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
#         ctx.kernel_size = kernel_size
#         ctx.group_size = group_size
#         ctx.scale_factor = scale_factor
#         ctx.feature_size = features.size()
#         ctx.mask_size = masks.size()
#
#         n, c, h, w = features.size()
#         output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
#         routput = features.new_zeros(output.size(), requires_grad=False)
#         rfeatures = features.new_zeros(features.size(), requires_grad=False)
#         rmasks = masks.new_zeros(masks.size(), requires_grad=False)
#         if features.is_cuda:
#             carafe_ext.forward(features, rfeatures, masks, rmasks, kernel_size,
#                                group_size, scale_factor, routput, output)
#         else:
#             raise NotImplementedError
#
#         if features.requires_grad or masks.requires_grad:
#             ctx.save_for_backward(features, masks, rfeatures)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         assert grad_output.is_cuda
#
#         features, masks, rfeatures = ctx.saved_tensors
#         kernel_size = ctx.kernel_size
#         group_size = ctx.group_size
#         scale_factor = ctx.scale_factor
#
#         rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
#         rgrad_input_hs = torch.zeros_like(grad_output, requires_grad=False)
#         rgrad_input = torch.zeros_like(features, requires_grad=False)
#         rgrad_masks = torch.zeros_like(masks, requires_grad=False)
#         grad_input = torch.zeros_like(features, requires_grad=False)
#         grad_masks = torch.zeros_like(masks, requires_grad=False)
#         carafe_ext.backward(grad_output.contiguous(), rfeatures, masks,
#                             kernel_size, group_size, scale_factor,
#                             rgrad_output, rgrad_input_hs, rgrad_input,
#                             rgrad_masks, grad_input, grad_masks)
#         return grad_input, grad_masks, None, None, None, None
#
#
# carafe = CARAFEFunction.apply
#
#
# class CARAFE(Module):
#     """ CARAFE: Content-Aware ReAssembly of FEatures
#
#     Please refer to https://arxiv.org/abs/1905.02188 for more details.
#
#     Args:
#         kernel_size (int): reassemble kernel size
#         group_size (int): reassemble group size
#         scale_factor (int): upsample ratio
#
#     Returns:
#         upsampled feature map
#     """
#
#     def __init__(self, kernel_size, group_size, scale_factor):
#         super(CARAFE, self).__init__()
#
#         assert isinstance(kernel_size, int) and isinstance(
#             group_size, int) and isinstance(scale_factor, int)
#         self.kernel_size = kernel_size
#         self.group_size = group_size
#         self.scale_factor = scale_factor
#
#     def forward(self, features, masks):
#         return CARAFEFunction.apply(features, masks, self.kernel_size,
#                                     self.group_size, self.scale_factor)