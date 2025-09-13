# MM-UNet: Ultrasound Image Segmentation

This repository provides the official PyTorch implementation of **MM-UNet**, a novel network for ultrasound medical image segmentation.  
MM-UNet integrates:
- **Multiplicative Residual Interaction (MRI)** module – strengthens boundary modeling through cross-scale feature interaction.  
- **Multi-Scale Attention Edge-Enhancement (MSAE)** module – improves sensitivity to multi-scale lesions and boundary preservation.  

✅ Outperforms state-of-the-art baselines on **BUSI**, **Dataset B**, and **DDTI** datasets.  
✅ Supports end-to-end training and evaluation.  

---
## 🚀 Quick Start
```bash
git clone https://github.com/JunYanUSST/MM-UNet.git
cd MM-UNet
python train.py 
