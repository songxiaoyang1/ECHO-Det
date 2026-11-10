```markdown
# ECHO‑Det
> ECHO‑Det: An RGB‑D Insulator Defect Detection Network Integrating Edge–Frequency Enhancement and Hierarchical Directional Modeling for Extreme Environments

[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](./paper.pdf)

## 📖 Introduction
Transmission‑line insulator defect detection under extreme environments suffers from tilted targets, background clutter, large feature gaps between **Broken (structural damage)** and **Corrosion (surface corrosion)** defects.

**ECHO‑Det** is a multimodal oriented bounding box(OBB) detection framework for RGB‑D insulator defect detection:
- 🧊 Depth branch: **ODAL‑Stem**, enhances geometric boundary and orientation priors for structural damage.
- 🎨 RGB branch: **EFCM (Edge‑Frequency Complementary Module)**, enhance edge contours and high‑frequency texture features for corrosion defects.
- 🔗 Cross‑modal fusion: **DCGRM (Difference‑Consistency Guided Refinement Module)**, adaptively fuse complementary information of RGB and depth via difference‑consistency decomposition & soft three‑region mask cross‑attention.
- 🎯 Oriented detection head based on D‑FINE, fit elongated tilted insulators better than horizontal bounding box(HBB).

Test‑set performance: `mAP=90.5%`, Broken AP=90.8%, Corrosion AP=90.3%, F1‑score=90.5%.
ECHO‑Det maintains robust detection performance under haze, dust, rain‑snow, strong light, low‑light, shadow and occlusion degradation scenarios.

## 📁 Project Architecture
```
ECHO‑Det
├── configs/                 # yaml config for training/inference
│   └── echo_det.yaml
├── datasets/
│   └── synthidia_rgb_d/     # SYNTHIDIA RGB‑D dataset loader
├── models/
│   ├── backbone/
│   ├── odal_stem.py         # Orientation‑Decoupled Adaptive LoG Stem
│   ├── efcm.py              # Edge‑Frequency Complementary Module
│   ├── dcgrm.py             # Difference‑Consistency Guided Refinement Module
│   └── echo_det.py          # Full ECHO‑Det network
├── utils/                   # loss, aug, metric, visualization(Grad‑CAM)
├── train.py                 # training entry
├── test.py                  # evaluation entry
├── demo.py                  # inference demo
└── README.md
```

## ⚙️ Environment Requirements
```bash
# install dependencies
pip install torch torchvision
pip install opencv-python numpy tqdm pycocotools pillow
pip install einops
```
- Python >= 3.8
- PyTorch >=2.0
- CUDA >=11.7

## 📊 SYNTHIDIA RGB‑D Dataset
We build a multimodal oriented object detection dataset based on SYNTHIDIA Blender project files.
- Synchronously render aligned RGB images、depth maps(npy format)、OBB annotations
- Simulated extreme conditions: `normal, haze, dust, rain‑snow, strong‑illumination, low‑illumination, shadow, occlusion`
- Two defect categories: `Broken`, `Corrosion`
- Total samples: **7488**
  - Train: 5248 | Val:1496 | Test:744

> Dataset download: [Google Drive / Baidu Yun placeholder]
Dataset format:
```
synthidia_rgb_d/
├── train/
│   ├── rgb/        # rgb image *.png
│   ├── depth/      # normalized depth map *.npy
│   └── annotations/# obb annotation *.json
├── val/
└── test/
```

## 🚀 Quick Start — Inference Demo
```bash
python demo.py \
--config configs/echo_det.yaml \
--weights weights/echo_det_best.pth \
--rgb ./demo/rgb_sample.png \
--depth ./demo/depth_sample.npy \
--output ./demo/output
```

## 🏋️‍♂️ Train
Default training hyper‑parameters are consistent with the paper:
- Input size: `640×640`
- Optimizer: AdamW, lr=1e‑4, weight‑decay=1e‑4
- Epoch:132, warm‑up:500 iter, EMA decay=0.9999
- Batch‑size: train=8, val=16

```bash
python train.py --config configs/echo_det.yaml
```

## 🧪 Test & Evaluation
```bash
python test.py \
--config configs/echo_det.yaml \
--weights weights/echo_det_best.pth
```
Metrics: AP(Broken), AP(Corrosion), mAP@0.5, Precision, Recall, F1‑score, Params, FLOPs, FPS.

## 🧩 Core Modules
1. **ODAL‑Stem**: Depth‑branch input stem. Isotropic LoG + multi‑orientation decoupled LoG filters + orientation competition gating. Enhance depth discontinuity and orientation prior for Broken defects.
2. **EFCM**: RGB‑branch feature enhancement. Split semantic‑appearance dual stream; Sobel edge attention + Laplacian high‑frequency attention + dual‑pooling channel attention + pixel‑wise gating. Target for corrosion texture.
3. **DCGRM**: Cross‑modal fusion. Compute difference term `|g'−x'|` and consistency term `g'⊙x'`; soft three‑region mask prediction; region‑based cross‑attention adaptive feature refinement.

## 📈 Main Experimental Results
### Comparison with SOTA methods
|Method|Broken AP(%)|Corrosion AP(%)|mAP(%)|F1‑score(%)|Params(M)|FPS|
|---|---|---|---|---|---|---|
|DAGE‑YOLO|80.1|80.3|80.2|80.0|18.2|88|
|MV3‑YOLO|76.4|77.2|76.8|76.3|10.5|125|
|EEG‑DETR|84.5|83.9|84.2|84.1|38.4|32|
|ISO‑DeTr|82.7|82.3|82.5|82.2|35.6|36|
|UD‑Fine|87.6|87.2|87.4|87.3|29.8|50|
|HF‑D‑FINE|86.8|85.4|86.1|86.1|28.5|54|
|**ECHO‑Det(Ours)**|**90.8**|**90.3**|**90.5**|**90.5**|**32.5**|**43**|

### Core Module Ablation
|Model|ODAL‑Stem|EFCM|DCGRM|Broken AP(%)|Corrosion AP(%)|mAP(%)|
|---|---|---|---|---|---|---|
|Baseline|×|×|×|86.7|86.1|86.4|
|M1|✓|×|×|88.9|86.5|87.8|
|M2|×|✓|×|87.2|88.6|87.9|
|M3|×|×|✓|88.3|87.9|88.1|
|M4|✓|✓|×|89.4|89.2|89.3|
|M5|✓|×|✓|89.9|88.4|89.1|
|M6|×|✓|✓|88.7|89.6|89.2|
|**ECHO‑Det**|✓|✓|✓|**90.8**|**90.3**|**90.5**|

### Extreme‑environment robustness
ECHO‑Det achieves superior mAP under dust/haze/snow/strong‑light/low‑light/shadow/occlusion compared with RGB‑only and simple RGB‑D baseline.

## 🔍 Visualization
- Detection visualization: compare HBB / OBB prediction, SOTA qualitative comparison.
- Grad‑CAM heatmap: feature activation visualization under adverse weather.

## 📝 Citation
If this work helps your research, please cite our paper:
```bibtex
@article{song2026echodet,
  title={ECHO‑Det: An RGB‑D Insulator Defect Detection Network Integrating Edge–Frequency Enhancement and Hierarchical Directional Modeling for Extreme Environments},
  author={Xiaoyang Song, Liqun Liu, Qingfeng Wu, Kai Yang, Jiebao Yang, Jianfeng Zhang},
  journal={},
  year={2026}
}
```

## 📜 License
MIT License

## 💬 Feedback
If you have any questions, open an issue or contact us. ⭐Star if this repo is helpful for you!
```
