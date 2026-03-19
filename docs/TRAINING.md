# Training Guide — SAHI-Aware Models

SAHI is not just an inference-time trick. To get the most out of sliced inference, the **training pipeline must be adapted** to match the slicing strategy used at inference time. This document describes the two training approaches used in this project and discusses how to push performance further.

## Why Training Matters for SAHI

A standard object detection model is trained on full images where objects span a wide range of scales. When SAHI slices a high-resolution frame into small tiles at inference time, each tile looks like a small cropped image — and the model may never have seen data like that during training.

The solution is to **train on sliced images** so the model learns to detect objects at the scale they will actually appear in each tile. This creates a virtuous cycle:

1. **Smaller network input** (e.g. 448×448 instead of 640×640) → faster inference per slice.
2. **Higher accuracy on slices** because the model was trained on data that looks like what it sees at inference.
3. **SAHI tiles the high-resolution frame** so small objects are never missed.
4. **GreedyNMM merges duplicates** at slice boundaries.

The key insight: **reducing the network input size does not hurt accuracy when combined with SAHI**, because SAHI guarantees that objects are always seen at the right scale. But it does **improve throughput** significantly because each slice is processed faster.

## Training Runs

Both models use the **YOLOv9 GELAN-C** architecture trained on the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset (11 classes).

### Stage 1 — Full-Frame Model (visdrone-full-640)

Trained from scratch on unmodified VisDrone images at 640×640.

| Parameter | Value |
|-----------|-------|
| Architecture | GELAN-C |
| Input size | 640×640 |
| Epochs | 140 |
| Batch size | 28 |
| GPUs | 2 |
| Optimizer | SGD (lr=0.01, momentum=0.937) |
| Augmentation | mosaic=0.5, scale=0.9, translate=0.1, flipud=0.4 |
| Close mosaic | Last 15 epochs |

#### Validation Metrics (epoch 140)

| Metric | Value |
|--------|-------|
| Precision | 0.569 |
| Recall | 0.490 |
| mAP@0.5 | 0.485 |
| mAP@0.5:0.95 | 0.242 |

These numbers reflect evaluation on **full-frame** VisDrone validation images, where small objects dominate and are inherently hard to detect at this resolution.

### Stage 2 — Sliced Model (visdrone-sliced-448)

Fine-tuned from the full-frame model (`last.pt`) on **pre-sliced** VisDrone images at 448×448. The VisDrone images were sliced offline using the SAHI slicing utility before training.

| Parameter | Value |
|-----------|-------|
| Architecture | GELAN-C (same weights as Stage 1) |
| Input size | 448×448 |
| Epochs | 40 |
| Batch size | 64 |
| GPUs | 2 |
| Optimizer | SGD (lr=0.01, momentum=0.937) |
| Augmentation | mosaic=0.1, scale=0.0, translate=0.0, flipud=0.1 |
| Close mosaic | Last 15 epochs |
| Init weights | `visdrone-full-640/last.pt` (fine-tuning) |

Augmentation was intentionally reduced — sliced patches already have limited spatial context, so aggressive augmentation like large-scale jitter or translation would be counterproductive.

#### Validation Metrics (epoch 40)

| Metric | Value |
|--------|-------|
| Precision | 0.926 |
| Recall | 0.785 |
| mAP@0.5 | 0.859 |
| mAP@0.5:0.95 | 0.641 |

Validated on **sliced** images matching the training distribution. The dramatic jump in metrics (mAP@0.5 from 0.485 to 0.859) reflects that the model now operates on patches where objects are large relative to the input — exactly what SAHI provides at inference time.

## End-to-End Comparison

When deployed with SAHI on 2560×1440 aerial video, both models achieve similar detection counts despite vastly different training strategies:

| Video | full-640 + SAHI | sliced-448 + SAHI | full-640 alone | sliced-448 alone |
|-------|----------------|------------------|----------------|-----------------|
| aerial_crowding_01 | 84.2 obj/frame | 85.3 obj/frame | 13.8 | 2.3 |
| aerial_crowding_02 | 664.7 | 614.9 | 206.2 | 35.9 |
| aerial_vehicles | 252.5 | 226.7 | 92.3 | 28.6 |

Without SAHI, the sliced-448 model is nearly useless on full-frame high-res video (2.3 mean objects vs 13.8 for full-640). **With SAHI, it matches or exceeds the full-frame model** — at lower computational cost per slice due to the smaller input.

## Room for Improvement

The sliced model was trained with only **40 epochs** as a quick fine-tuning experiment. There is significant room to improve:

1. **More epochs.** 40 epochs is minimal for fine-tuning. Training for 100–200 epochs with proper learning rate scheduling would likely improve mAP further.

2. **Smaller input sizes.** The 448×448 input was chosen conservatively. With SAHI guaranteeing that objects always appear at an appropriate scale within each slice, it may be possible to go down to **320×320 or even 256×256** while maintaining high accuracy. Smaller inputs mean:
   - Faster inference per slice (quadratic reduction in compute).
   - More slices fit in a batch.
   - Potentially higher overall throughput.

3. **Lighter architectures.** The choice of GELAN-C in this project was intentional — not because it is the right fit for this use case, but because a mid-sized, well-understood model was needed to **isolate and validate the plugins themselves**. The real opportunity goes in the opposite direction: lighter, purpose-built architectures. Recent work already points this way — for example, [Zhu & Xie (2026)](https://www.nature.com/articles/s41598-026-35301-2) propose an enhanced YOLOv11n for UAV small-object detection on VisDrone that gains **+4.6% mAP50 while reducing parameter count by ~8.5%** through multi-scale edge-adaptive fusion and a redesigned neck with a P2 detection head. Combined with SAHI slicing, an architecture like that could deliver better precision with significantly less GPU per inference — which is exactly what makes this approach compelling for edge deployments. The SAHI plugins are architecture-agnostic: any model that works with `nvinfer` benefits from sliced inference.

4. **Optimized slice parameters.** The overlap ratio, slice size, and full-frame inclusion can all be tuned per deployment scenario. Dense urban scenes may benefit from more overlap; highway surveillance may need less.

5. **Training on the target domain.** These models were trained on VisDrone. Fine-tuning on data from the actual deployment camera angles, resolutions, and object types would yield the largest gains.

## Key Takeaway

SAHI is a **full-pipeline strategy**, not just a post-hoc inference trick. The best results come from:

1. **Slicing the training data** to match the inference-time slice parameters.
2. **Training (or fine-tuning) the model** on sliced patches so it learns the right scale distribution.
3. **Using a smaller, faster network** because SAHI removes the need for the model to handle extreme scale variation.
4. **Deploying with the SAHI plugins** (`nvsahipreprocess` + `nvsahipostprocess`) for zero-copy slicing and merging in DeepStream.

The combination of slice-aware training + smaller network + SAHI inference delivers both **higher accuracy** and **better throughput** than a large model running on full frames.

## Training Artifacts

The `train_yolov9_visdrone/` directory contains the training outputs for both runs:

```
train_yolov9_visdrone/
├── yolov9_gelan_c_visdrone_full_640/    # Full-frame model (140 epochs)
│   ├── opt.yaml                         # Training configuration
│   ├── hyp.yaml                         # Hyperparameters
│   ├── results.csv                      # Per-epoch training metrics
│   ├── results.png                      # Training curves
│   ├── confusion_matrix.png             # Confusion matrix
│   ├── F1_curve.png                     # F1 score curve
│   ├── PR_curve.png                     # Precision-Recall curve
│   ├── P_curve.png / R_curve.png        # Precision / Recall curves
│   └── labels.jpg / labels_correlogram.jpg
└── yolov9_gelan_c_visdrone_slice_448/   # Sliced model (40 epochs, fine-tuned)
    └── (same structure)
```

> **Note:** Model weights (`.pt` files) are not included in this repository due to size. They are available for download on **[Google Drive](https://drive.google.com/drive/folders/13xUVFYX1bd6LqRFrVAR3HPrLzhSbUH-n)**. The training was performed using the [YOLOv9](https://github.com/WongKinYiu/yolov9) codebase.
