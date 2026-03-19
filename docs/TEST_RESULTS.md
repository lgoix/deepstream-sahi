# Test Results

This document presents the evaluation results comparing **SAHI** (Slicing Aided Hyper Inference) against standard full-frame inference on three aerial surveillance videos using VisDrone GELAN-C models.

## Test Environment

| Component | Version |
|-----------|---------|
| DeepStream SDK | 8.0 |
| TensorRT | 10.9 |
| GPU | NVIDIA RTX 5080 (SM 1.20) |
| Precision | FP16 |
| Batch Size | 16 |

## Models Tested

| Model ID | Training Strategy | Input Size | Classes | Dataset |
|----------|-------------------|-----------|---------|---------|
| `visdrone-full-640` | Full-frame | 640x640 | 11 | VisDrone |
| `visdrone-sliced-448` | Sliced | 448x448 | 11 | VisDrone |

## Test Videos

Available for download at **[Google Drive](https://drive.google.com/drive/folders/1CRLnuH9AtTwmxRz7z-Mtu6ErKx__VMK4)**.

| Video | Resolution | Frames | Size | Scene | Results on YouTube |
|-------|-----------|--------|------|-------|--------------------|
| `aerial_crowding_01.mp4` | 2560x1440 | 15,794 | 274.6 MB | Dense pedestrian crowd | [Watch results](https://www.youtube.com/watch?v=_W_wBDvpzzY&list=PLJMGcwo73q30LtZaCw1VQ7UGPvGsmVlco) |
| `aerial_crowding_02.mp4` | 2560x1440 | 1,114 | 20.1 MB | Very dense crowd with motorcycles | [Watch results](https://www.youtube.com/watch?v=RFX8hIWscgw&list=PLJMGcwo73q33YfssoIGBIMu51EPJBobxf) |
| `aerial_vehicles.mp4` | 2560x1440 | 482 | 8.5 MB | Dense vehicle traffic | [Watch results](https://www.youtube.com/watch?v=3CxEp90Jy60&list=PLJMGcwo73q33HWQfjHUD_exEVOUSnlbsA) |

## SAHI Parameters

All SAHI tests used the following default parameters:

| Parameter | Value |
|-----------|-------|
| Overlap Width Ratio | 0.2 |
| Overlap Height Ratio | 0.2 |
| Full-Frame Slice | Enabled |
| Match Metric | IoS (Intersection over Smaller) |
| Match Threshold | 0.5 |

---

## Results Summary

### SAHI vs No SAHI — Detection Improvement

The table below shows the mean objects detected per frame with and without SAHI, along with the improvement factor.

#### visdrone-full-640 (640x640, full-frame trained)

| Video | SAHI Mean | No SAHI Mean | Improvement |
|-------|----------|-------------|-------------|
| aerial_crowding_01 | **84.2** | 13.8 | **+510%** |
| aerial_crowding_02 | **664.7** | 206.2 | **+222%** |
| aerial_vehicles | **252.5** | 92.3 | **+174%** |

#### visdrone-sliced-448 (448x448, slice trained)

| Video | SAHI Mean | No SAHI Mean | Improvement |
|-------|----------|-------------|-------------|
| aerial_crowding_01 | **85.3** | 2.3 | **+3,619%** |
| aerial_crowding_02 | **614.9** | 35.9 | **+1,613%** |
| aerial_vehicles | **226.7** | 28.6 | **+694%** |

> **Key insight:** The sliced-trained model (448x448) benefits dramatically more from SAHI because its native resolution is too small for the 2560x1440 input. Without SAHI, small objects are effectively invisible. With SAHI slicing, the model operates at its intended scale.

### SAHI Model Comparison — Full-Frame vs Sliced Training

When both models use SAHI, their detection counts are remarkably similar:

| Video | full-640 + SAHI | sliced-448 + SAHI | Difference |
|-------|----------------|------------------|------------|
| aerial_crowding_01 | 84.2 | 85.3 | +1.3% |
| aerial_crowding_02 | 664.7 | 614.9 | -7.5% |
| aerial_vehicles | 252.5 | 226.7 | -10.2% |

> **Key insight:** With SAHI enabled, both training strategies achieve comparable detection performance. The full-frame trained model has a slight edge because the full-frame slice captures large objects that may span multiple slices.

---

## Detailed Results by Video

### aerial_crowding_01 — Dense Pedestrian Crowd

**15,794 frames | 2560x1440 | Primarily pedestrians** — [Watch on YouTube](https://www.youtube.com/watch?v=_W_wBDvpzzY&list=PLJMGcwo73q30LtZaCw1VQ7UGPvGsmVlco)

#### SAHI vs No SAHI (visdrone-full-640)

| | SAHI | No SAHI |
|---|---|---|
| Mean objects/frame | 84.2 | 13.8 |
| Median objects/frame | 60 | 4 |
| Max objects/frame | 300 | 87 |

Top class improvements with SAHI:
| Class | SAHI | No SAHI | Gain |
|-------|------|---------|------|
| pedestrian | 75.53 | 12.35 | +63.18 |
| people | 2.50 | 0.19 | +2.31 |
| motor | 2.18 | 0.43 | +1.74 |
| car | 1.89 | 0.44 | +1.45 |

#### SAHI vs No SAHI (visdrone-sliced-448)

| | SAHI | No SAHI |
|---|---|---|
| Mean objects/frame | 85.3 | 2.3 |
| Median objects/frame | 61 | 0 |
| Max objects/frame | 332 | 19 |

Top class improvements with SAHI:
| Class | SAHI | No SAHI | Gain |
|-------|------|---------|------|
| pedestrian | 70.24 | 1.71 | +68.53 |
| people | 6.45 | 0.02 | +6.43 |
| motor | 3.96 | 0.19 | +3.78 |
| car | 2.23 | 0.28 | +1.95 |

---

### aerial_crowding_02 — Very Dense Crowd

**1,114 frames | 2560x1440 | Extremely dense scene with pedestrians and motorcycles** — [Watch on YouTube](https://www.youtube.com/watch?v=RFX8hIWscgw&list=PLJMGcwo73q33YfssoIGBIMu51EPJBobxf)

#### SAHI vs No SAHI (visdrone-full-640)

| | SAHI | No SAHI |
|---|---|---|
| Mean objects/frame | 664.7 | 206.2 |
| Median objects/frame | 664 | 210 |
| Max objects/frame | 759 | 281 |

Top class improvements with SAHI:
| Class | SAHI | No SAHI | Gain |
|-------|------|---------|------|
| pedestrian | 473.05 | 160.82 | +312.23 |
| motor | 158.18 | 43.05 | +115.14 |
| people | 16.27 | 0.22 | +16.05 |
| bicycle | 9.35 | 1.49 | +7.86 |

#### SAHI vs No SAHI (visdrone-sliced-448)

| | SAHI | No SAHI |
|---|---|---|
| Mean objects/frame | 614.9 | 35.9 |
| Median objects/frame | 644 | 37 |
| Max objects/frame | 733 | 72 |

Top class improvements with SAHI:
| Class | SAHI | No SAHI | Gain |
|-------|------|---------|------|
| pedestrian | 400.68 | 26.32 | +374.36 |
| motor | 120.28 | 8.23 | +112.04 |
| people | 85.65 | 0.05 | +85.59 |
| bicycle | 2.48 | 0.00 | +2.48 |

---

### aerial_vehicles — Dense Vehicle Traffic

**482 frames | 2560x1440 | Primarily cars, vans, trucks** — [Watch on YouTube](https://www.youtube.com/watch?v=3CxEp90Jy60&list=PLJMGcwo73q33HWQfjHUD_exEVOUSnlbsA)

#### SAHI vs No SAHI (visdrone-full-640)

| | SAHI | No SAHI |
|---|---|---|
| Mean objects/frame | 252.5 | 92.3 |
| Median objects/frame | 290 | 92 |
| Max objects/frame | 346 | 150 |

Top class improvements with SAHI:
| Class | SAHI | No SAHI | Gain |
|-------|------|---------|------|
| car | 175.59 | 70.81 | +104.78 |
| van | 53.88 | 16.81 | +37.07 |
| truck | 12.22 | 1.43 | +10.79 |
| bus | 5.83 | 3.16 | +2.67 |

#### SAHI vs No SAHI (visdrone-sliced-448)

| | SAHI | No SAHI |
|---|---|---|
| Mean objects/frame | 226.7 | 28.6 |
| Median objects/frame | 258 | 36 |
| Max objects/frame | 327 | 74 |

Top class improvements with SAHI:
| Class | SAHI | No SAHI | Gain |
|-------|------|---------|------|
| car | 190.85 | 26.43 | +164.42 |
| van | 15.92 | 0.68 | +15.24 |
| truck | 13.96 | 0.06 | +13.90 |
| bus | 4.81 | 1.07 | +3.74 |

---

## Conclusions

1. **SAHI dramatically increases detection recall** across all test scenarios. In pedestrian-heavy scenes, SAHI detects **5x to 36x** more objects than standard full-frame inference.

2. **Sliced-trained models benefit the most from SAHI.** The `visdrone-sliced-448` model goes from nearly zero detections (2.3 mean on aerial_crowding_01) to 85.3 with SAHI — a 3,619% improvement. This is expected: models trained on sliced data expect small input patches and fail when presented with a full high-resolution frame.

3. **Full-frame trained models also benefit significantly.** Even the `visdrone-full-640` model, designed for larger inputs, sees 174–510% improvement with SAHI on 2560x1440 video. Small objects at this resolution are still below the model's effective receptive field.

4. **With SAHI, both training strategies converge.** The difference between full-frame and sliced models drops to 1–10% when both use SAHI, suggesting that the slicing strategy successfully normalizes the scale problem.

5. **GreedyNMM post-processing effectively merges duplicates.** The IoS (Intersection over Smaller) metric with a 0.5 threshold successfully suppresses duplicate detections at slice boundaries without losing valid nearby objects.

## Raw Data

The complete per-frame detection CSVs and detailed comparison reports (including charts) are available in the [`test_results/`](../test_results/) directory.
