# Plugin Reference

Complete documentation for the two GStreamer plugins provided by this project.

---

## nvsahipreprocess

**SAHI dynamic-slice pre-processor for DeepStream.**

Replaces static ROI groups used by NVIDIA's `nvdspreprocess` with dynamic per-frame SAHI slice computation. For each incoming frame, the plugin divides the image into overlapping tiles (slices), crops and scales each tile to the network input resolution via `NvBufSurfTransform`, then delegates tensor preparation to the same custom library interface used by `nvdspreprocess`. The resulting `GstNvDsPreProcessBatchMeta` is consumed by `nvinfer` with `input-tensor-meta=1`.

### Pipeline Position

```
nvstreammux → nvsahipreprocess → nvinfer (input-tensor-meta=1) → queue → nvsahipostprocess → ...
```

### Pad Templates

| Pad  | Direction | Availability | Caps |
|------|-----------|-------------|------|
| sink | Sink      | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |
| src  | Source    | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |

### GStreamer Element Properties

These properties are set directly on the element in the pipeline (e.g. via Python `set_property()` or `gst-launch` command line).

| Property | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `unique-id` | uint | 15 | 0 – UINT_MAX | Unique identifier for this element instance. Used to match with `target-unique-ids` in downstream elements. |
| `enable` | boolean | true | — | Enable the plugin. When `false`, the element operates in passthrough mode (buffers flow through untouched). |
| `gpu-id` | uint | 0 | 0 – UINT_MAX | GPU device ID to use for scaling and tensor operations. |
| `config-file` | string | `""` | — | Path to the configuration file for tensor preparation parameters (see [Configuration File](#configuration-file) below). **Required.** |
| `slice-width` | uint | 640 | 1 – UINT_MAX | Width of each SAHI slice in pixels. |
| `slice-height` | uint | 640 | 1 – UINT_MAX | Height of each SAHI slice in pixels. |
| `overlap-width-ratio` | float | 0.2 | 0.0 – 0.99 | Horizontal overlap between adjacent slices, expressed as a fraction of `slice-width`. |
| `overlap-height-ratio` | float | 0.2 | 0.0 – 0.99 | Vertical overlap between adjacent slices, expressed as a fraction of `slice-height`. |
| `enable-full-frame` | boolean | true | — | Append the entire frame as an additional slice. This ensures large objects spanning multiple slices are still detected — standard SAHI behaviour. Disable only for debugging. |
| `target-unique-ids` | string | `""` | — | Semicolon-separated list of downstream GIE `unique-id` values for which tensors are prepared (e.g. `"1"` or `"3;4;5"`). |

### Configuration File

The config file uses GLib key-file (INI) format and contains a mandatory `[property]` section and an optional `[user-configs]` section. Unlike `nvdspreprocess`, there are **no `[group-N]` / ROI sections** — slices are computed dynamically from the element properties above.

#### `[property]` Section — Required Keys

| Key | Type | Description |
|-----|------|-------------|
| `enable` | int (0/1) | Enable or disable the plugin. |
| `target-unique-ids` | int list (`;`-separated) | GIE unique-ids to target. |
| `processing-width` | int | Width to scale each slice to before tensor preparation. Must match the network input width. |
| `processing-height` | int | Height to scale each slice to before tensor preparation. Must match the network input height. |
| `maintain-aspect-ratio` | int (0/1) | Maintain aspect ratio during scaling, padding with black. |
| `symmetric-padding` | int (0/1) | When `maintain-aspect-ratio=1`, pad symmetrically (center the image) instead of padding only right/bottom. |
| `network-input-order` | int | Tensor layout: `0` = NCHW, `1` = NHWC. |
| `network-input-shape` | int list (`;`-separated) | Full tensor shape. For NCHW: `B;C;H;W` (e.g. `16;3;640;640`). The batch dimension (`B`) determines the maximum number of slices processed in a single GPU batch. |
| `network-color-format` | int | Color format: `0` = RGB, `1` = BGR, `2` = GRAY. |
| `tensor-data-type` | int | Data type: `0` = FP32, `1` = UINT8, `2` = INT8, `3` = UINT32, `4` = INT32, `5` = FP16. |
| `tensor-name` | string | Name of the input tensor (must match the model's input layer name, e.g. `images`). |
| `scaling-filter` | int | Interpolation filter: `0` = Nearest, `1` = Bilinear. |
| `scaling-pool-memory-type` | int | Surface memory type: `0` = Default, `1` = CUDA Pinned, `2` = CUDA Device, `3` = CUDA Unified, `4` = Surface Array. |
| `scaling-pool-compute-hw` | int | Compute backend for scaling: `0` = Default, `1` = GPU, `2` = VIC (Jetson only). |
| `custom-lib-path` | string | Absolute path to the shared library for tensor preparation (e.g. `/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so`). |
| `custom-tensor-preparation-function` | string | Name of the exported function in the custom library (e.g. `CustomTensorPreparation`). |

#### `[property]` Section — Optional Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `unique-id` | int | 15 | Plugin unique-id (can also be set via element property). |
| `gpu-id` | int | 0 | GPU device ID (can also be set via element property). |
| `scaling-buf-pool-size` | int | 6 | Number of buffers in the scaling surface pool. Increase if the pipeline stalls waiting for buffers. |
| `tensor-buf-pool-size` | int | 6 | Number of buffers in the tensor pool. |

#### `[user-configs]` Section

Arbitrary key-value pairs passed to the custom library's `initLib()` function as a `std::unordered_map<std::string, std::string>`.

| Key | Example Value | Description |
|-----|--------------|-------------|
| `pixel-normalization-factor` | `0.003921568` | Per-pixel normalization (1/255). Used by the default `libcustom2d_preprocess.so`. |

#### Example Config File

```ini
[property]
enable=1
target-unique-ids=1
network-input-order=0
maintain-aspect-ratio=1
symmetric-padding=1
processing-width=640
processing-height=640
scaling-buf-pool-size=6
tensor-buf-pool-size=6
network-input-shape=16;3;640;640
network-color-format=0
tensor-data-type=0
tensor-name=images
scaling-pool-memory-type=0
scaling-pool-compute-hw=0
scaling-filter=0
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so
custom-tensor-preparation-function=CustomTensorPreparation

[user-configs]
pixel-normalization-factor=0.003921568
```

### Slice Computation Algorithm

The slicing algorithm is ported from SAHI's Python `slicing.py:get_slice_bboxes()`:

1. Compute step sizes: `w_step = slice_width - (slice_width × overlap_width_ratio)`, same for height.
2. Iterate over the frame in a grid with those steps.
3. At each position, create a crop rectangle of `slice_width × slice_height`.
4. When a slice would extend past the frame boundary, shift it back so it ends exactly at the edge (ensures full coverage without padding).
5. If `enable-full-frame=true`, append one additional slice covering the entire frame (0, 0, frame_width, frame_height).

For a 1920×1080 frame with 640×640 slices and 0.2 overlap:
- Horizontal step = 640 − 128 = 512 → positions at x = 0, 512, 1024, 1280
- Vertical step = 640 − 128 = 512 → positions at y = 0, 440
- **Result**: 4×2 = 8 slices + 1 full-frame = **9 total** inference regions per frame.

### Architecture

The plugin uses a two-thread architecture (inherited from `nvdspreprocess`):

1. **Submission thread** (`submit_input_buffer`): computes slices, crops and scales each slice via batched `NvBufSurfTransformAsync`, then enqueues the batch.
2. **Output thread** (`gst_nvsahipreprocess_output_loop`): dequeues batches, waits for async transforms to complete, calls the custom tensor preparation function, and attaches `GstNvDsPreProcessBatchMeta` to the buffer before pushing downstream.

### Metadata Output

The plugin attaches `GstNvDsPreProcessBatchMeta` (meta type `NVDS_PREPROCESS_BATCH_META`) to each buffer. This metadata contains:

- `roi_vector` — one `NvDsRoiMeta` per slice with the original crop coordinates, scale ratios, and offsets.
- `tensor_meta` — pointer to the prepared GPU tensor buffer, shape, data type, and tensor name.
- `target_unique_ids` — the GIE IDs this tensor is intended for.

When `nvinfer` has `input-tensor-meta=1`, it reads this metadata directly instead of performing its own scaling, enabling zero-copy tensor handoff.

---

## nvsahipostprocess

**SAHI GreedyNMM post-processor for DeepStream.**

Merges duplicate detections produced by sliced inference. Objects at slice boundaries appear in multiple overlapping slices, producing near-duplicate bounding boxes. This plugin applies the GreedyNMM algorithm (ported from SAHI's `combine.py`) to suppress or merge those duplicates. It operates entirely on `NvDsObjectMeta` — no tensor access, no CUDA kernels.

### Pipeline Position

```
nvinfer → queue → nvsahipostprocess → nvtracker → nvdsosd → sink
```

### Pad Templates

| Pad  | Direction | Availability | Caps |
|------|-----------|-------------|------|
| sink | Sink      | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |
| src  | Source    | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |

### GStreamer Element Properties

| Property | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `gie-id` | int | -1 | -1 – INT_MAX | Only process detections from this GIE `unique-component-id`. Set to `-1` to merge all detections regardless of source GIE. |
| `match-metric` | uint | 1 (IoS) | 0 – 1 | Overlap metric used to determine if two detections are duplicates. `0` = **IoU** (Intersection over Union). `1` = **IoS** (Intersection over Smaller area). IoS is recommended for SAHI because small objects near slice boundaries often have very different areas, making IoU unreliable. |
| `match-threshold` | float | 0.5 | 0.0 – 1.0 | Overlap value above which two detections are considered duplicates. Lower values merge more aggressively; higher values are more conservative. |
| `class-agnostic` | boolean | false | — | If `true`, compare detections across different class IDs. If `false`, only compare detections within the same class. |
| `enable-merge` | boolean | true | — | If `true`, when a detection is suppressed, its bounding box is merged into the surviving detection (GreedyNMM — the survivor's box expands to encompass both). If `false`, suppressed detections are simply removed (standard NMS behaviour). |

### GreedyNMM Algorithm

The algorithm processes each frame independently:

1. **Collect** all `NvDsObjectMeta` from the frame (optionally filtering by `gie-id`).
2. **Sort** detections by confidence score in descending order.
3. **Iterate** in score order. For each non-suppressed detection `i`:
   - Compare with every subsequent non-suppressed detection `j`.
   - If `class-agnostic=false` and classes differ, skip.
   - Compute overlap using the chosen metric.
   - If overlap ≥ `match-threshold`, mark `j` as suppressed.
   - If `enable-merge=true`, expand `i`'s bounding box to encompass `j`'s box. Take the maximum confidence of both.
4. **Apply results**:
   - Suppressed detections: removed from `NvDsFrameMeta.obj_meta_list`.
   - Merged survivors: `rect_params` and `detector_bbox_info.org_bbox_coords` updated with the expanded bounding box.

### Match Metrics Explained

#### IoU (Intersection over Union)

```
IoU = intersection_area / (area_A + area_B - intersection_area)
```

Standard NMS metric. Works well when both boxes have similar sizes. Less effective for SAHI because a small object at a slice boundary may overlap with a much larger version from the full-frame slice.

#### IoS (Intersection over Smaller) — Recommended for SAHI

```
IoS = intersection_area / min(area_A, area_B)
```

Measures how much of the smaller box is contained within the larger one. This handles the common SAHI scenario where the full-frame detection is much larger than the slice detection of the same object. A high IoS means the small detection is mostly inside the large one, indicating a duplicate.

### Example Pipeline (gst-launch)

```bash
gst-launch-1.0 \
  uridecodebin uri=file:///path/to/video.mp4 ! \
  nvstreammux batch-size=1 width=2560 height=1440 ! \
  nvsahipreprocess \
    config-file=preprocess_640.txt \
    slice-width=640 slice-height=640 \
    overlap-width-ratio=0.2 overlap-height-ratio=0.2 \
    enable-full-frame=true ! \
  nvinfer config-file-path=pgie_config.txt input-tensor-meta=1 ! \
  queue ! \
  nvsahipostprocess \
    gie-id=1 \
    match-metric=1 \
    match-threshold=0.5 \
    class-agnostic=false \
    enable-merge=true ! \
  nvtracker ... ! \
  nvdsosd ! \
  fakesink
```

> Replace `fakesink` with `nveglglessink` if a display server is available.

### Example (Python — GstElement Properties)

```python
# Pre-process
preprocess = Gst.ElementFactory.make("nvsahipreprocess", "sahi-pre")
preprocess.set_property("config-file", "config/preprocess/preprocess_640.txt")
preprocess.set_property("slice-width", 640)
preprocess.set_property("slice-height", 640)
preprocess.set_property("overlap-width-ratio", 0.2)
preprocess.set_property("overlap-height-ratio", 0.2)
preprocess.set_property("enable-full-frame", True)

# Post-process
postprocess = Gst.ElementFactory.make("nvsahipostprocess", "sahi-post")
postprocess.set_property("gie-id", 1)
postprocess.set_property("match-metric", 1)       # IoS
postprocess.set_property("match-threshold", 0.5)
postprocess.set_property("class-agnostic", False)
postprocess.set_property("enable-merge", True)
```

---

## Tuning Guide

### Slice Size

- **Smaller slices** (e.g. 320×320): better for very small objects, but produces more slices per frame → higher GPU cost.
- **Larger slices** (e.g. 960×960): fewer slices, faster, but small objects may still be missed.
- **Rule of thumb**: set `slice-width` and `slice-height` close to the model's native input resolution (e.g. 640 for a 640×640 YOLO model).

### Overlap Ratio

- **0.2** (20%) is a good default. Ensures objects at slice boundaries appear in at least two slices.
- Increase to **0.3–0.4** if you see missed detections at boundaries.
- Higher overlap = more slices = more inference cost.

### Batch Size (network-input-shape[0])

The first dimension of `network-input-shape` in the config file controls how many slices are batched per GPU inference call. Set this to at least the number of slices per frame for optimal throughput. For 9 slices, use `16` (next power-of-two is common for GPU efficiency).

### Match Threshold

- **0.5** is a good starting point for IoS.
- If you see leftover duplicate boxes, lower to **0.3–0.4**.
- If valid nearby detections are being incorrectly merged, raise to **0.6–0.7**.

### Full-Frame Slice

Keep `enable-full-frame=true` (default). Without it, large objects spanning multiple slices may be partially detected in each slice but never seen as a whole. The full-frame slice provides a global view that catches these objects, while GreedyNMM merges the duplicates.
