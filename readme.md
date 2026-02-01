# Improved Pose & Face Detection for Wan Video 2.2 Animate

> Based on GitHub Issue **#1410** (WanVideoPreProcessor) and **#10** (WanAnimatePreProcessor)
>
> **Special thanks:** [@kijai](https://github.com/kijai) and [@steven850](https://github.com/steven850)

---

## 📌 Overview

This document describes improvements to pose and face detection used in **Wan Video 2.2 Animate preprocessing**, focusing on:

* More stable pose tracking (reduced jitter / vanishing limbs)
* Robust face bounding boxes (no eyebrow/chin clipping)
* Better performance in difficult lighting (day/night)
* Cleaner face crops for Animate expression reading

The changes introduce **adaptive preprocessing**, **dynamic temporal smoothing**, and **pose-aware face box stabilization**.

---

## ⚙️ Detection & Pose Parameters

### YOLO Detection Threshold

Controls filtering during YOLO post-processing.

| Value  | Effect                                         |
| ------ | ---------------------------------------------- |
| `0.01` | More detections (better for partial occlusion) |
| `0.1`  | Fewer detections (may increase flicker)        |

> **Note:** Defaults are well tuned for most videos. Adjust only when necessary.

---

### Pose Threshold

Zeroes out low-confidence keypoints **before drawing**.

* Affects stored pose data
* Defaults work well
* Adjust only if pose loss persists after other fixes

---

## 🎨 Preprocessing Enhancements

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

ViTPose is extremely sensitive to:

* Lighting
* Contrast
* Subject/background similarity

**CLAHE dynamically enhances contrast**, improving detection—especially when subject and background colors are similar.

---

### Gaussian Blur for Pose Detection

Pose models prefer **low-frequency input**.

#### Why blur helps

ViTPose (and similar models like HRNet, SimpleBaseline):

* Predict 2D heatmaps per joint
* Are trained on normalized, rescaled human figures
* Focus on *shape & structure*, not pixel-level texture

High-frequency noise causes:

* Multiple weak heatmap activations
* Vanishing or jittering limbs

A **mild Gaussian blur**:

* Suppresses texture noise (fabric, hair, compression artifacts)
* Produces cleaner joint blobs
* Stabilizes YOLO bounding boxes and ViTPose output

> ⚠️ Blur is **only applied to detection & pose input**, not to final face crops.

#### Recommended settings

* **Blur radius:** `5`
* **Blur sigma:** `2`

Result: almost all bone jitter removed; limbs no longer vanish.

---

## 🙂 Face Bounding Box Stabilization

### Face Smoothing (Temporal)

Prevents face crops from jittering between frames.

* Uses **dynamic exponential smoothing**
* Adjusts smoothing strength per frame based on motion

| Movement                | Result                     |
| ----------------------- | -------------------------- |
| Face still              | Very smooth, nearly static |
| Small motion            | Mild smoothing             |
| Large motion (turn/nod) | Immediate tracking, no lag |

**face_smoothing_strength**:

* `0.0` → Disabled
* `1.0` → Heavy smoothing

---

### Constant Face Box (Pose-Aware)

#### Problem

ViTPose face keypoints cover only:

* Eyes
* Nose
* Mouth

They **do not include full head geometry**, causing:

* Forehead or chin clipping
* Poor crops during head tilt or profile views

#### Solution

Use **body keypoints** to stabilize the face box:

* Include neck + shoulders
* Track previous frame center
* Keep box size constant
* Smooth center motion

#### Algorithm

1. Compute bbox from visible face keypoints
2. Compute weighted center toward expected head position
3. Incorporate body joints (neck, shoulders)
4. Stabilize against previous frame
5. Expand symmetrically
6. Clamp to image bounds

> Box size remains constant; only center moves smoothly.

---

### Face Box Size Guide

This acts as a *zoom level*. Output face crops are **always 512×512** (Animate requirement).

| Subject Framing        | Suggested `face_box_size_px` | Notes                       |
| ---------------------- | ---------------------------- | --------------------------- |
| Close-up               | `320–448`                    | Keeps forehead & chin       |
| Medium shot (chest-up) | `256–320`                    | `224–256` also works        |
| Full-body              | `192–256`                    | Still captures head turns   |
| Wide shot (small head) | `128–192`                    | Increase if clipping occurs |

---

## 🦴 Pose Drawing

### Pose Draw Threshold

Controls whether joints/limbs are rendered.

* Recommended: `0.05–0.2`
* Lower values prevent limbs from disappearing

---

## 🔄 Processing Pipeline

```text
Input Frames (B × H × W × C)
        │
        ▼
[ Stage 1: Preprocess ]
- Optional CLAHE
- Optional Gaussian Blur
→ Produces:
  • Blurred frames (detection & pose)
  • Original frames (final output)
        │
        ▼
[ Stage 2: YOLO Detection ]
- Runs on blurred frames
- Outputs person bounding boxes
        │
        ▼
[ Stage 3: ViTPose Inference ]
- Crop & rescale YOLO boxes
- Optional CLAHE
- Normalize
- Predict body + face joints
        │
        ▼
[ Stage 4: Pose Meta Extraction ]
- Convert keypoints to pose metadata
        │
        ▼
[ Stage 5: Face BBox Stabilization ]
- Compute pose-aware face bbox
- Constant size, adaptive center
        │
        ▼
[ Stage 6: Temporal Smoothing ]
- Dynamic exponential smoothing
        │
        ▼
[ Stage 7: Face Cropping ]
- Crop from original (unblurred) frame
- Resize to 512×512
        │
        ▼
[ Stage 8: Packaging ]
- Pose data
- Face crops
- Face bboxes
- JSON keypoints
```

---

## 📦 Output Summary

| Stage | Purpose   | Input          | Output                  |
| ----- | --------- | -------------- | ----------------------- |
| 1     | Enhance   | Raw image      | Blurred / enhanced copy |
| 2     | Detect    | Blurred        | Person bbox             |
| 3     | Pose      | Blurred crops  | Keypoints               |
| 4     | Convert   | Keypoints      | Pose metadata           |
| 5     | Stabilize | Pose meta      | Centered face region    |
| 6     | Smooth    | BBox sequence  | Stable motion           |
| 7     | Crop      | Original image | Sharp 512×512 face      |
| 8     | Package   | All            | PoseData + JSON         |

---

## 🧠 ComfyUI Helper Nodes

Helper nodes for:

* ViTPose inference
* Face crops
* Keypoint extraction for **SAM2 segmentation**

Repository:
[https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)

---

## 📥 Model Downloads

### YOLO (ONNX)

Place in:

```
ComfyUI/models/detection
```

Model:
[https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/process_checkpoint/det/yolov10m.onnx](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/process_checkpoint/det/yolov10m.onnx)

---

### ViTPose (ONNX)

#### Option 1: Large Model

[https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/wholebody](https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/wholebody)

#### Option 2: Huge Model (Original)

Split due to ONNX size limits. **Both files must be in the same directory**.

* `vitpose_h_wholebody_model.onnx`
* `vitpose_h_wholebody_data.bin`

[https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx](https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx)

Select the `.onnx` file in the model loader.

---

## 🌗 Results

Tracking quality shows a **day-and-night improvement**:

* Stable bones
* No limb vanishing
* Clean face crops
* Robust performance under difficult lighting

> Attached: `nodes.py`, `onnx_models.py` (minor changes to pass thresholds correctly)

---

🙏 **Thanks for all the work on this project.**
This contribution is shared in the hope it can be merged upstream and help improve Animate preprocessing for everyone.

![Example Output](example.png)
