# NOTICE — ComfyUI-WanAnimatePreprocessV2

Copyright (c) 2025-2026 Code2Collapse (https://github.com/Code2Collapse)  
License: Apache License 2.0 (see LICENSE)

**Project author**: Code2Collapse  

This project is a **derivative work** based on:

1. **kijai's `ComfyUI-WanAnimatePreprocess`** — original YOLO+ViTPose+Wan pipeline  
   Author: kijai (Jukka Seppänen) — <https://github.com/kijai>  
   Repository: <https://github.com/kijai/ComfyUI-WanAnimatePreprocess>  
   License: Apache 2.0

2. **steven850's improved nodes** — added CLAHE preprocessing, Gaussian blur,  
   temporal face-bbox smoothing, constant-size face crop, detection threshold  
   controls. Submitted as an attachment to issue #10 of kijai's repo.  
   Author: steven850 — <https://github.com/steven850>  
   Issue: <https://github.com/kijai/ComfyUI-WanAnimatePreprocess/issues/10>  
   License: Apache 2.0 (contribution to an Apache-2.0 repository)

**Additions by Code2Collapse** on top of the kijai/steven850 base:
- Iris/pupil detection (gradient voting, Timm-Barth 2011 inspired multi-strategy fallback)
- MediaPipe FaceMesh 478-point pipeline with iris/gaze/lip tracking
- Protobuf ≥5.x compatibility fix for mediapipe ≤0.10.x
- Lip openness ratio output
- Renamed to V2 namespace with additional RETURN_TYPES

This project includes code and data derived from the following third-party
works. Their copyrights and licenses are listed below.

---

## 1. Wan — Alibaba Wan Team

**Repository owners**: The Alibaba Wan Team Authors  
**Contact**: https://github.com/Wan-Video

The following files are **directly derived from** the Wan open-source project
and carry their original copyright notices intact (as required by Apache 2.0):

- `pose_utils/human_visualization.py` — Copyright 2024-2025 The Alibaba Wan Team Authors.
- `pose_utils/pose2d_utils.py`         — Copyright 2024-2025 The Alibaba Wan Team Authors.
- `models/onnx_models.py`              — Copyright 2024-2025 The Alibaba Wan Team Authors.
- `retarget_pose.py`                   — Copyright 2024-2025 The Alibaba Wan Team Authors.
- `utils.py`                           — Copyright 2024-2025 The Alibaba Wan Team Authors.

**Repository**: <https://github.com/Wan-Video/Wan2.1>  
**License**: Apache License 2.0  
<https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE>

---

## 2. kijai — ComfyUI-WanAnimatePreprocess

**Author**: kijai (Jukka Seppänen) — <https://github.com/kijai>  
**Repository**: <https://github.com/kijai/ComfyUI-WanAnimatePreprocess>  
**License**: Apache License 2.0

`nodes.py` and `models/onnx_models.py` in this pack are derivative works
based on kijai's original ComfyUI node wrappers for the Wan Animate preprocessing
pipeline. kijai's code wraps the Alibaba Wan Team's preprocess logic as
ComfyUI nodes.

---

## 3. steven850 — Improved pose and face detection

**Author**: steven850 — <https://github.com/steven850>  
**Source**: Issue #10 of kijai/ComfyUI-WanAnimatePreprocess  
<https://github.com/kijai/ComfyUI-WanAnimatePreprocess/issues/10>  
**License**: Apache License 2.0 (contribution to Apache-2.0 repository)

The following improvements in `nodes.py` originate from steven850's
contribution posted as issue #10:

- CLAHE contrast enhancement preprocessing
- Gaussian blur preprocessing for YOLO/ViTPose stability
- Motion-adaptive temporal smoothing for face bounding boxes
- Constant-size face crop with center-tracking
- Detection threshold and pose threshold parameters
- Core 8-stage pipeline architecture (YOLO → ViTPose → pose metas → face bbox → smooth → crop → package)

---

## 4. One-to-All-Animation (CVPR 2026)

**Repository owners / Authors**:  
Shijun Shi (Jiangnan University), Jing Xu (USTC), Zhihang Li (CAS),  
Chunli Peng (BUPT), Xiaoda Yang (Zhejiang University), Lijing Lu (CAS),  
Kai Hu (Jiangnan University), Jiangning Zhang (Zhejiang University)  
**Contact**: ssj180123@gmail.com  
**Repository**: <https://github.com/ssj9596/One-to-All-Animation>

The following files are adapted from this project:

- `onetoall/infer_function.py` — pose format conversion utilities
- `onetoall/utils.py` — pose retargeting and scale-and-translate logic

**License**: Apache License 2.0  
<https://github.com/ssj9596/One-to-All-Animation/blob/main/LICENSE>

**Modifications by Code2Collapse (2025-2026)**:
- Adapted for ComfyUI-WanAnimatePreprocessV2 node integration
- Added `aaposemeta_obj_to_dwpose` for AAPoseMeta object support

**Citation** (if used in research):
```bibtex
@article{shi2025one,
  title={One-to-All Animation: Alignment-Free Character Animation and Image Pose Transfer},
  author={Shi, Shijun and Xu, Jing and Li, Zhihang and Peng, Chunli and Yang, Xiaoda and
          Lu, Lijing and Hu, Kai and Zhang, Jiangning},
  journal={arXiv preprint arXiv:2511.22940},
  year={2025}
}
```

---

## 5. MediaPipe (Google LLC)

Used for 478-point FaceMesh landmark detection in `nodes.py`.

**Repository**: <https://github.com/google-ai-edge/mediapipe>  
**Copyright**: Copyright 2023 The MediaPipe Authors  
**License**: Apache License 2.0  
<https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>

---

## 6. ViTPose (MMLAB / OpenMMLab)

ONNX models for 2D human pose estimation are derived from the ViTPose project.

**Repository**: <https://github.com/ViTAE-Transformer/ViTPose>  
**Copyright**: Copyright (c) 2022 ViTAE-Transformer  
**License**: Apache License 2.0  
<https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE>

---

## 7. ONNX Runtime (Microsoft)

Used as the ONNX inference backend for ViTPose and YOLO models.

**Repository**: <https://github.com/microsoft/onnxruntime>  
**Copyright**: Copyright (c) Microsoft Corporation  
**License**: MIT License  
<https://github.com/microsoft/onnxruntime/blob/main/LICENSE>

---

## 8. DWPose / ControlNet Auxiliary Preprocessors

The pose data structures in `onetoall/infer_function.py` use the DWPose
format developed by IDEA Research / ControlNet contributors.

**Repository**: <https://github.com/IDEA-Research/DWPose>  
**Copyright**: Copyright (c) 2023 IDEA Research  
**License**: Apache License 2.0

---

## 9. OpenCV

**Repository**: <https://github.com/opencv/opencv>  
**License**: Apache License 2.0 (since OpenCV 4.5)

---

## 10. PyTorch

**Repository**: <https://github.com/pytorch/pytorch>  
**Copyright**: Copyright (c) 2016-2024 Facebook, Inc. and its affiliates  
**License**: BSD-style license — <https://github.com/pytorch/pytorch/blob/main/LICENSE>

---

## Model Weights Notice

ViTPose and YOLO ONNX model weights are third-party assets. Users are
responsible for reviewing the license of each model checkpoint before
commercial use.
