# NOTICE — ComfyUI-WanAnimatePreprocessV2

Copyright (c) 2025-2026 Likhith-24 (https://github.com/Likhith-24)  
License: Apache License 2.0 (see LICENSE)

**Project author**: Likhith-24  
**Original nodes.py pipeline** was written from scratch, inspired by
discussions in ComfyUI-WanVideoWrapper issues #1410 and #10 by
[@kijai](https://github.com/kijai) and [@steven850](https://github.com/steven850).
No code was copied from their repositories.

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

## 2. One-to-All-Animation (CVPR 2026)

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

**Modifications by Likhith-24 (2025-2026)**:
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

## 3. MediaPipe (Google LLC)

Used for 478-point FaceMesh landmark detection in `nodes.py`.

**Repository**: <https://github.com/google-ai-edge/mediapipe>  
**Copyright**: Copyright 2023 The MediaPipe Authors  
**License**: Apache License 2.0  
<https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>

---

## 4. ViTPose (MMLAB / OpenMMLab)

ONNX models for 2D human pose estimation are derived from the ViTPose project.

**Repository**: <https://github.com/ViTAE-Transformer/ViTPose>  
**Copyright**: Copyright (c) 2022 ViTAE-Transformer  
**License**: Apache License 2.0  
<https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE>

---

## 5. ONNX Runtime (Microsoft)

Used as the ONNX inference backend for ViTPose and YOLO models.

**Repository**: <https://github.com/microsoft/onnxruntime>  
**Copyright**: Copyright (c) Microsoft Corporation  
**License**: MIT License  
<https://github.com/microsoft/onnxruntime/blob/main/LICENSE>

---

## 6. DWPose / ControlNet Auxiliary Preprocessors

The pose data structures in `onetoall/infer_function.py` use the DWPose
format developed by IDEA Research / ControlNet contributors.

**Repository**: <https://github.com/IDEA-Research/DWPose>  
**Copyright**: Copyright (c) 2023 IDEA Research  
**License**: Apache License 2.0

---

## 7. OpenCV

**Repository**: <https://github.com/opencv/opencv>  
**License**: Apache License 2.0 (since OpenCV 4.5)

---

## 8. PyTorch

**Repository**: <https://github.com/pytorch/pytorch>  
**Copyright**: Copyright (c) 2016-2024 Facebook, Inc. and its affiliates  
**License**: BSD-style license — <https://github.com/pytorch/pytorch/blob/main/LICENSE>

---

## Model Weights Notice

ViTPose and YOLO ONNX model weights are third-party assets. Users are
responsible for reviewing the license of each model checkpoint before
commercial use.
