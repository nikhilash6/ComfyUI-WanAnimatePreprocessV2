import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging

script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import get_face_bboxes, padding_resize
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new


# ---------------------------------------------------
# Image enhancement utilities
# ---------------------------------------------------
def preprocess_for_pose(img, use_clahe=True):
    """Optional CLAHE contrast enhancement for ViTPose inputs."""
    if not use_clahe:
        return img

    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_uint8 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return img_uint8.astype(np.float32) / 255.0


# ---------------------------------------------------
# ONNX model loader
# ---------------------------------------------------
class OnnxDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "Models loaded from ComfyUI/models/detection"}),
                "yolo_model": (folder_paths.get_filename_list("detection"), {"tooltip": "Models loaded from ComfyUI/models/detection"}),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider"}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess"

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):
        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)
        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)
        return ({"vitpose": vitpose, "yolo": yolo},)


# ---------------------------------------------------
# Pose and Face Detection
# ---------------------------------------------------
class PoseAndFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048}),
                "detection_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0}),
                "pose_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                # Enhancement options
                "use_clahe": ("BOOLEAN", {"default": True, "tooltip": "Apply CLAHE contrast enhancement for pose detection."}),
                "use_blur_for_pose": ("BOOLEAN", {"default": True, "tooltip": "Apply Gaussian blur internally for YOLO and ViTPose."}),
                "blur_radius": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                # Face smoothing
                "use_face_smoothing": ("BOOLEAN", {"default": True, "tooltip": "Smooth face bounding box center over time."}),
                "face_smoothing_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Higher = more smoothing"}),
                # Constant-size face box
                "use_constant_face_box": ("BOOLEAN", {"default": True, "tooltip": "Keep a constant pixel size face crop; position adapts."}),
                "face_box_size_px": ("INT", {"default": 224, "min": 64, "max": 1024, "step": 16, "tooltip": "Pixel size of the square face crop when constant mode is on."}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX,")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"

    def process(
        self,
        model,
        images,
        width,
        height,
        detection_threshold,
        pose_threshold,
        use_clahe,
        use_blur_for_pose,
        blur_radius,
        blur_sigma,
        use_face_smoothing,
        face_smoothing_strength,
        use_constant_face_box,
        face_box_size_px,
    ):
        detector = model["yolo"]
        pose_model = model["vitpose"]

        if hasattr(detector, "threshold_conf"):
            detector.threshold_conf = detection_threshold

        B, H, W, C = images.shape
        shape = np.array([H, W])[None]
        images_np = images.numpy()

        # --- Prepare blurred version for detection & pose ---
        if use_blur_for_pose:
            ksize = int(blur_radius) * 2 + 1
            images_blurred = np.stack([
                cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
                for img in images_np
            ])
        else:
            images_blurred = images_np

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        comfy_pbar = ProgressBar(B * 2)
        progress = 0
        bboxes = []

        # --- YOLO detection (on blurred) ---
        for img in tqdm(images_blurred, total=B, desc="Detecting bboxes"):
            detections = detector(cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape)[0]
            if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
                bboxes.append(detections[0]["bbox"])
            else:
                bboxes.append(None)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        # --- Pose detection (on blurred) ---
        kp2ds = []
        for img, bbox in tqdm(zip(images_blurred, bboxes), total=B, desc="Extracting keypoints"):
            if (
                bbox is None
                or len(bbox) < 5
                or bbox[4] <= 0
                or (bbox[2] - bbox[0]) < 10
                or (bbox[3] - bbox[1]) < 10
            ):
                bbox_use = np.array([0, 0, img.shape[1], img.shape[0], 1.0], dtype=np.float32)
            else:
                bbox_use = bbox

            center, scale = bbox_from_detector(bbox_use, input_resolution, rescale=rescale)
            img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_crop = preprocess_for_pose(img_crop, use_clahe)
            img_norm = (img_crop - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)

            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()
        kp2ds = np.concatenate(kp2ds, 0)

        # --- Confidence threshold for keypoints ---
        if pose_threshold > 0.0:
            kp2ds[..., 2] = np.where(kp2ds[..., 2] < pose_threshold, 0, kp2ds[..., 2])

        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        # --- Raw face bboxes (from blurred pose keypoints; values are in pixel space) ---
        raw_face_bboxes = []
        for meta in pose_metas:
            bbox_face = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))
            # Ensure ints and within bounds
            x1, x2, y1, y2 = map(int, bbox_face)
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H, y2))
            # Fallback if invalid
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, min(W, 128), min(H, 128)
            raw_face_bboxes.append((x1, x2, y1, y2))

        # --- Convert to centers (for better, rotation-robust smoothing) ---
        raw_centers = []
        for (x1, x2, y1, y2) in raw_face_bboxes:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            raw_centers.append(np.array([cx, cy], dtype=np.float32))

        # --- Temporal smoothing for centers (motion-adaptive) ---
        if use_face_smoothing and len(raw_centers) > 1:
            base_strength = float(np.clip(face_smoothing_strength, 0.0, 1.0))
            smoothed_centers = [raw_centers[0].copy()]
            norm = max(1.0, (W + H) / 2.0)
            for i in range(1, len(raw_centers)):
                curr = raw_centers[i]
                prev = smoothed_centers[-1]
                motion = float(np.mean(np.abs(curr - prev)) / norm)
                # More motion -> less smoothing
                k = 5.0
                dynamic_strength = base_strength * np.exp(-motion * k)
                alpha = 1.0 - dynamic_strength  # 1=no smoothing, 0=full smoothing
                smoothed = alpha * curr + (1.0 - alpha) * prev
                smoothed_centers.append(smoothed.astype(np.float32))
        else:
            smoothed_centers = raw_centers

        # --- Build final face bboxes from smoothed centers ---
        face_bboxes = []
        if use_constant_face_box:
            half = face_box_size_px / 2.0
            for c in smoothed_centers:
                cx, cy = float(c[0]), float(c[1])
                # Clamp so the square stays in bounds
                x1 = int(np.clip(cx - half, 0, W - face_box_size_px))
                y1 = int(np.clip(cy - half, 0, H - face_box_size_px))
                x2 = x1 + int(face_box_size_px)
                y2 = y1 + int(face_box_size_px)
                face_bboxes.append((x1, x2, y1, y2))
        else:
            # If not constant size, just slightly pad the original (helps tilted heads)
            for (x1, x2, y1, y2), c in zip(raw_face_bboxes, smoothed_centers):
                w = x2 - x1
                h = y2 - y1
                x_pad = int(w * 0.2)
                y_pad = int(h * 0.3)
                # Recenter to smoothed center but keep variable size
                cx, cy = float(c[0]), float(c[1])
                half_w = (w / 2.0) + x_pad
                half_h = (h / 2.0) + y_pad
                nx1 = int(np.clip(cx - half_w, 0, W - 1))
                ny1 = int(np.clip(cy - half_h, 0, H - 1))
                nx2 = int(np.clip(cx + half_w, 0, W))
                ny2 = int(np.clip(cy + half_h, 0, H))
                if nx2 <= nx1 or ny2 <= ny1:
                    nx1, ny1, nx2, ny2 = x1, y1, x2, y2  # fallback to raw
                face_bboxes.append((nx1, nx2, ny1, ny2))

        # --- Face crops from sharp original frames ---
        face_images = []
        for idx, (x1, x2, y1, y2) in enumerate(face_bboxes):
            face_image = images_np[idx][y1:y2, x1:x2]
            if face_image.size == 0:
                fallback_size = int(min(H, W) * 0.3)
                fx1 = (W - fallback_size) // 2
                fx2 = fx1 + fallback_size
                fy1 = int(H * 0.1)
                fy2 = fy1 + fallback_size
                face_image = images_np[idx][fy1:fy2, fx1:fx2]
                if face_image.size == 0:
                    face_image = np.zeros((fallback_size, fallback_size, C), dtype=images_np.dtype)
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)

        retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        # use first bbox for return (legacy)
        bbox0 = bboxes[0]
        bbox = np.array(bbox0).flatten() if bbox0 is not None else np.array([0, 0, 0, 0])
        bbox_ints = tuple(int(v) for v in bbox[:4]) if bbox.shape[0] >= 4 else (0, 0, 0, 0)

        # key frame points (unchanged)
        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
        body_key_points = pose_metas[0]['keypoints_body']
        keypoints_body = np.array([body_key_points[i] for i in key_points_index if body_key_points[i] is not None])[:, :2]
        wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
        points = (keypoints_body * wh).astype(np.int32)
        points_dict_list = [{"x": int(p[0]), "y": int(p[1])} for p in points]

        pose_data = {"pose_metas": retarget_pose_metas, "pose_metas_original": pose_metas}
        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints], face_bboxes)


# ---------------------------------------------------
# Draw ViTPose
# ---------------------------------------------------
class DrawViTPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20}),
                "draw_head": ("BOOLEAN", {"default": "True"}),
                "pose_draw_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, pose_draw_threshold, retarget_padding=64):
        pose_metas = pose_data["pose_metas"]
        draw_hand = hand_stick_width != 0

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(
                canvas,
                meta,
                draw_hand=draw_hand,
                draw_head=draw_head,
                body_stick_width=body_stick_width,
                hand_stick_width=hand_stick_width,
                threshold=pose_draw_threshold,
            )
            pose_image = padding_resize(pose_image, height, width)
            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0
        return (pose_images_tensor, )


NODE_CLASS_MAPPINGS = {
    "OnnxDetectionModelLoader": OnnxDetectionModelLoader,
    "PoseAndFaceDetection": PoseAndFaceDetection,
    "DrawViTPose": DrawViTPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoader": "ONNX Detection Model Loader",
    "PoseAndFaceDetection": "Pose and Face Detection",
    "Draw ViT Pose": "Draw ViT Pose",
}
