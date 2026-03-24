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
# Iris / pupil estimation (image-based)
# ---------------------------------------------------
# Eye contour landmark indices within the 69-point face array.
# face array = kp2ds[22:91]; index 0 is body, indices 1-68 are
# standard 68-face landmarks (standard N -> face[N+1]).
#
# Right eye (standard 36-41):
#   37=outer  38=upper_outer  39=upper_inner
#   40=inner  41=lower_inner  42=lower_outer
# Left eye (standard 42-47):
#   43=inner  44=upper_inner  45=upper_outer
#   46=outer  47=lower_outer  48=lower_inner
_RIGHT_EYE_IDX = [37, 38, 39, 40, 41, 42]
_LEFT_EYE_IDX  = [43, 44, 45, 46, 47, 48]
_EYE_CONTOUR_INDICES = [_RIGHT_EYE_IDX, _LEFT_EYE_IDX]


def _gradient_vote_pupil(roi_gray, mask, gc_local, eye_w, eye_h):
    """Gradient-based pupil centre detection (Timm-Barth 2011 inspired).

    Edge gradients around the circular pupil boundary point radially outward.
    By casting rays in the *negative* gradient direction from every strong-edge
    pixel we accumulate votes at the true centre.

    Returns (local_cx, local_cy, score) or None.
    """
    h, w = roi_gray.shape
    gx = cv2.Sobel(roi_gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    mag_masked = mag.copy()
    mag_masked[mask == 0] = 0
    vals = mag_masked[mask > 0]
    if len(vals) < 8 or vals.max() < 1:
        return None

    thresh = float(np.percentile(vals, 70))  # top 30 % of gradients
    strong = (mag > thresh) & (mask > 0)
    if np.count_nonzero(strong) < 8:
        return None

    ys, xs = np.where(strong)
    gx_s = gx[ys, xs]
    gy_s = gy[ys, xs]
    mag_s = mag[ys, xs]

    # Normalise & negate  (point *toward* centre)
    gx_n = -gx_s / (mag_s + 1e-10)
    gy_n = -gy_s / (mag_s + 1e-10)

    accumulator = np.zeros((h, w), np.float64)
    max_t = max(int(max(eye_w, eye_h) * 0.5), 5)

    for t in range(1, max_t + 1):
        px = (xs + gx_n * t + 0.5).astype(np.int32)
        py = (ys + gy_n * t + 0.5).astype(np.int32)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        pxv, pyv, magv = px[valid], py[valid], mag_s[valid]
        in_mask = mask[pyv, pxv] > 0
        np.add.at(accumulator, (pyv[in_mask], pxv[in_mask]), magv[in_mask])

    # Weight by darkness (pupil region is darker than sclera)
    dark_w = (255.0 - roi_gray.astype(np.float64)) / 255.0
    accumulator *= (0.5 + 0.5 * dark_w)
    accumulator[mask == 0] = 0

    if accumulator.max() < 1:
        return None

    acc_smooth = cv2.GaussianBlur(accumulator, (5, 5), 1.0)
    _, max_val, _, max_loc = cv2.minMaxLoc(acc_smooth, mask=mask)
    cx, cy = float(max_loc[0]), float(max_loc[1])

    mean_acc = float(np.mean(accumulator[mask > 0]) + 1e-6)
    score = min(1.0, max_val / (mean_acc * 5))
    return cx, cy, score


def _find_pupil_center(eye_pts_px, img_gray, W, H):
    """Locate the pupil/iris centre inside one eye.

    Pipeline
    --------
    1. Build a tight eye-region mask from 6 contour landmarks.
    2. Restrict search to the **upper 65 %** of the lid opening to avoid
       eyelid / eyelash shadow contamination.
    3. Apply CLAHE for better pupil-iris-sclera contrast.
    4. **Primary** – gradient-based centre voting (Timm-Barth inspired):
       robust to lighting, threshold-free.
    5. **Secondary** – multi-threshold contour moments with asymmetric
       vertical scoring.
    6. **Tertiary** – weighted dark-pixel centroid with upper-region bias.
    7. Fallback – geometric centre of the eye contour.
    """
    geo_center = np.mean(eye_pts_px, axis=0)

    # --- Eye Aspect Ratio (EAR) – skip closed eyes ---
    v1 = np.linalg.norm(eye_pts_px[1] - eye_pts_px[5])
    v2 = np.linalg.norm(eye_pts_px[2] - eye_pts_px[4])
    horiz = np.linalg.norm(eye_pts_px[0] - eye_pts_px[3])
    if horiz < 3:
        return float(geo_center[0]), float(geo_center[1]), 0.0
    ear = (v1 + v2) / (2.0 * horiz)
    if ear < 0.12:
        return float(geo_center[0]), float(geo_center[1]), 0.05

    # --- Tight padded ROI ---
    min_xy = np.min(eye_pts_px, axis=0)
    max_xy = np.max(eye_pts_px, axis=0)
    eye_w = max_xy[0] - min_xy[0]
    eye_h = max_xy[1] - min_xy[1]
    pad = max(int(eye_w * 0.15), 2)
    rx1 = max(0, int(min_xy[0]) - pad)
    ry1 = max(0, int(min_xy[1]) - pad)
    rx2 = min(W, int(max_xy[0]) + pad)
    ry2 = min(H, int(max_xy[1]) + pad)
    roi = img_gray[ry1:ry2, rx1:rx2]
    if roi.size < 20:
        return float(geo_center[0]), float(geo_center[1]), 0.1
    h_roi, w_roi = roi.shape

    # --- Eye contour mask ---
    pts_local = eye_pts_px.astype(np.int32).copy()
    pts_local[:, 0] -= rx1
    pts_local[:, 1] -= ry1
    mask_full = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, pts_local, 255)

    # --- Restrict to upper 65 % of lid opening ---
    eye_top_l = max(0, int(min_xy[1]) - ry1)
    eye_bot_l = min(h_roi, int(max_xy[1]) - ry1)
    cutoff = int(eye_top_l + 0.65 * (eye_bot_l - eye_top_l))
    mask = mask_full.copy()
    mask[cutoff:, :] = 0

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_inner = cv2.erode(mask, kern, iterations=2)
    if np.count_nonzero(mask_inner) < 5:
        mask_inner = cv2.erode(mask, kern, iterations=1)
    if np.count_nonzero(mask_inner) < 5:
        mask_inner = mask

    # --- CLAHE + gentle blur ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    roi_eq = clahe.apply(roi)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0.7)

    masked_pixels = roi_blur[mask_inner > 0]
    if len(masked_pixels) < 5:
        return float(geo_center[0]), float(geo_center[1]), 0.1

    gc_local = geo_center - np.array([rx1, ry1])
    mask_area = float(max(np.count_nonzero(mask_inner), 1))

    # ================================================================
    # Strategy 1 – gradient-based centre voting  (primary)
    # ================================================================
    grad = _gradient_vote_pupil(roi_blur, mask_inner, gc_local, eye_w, eye_h)
    if grad is not None:
        gcx, gcy, gscore = grad
        if gscore > 0.20:
            conf = float(np.clip(ear * 2.5 * gscore, 0.1, 1.0))
            return float(gcx) + rx1, float(gcy) + ry1, conf

    # ================================================================
    # Strategy 2 – multi-threshold contour moments  (secondary)
    # ================================================================
    best_cx, best_cy, best_score = None, None, -1.0
    for pct in (10, 20, 30, 40):
        thresh_val = int(np.percentile(masked_pixels, pct))
        binary = np.zeros_like(roi_blur)
        binary[(roi_blur <= thresh_val) & (mask_inner > 0)] = 255
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kern)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1:
                continue
            cx_l = M["m10"] / M["m00"]
            cy_l = M["m01"] / M["m00"]
            ix, iy = int(cx_l), int(cy_l)
            if not (0 <= ix < w_roi and 0 <= iy < h_roi):
                continue
            if mask_full[iy, ix] == 0:
                continue

            # Circularity
            perim = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (perim ** 2 + 1e-6)
            circ_score = min(circ, 1.0)

            # Proximity – asymmetric vertical penalty
            dx = abs(cx_l - gc_local[0])
            dy = cy_l - gc_local[1]          # positive = below centre
            max_dx = max(eye_w * 0.45, 1)
            h_prox = max(0.0, 1.0 - dx / max_dx)
            max_dy_up = max(eye_h * 0.4, 1)
            max_dy_dn = max(eye_h * 0.25, 1)  # tighter below
            v_prox = max(0.0, 1.0 - abs(dy) / (max_dy_dn if dy > 0 else max_dy_up))
            prox_score = 0.5 * h_prox + 0.5 * v_prox

            # Size
            ratio = area / mask_area
            if ratio < 0.03 or ratio > 0.70:
                size_score = 0.1
            elif 0.08 <= ratio <= 0.45:
                size_score = 1.0
            else:
                size_score = 0.5

            score = circ_score * 0.25 + prox_score * 0.45 + size_score * 0.30
            if score > best_score:
                best_score = score
                best_cx = cx_l + rx1
                best_cy = cy_l + ry1

    if best_cx is not None and best_score > 0.25:
        conf = float(np.clip(ear * 2.5 * best_score, 0.1, 1.0))
        return float(best_cx), float(best_cy), conf

    # ================================================================
    # Strategy 3 – weighted dark-pixel centroid with upper-region bias
    # ================================================================
    thresh_val = int(np.percentile(masked_pixels, 25))
    dark = (roi_blur <= thresh_val) & (mask_inner > 0)
    ys, xs = np.where(dark)
    if len(xs) > 3:
        weights = (255.0 - roi_blur[dark]).astype(np.float64)
        vert_bias = np.clip(
            1.0 - (ys - gc_local[1]) / max(eye_h * 0.3, 1), 0.3, 1.5)
        weights *= vert_bias
        total = weights.sum()
        if total > 0:
            cx = float(np.sum(xs * weights) / total) + rx1
            cy = float(np.sum(ys * weights) / total) + ry1
            return cx, cy, float(np.clip(ear * 1.5, 0.1, 0.7))

    # --- Fallback: geometric centre ---
    return float(geo_center[0]), float(geo_center[1]), 0.1


def estimate_iris_positions(face_kps, image_np, img_width, img_height):
    """Estimate iris centres for both eyes using image-based pupil detection.

    Args:
        face_kps: (69, 3) normalised keypoints [x/W, y/H, conf]
        image_np: (H, W, 3) float32 RGB image [0, 1]
        img_width, img_height: pixel dimensions

    Returns:
        dict with right_iris, left_iris, right_gaze, left_gaze.
    """
    W, H = img_width, img_height
    kps_px = face_kps[:, :2].copy() * np.array([W, H])
    kps_conf = face_kps[:, 2].copy()

    img_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

    results = {}
    for eye_name, eye_idx in [('right', _RIGHT_EYE_IDX),
                               ('left', _LEFT_EYE_IDX)]:
        pts = kps_px[eye_idx]      # (6, 2)
        confs = kps_conf[eye_idx]
        mc = float(np.mean(confs))
        geo = np.mean(pts, axis=0)

        if mc < 0.05:
            results[f'{eye_name}_iris'] = {
                'x': float(geo[0]), 'y': float(geo[1]), 'confidence': 0.0}
            results[f'{eye_name}_gaze'] = {'dx': 0.0, 'dy': 0.0}
            continue

        ix, iy, ic = _find_pupil_center(pts, img_gray, W, H)
        results[f'{eye_name}_iris'] = {'x': ix, 'y': iy, 'confidence': ic}

        dx = ix - float(geo[0])
        dy = iy - float(geo[1])
        norm = max(np.hypot(dx, dy), 1e-6)
        results[f'{eye_name}_gaze'] = {
            'dx': round(dx / norm, 4), 'dy': round(dy / norm, 4)}

    return results


# ---------------------------------------------------
# Debug visualisation overlay
# ---------------------------------------------------
# Colour palette for 68-landmark regions (face-array index ranges)
_LANDMARK_COLORS = [
    (1,  17, (255, 200, 0)),    # jawline
    (18, 22, (200, 255, 0)),    # right eyebrow
    (23, 27, (200, 255, 0)),    # left eyebrow
    (28, 36, (0, 0, 255)),      # nose
    (37, 42, (0, 255, 0)),      # right eye
    (43, 48, (0, 255, 0)),      # left eye
    (49, 60, (0, 255, 255)),    # outer mouth
    (61, 68, (0, 255, 200)),    # inner mouth
]


def draw_debug_overlay(frame_uint8, face_kps_norm, iris_data,
                       face_bbox, body_bbox, W, H):
    """Draw face landmarks, iris positions and bounding boxes for debugging.

    Args:
        frame_uint8:   (H, W, 3) uint8 RGB
        face_kps_norm: (69, 3) normalised keypoints
        iris_data:     dict from estimate_iris_positions
        face_bbox:     (x1, x2, y1, y2) or None
        body_bbox:     [x1, y1, x2, y2, ...] array or None
        W, H:          image pixel dimensions

    Returns:
        vis: (H, W, 3) uint8 RGB image with annotations
    """
    vis = frame_uint8.copy()
    kps_px = face_kps_norm[:, :2] * np.array([W, H])
    kps_conf = face_kps_norm[:, 2]

    # --- Face landmarks ---
    for idx in range(1, min(69, len(kps_px))):
        if kps_conf[idx] < 0.05:
            continue
        x, y = int(kps_px[idx, 0]), int(kps_px[idx, 1])
        if not (0 <= x < W and 0 <= y < H):
            continue
        color = (180, 180, 180)
        for lo, hi, c in _LANDMARK_COLORS:
            if lo <= idx <= hi:
                color = c
                break
        cv2.circle(vis, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 2, color, -1, cv2.LINE_AA)

    # --- Eye contour polylines ---
    for eye_indices in _EYE_CONTOUR_INDICES:
        pts = []
        for i in eye_indices:
            if i < len(kps_px) and kps_conf[i] > 0.05:
                pts.append([int(kps_px[i, 0]), int(kps_px[i, 1])])
        if len(pts) >= 4:
            cv2.polylines(vis, [np.array(pts, np.int32)], True,
                          (0, 255, 0), 1, cv2.LINE_AA)

    # --- Iris markers + gaze arrows ---
    for eye_key, gaze_key in [('right_iris', 'right_gaze'),
                               ('left_iris', 'left_gaze')]:
        iris = iris_data.get(eye_key)
        gaze = iris_data.get(gaze_key)
        if iris is None or iris['confidence'] < 0.05:
            continue
        ix, iy = int(iris['x']), int(iris['y'])
        if 0 <= ix < W and 0 <= iy < H:
            cv2.drawMarker(vis, (ix, iy), (255, 0, 255),
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 5, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"{iris['confidence']:.2f}",
                        (ix + 8, iy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 0, 255), 1, cv2.LINE_AA)
        if gaze and (abs(gaze['dx']) > 1e-4 or abs(gaze['dy']) > 1e-4):
            arrow_len = 35
            ex = int(ix + gaze['dx'] * arrow_len)
            ey = int(iy + gaze['dy'] * arrow_len)
            cv2.arrowedLine(vis, (ix, iy), (ex, ey),
                            (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.3)

    # --- Face bounding box (cyan) ---
    if face_bbox is not None:
        x1, x2, y1, y2 = face_bbox
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "FACE", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # --- Body bounding box (green) ---
    if body_bbox is not None:
        bb = np.asarray(body_bbox).flatten()
        if len(bb) >= 4:
            cv2.rectangle(vis, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, "BODY", (int(bb[0]), int(bb[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------
# ONNX model loader
# ---------------------------------------------------
class OnnxDetectionModelLoaderV2:
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
    CATEGORY = "WanAnimatePreprocess_V2"

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):
        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)
        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)
        return ({"vitpose": vitpose, "yolo": yolo},)


# ---------------------------------------------------
# Pose and Face Detection
# ---------------------------------------------------
class PoseAndFaceDetectionV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048}),
                "detection_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
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
                # Iris estimation
                "use_iris_smoothing": ("BOOLEAN", {"default": True, "tooltip": "Temporally smooth iris positions across frames."}),
                "iris_smoothing_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Higher = more temporal smoothing for iris."}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes", "iris_data", "debug_image")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

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
        use_iris_smoothing,
        iris_smoothing_strength,
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

        # --- Iris estimation (image-based pupil detection) ---
        all_iris = []
        for idx, meta in enumerate(pose_metas):
            iris_result = estimate_iris_positions(
                meta['keypoints_face'], images_np[idx], W, H,
            )
            all_iris.append(iris_result)

        # --- Temporal smoothing for iris positions ---
        if use_iris_smoothing and len(all_iris) > 1:
            strength = float(np.clip(iris_smoothing_strength, 0.0, 1.0))
            for eye_key in ('right_iris', 'left_iris'):
                prev_x = all_iris[0][eye_key]['x']
                prev_y = all_iris[0][eye_key]['y']
                for i in range(1, len(all_iris)):
                    cur = all_iris[i][eye_key]
                    alpha = 1.0 - strength
                    cur['x'] = alpha * cur['x'] + strength * prev_x
                    cur['y'] = alpha * cur['y'] + strength * prev_y
                    prev_x, prev_y = cur['x'], cur['y']

        # Build per-frame iris output
        iris_output = []
        for idx, iris in enumerate(all_iris):
            iris_output.append({
                'frame': idx,
                'right_iris': iris.get('right_iris'),
                'left_iris': iris.get('left_iris'),
                'right_gaze': iris.get('right_gaze'),
                'left_gaze': iris.get('left_gaze'),
            })

        pose_data = {
            "pose_metas": retarget_pose_metas,
            "pose_metas_original": pose_metas,
            "iris_data": all_iris,
        }

        # --- Debug visualisation ---
        debug_frames = []
        for idx in range(B):
            frame = images_np[idx]
            if frame.dtype != np.uint8:
                frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame_u8 = frame.copy()
            vis = draw_debug_overlay(
                frame_u8, pose_metas[idx]['keypoints_face'],
                all_iris[idx], face_bboxes[idx], bboxes[idx], W, H,
            )
            debug_frames.append(vis)
        debug_np = np.stack(debug_frames, 0).astype(np.float32) / 255.0
        debug_tensor = torch.from_numpy(debug_np)

        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints], face_bboxes, json.dumps(iris_output), debug_tensor)


# ---------------------------------------------------
# Draw ViTPose
# ---------------------------------------------------
class DrawViTPoseV2:
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
                "pose_draw_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

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
    "OnnxDetectionModelLoaderV2": OnnxDetectionModelLoaderV2,
    "PoseAndFaceDetectionV2": PoseAndFaceDetectionV2,
    "DrawViTPoseV2": DrawViTPoseV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoaderV2": "ONNX Detection Model Loader (V2)",
    "PoseAndFaceDetectionV2": "Pose and Face Detection (V2)",
    "DrawViTPoseV2": "Draw ViT Pose (V2)",
}
