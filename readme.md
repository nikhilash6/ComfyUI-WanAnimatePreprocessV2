## THIS IS FROM THE GITHUB ISSUE #1410 of WanVideoPreProccessor and Improved pose and face detection `#10` issue of WanAnimatePreProcessor

Special thanks to [@kijai](https://github.com/kijai) and [@steven850](https://github.com/steven850)


## NOTES FOR USAGE

Detection threshold: affects filtering inside YOLO’s postprocessing.
At 0.01 → more detections, even when people are partly occluded.
At 0.1 → fewer detections, possibly more flicker.
(the defaults I set work well and dont need to be adjusted for most videos, but handy to have just in case)

Pose Threshold: zeroes out low-confidence keypoints in the stored pose data (pre-draw). (Again applied defaults work well here. Only adjust if all else fails.)

use CLAHE: ViTPose is extremely sensitive to lighting, contrast, and body scale. So I added CALHE, its dynamic contrast enhancement, helps ViT detect a bit better especially on videos where the subject and background color is similar.

Use blur for pose: Pose models like slightly blurred, low-frequency input.
It suppresses texture noise that confuses keypoint heatmaps — so limbs stop flickering and vanishing. blurring stabilizes ViTPose and YOLO’s bounding boxes.
ViTPose (and its close cousins like HRNet and SimpleBaseline) predict 2-D heatmaps for each joint.
Those heatmaps are trained on images where the human figure was rescaled and normalized (not high-frequency, over-sharpened detail).
The backbone (a Vision Transformer) is tuned to capture shape and structure, not pixel-level texture.

So, when you feed the model frames with:
strong texture (noise, fabric patterns, hair detail), or camera compression artifacts,
the self-attention layers get confused.
You end up with multiple small heatmap activations instead of one confident blob per joint → “vanishing limbs.”

A mild Gaussian blur suppresses those noisy, high-frequency features, making the joint blobs cleaner and more stable.
this gets rid of almost all jitter on the bones, and stops them vanishing as the system can now get clean track without getting distracted by noise. (its only applied to the images fed to the YOLO and ViT, the face crop is performed on the original unblurred images.)
Blur radius: Setting for blur radius, 5 is good for most videos.
blur sigma: Sigma setting, 2 seems fine.
Use face smoothing: A temporal smoothing system for the face bounding boxes, stop the face crops from jittering so animate can get a clean read on the expressions.
by dynamically adjusting the smoothing strength per frame based on how much the face bbox moved from the previous frame.
If the movement is small → apply strong smoothing.
If the movement is large → reduce smoothing so the face doesn’t “lag” behind.

Movement | Dynamic result -- | -- Face still | Very smooth, barely moves Small motion | Mild smoothing, still steady Big motion (turn/nod) | Quickly adapts, tracks face immediately
face smoothing strength: 0, no smoothing, 1 heavy smoothing.
Use constant face box: The issue is that ViTPose’s face keypoints only cover a small region — mainly around the eyes, nose, and mouth (not the full head outline).
So when the head tilts or rotates, the top (forehead/hair) and bottom (chin/neck) move outside that tight cluster of keypoints, and the bbox doesn’t follow.
That’s why in profile or tilted poses, you see the face crop “cut off” at the eyebrows or chin. ViTPose also predicts body joints — and the “head top” / “neck” points are available in keypoints_body.
so by including those in the face bbox computation, you get a much more stable crop that doesnt cut off parts of the face.
It works as follows:
Compute the current bbox from the visible keypoints.
Compute a weighted center that nudges toward the expected head center.
Use additional cues from body keypoints (neck, shoulders).
If available, track the previous frame’s center and keep it stable unless the movement is large.
Expand the box symmetrically around that center, keeping its size constant.
Clamp to image bounds.
That way, the box slides smoothly to follow the face, instead of shrinking or drifting.
Also runs off the face smoothing parameter, so it also adjust the smoothing amount on the face box.

Face box size: Sets the size of the face box, 224 works as a starting point, just have to look at the face images output and adjust if needed to encompass the entire face.

Subject framing | Suggested face_box_size_px | Notes -- | -- | -- Close-up (head fills frame) | 320–448 | Ensures forehead + chin stay in Medium shot (chest-up) | 256–320 | Your “224–256” sweet spot also fine depending on subject Full-body | 192–256 | Zooms in more, but still catches head turns Wide (small head) | 128–192 | Tight face crops; increase if you see clipping This is essentially a zoom, the output of the face crops is always 512x512 regardless of what size this is set to, as this is what animate expects.
Draw ViT Pose:

Image
Pose draw threshold: the one that actually controls whether a joint/limb is drawn.
Set this lower (e.g., 0.05–0.2) to keep limbs from vanishing.

PoseAndFaceDetection Data Flow

 ┌────────────────────────────┐
 │         Input:             │
 │  images (B × H × W × C)    │
 └────────────┬───────────────┘
              │
              ▼
      [ Stage 1: Preprocess ]
 ┌────────────────────────────┐
 │ Optional CLAHE (contrast)  │
 │ Optional Gaussian Blur     │
 │ → produces two streams:    │
 │   • images_blurred (for detection & pose)  
 │   • images_np (original for outputs)      │
 └────────────┬───────────────┘
              │
              ▼
     [ Stage 2: YOLO Detection ]
 ┌────────────────────────────┐
 │ YOLO runs on images_blurred│
 │ → returns bounding boxes   │
 │   for each detected person │
 └────────────┬───────────────┘
              │
              ▼
     [ Stage 3: ViTPose Inference ]
 ┌────────────────────────────┐
 │ For each YOLO bbox:        │
 │   - crop & rescale region  │
 │   - CLAHE (optional)       │
 │   - normalize              │
 │ ViTPose predicts joints    │
 │ → keypoints per frame      │
 └────────────┬───────────────┘
              │
              ▼
     [ Stage 4: Pose Meta Extraction ]
 ┌────────────────────────────┐
 │ Convert ViTPose keypoints  │
 │ → human pose metadata      │
 │ Includes body + face       │
 └────────────┬───────────────┘
              │
              ▼
     [ Stage 5: Face BBox Stabilization ]
 ┌──────────────────────────────────────────────────┐
 │ Compute bbox around face + upper-body joints     │
 │ Centered using `stabilize_face_bbox()`           │
 │ → Keeps box constant-size (512×512)              │
 │ → Adapts center for tilt / lying-down poses      │
 │ → Temporally smoothed center                    │
 └────────────┬─────────────────────────────────────┘
              │
              ▼
     [ Stage 6: Face BBox Temporal Smoothing ]
 ┌──────────────────────────────────────┐
 │ Exponential smoothing of bbox motion │
 │   α = f(face_smoothing_strength, motion) │
 │ → Dynamic adaptation: stable when still, │
 │   responsive when moving                │
 └────────────┬────────────────────────────┘
              │
              ▼
     [ Stage 7: Face Cropping (final) ]
 ┌──────────────────────────────────────┐
 │ Crop from *unblurred* original image │
 │ using stabilized/smoothed bbox       │
 │ Resize to constant 512×512           │
 └────────────┬──────────────────────────┘
              │
              ▼
     [ Stage 8: Pose Data Packaging ]
 ┌────────────────────────────┐
 │ pose_data (keypoints, etc) │
 │ face_images (512×512)      │
 │ face_bboxes (coords)       │
 │ key_frame_body_points JSON │
 └────────────────────────────┘
🧩 Summary
Stage	Purpose	Input	Output
1	CLAHE + Blur	Raw image	Enhanced/blurred copy
2	YOLO detect	Blurred	Person bbox
3	ViTPose	Blurred crops	Keypoints
4	Convert	Keypoints	Pose metas
5	Stabilize bbox	Pose metas	Centered 512×512 region
6	Temporal smooth	bbox sequence	Stable motion
7	Crop face	Original image	Sharp 512×512 face
8	Package	All results	PoseData, Faces, JSON
Attached is the nodes.py and onnx_models.py(a few small changes to ensure threshold settings are passed)
Day and night difference on tracking quality. I hope you like it and add it to the repo.
I really appreciate all the work that you do, I figured this was the best way for me to help.


## ComfyUI helper nodes for [Wan video 2.2 Animate preprocessing](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)

Nodes to run the ViTPose model, get face crops and keypoint list for SAM2 segmentation.

Models:

to `ComfyUI/models/detection` (subject to change in the future)

YOLO:

https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/process_checkpoint/det/yolov10m.onnx

ViTPose ONNX:

Use either the Large model from here:

https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/wholebody

Or the Huge model like in the original code, it's split into two files due to ONNX file size limit:

Both files need to be in same directory, and the onnx file selected in the model loader:

`vitpose_h_wholebody_data.bin` and `vitpose_h_wholebody_model.onnx`

https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx


![example](example.png)
