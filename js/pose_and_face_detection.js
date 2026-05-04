// PoseAndFaceDetectionV2: conditional widget visibility for blur/smoothing/constant-box/iris settings.
import { app } from "../../scripts/app.js";

const ALWAYS = new Set([
    "width","height","detection_threshold","pose_threshold",
    "use_clahe","use_blur_for_pose","use_face_smoothing","use_constant_face_box",
    "use_iris_smoothing","use_mediapipe_face",
]);

function setHidden(w, hidden) {
    if (hidden) {
        if (!w.__origComputeSize) {
            w.__origComputeSize = w.computeSize;
            w.__origType = w.type;
        }
        w.computeSize = () => [0, -4];
        w.hidden = true;
        w.type = "hidden";
    } else {
        if (w.__origComputeSize) {
            w.computeSize = w.__origComputeSize;
            delete w.__origComputeSize;
        }
        w.hidden = false;
        w.type = w.__origType || w.type;
    }
}

function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);
    const useBlur     = !!get("use_blur_for_pose")?.value;
    const faceSmooth  = !!get("use_face_smoothing")?.value;
    const constBox    = !!get("use_constant_face_box")?.value;
    const irisSmooth  = !!get("use_iris_smoothing")?.value;
    const conditional = {
        blur_radius:              useBlur,
        blur_sigma:               useBlur,
        face_smoothing_strength:  faceSmooth,
        face_box_size_px:         constBox,
        iris_smoothing_strength:  irisSmooth,
    };
    for (const w of node.widgets) {
        if (ALWAYS.has(w.name)) { setHidden(w, false); continue; }
        if (w.name in conditional) { setHidden(w, !conditional[w.name]); continue; }
        setHidden(w, false);
    }
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = (v, ...rest) => {
        const r = orig?.call(w, v, ...rest);
        applyVisibility(node);
        return r;
    };
}

app.registerExtension({
    name: "WanAnimateV2.PoseAndFaceDetectionV2.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PoseAndFaceDetectionV2") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            for (const n of ["use_blur_for_pose","use_face_smoothing","use_constant_face_box","use_iris_smoothing"]) {
                hookWidget(this, n);
            }
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
    },
});
