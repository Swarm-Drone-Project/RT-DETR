"""
ONNX adapter — for .onnx exports.

Handles the three layouts you'll actually meet:
  A) (1, 4+nc, N)  — YOLOv8/v11 raw export, no NMS baked in
  B) (1, N, 4+nc)  — same but transposed
  C) (1, N, 6)     — NMS already baked in (end2end / YOLOv10), [x1,y1,x2,y2,score,cls]

If your export does something else, copy custom_template.py instead of
fighting this file.
"""

from __future__ import annotations

import numpy as np

from .base import EMPTY, DetectorAdapter, letterbox, nms_numpy, scale_boxes


class OnnxAdapter(DetectorAdapter):
    format_name = "onnx"

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("pip install onnxruntime-gpu  (or onnxruntime)") from e

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if str(self.device).startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self.sess = ort.InferenceSession(str(self.weights), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

        # Trust the graph's own input size over the CLI flag when it is static.
        shape = self.sess.get_inputs()[0].shape
        if isinstance(shape[-1], int) and isinstance(shape[-2], int):
            self.imgsz = int(shape[-1])
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        orig_hw = image_bgr.shape[:2]
        img, gain, pad = letterbox(image_bgr, self.imgsz)

        blob = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None]

        out = self.sess.run(None, {self.input_name: blob})[0]
        dets = self._decode(np.asarray(out))
        if dets.size == 0:
            return EMPTY

        dets[:, :4] = scale_boxes(dets[:, :4], gain, pad, orig_hw)
        return dets.astype(np.float32)

    # ── output-layout handling ────────────────────────────────────────────────

    def _decode(self, out: np.ndarray) -> np.ndarray:
        out = out[0] if out.ndim == 3 else out  # drop batch

        # Layout C: NMS already applied
        if out.ndim == 2 and out.shape[1] == 6:
            keep = out[:, 4] >= self.conf
            return out[keep]

        # Layouts A/B: raw. N (e.g. 8400) always dwarfs 4+nc, so orient by size.
        if out.shape[0] < out.shape[1]:
            out = out.T  # (4+nc, N) -> (N, 4+nc)

        boxes_cxcywh, scores_all = out[:, :4], out[:, 4:]
        cls = scores_all.argmax(1)
        score = scores_all.max(1)

        keep = score >= self.conf
        if not keep.any():
            return EMPTY
        boxes_cxcywh, score, cls = boxes_cxcywh[keep], score[keep], cls[keep]

        xyxy = np.empty_like(boxes_cxcywh)
        xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        idx = nms_numpy(xyxy, score, cls, self.nms_iou)[: self.max_det]
        return np.concatenate(
            [xyxy[idx], score[idx, None], cls[idx, None].astype(np.float32)], axis=1
        )
