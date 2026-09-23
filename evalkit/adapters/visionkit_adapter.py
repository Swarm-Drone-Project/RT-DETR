"""
VisionKit adapter — for the team's own native plugins
(yolo, yolo_fire, yolo_mobile, rtdetr, and anything added later).

Requires the visionkit package to be importable and --model-name to be passed:

    python run_eval.py --weights weights/yolo_fire/best.pt \\
                       --format visionkit --model-name yolo_fire

Note: the old benchmarker hardcoded load_model("yolo"), so every non-yolo
native checkpoint was silently scored against the baseline architecture.
This honours the name you pass.
"""

from __future__ import annotations

import numpy as np

from .base import EMPTY, DetectorAdapter, letterbox, nms_numpy, scale_boxes


class VisionKitAdapter(DetectorAdapter):
    format_name = "visionkit"

    def load(self) -> None:
        import torch

        from visionkit.models import load_model, load_model_weights

        self.torch = torch
        model_name = self.extra.get("model_name")
        if not model_name:
            raise ValueError(
                "--format visionkit requires --model-name "
                "(e.g. yolo, yolo_fire, yolo_mobile, rtdetr)"
            )

        nc = int(self.extra.get("nc", 1))
        model = load_model(model_name, nc=nc)
        model = load_model_weights(model, self.weights, map_location=str(self.device))
        self.model = model.to(self.device).eval()
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        orig_hw = image_bgr.shape[:2]
        img, gain, pad = letterbox(image_bgr, self.imgsz)

        blob = img[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        t = self.torch.from_numpy(blob)[None].to(self.device)

        with self.torch.no_grad():
            out = self.model(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.float().cpu().numpy()[0]  # (N, 5+nc)

        if out.shape[0] < out.shape[1]:
            out = out.T

        cxcywh, obj, cls_scores = out[:, :4], out[:, 4:5], out[:, 5:]
        score_all = obj * cls_scores
        cls = score_all.argmax(1)
        score = score_all.max(1)

        keep = score >= self.conf
        if not keep.any():
            return EMPTY
        cxcywh, score, cls = cxcywh[keep], score[keep], cls[keep]

        xyxy = np.empty_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

        idx = nms_numpy(xyxy, score, cls, self.nms_iou)[: self.max_det]
        dets = np.concatenate(
            [xyxy[idx], score[idx, None], cls[idx, None].astype(np.float32)], axis=1
        )
        dets[:, :4] = scale_boxes(dets[:, :4], gain, pad, orig_hw)
        return dets.astype(np.float32)

    def info(self) -> dict:
        base = super().info()
        try:
            base["params"] = int(sum(p.numel() for p in self.model.parameters()))
        except Exception:
            pass
        return base

    def sync(self) -> None:
        if self.torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
