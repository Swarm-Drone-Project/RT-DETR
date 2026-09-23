"""
COPY THIS FILE if your model isn't one of the standard formats.

    cp adapters/custom_template.py adapters/my_model.py

Edit the two methods below, then run:

    python run_eval.py --weights path/to/best.pt --format my_model

The file name (minus .py) becomes the --format value. The harness finds it
automatically — no registration step.

You only need to get one thing right: predict() returns boxes in ORIGINAL
IMAGE PIXELS. If you resized the image, map them back (scale_boxes does this
for letterbox). Everything else is up to you.
"""

from __future__ import annotations

import numpy as np

from .base import EMPTY, DetectorAdapter, letterbox, nms_numpy, scale_boxes


class CustomAdapter(DetectorAdapter):
    # Must match the filename. `my_model.py` -> "my_model"
    format_name = "custom_template"

    def load(self) -> None:
        """Build the architecture and load weights. Runs once."""
        import torch

        self.torch = torch

        # ── EDIT ME ──────────────────────────────────────────────────────────
        # from my_repo.models import MyDetector
        # self.model = MyDetector(num_classes=1)
        # ckpt = torch.load(self.weights, map_location="cpu")
        # self.model.load_state_dict(ckpt["state_dict"])
        # self.model.to(self.device).eval()
        # ─────────────────────────────────────────────────────────────────────
        raise NotImplementedError("Fill in load()")

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        (H, W, 3) uint8 BGR  ->  (N, 6) [x1, y1, x2, y2, score, class_id]
        in ORIGINAL image pixels. Return EMPTY when nothing is found.

        Include preprocessing and NMS here — the latency timer wraps this whole
        call, and that is what your drone actually pays for.
        """
        orig_hw = image_bgr.shape[:2]

        # ── 1. preprocess (EDIT to match your training pipeline exactly) ──────
        img, gain, pad = letterbox(image_bgr, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(blob)[None].to(self.device)

        # ── 2. forward ───────────────────────────────────────────────────────
        with self.torch.no_grad():
            raw = self.model(tensor)

        # ── 3. decode to xyxy + score + class (EDIT for your output format) ──
        # Example assumes raw is (1, N, 5+nc) with cxcywh + objectness + classes.
        out = raw[0].float().cpu().numpy()
        cxcywh, obj, cls_scores = out[:, :4], out[:, 4:5], out[:, 5:]
        score_all = obj * cls_scores
        cls, score = score_all.argmax(1), score_all.max(1)

        keep = score >= self.conf
        if not keep.any():
            return EMPTY
        cxcywh, score, cls = cxcywh[keep], score[keep], cls[keep]

        xyxy = np.empty_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

        # ── 4. NMS ───────────────────────────────────────────────────────────
        idx = nms_numpy(xyxy, score, cls, self.nms_iou)[: self.max_det]
        dets = np.concatenate(
            [xyxy[idx], score[idx, None], cls[idx, None].astype(np.float32)], axis=1
        )

        # ── 5. map back to original image pixels (DO NOT SKIP) ───────────────
        dets[:, :4] = scale_boxes(dets[:, :4], gain, pad, orig_hw)
        return dets.astype(np.float32)

    def info(self) -> dict:
        base = super().info()
        try:
            base["params"] = int(sum(p.numel() for p in self.model.parameters()))
        except Exception:
            pass
        return base
