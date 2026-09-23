"""
Ultralytics adapter — YOLOv5u/v8/v9/v10/v11/v12 and RT-DETR .pt checkpoints.

Zero user code required. We hand Ultralytics the raw BGR frame and let it do
its own letterboxing and NMS, which is exactly what happens in deployment, so
the numbers reflect the real pipeline rather than our reimplementation of it.
"""

from __future__ import annotations

import numpy as np

from .base import EMPTY, DetectorAdapter


class UltralyticsAdapter(DetectorAdapter):
    format_name = "ultralytics"

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "pip install ultralytics  (needed for --format ultralytics)"
            ) from e

        self.model = YOLO(str(self.weights))
        self.model.to(self.device)
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        # source=ndarray -> Ultralytics treats it as BGR, does its own letterbox.
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.nms_iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return EMPTY

        # .xyxy is already mapped back to original image pixels by Ultralytics.
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)[:, None]
        cls = boxes.cls.cpu().numpy().astype(np.float32)[:, None]
        return np.concatenate([xyxy, conf, cls], axis=1)

    def info(self) -> dict:
        base = super().info()
        try:
            params = sum(p.numel() for p in self.model.model.parameters())
            base["params"] = int(params)
            names = self.model.model.names
            base["class_names"] = (
                list(names.values()) if isinstance(names, dict) else list(names)
            )
        except Exception:
            pass
        return base
