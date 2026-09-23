"""
TorchScript adapter — for .torchscript / traced .pt files.

Assumes the traced graph takes (1, 3, H, W) float32 in [0, 1] and emits a raw
YOLO-style tensor. Decode is shared with the ONNX adapter.
"""

from __future__ import annotations

import numpy as np

from .base import EMPTY, DetectorAdapter, letterbox, scale_boxes
from .onnx_adapter import OnnxAdapter


class TorchScriptAdapter(DetectorAdapter):
    format_name = "torchscript"

    def load(self) -> None:
        import torch

        self.torch = torch
        self.model = torch.jit.load(str(self.weights), map_location=self.device)
        self.model.eval()
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
        out = out.float().cpu().numpy()

        dets = OnnxAdapter._decode(self, out)
        if dets.size == 0:
            return EMPTY
        dets[:, :4] = scale_boxes(dets[:, :4], gain, pad, orig_hw)
        return dets.astype(np.float32)

    def sync(self) -> None:
        if self.torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
