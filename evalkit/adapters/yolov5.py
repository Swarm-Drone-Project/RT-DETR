"""
YOLOv5 adapter — checkpoints trained with this repo's own train.py.

Covers the teacher/student KD models, grayscale and RGB, any yolov5*.yaml
variant. These are NOT loadable by the `ultralytics` pip package: the pickled
class is this repo's models.yolo.DetectionModel, and the grayscale ones take
1-channel input, which Ultralytics would silently feed 3 channels.

Nothing needs configuring. The number of input channels, the training image
size, and the class names are all read out of the checkpoint (see
defaults_from_weights below), so `--weights` alone is enough.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .base import EMPTY, DetectorAdapter

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../teacher_student_KD


def _ensure_repo_on_path() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


# At import time, not just inside load(): unpickling one of these checkpoints
# needs models.yolo importable, and torch.load runs during format auto-detection
# before any adapter is instantiated.
_ensure_repo_on_path()


def _input_channels(model) -> int | None:
    """Channels the network actually expects, from its first conv layer."""
    y = getattr(model, "yaml", None)
    if isinstance(y, dict) and y.get("ch"):
        return int(y["ch"])
    for p in model.parameters():
        if p.ndim == 4:  # first conv weight: (out, in, k, k)
            return int(p.shape[1])
    return None


class Yolov5Adapter(DetectorAdapter):
    format_name = "yolov5"

    @classmethod
    def defaults_from_weights(cls, weights: str | Path) -> dict:
        """Pull channels / imgsz / class names straight out of the checkpoint."""
        _ensure_repo_on_path()
        import torch

        out: dict = {}
        try:
            ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            return out

        model = ckpt.get("ema") or ckpt.get("model")
        if model is not None:
            ch = _input_channels(model)
            if ch:
                out["channels"] = ch
            names = getattr(model, "names", None)
            if names:
                out["class_names"] = (
                    list(names.values()) if isinstance(names, dict) else list(names)
                )

        opt = ckpt.get("opt")
        if isinstance(opt, dict) and opt.get("imgsz"):
            imgsz = opt["imgsz"]
            out["imgsz"] = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)

        return out

    def load(self) -> None:
        _ensure_repo_on_path()
        import torch

        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_boxes
        from utils.torch_utils import select_device

        self.torch = torch
        self._letterbox = letterbox
        self._nms = non_max_suppression
        self._scale_boxes = scale_boxes

        dev = str(self.device).replace("cuda:", "").replace("cuda", "0")
        self.torch_device = select_device(dev if dev else "cpu")

        self.model = DetectMultiBackend(
            str(self.weights), device=self.torch_device, fp16=False
        )
        self.stride = max(int(self.model.stride), 32)
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        # `channels` may be forced by the caller (--color); otherwise trust the net.
        forced = self.extra.get("channels")
        self.channels = int(forced) if forced else (_input_channels(self.model.model) or 3)
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        orig_hw = image_bgr.shape[:2]

        im, _, _ = self._letterbox(
            image_bgr, self.imgsz, stride=self.stride, auto=True
        )

        if self.channels == 1:
            # True luminance, matching how training read images
            # (utils/dataloaders.py uses cv2.IMREAD_GRAYSCALE on a BGR file).
            blob = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[None]
        else:
            blob = im.transpose(2, 0, 1)[::-1]  # BGR->RGB, HWC->CHW

        blob = np.ascontiguousarray(blob)
        tensor = self.torch.from_numpy(blob).to(self.torch_device).float() / 255.0
        tensor = tensor[None]

        with self.torch.no_grad():
            pred = self.model(tensor)

        det = self._nms(pred, self.conf, self.nms_iou, max_det=self.max_det)[0]
        if det is None or len(det) == 0:
            return EMPTY

        det = det.clone()
        det[:, :4] = self._scale_boxes(tensor.shape[2:], det[:, :4], orig_hw).round()
        return det.cpu().numpy().astype(np.float32)

    def info(self) -> dict:
        base = super().info()
        base["input_channels"] = getattr(self, "channels", None)
        try:
            base["params"] = int(sum(p.numel() for p in self.model.model.parameters()))
            names = self.model.names
            base["class_names"] = (
                list(names.values()) if isinstance(names, dict) else list(names)
            )
        except Exception:  # noqa: BLE001
            pass
        return base
