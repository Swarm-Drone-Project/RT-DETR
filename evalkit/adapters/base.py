"""
base.py — the one interface every model must satisfy.

THE ENTIRE CONTRACT
-------------------
Given a raw BGR image (whatever size it is on disk), return an (N, 6) float32
array of detections in ORIGINAL IMAGE PIXEL COORDINATES:

    [x1, y1, x2, y2, score, class_id]

That's it. How you resize, normalise, batch, decode, or run NMS is entirely
your business — the harness never looks inside. This is deliberate: if the
harness did the preprocessing, we would be benchmarking the harness, not the
model. A model trained with stretch-resize at 512 must be free to do exactly
that at eval time, or its numbers are meaningless.

Boxes must be in original image pixels. If you resized the image to run the
model, you MUST map the boxes back. `scale_boxes()` below does this for the
standard letterbox case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

EMPTY = np.zeros((0, 6), dtype=np.float32)


class DetectorAdapter(ABC):
    """Wraps one model so the harness can treat every model identically."""

    #: short string used in --format and in output filenames
    format_name: str = "base"

    #: Set True if your model needs frames in the order they were filmed —
    #: motion detection, tracking, temporal smoothing, anything with state that
    #: carries between frames. evalkit then walks the split video by video in
    #: frame order and calls reset() at every boundary, instead of treating the
    #: images as an unordered bag. Leave False for a plain per-image detector.
    sequential: bool = False

    def __init__(
        self,
        weights: str | Path,
        imgsz: int = 640,
        conf: float = 0.001,
        nms_iou: float = 0.65,
        device: str = "cuda",
        max_det: int = 300,
        **extra,
    ):
        self.weights = Path(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.nms_iou = float(nms_iou)
        self.device = device
        self.max_det = int(max_det)
        self.extra = extra
        self._loaded = False

    # ── must implement ────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the device. Called once before any predict()."""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: (H, W, 3) uint8 BGR, original resolution straight from disk.

        Returns:
            (N, 6) float32 — [x1, y1, x2, y2, score, class_id] in ORIGINAL
            image pixels. Return EMPTY (0, 6) when nothing is detected.

        Must include everything the deployed system pays for: preprocessing,
        forward pass, decode, and NMS. The latency timer wraps this call.
        """

    # ── optional: self-configuration ──────────────────────────────────────────

    @classmethod
    def defaults_from_weights(cls, weights: str | Path) -> dict:
        """
        Read settings the checkpoint already knows about so the user doesn't
        have to type them. May return any of: imgsz, channels, class_names.

        Anything returned here is a DEFAULT — an explicit CLI flag always wins.
        Returning {} means "nothing detectable", and the harness falls back to
        its own defaults.
        """
        return {}

    # ── provided ──────────────────────────────────────────────────────────────

    def warmup(self, image_bgr: np.ndarray, n: int = 10) -> None:
        """
        Amortise CUDA kernel compilation / lazy graph building.

        Note for sequential models: this feeds the SAME frame n times, which
        would leave a tracker convinced of a stationary target. evalkit calls
        reset() immediately afterwards, so any state built here is discarded —
        but if warmup itself is harmful for your model, override it to a no-op.
        """
        for _ in range(n):
            self.predict(image_bgr)

    def reset(self) -> None:
        """
        Clear per-sequence state. Called before the first frame of every video
        when `sequential = True`, and after warmup.

        Override it if you keep a previous frame, a track, a Kalman filter or a
        frame counter. Without this, a tracker carries its target across the cut
        between two unrelated videos and scores the next clip against a stale
        position. No-op for stateless detectors.
        """

    def info(self) -> dict:
        """Model size and parameter count. Override if you can do better."""
        size_mb = self.weights.stat().st_size / 1e6 if self.weights.exists() else 0.0
        return {"params": None, "weights_mb": round(size_mb, 2)}

    def sync(self) -> None:
        """Block until the device has finished. Override for non-CUDA backends."""
        try:
            import torch

            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        except ImportError:
            pass


# ── helpers for adapters that do their own preprocessing ─────────────────────


def letterbox(
    img: np.ndarray, new_size: int = 640, color: tuple = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize preserving aspect ratio, pad the remainder.
    Returns (padded_image, gain, (pad_w, pad_h)) — feed gain/pad to scale_boxes().
    """
    h, w = img.shape[:2]
    gain = min(new_size / h, new_size / w)
    nh, nw = int(round(h * gain)), int(round(w * gain))
    pad_h, pad_w = (new_size - nh) / 2, (new_size - nw) / 2

    if (h, w) != (nh, nw):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, gain, (pad_w, pad_h)


def scale_boxes(
    boxes: np.ndarray, gain: float, pad: tuple[float, float], orig_hw: tuple[int, int]
) -> np.ndarray:
    """Undo letterbox: model-input xyxy -> original-image xyxy, clipped to frame."""
    if boxes.size == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    h, w = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float
) -> np.ndarray:
    """Class-aware greedy NMS. Returns indices to keep. Fallback when no torchvision."""
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    # offset boxes by class so different classes never suppress each other
    offset = classes.astype(np.float32) * (boxes.max() + 1.0)
    b = boxes + offset[:, None]

    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]

    return np.array(keep, dtype=np.int64)
