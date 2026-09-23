"""
RT-DETRv2 adapter — checkpoints trained with this repo's rtdetrv2_pytorch.

Run it with --format rtdetrv2. The model is built from the
RT-DETR code in ../rtdetrv2_pytorch (src/ + configs/): a .pth here is a bare
state_dict and cannot be loaded without it. Clone the whole RT-DETR repo, or
point RTDETR_ROOT at a rtdetrv2_pytorch/ folder somewhere else.

Usage (from the RT-DETR repo root):

    python evalkit/run_eval.py \
        --format  rtdetrv2 \
        --weights train_robo.pth \
        --dataset data_set/indrones_full_test \
        --split   test \
        --device  cuda

WHY NO LETTERBOX
-----------------
RT-DETR is a set-prediction model: no NMS, and no letterbox. Its
postprocessor (RTDETRPostProcessor.forward, in
src/zoo/rtdetr/rtdetr_postprocessor.py) takes normalised cxcywh boxes and
multiplies them directly by the ORIGINAL image size:

    bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

That only produces correct pixel coordinates if the input was a plain
stretch-resize to (imgsz, imgsz) — which is what every drone config's
val_dataloader does (`{type: Resize, size: [640, 640]}`, no aspect-ratio
padding). So predict() below stretch-resizes too, and boxes come back in
original pixels with nothing left to undo — no scale_boxes() call needed.

CONFIG <-> CHECKPOINT PAIRING
------------------------------
A .pth here is a bare state_dict, not a self-describing export — it does not
say which config built it. train_robo.pth was identified by inspecting its
state_dict directly (PResNet stem takes 1 input channel, 2 basic blocks per
res_layer stage with no branch2c => depth=18, decoder score heads output
dim 1 => num_classes=1, no temporal_fusion keys), which matches
configs/rtdetrv2/rtdetrv2_r18vd_drone_gray1ch.yml exactly. If you add another
checkpoint, add its config below — guessing wrong fails load_state_dict
loudly (shape mismatch), it does not silently score garbage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .base import EMPTY, DetectorAdapter

# .../RT-DETR/rtdetrv2_pytorch, unless RTDETR_ROOT says otherwise
_REPO_ROOT = Path(
    os.environ.get("RTDETR_ROOT")
    or Path(__file__).resolve().parents[2] / "rtdetrv2_pytorch"
).expanduser().resolve()


def _ensure_repo_on_path() -> None:
    # Only inside load(): rtdetrv2_pytorch/ has top-level folders (dataset/,
    # tools/) that should not shadow anything for runs of other formats.
    if not (_REPO_ROOT / "src" / "core").is_dir():
        raise RuntimeError(
            f"RT-DETR code not found at {_REPO_ROOT}. Clone the full RT-DETR "
            f"repo, or set RTDETR_ROOT=/path/to/RT-DETR/rtdetrv2_pytorch"
        )
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

_CONFIG_FOR_WEIGHTS = {
    "train_robo.pth": "configs/rtdetrv2/rtdetrv2_r18vd_drone_gray1ch.yml",
}
_DEFAULT_CONFIG = "configs/rtdetrv2/rtdetrv2_r18vd_drone_gray1ch.yml"


class RTDETRv2Adapter(DetectorAdapter):
    format_name = "rtdetrv2"

    @classmethod
    def defaults_from_weights(cls, weights) -> dict:
        # This repo's drone configs are all single-class grayscale at 640.
        # If you add an RGB or multi-class config, make this read the yaml
        # instead of hardcoding.
        return {"imgsz": 640, "channels": 1, "class_names": ["drone"]}

    def load(self) -> None:
        _ensure_repo_on_path()
        import torch

        from src.core import YAMLConfig

        self.torch = torch

        config_name = _CONFIG_FOR_WEIGHTS.get(self.weights.name, _DEFAULT_CONFIG)
        config_path = _REPO_ROOT / config_name
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No config mapped for {self.weights.name}. Edit "
                f"_CONFIG_FOR_WEIGHTS in {__file__}."
            )

        # pretrained=False: we load a full checkpoint below, so skip
        # PResNet's own ImageNet-weights download (presnet.py calls
        # torch.hub.load_state_dict_from_url when pretrained=True) — it
        # would just be overwritten, and may not have internet at eval time.
        cfg = YAMLConfig(str(config_path), PResNet={"pretrained": False})

        ckpt = torch.load(str(self.weights), map_location="cpu", weights_only=False)
        state = ckpt["ema"]["module"] if ckpt.get("ema") else ckpt["model"]
        cfg.model.load_state_dict(state)

        self.model = cfg.model.deploy().to(self.device)
        self.postprocessor = cfg.postprocessor.deploy().to(self.device)
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]

        resized = cv2.resize(
            image_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blob = np.ascontiguousarray(gray[None], dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(blob)[None].to(self.device)
        orig_size = self.torch.tensor([[w, h]], dtype=self.torch.float32).to(self.device)

        with self.torch.no_grad():
            outputs = self.model(tensor)
            labels, boxes, scores = self.postprocessor(outputs, orig_size)

        labels, boxes, scores = labels[0], boxes[0], scores[0]
        keep = scores >= self.conf
        if not bool(keep.any()):
            return EMPTY

        dets = self.torch.cat(
            [boxes[keep], scores[keep, None], labels[keep, None].float()], dim=1
        )
        return dets.cpu().numpy().astype(np.float32)

    def info(self) -> dict:
        base = super().info()
        base["input_channels"] = 1
        try:
            base["params"] = int(sum(p.numel() for p in self.model.parameters()))
        except Exception:  # noqa: BLE001
            pass
        return base
