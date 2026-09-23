"""
COPY THIS FILE INTO YOUR OWN REPO, next to your own model code.

    cp evalkit/templates/external_predictor.py ~/my_detector/evalkit_predictor.py

Then edit the two marked methods and run, from YOUR environment with YOUR
dependencies installed:

    python /path/to/evalkit/run_eval.py \
        --predictor ~/my_detector/evalkit_predictor.py \
        --weights ~/my_detector/checkpoints/best.pt \
        --dataset /shared/drone_dataset

You never edit anything inside evalkit, and evalkit never needs your packages
installed in its own environment — this file is imported from where it lives, so
`import my_detector` works because your repo is on the path.

You only have to get ONE thing right:

    predict(image_bgr) -> (N, 6) array of [x1, y1, x2, y2, score, class_id]
                          in ORIGINAL image pixels

If you resized the image to run the model, map the boxes back. `scale_boxes()`
does it for the standard letterbox case. Everything else — the resize strategy,
normalisation, colour space, batching, NMS — is yours, and should match what you
do in deployment. evalkit deliberately does not standardise preprocessing: if it
did, a model trained at 512 with stretch-resize would look artificially bad and
you'd be comparing preprocessing instead of models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make evalkit's helpers importable no matter where this file ended up.
# Walks up from here looking for the evalkit folder, so all the usual layouts
# work with no editing: evalkit inside your project, beside it, or further up.
def _find_evalkit(start: Path) -> Path | None:
    for d in (start, *start.parents):
        for cand in (d / "evalkit", d):
            if (cand / "adapters" / "base.py").is_file():
                return cand
    return None


_EVALKIT = _find_evalkit(Path(__file__).resolve().parent)
if _EVALKIT is None:
    raise ImportError(
        "Could not find evalkit from " + str(Path(__file__).resolve().parent) + ".\n"
        "Either put the evalkit folder somewhere above this file, or replace this "
        "block with:  sys.path.insert(0, '/absolute/path/to/evalkit')"
    )
sys.path.insert(0, str(_EVALKIT))

from adapters.base import (  # noqa: E402
    EMPTY,
    DetectorAdapter,
    letterbox,
    nms_numpy,
    scale_boxes,
)


class MyModelAdapter(DetectorAdapter):
    """Rename this to whatever you like — evalkit finds it by subclass, not name."""

    #: Shows up in results and on the leaderboard. Make it recognisable.
    format_name = "my_model"

    # ── OPTIONAL, but nice: let evalkit read settings out of your checkpoint so
    #    you don't have to pass --imgsz / --color. Delete if not applicable.
    @classmethod
    def defaults_from_weights(cls, weights):
        # e.g. return {"imgsz": 960, "channels": 1, "class_names": ["drone"]}
        return {}

    # ── EDIT ME ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        """
        Build your architecture and load weights. Called once, before predict().

        Available to you here:
            self.weights   Path to the checkpoint (--weights)
            self.imgsz     int (--imgsz, or from defaults_from_weights)
            self.conf      float, confidence floor set by the frozen protocol
            self.nms_iou   float, NMS IoU set by the frozen protocol
            self.max_det   int
            self.device    "cuda" / "cuda:0" / "cpu"
        """
        import torch

        self.torch = torch

        # from my_detector.models import MyDetector
        # self.model = MyDetector(num_classes=1)
        # ckpt = torch.load(self.weights, map_location="cpu")
        # self.model.load_state_dict(ckpt["state_dict"])
        # self.model.to(self.device).eval()

        raise NotImplementedError("Fill in load()")

    # ── EDIT ME ──────────────────────────────────────────────────────────────
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        (H, W, 3) uint8 BGR at original resolution
            -> (N, 6) float32 [x1, y1, x2, y2, score, class_id] in original pixels

        Return EMPTY when nothing is found. Include preprocessing and NMS here:
        the latency timer wraps this whole call, and that is what your drone
        actually pays for.

        The example below is a plain YOLO-style pipeline. Replace the parts that
        don't match your model.
        """
        orig_hw = image_bgr.shape[:2]

        # 1. preprocess — must match how you trained
        img, gain, pad = letterbox(image_bgr, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(blob)[None].to(self.device)

        # 2. forward
        with self.torch.no_grad():
            raw = self.model(tensor)

        # 3. decode to xyxy + score + class.
        #    This example assumes (1, N, 5+nc) with cxcywh + obj + class scores.
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

        # 4. NMS (skip if your model already did it)
        idx = nms_numpy(xyxy, score, cls, self.nms_iou)[: self.max_det]
        dets = np.concatenate(
            [xyxy[idx], score[idx, None], cls[idx, None].astype(np.float32)], axis=1
        )

        # 5. back to original image pixels — DO NOT SKIP
        dets[:, :4] = scale_boxes(dets[:, :4], gain, pad, orig_hw)
        return dets.astype(np.float32)

    def info(self) -> dict:
        base = super().info()
        try:
            base["params"] = int(sum(p.numel() for p in self.model.parameters()))
        except Exception:  # noqa: BLE001
            pass
        return base
