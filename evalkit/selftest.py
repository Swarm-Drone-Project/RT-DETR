#!/usr/bin/env python3
"""
selftest.py — verify the harness works without needing a real model or dataset.

Builds a synthetic drone dataset (small bright blobs on noisy sky, including
empty frames), then runs a deliberately imperfect fake detector through the
full pipeline. Confirms the metrics are sane and every plot renders.

    python selftest.py

Run this after cloning to check your install, and after editing metrics.py.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.base import EMPTY, DetectorAdapter  # noqa: E402
from core.dataset import EvalDataset  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from core.plots import generate_all  # noqa: E402
from core.predict import measure_latency, run_predictions  # noqa: E402

RNG = np.random.default_rng(0)


def build_dataset(root: Path, n_images: int = 60, n_empty: int = 15) -> None:
    if root.exists():
        shutil.rmtree(root)
    img_dir, lbl_dir = root / "images" / "val", root / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (root / "data.yaml").write_text("nc: 1\nnames: ['drone']\n")

    H, W = 480, 640
    for i in range(n_images):
        img = RNG.integers(140, 190, (H, W, 3), dtype=np.uint8)  # grey sky
        rows = []
        if i >= n_empty:
            # 1-3 targets, deliberately spanning tiny -> large
            for _ in range(int(RNG.integers(1, 4))):
                size = int(RNG.choice([8, 14, 22, 40, 70]))
                cx = int(RNG.integers(size, W - size))
                cy = int(RNG.integers(size, H - size))
                cv2.circle(img, (cx, cy), size // 2, (40, 40, 45), -1)
                rows.append(f"0 {cx/W:.6f} {cy/H:.6f} {size/W:.6f} {size/H:.6f}")
        cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)
        (lbl_dir / f"img_{i:03d}.txt").write_text("\n".join(rows))


class FakeDetector(DetectorAdapter):
    """
    Blob detector with realistic failure modes: jittered boxes, confidence that
    drops with object size, occasional misses, and spurious detections. Exists
    so the metrics have something imperfect to score.
    """

    format_name = "fake"

    def load(self) -> None:
        self.rng = np.random.default_rng(1)
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey, 90, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 3 or h < 3:
                continue
            side = max(w, h)
            if side < 12 and self.rng.random() < 0.45:  # miss small targets often
                continue
            j = side * 0.10
            box = [
                x + self.rng.normal(0, j), y + self.rng.normal(0, j),
                x + w + self.rng.normal(0, j), y + h + self.rng.normal(0, j),
            ]
            score = float(np.clip(0.35 + side / 90 + self.rng.normal(0, 0.10), 0.02, 0.99))
            out.append(box + [score, 0.0])

        if self.rng.random() < 0.30:  # spurious detection
            x = self.rng.uniform(0, 560); y = self.rng.uniform(0, 400)
            out.append([x, y, x + 25, y + 25, float(self.rng.uniform(0.05, 0.45)), 0.0])

        if not out:
            return EMPTY
        d = np.array(out, dtype=np.float32)
        return d[d[:, 4] >= self.conf]


def main() -> int:
    tmp = Path("_selftest")
    ds_root, out_dir = tmp / "dataset", tmp / "results" / "fake_model"

    print("building synthetic dataset ...")
    build_dataset(ds_root)

    ds = EvalDataset(ds_root, "val")
    print(f"  {ds.summary()}")
    assert ds.nc == 1 and ds.class_names == ["drone"], "class parsing failed"
    assert ds.n_empty == 15, f"expected 15 empty frames, got {ds.n_empty}"

    adapter = FakeDetector(weights=Path("fake.pt"), imgsz=640, conf=0.01, device="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_gt(out_dir / "gt.json")

    dets, raw = run_predictions(adapter, ds, out_dir / "preds.json", warmup=1)
    print(f"  {len(dets)} detections")
    assert dets, "detector produced nothing"

    m = compute_metrics(ds, dets, raw, operating_conf=0.25)
    a, bs, c, op = m["accuracy"], m["by_size"], m["curve"], m["operating_point"]

    print("\nmetrics")
    print(f"  mAP@0.50:0.95   {a['map50_95']:.4f}")
    print(f"  mAP@0.50        {a['map50']:.4f}")
    print(f"  AR@100          {a['ar_100']:.4f}")
    print("  by size:", {k: v["map50"] for k, v in bs.items()})
    print(f"  best F1         {c['best_f1']:.4f} @ conf {c['best_f1_conf']:.3f}")
    print(f"  R@P95           {c['recall_at_p95']:.4f}")
    print(f"  FP/empty frame  {op['fp_per_empty_frame']:.3f}")
    print(f"  confusion       {op['confusion_matrix']}")

    # sanity assertions
    assert 0.0 < a["map50"] <= 1.0, "mAP@0.50 out of range"
    assert a["map50"] >= a["map50_95"], "mAP@0.50 must be >= mAP@0.50:0.95"
    assert a["map50"] >= a["map75"], "mAP@0.50 must be >= mAP@0.75"
    assert len(m["_plot_data"]["iou_thrs"]) == 10, "COCO sweep must be 10 IoU thresholds"
    assert abs(m["_plot_data"]["iou_thrs"][0] - 0.50) < 1e-6
    assert abs(m["_plot_data"]["iou_thrs"][-1] - 0.95) < 1e-6
    assert bs["large"]["map50"] >= bs["tiny"]["map50"], \
        "detector misses small targets by design, so large AP should exceed tiny AP"
    assert 0.0 <= c["recall_at_p95"] <= 1.0
    assert np.array(op["confusion_matrix"]).shape == (2, 2), "cm must be (nc+1, nc+1)"

    lat = measure_latency(adapter, ds, n_images=25, warmup=3)
    print(f"\nlatency  mean={lat['mean_ms']:.2f} ms  p95={lat['p95_ms']:.2f} ms  "
          f"{lat['fps_mean']:.1f} FPS")
    assert lat["p99_ms"] >= lat["p50_ms"], "percentiles inconsistent"

    made = generate_all(m, lat, out_dir / "plots", "fake_model")
    print(f"\n{len(made)} plots: {[p.name for p in made]}")
    assert len(made) == 7, f"expected 7 plots, got {len(made)}"
    for p in made:
        assert p.stat().st_size > 1000, f"{p.name} looks empty"

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
