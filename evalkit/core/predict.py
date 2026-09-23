"""
predict.py — run a model over the split and dump predictions to COCO JSON.

Deliberately separate from scoring. Once preds.json exists you can recompute
every metric in seconds without touching the GPU, add a new metric months later
without re-running anyone's model, and diff two runs box-for-box when the
numbers disagree.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .dataset import EvalDataset


def run_predictions(
    adapter, dataset: EvalDataset, out_path: Path, warmup: int = 10
) -> tuple[list[dict], dict]:
    """
    Returns (coco_detections, raw_by_image).
    raw_by_image maps image_id -> (N, 6) array, used for the confusion matrix
    and empty-frame analysis without re-parsing the JSON.
    """
    if not adapter._loaded:
        adapter.load()

    # Warm up on a real frame so the first timed image isn't paying for
    # CUDA kernel compilation.
    first = next(iter(dataset))[2]
    adapter.warmup(first, n=warmup)

    detections: list[dict] = []
    raw_by_image: dict[int, np.ndarray] = {}

    errors: list[tuple] = []
    for img_id, _path, img in tqdm(
        dataset, total=len(dataset), desc="Inference", dynamic_ncols=True
    ):
        try:
            dets = adapter.predict(img)
        except Exception as exc:  # noqa: BLE001
            errors.append((img_id, f"{type(exc).__name__}: {exc}"))
            dets = np.zeros((0, 6), dtype=np.float32)
        raw_by_image[img_id] = dets

        for x1, y1, x2, y2, score, cls in dets:
            detections.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(cls),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(detections, f)

    if errors:
        n = len(dataset)
        print(f"\n[evalkit] WARNING: the model raised on {len(errors)}/{n} frames "
              f"({len(errors)/n*100:.1f}%). Those frames scored as no-detection.")
        print(f"[evalkit] first failure, image_id {errors[0][0]}: {errors[0][1]}")

    return detections, raw_by_image


def run_predictions_sequential(
    adapter,
    dataset: EvalDataset,
    out_path: Path,
    sequences: list,
    warmup: int = 10,
) -> tuple[list[dict], dict, dict]:
    """
    Inference for models that need frames in filming order.

    Differs from run_predictions in three ways that matter:

      * Frames are walked video by video, in frame order, so motion and tracking
        see the continuity they were designed for.
      * adapter.reset() is called at every sequence boundary, so a target from
        the end of one clip is not carried into the start of the next.
      * Latency is timed inline rather than in a second random-access pass. A
        stateful model cannot be re-run over a sample of frames without either
        corrupting its state or measuring the wrong thing, so the timing here
        comes from the real run. It is per-frame and CUDA-synced, but unlike the
        dedicated pass it includes whatever the model does on first sight of a
        new sequence.

    Returns (coco_detections, raw_by_image, latency_stats).
    """
    if not adapter._loaded:
        adapter.load()

    first = next(iter(dataset))[2]
    adapter.warmup(first, n=warmup)
    adapter.reset()          # discard whatever warmup left behind

    by_path = {img_id: p for img_id, p in zip(dataset.image_ids, dataset.image_paths)}

    detections: list[dict] = []
    raw_by_image: dict[int, np.ndarray] = {}
    samples: list[float] = []
    errors: list[tuple] = []

    total = sum(len(ids) for _, ids in sequences)
    bar = tqdm(total=total, desc="Inference (sequential)", dynamic_ncols=True)

    for seq_name, image_ids in sequences:
        adapter.reset()
        bar.set_postfix_str(seq_name[:28], refresh=False)

        for img_id in image_ids:
            img = cv2.imread(str(by_path[img_id]))
            if img is None:
                raw_by_image[img_id] = np.zeros((0, 6), dtype=np.float32)
                bar.update(1)
                continue

            adapter.sync()
            t0 = time.perf_counter()
            try:
                dets = adapter.predict(img)
            except Exception as exc:  # noqa: BLE001
                # One bad frame must not destroy an hour of work. Record it,
                # score the frame as "no detection", and keep going — but the
                # count is surfaced in the report, because a model that threw on
                # 30% of frames is broken, not merely unlucky.
                errors.append((img_id, f"{type(exc).__name__}: {exc}"))
                dets = np.zeros((0, 6), dtype=np.float32)
            adapter.sync()
            samples.append((time.perf_counter() - t0) * 1000.0)

            raw_by_image[img_id] = dets
            for x1, y1, x2, y2, score, cls in dets:
                detections.append({
                    "image_id": int(img_id),
                    "category_id": int(cls),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
            bar.update(1)
    bar.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(detections, f)

    if errors:
        print(f"\n[evalkit] WARNING: the model raised on {len(errors)}/{total} frames "
              f"({len(errors)/total*100:.1f}%). Those frames scored as no-detection.")
        print(f"[evalkit] first failure, image_id {errors[0][0]}: {errors[0][1]}")
        if len(errors) / total > 0.05:
            print("[evalkit] more than 5% of frames failed — treat these numbers as "
                  "a lower bound and fix the model before comparing it to anything.")

    lat = _latency_stats(np.array(samples, dtype=np.float64), samples) if samples else None
    if lat:
        lat["measured_inline"] = True
        lat["frames_errored"] = len(errors)
    return detections, raw_by_image, lat


def _latency_stats(lat: np.ndarray, samples: list[float]) -> dict:
    stats = {
        "n_samples": int(lat.size),
        "mean_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "min_ms": float(lat.min()),
        "max_ms": float(lat.max()),
        "std_ms": float(lat.std(ddof=1)) if lat.size > 1 else 0.0,
        "fps_mean": float(1000.0 / lat.mean()) if lat.mean() > 0 else 0.0,
        "fps_p99": float(1000.0 / np.percentile(lat, 99)),
        "samples_ms": [round(x, 4) for x in samples],
    }
    try:
        import torch

        if torch.cuda.is_available():
            stats["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            stats["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return stats


def measure_latency(
    adapter, dataset: EvalDataset, n_images: int = 200, warmup: int = 20
) -> dict:
    """
    End-to-end batch=1 latency: preprocess + forward + decode + NMS.

    This is a separate pass on purpose. The accuracy run is batched and uses
    conf=0.001 to trace the full PR curve — timings taken from it would be
    throughput, not the per-frame latency a tracking loop actually pays.
    """
    if not adapter._loaded:
        adapter.load()

    images = []
    for i, (_id, _p, img) in enumerate(dataset):
        if i >= n_images:
            break
        images.append(img)

    adapter.warmup(images[0], n=warmup)

    samples = []
    for img in tqdm(images, desc="Latency (batch=1)", dynamic_ncols=True):
        adapter.sync()
        t0 = time.perf_counter()
        adapter.predict(img)
        adapter.sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(samples, dtype=np.float64)
    stats = {
        "n_samples": int(lat.size),
        "mean_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "min_ms": float(lat.min()),
        "max_ms": float(lat.max()),
        "std_ms": float(lat.std(ddof=1)) if lat.size > 1 else 0.0,
        "fps_mean": float(1000.0 / lat.mean()) if lat.mean() > 0 else 0.0,
        "fps_p99": float(1000.0 / np.percentile(lat, 99)),
        "samples_ms": [round(x, 4) for x in samples],
    }

    try:
        import torch

        if torch.cuda.is_available():
            stats["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            stats["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return stats
