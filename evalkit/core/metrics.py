"""
metrics.py — scoring. Identical code path for every model, always.

AP comes from pycocotools rather than a hand-rolled implementation. That ends
any argument about whether the numbers are right, and makes them directly
comparable to published results.

Two things are tuned for drone-vs-drone specifically:

  1. Size buckets. COCO's small/medium/large split at 32x32 and 96x96 is far
     too coarse when most targets are under 30px. We use 12/24/48px so you can
     see the accuracy cliff where it actually happens.

  2. False positives on empty frames. mAP barely reflects false alarms, but for
     an anti-drone system a false alarm is expensive. We report FP/frame on
     images with no ground truth, at your intended operating confidence.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

# Drone-tuned area ranges in px^2. Edit AREA_LABELS/AREA_RNG together.
AREA_LABELS = ["all", "tiny", "small", "medium", "large"]
AREA_RNG = [
    [0, 1e10],       # all
    [0, 12**2],      # tiny    : < 12px
    [12**2, 24**2],  # small   : 12-24px
    [24**2, 48**2],  # medium  : 24-48px
    [48**2, 1e10],   # large   : > 48px
]
MAX_DETS = [1, 10, 100]


def _build_cocoeval(coco_gt_dict: dict, detections: list[dict]):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco_gt_dict
        gt.createIndex()

        if not detections:
            return None, None

        dt = gt.loadRes(list(detections))
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.areaRng = AREA_RNG
        ev.params.areaRngLbl = AREA_LABELS
        ev.params.maxDets = MAX_DETS
        ev.evaluate()
        ev.accumulate()

    return ev, gt


def _ap(ev, iou=None, area_idx=0, maxdet_idx=2, cls_idx=None) -> float:
    """Pull AP out of the accumulated precision array [T, R, K, A, M]."""
    p = ev.eval["precision"]
    if iou is not None:
        t = int(np.argmin(np.abs(ev.params.iouThrs - iou)))
        p = p[t : t + 1]
    if cls_idx is not None:
        p = p[:, :, cls_idx : cls_idx + 1, :, :]
    p = p[:, :, :, area_idx, maxdet_idx]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def _ar(ev, area_idx=0, maxdet_idx=2) -> float:
    r = ev.eval["recall"][:, :, area_idx, maxdet_idx]
    r = r[r > -1]
    return float(r.mean()) if r.size else float("nan")


def _pr_curve(ev, iou=0.50):
    """Mean precision across classes at each of the 101 recall points."""
    t = int(np.argmin(np.abs(ev.params.iouThrs - iou)))
    prec = ev.eval["precision"][t, :, :, 0, 2]  # [R, K]
    scores = ev.eval["scores"][t, :, :, 0, 2]

    valid = prec > -1
    with np.errstate(invalid="ignore"):
        p_mean = np.where(valid, prec, np.nan)
        p_mean = np.nanmean(p_mean, axis=1)
        s_mean = np.nanmean(np.where(valid, scores, np.nan), axis=1)

    return ev.params.recThrs, np.nan_to_num(p_mean), np.nan_to_num(s_mean)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def operating_point(dataset, raw_by_image: dict, conf: float, iou_thr: float = 0.5) -> dict:
    """
    Confusion matrix, per-class TP/FP/FN, and empty-frame false alarms at a
    single confidence threshold — i.e. how the model behaves as deployed.
    """
    nc = dataset.nc
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)  # last row/col = background
    tp = np.zeros(nc, dtype=np.int64)
    fp = np.zeros(nc, dtype=np.int64)
    fn = np.zeros(nc, dtype=np.int64)

    fp_on_empty = 0
    n_empty = 0
    tp_confs, fp_confs = [], []

    for image_id in dataset.image_ids:
        dets = raw_by_image.get(image_id, np.zeros((0, 6), dtype=np.float32))
        dets = dets[dets[:, 4] >= conf] if dets.size else dets
        # highest confidence first — matching must be greedy by score
        if dets.size:
            dets = dets[np.argsort(-dets[:, 4])]

        gt = dataset.gt_boxes(image_id)

        if gt.shape[0] == 0:
            n_empty += 1
            fp_on_empty += dets.shape[0]

        ious = _iou_matrix(dets[:, :4], gt[:, :4]) if dets.size and gt.size else None
        matched_gt: set[int] = set()

        for di in range(dets.shape[0]):
            d_cls = int(dets[di, 5])
            best_j, best_iou = -1, iou_thr
            if ious is not None:
                for gj in range(gt.shape[0]):
                    if gj in matched_gt:
                        continue
                    if ious[di, gj] >= best_iou:
                        best_iou, best_j = ious[di, gj], gj

            if best_j >= 0:
                g_cls = int(gt[best_j, 4])
                cm[min(d_cls, nc), min(g_cls, nc)] += 1
                matched_gt.add(best_j)
                if d_cls == g_cls:
                    tp[d_cls] += 1
                    tp_confs.append(float(dets[di, 4]))
                else:
                    fp[min(d_cls, nc - 1)] += 1
                    fp_confs.append(float(dets[di, 4]))
            else:
                cm[min(d_cls, nc), nc] += 1
                fp[min(d_cls, nc - 1)] += 1
                fp_confs.append(float(dets[di, 4]))

        for gj in range(gt.shape[0]):
            if gj not in matched_gt:
                g_cls = int(gt[gj, 4])
                cm[nc, min(g_cls, nc)] += 1
                fn[min(g_cls, nc - 1)] += 1

    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    return {
        "conf": conf,
        "iou": iou_thr,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            dataset.class_names[i]: {
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
            }
            for i in range(nc)
        },
        "precision": round(float(prec.mean()), 4),
        "recall": round(float(rec.mean()), 4),
        "f1": round(float(f1.mean()), 4),
        "empty_frames": int(n_empty),
        "fp_on_empty_frames": int(fp_on_empty),
        "fp_per_empty_frame": round(fp_on_empty / max(n_empty, 1), 4),
        "_tp_confs": tp_confs,
        "_fp_confs": fp_confs,
    }


def tracking_metrics(
    dataset, raw_by_image: dict, conf: float = 0.25, iou_thr: float = 0.5
) -> dict:
    """
    Single-target metrics, for models that answer "where is the MAV" with one
    box per frame rather than a ranked list.

    mAP is the wrong instrument for those. It needs a score distribution to
    trace a PR curve, and a tracker that emits its single best guess has none —
    it would score poorly for reasons that say nothing about how well it tracks.
    What matters instead is: on the frames where a target exists, how often is
    the answer right, and how tight is the box.

    This mirrors what a tracking validation script conventionally reports, so a
    model's own harness and evalkit can be compared directly:

        hit          top-scoring box overlaps the GT by >= iou_thr
        miss         GT present, no box or the box is wrong
        false alarm  a box on a frame with no GT
        mean_iou     averaged over frames that have a GT and a prediction

    Reported alongside mAP, never instead of it — a multi-box detector should be
    read from the mAP block, a single-target tracker from this one.
    """
    hits = misses = false_alarms = 0
    frames_with_gt = frames_empty = 0
    ious: list[float] = []
    predicted_frames = 0

    for image_id in dataset.image_ids:
        dets = raw_by_image.get(image_id, np.zeros((0, 6), dtype=np.float32))
        if dets.size:
            dets = dets[dets[:, 4] >= conf]
        gt = dataset.gt_boxes(image_id)

        # the model's single best answer for this frame
        best = dets[np.argmax(dets[:, 4])] if dets.size else None
        if best is not None:
            predicted_frames += 1

        if gt.shape[0] == 0:
            frames_empty += 1
            if best is not None:
                false_alarms += 1
            continue

        frames_with_gt += 1
        if best is None:
            misses += 1
            continue

        # best IoU against any GT in the frame
        m = _iou_matrix(best[None, :4], gt[:, :4])
        best_iou = float(m.max()) if m.size else 0.0
        ious.append(best_iou)
        if best_iou >= iou_thr:
            hits += 1
        else:
            misses += 1

    precision = hits / max(predicted_frames, 1)
    recall = hits / max(frames_with_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "conf": conf,
        "iou": iou_thr,
        "frames_total": len(dataset.image_ids),
        "frames_with_gt": frames_with_gt,
        "frames_empty": frames_empty,
        "frames_predicted": predicted_frames,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "success_rate": round(hits / max(frames_with_gt, 1), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "false_alarm_rate": round(false_alarms / max(frames_empty, 1), 4),
    }


def tracking_metrics_per_sequence(
    dataset, raw_by_image: dict, sequences: list, conf: float = 0.25,
    iou_thr: float = 0.5
) -> dict:
    """
    tracking_metrics(), but per video.

    Pooling a tracker's score across videos hides the thing you most need to
    know. A model that holds lock on one clip and never acquires on another is
    not "40% accurate" — it is two different behaviours averaged into a number
    that describes neither, and the fix for each is different. Worse, a short
    clip and a long one contribute unequally to the pooled figure, so the mean
    drifts with clip length rather than with quality.

    Returns {sequence_name: <same fields as tracking_metrics>}.
    """
    id_to_seq = {}
    for name, ids in sequences:
        for i in ids:
            id_to_seq[i] = name

    out: dict[str, dict] = {}
    for name, ids in sequences:
        sub = _SubsetView(dataset, ids)
        out[name] = tracking_metrics(sub, raw_by_image, conf=conf, iou_thr=iou_thr)
    return out


class _SubsetView:
    """Minimal dataset view over a subset of image_ids, for per-sequence scoring."""

    def __init__(self, dataset, image_ids):
        self._ds = dataset
        self.image_ids = image_ids
        self.nc = dataset.nc
        self.class_names = dataset.class_names

    def gt_boxes(self, image_id):
        return self._ds.gt_boxes(image_id)


def compute_metrics(
    dataset, detections: list[dict], raw_by_image: dict, operating_conf: float = 0.25
) -> dict:
    """The full metric set. Every model goes through exactly this."""
    ev, _gt = _build_cocoeval(dataset.coco_gt, detections)

    if ev is None:
        return {
            "error": "model produced zero detections above the confidence floor",
            "map50_95": 0.0,
            "map50": 0.0,
        }

    nc = dataset.nc
    names = dataset.class_names

    # ── AP / AR ──────────────────────────────────────────────────────────────
    accuracy = {
        "map50_95": round(_ap(ev), 4),                      # true COCO: 10 IoUs, .50:.95
        "map50": round(_ap(ev, iou=0.50), 4),
        "map75": round(_ap(ev, iou=0.75), 4),
        "ar_1": round(_ar(ev, maxdet_idx=0), 4),
        "ar_10": round(_ar(ev, maxdet_idx=1), 4),
        "ar_100": round(_ar(ev, maxdet_idx=2), 4),
    }

    by_size = {
        lbl: {
            "map50_95": round(_ap(ev, area_idx=i), 4),
            "map50": round(_ap(ev, iou=0.50, area_idx=i), 4),
            "px_range": f"{int(np.sqrt(AREA_RNG[i][0]))}-"
            + ("inf" if AREA_RNG[i][1] > 1e9 else str(int(np.sqrt(AREA_RNG[i][1])))),
        }
        for i, lbl in enumerate(AREA_LABELS)
        if lbl != "all"
    }

    per_class = {
        names[k]: {
            "map50_95": round(_ap(ev, cls_idx=k), 4),
            "map50": round(_ap(ev, iou=0.50, cls_idx=k), 4),
        }
        for k in range(nc)
    }

    # ── PR curve, best-F1 operating point, recall at fixed precision ─────────
    rec_thrs, prec_curve, score_curve = _pr_curve(ev, iou=0.50)
    f1_curve = 2 * prec_curve * rec_thrs / np.maximum(prec_curve + rec_thrs, 1e-9)
    best = int(np.argmax(f1_curve))

    curve = {
        "best_f1": round(float(f1_curve[best]), 4),
        "best_f1_conf": round(float(score_curve[best]), 4),
        "best_f1_precision": round(float(prec_curve[best]), 4),
        "best_f1_recall": round(float(rec_thrs[best]), 4),
    }
    for target in (0.90, 0.95, 0.99):
        ok = np.where(prec_curve >= target)[0]
        curve[f"recall_at_p{int(target * 100)}"] = (
            round(float(rec_thrs[ok].max()), 4) if ok.size else 0.0
        )
        idx = ok.max() if ok.size else None
        curve[f"conf_at_p{int(target * 100)}"] = (
            round(float(score_curve[idx]), 4) if idx is not None else None
        )

    # ── deployed behaviour at a single threshold ─────────────────────────────
    op = operating_point(dataset, raw_by_image, conf=operating_conf)
    tp_confs = op.pop("_tp_confs")
    fp_confs = op.pop("_fp_confs")

    return {
        "accuracy": accuracy,
        "by_size": by_size,
        "per_class": per_class,
        "curve": curve,
        "operating_point": op,
        "_plot_data": {
            "recall": rec_thrs.tolist(),
            "precision": prec_curve.tolist(),
            "conf": score_curve.tolist(),
            "f1": f1_curve.tolist(),
            "tp_confs": tp_confs,
            "fp_confs": fp_confs,
            "iou_thrs": ev.params.iouThrs.tolist(),
            "map_per_iou": [
                round(_ap(ev, iou=float(t)), 4) for t in ev.params.iouThrs
            ],
        },
    }
