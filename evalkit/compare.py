#!/usr/bin/env python3
"""
compare.py — collapse every run into one leaderboard.

    python compare.py results/
    python compare.py results/ --sort map50_95 --csv leaderboard.csv

Reads results/*/metrics.json. Any run in the folder shows up; nothing to
configure. This is the artifact you actually take to a team meeting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    ("name", "model", "{}"),
    ("map50_95", "mAP@50:95", "{:.4f}"),
    ("map50", "mAP@50", "{:.4f}"),
    ("map50_tiny", "AP tiny", "{:.4f}"),
    ("map50_small", "AP small", "{:.4f}"),
    ("success_rate", "succ", "{:.3f}"),
    ("mean_iou", "mIoU", "{:.3f}"),
    ("best_f1", "best F1", "{:.4f}"),
    ("recall_at_p95", "R@P95", "{:.4f}"),
    ("fp_per_empty_frame", "FP/empty", "{:.3f}"),
    ("p95_ms", "p95 ms", "{:.2f}"),
    ("fps_mean", "FPS", "{:.1f}"),
    ("params_m", "params M", "{:.2f}"),
]


def flatten(payload: dict) -> dict:
    m = payload.get("metrics", {})
    a = m.get("accuracy", {})
    bs = m.get("by_size", {})
    c = m.get("curve", {})
    op = m.get("operating_point", {})
    trk = m.get("tracking") or {}
    lat = payload.get("latency") or {}
    mi = payload.get("model", {})
    params = mi.get("params")
    run = payload.get("run", {})
    pr = payload.get("protocol", {})

    # A run whose scoring settings were changed is not on the same footing as
    # the rest, so mark it in the one column everybody reads.
    name = run.get("name", "?")
    if pr and not pr.get("comparable", True):
        name += " (!)"

    return {
        "name": name,
        "format": run.get("format", "?"),
        "dataset": run.get("dataset"),
        "split": run.get("split"),
        "protocol_version": pr.get("version"),
        "comparable": pr.get("comparable", None),
        "gpu": lat.get("gpu_name"),
        "map50_95": a.get("map50_95"),
        "map50": a.get("map50"),
        "map75": a.get("map75"),
        "map50_tiny": bs.get("tiny", {}).get("map50"),
        "map50_small": bs.get("small", {}).get("map50"),
        "map50_medium": bs.get("medium", {}).get("map50"),
        "map50_large": bs.get("large", {}).get("map50"),
        "success_rate": trk.get("success_rate"),
        "mean_iou": trk.get("mean_iou"),
        "tracking_hits": trk.get("hits"),
        "tracking_frames_with_gt": trk.get("frames_with_gt"),
        "best_f1": c.get("best_f1"),
        "best_f1_conf": c.get("best_f1_conf"),
        "recall_at_p95": c.get("recall_at_p95"),
        "precision": op.get("precision"),
        "recall": op.get("recall"),
        "fp_per_empty_frame": op.get("fp_per_empty_frame"),
        "mean_ms": lat.get("mean_ms"),
        "p95_ms": lat.get("p95_ms"),
        "p99_ms": lat.get("p99_ms"),
        "fps_mean": lat.get("fps_mean"),
        "peak_vram_mb": lat.get("peak_vram_mb"),
        "params_m": params / 1e6 if params else None,
        "weights_mb": mi.get("weights_mb"),
    }


def fmt(val, spec: str) -> str:
    if val is None:
        return "—"
    if isinstance(val, float) and val != val:  # NaN: no objects in that bucket
        return "n/a"
    try:
        return spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def comparability_warnings(rows: list[dict]) -> list[str]:
    """
    The ways a leaderboard quietly becomes meaningless. Better to say so under
    the table than to let someone screenshot it and pick a model.
    """
    out = []

    modified = [r["name"] for r in rows if r.get("comparable") is False]
    if modified:
        out.append(
            f"\n(!) {len(modified)} run(s) changed the frozen scoring settings and are "
            f"NOT comparable: {', '.join(modified)}\n"
            f"    Re-run them without the overriding flags."
        )

    def spread(key: str) -> list:
        return sorted({str(r.get(key)) for r in rows if r.get(key) is not None})

    datasets, splits = spread("dataset"), spread("split")
    if len(datasets) > 1 or len(splits) > 1:
        out.append(
            "\n(!) These runs did NOT all use the same data, so the accuracy "
            "columns cannot be compared:"
        )
        for d in datasets:
            out.append(f"      dataset: {d}")
        if len(splits) > 1:
            out.append(f"      splits:  {', '.join(splits)}")

    versions = spread("protocol_version")
    if len(versions) > 1:
        out.append(
            f"\n(!) Mixed protocol versions ({', '.join(versions)}). The scoring "
            f"rules changed between them — re-run the older ones."
        )

    gpus = spread("gpu")
    if len(gpus) > 1:
        out.append(
            "\n(i) Latency was measured on different GPUs, so the ms/FPS columns "
            "are not comparable (accuracy still is):"
        )
        for g in gpus:
            out.append(f"      {g}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Leaderboard across evalkit runs")
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--sort", default="map50_95", help="Column to sort by, descending")
    ap.add_argument("--csv", default=None, help="Also write full table to this CSV")
    args = ap.parse_args()

    files = sorted(Path(args.results_dir).glob("*/metrics.json"))
    if not files:
        return print(f"No metrics.json found under {args.results_dir}/") or 1

    rows = []
    for f in files:
        try:
            rows.append(flatten(json.loads(f.read_text())))
        except Exception as e:  # noqa: BLE001
            print(f"  skipped {f}: {e}")

    ascending = args.sort in ("mean_ms", "p95_ms", "p99_ms", "fp_per_empty_frame",
                              "params_m", "weights_mb", "peak_vram_mb")
    rows.sort(key=lambda r: (r.get(args.sort) is None,
                             r.get(args.sort) if ascending else -(r.get(args.sort) or 0)))

    headers = [h for _, h, _ in COLUMNS]
    table = [[fmt(r.get(k), s) for k, _, s in COLUMNS] for r in rows]
    widths = [max(len(headers[i]), *(len(row[i]) for row in table)) for i in range(len(headers))]

    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"\n{line}")
    print("-" * len(line))
    for row in table:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))

    print(f"\n{len(rows)} runs. Sorted by {args.sort}.")
    print("AP tiny = targets under 12px — usually the number that decides an anti-drone model.")
    print("succ / mIoU = single-target: top-scoring box per frame vs ground truth. "
          "Computed for EVERY model,")
    print("              and the only fair basis for comparing a tracker against a detector.")

    for warning in comparability_warnings(rows):
        print(warning)

    if args.csv:
        keys = list(rows[0])
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Full table -> {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
