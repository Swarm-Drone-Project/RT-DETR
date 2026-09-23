"""
dashboard.py — one page you can put on a slide.

Everything here is drawn from metrics.json; nothing is recomputed. The layout
foregrounds the four numbers that actually decide an anti-drone model — AP on
tiny targets, recall at high precision, false alarms per empty frame, and p95
latency — rather than leading with a single aggregate mAP that hides all four.

Two deliberate rules, both learned from reading a dashboard that broke them:

  1. Every axis says what it actually contains. The mAP-vs-IoU panel prints the
     thresholds it averaged over, so a reader can see at a glance that
     "mAP@0.50:0.95" really does span 0.50 to 0.95.
  2. No PR curve. It is the one panel that can actively mislead: pycocotools
     sorts detections by score, and when a model emits a single constant score
     the sort falls back to insertion order — so the "curve" traces the order of
     your image files rather than any behaviour of the model. Every other
     confidence-swept panel is drawn for every model; only this one is dropped,
     for all of them, so the page is identical whatever you evaluate.
  3. No panel is ever left blank. When something genuinely was not measured —
     latency on a submitted-predictions run, a size bucket with no objects —
     the panel says so in words instead of rendering an empty pair of axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# One restrained palette, reused everywhere so panels read as a set.
INK = "#14181D"
MUTE = "#6E7A87"
RULE = "#D9DEE4"
PRIMARY = "#1B6E9E"
GOOD = "#1C7048"
WARN = "#AC5A08"
BAD = "#A32118"
FILL = "#E8F1F7"
SIZE_COLORS = ["#0B3C5D", "#1B6E9E", "#4E9EC4", "#9CC7DD"]


def _fmt(v, spec="{:.4f}", dash="n/a"):
    if v is None:
        return dash
    if isinstance(v, float) and v != v:  # NaN
        return dash
    return spec.format(v)


def _blank(ax, message: str, title: str = "") -> None:
    """A panel with nothing to show says why, rather than showing empty axes."""
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9,
            color=MUTE, wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _tile(ax, label: str, value: str, sub: str = "", accent: str = INK) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side, s in ax.spines.items():
        s.set_visible(True)
        s.set_color(RULE)
        s.set_linewidth(0.8)
    ax.text(0.06, 0.80, label.upper(), fontsize=7.2, color=MUTE,
            transform=ax.transAxes, ha="left", va="top", family="monospace")
    ax.text(0.06, 0.52, value, fontsize=17, color=accent, fontweight="bold",
            transform=ax.transAxes, ha="left", va="center")
    if sub:
        ax.text(0.06, 0.17, sub, fontsize=7.2, color=MUTE,
                transform=ax.transAxes, ha="left", va="center")


def generate_dashboard(payload: dict, out_path: Path) -> Path | None:
    """Render the single-page summary. Returns the path, or None if unrenderable."""
    m = payload.get("metrics", {})
    d = m.get("_plot_data", {})
    if not m or "accuracy" not in m:
        return None

    acc = m["accuracy"]
    by_size = m.get("by_size", {})
    curve = m.get("curve", {})
    op = m.get("operating_point", {})
    lat = payload.get("latency")
    run = payload.get("run", {})
    mdl = payload.get("model", {})
    ds = payload.get("dataset", {})
    proto = payload.get("protocol", {})
    trk = m.get("tracking")
    # A sequential run is a tracker: one box per frame, no calibrated score.
    # mAP is not the right read, so the page has to say so and lead with the
    # single-target numbers instead.
    is_tracking = bool(run.get("sequential")) and bool(trk)

    fig = plt.figure(figsize=(19, 15.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        5, 24, figure=fig,
        height_ratios=[0.46, 0.60, 1.5, 1.5, 1.5],
        hspace=0.48, wspace=0.32,
        left=0.045, right=0.975, top=0.985, bottom=0.075,
    )

    # ── Title bar ────────────────────────────────────────────────────────────
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.72, run.get("name", "evalkit run"), fontsize=25,
              fontweight="bold", color=INK, va="top")

    bits = [
        f"{run.get('format', '?')}",
        f"{ds.get('images', '?')} images / {ds.get('annotations', '?')} objects"
        f" / {ds.get('empty_images', 0)} empty",
        f"{run.get('dataset', '')}/{run.get('split', '')}",
    ]
    if run.get("imgsz"):
        bits.append(f"imgsz {run['imgsz']}")
    if mdl.get("input_channels"):
        bits.append("grayscale" if mdl["input_channels"] == 1 else "RGB")
    head.text(0, 0.16, "   ·   ".join(str(b) for b in bits), fontsize=9.5,
              color=MUTE, va="top")

    if is_tracking:
        head.text(0, -0.34,
                  "SINGLE-TARGET MODEL (one box per frame, uncalibrated score)  —  every metric below is computed identically to every other model, but for THIS model the "
                  "mAP and PR panels are not informative. Compare it on success rate, mean IoU and FP/empty frame.",
                  fontsize=9.5, color=WARN, va="top", fontweight="bold")

    if proto:
        ok = proto.get("comparable", True)
        head.text(1.0, 0.72,
                  f"protocol v{proto.get('version', '?')}  "
                  f"{'COMPARABLE' if ok else 'NOT COMPARABLE'}",
                  fontsize=10, fontweight="bold", color=GOOD if ok else BAD,
                  ha="right", va="top", family="monospace")
        if not ok:
            head.text(1.0, 0.20,
                      "scoring settings were changed — do not compare this run",
                      fontsize=8.5, color=BAD, ha="right", va="top")

    # ── Stat tiles: the numbers that decide the model ────────────────────────
    tiny = by_size.get("tiny", {}).get("map50")
    fp_ef = op.get("fp_per_empty_frame")
    r95 = curve.get("recall_at_p95")

    def _T(i):
        return fig.add_subplot(gs[1, i * 3:(i + 1) * 3])

    sr = (trk or {}).get("success_rate")
    miou = (trk or {}).get("mean_iou")

    _tile(_T(0), "mAP@0.50:0.95", _fmt(acc.get("map50_95")), "10 IoUs, 0.50-0.95")
    _tile(_T(1), "mAP@0.50", _fmt(acc.get("map50")),
          f"mAP@0.75 {_fmt(acc.get('map75'))}")
    _tile(_T(2), "AP tiny (<12px)", _fmt(tiny), "decides detection range",
          accent=INK if tiny is None or tiny != tiny else
          (GOOD if tiny >= .5 else WARN if tiny >= .2 else BAD))
    _tile(_T(3), "recall @ P>=95%", _fmt(r95), "usable operating point",
          accent=BAD if r95 == 0 else GOOD if (r95 or 0) >= .8 else WARN)
    _tile(_T(4), "success rate", _fmt(sr),
          f"{(trk or {}).get('hits', 0):,} hits / {(trk or {}).get('frames_with_gt', 0):,} GT frames",
          accent=GOOD if (sr or 0) >= .7 else WARN if (sr or 0) >= .3 else BAD)
    _tile(_T(5), "mean IoU", _fmt(miou), "box tightness when it hits",
          accent=GOOD if (miou or 0) >= .6 else WARN if (miou or 0) >= .3 else BAD)
    n_empty = op.get("empty_frames", 0)
    if n_empty:
        _tile(_T(6), "FP / empty frame", _fmt(fp_ef, "{:.3f}"),
              f"over {n_empty:,} empty frames",
              accent=GOOD if (fp_ef or 0) <= .05 else WARN if (fp_ef or 0) <= .3 else BAD)
    else:
        _tile(_T(6), "FP / empty frame", "n/a", "split has no empty frames", accent=MUTE)
    if lat:
        _tile(_T(7), "p95 latency", f"{lat['p95_ms']:.2f} ms",
              f"{lat['fps_mean']:.0f} FPS mean")
    else:
        _tile(_T(7), "p95 latency", "—", "not measured")

    # ── Panel 1: mAP vs IoU, labelled with what it averaged ──────────────────
    ax = fig.add_subplot(gs[2, 0:8])
    thr, mpi = d.get("iou_thrs"), d.get("map_per_iou")
    if thr and mpi:
        ax.plot(thr, mpi, "o-", color=PRIMARY, lw=2, ms=4.5)
        ax.fill_between(thr, mpi, color=FILL, alpha=.85)
        mean = float(np.mean(mpi))
        ax.axhline(mean, ls="--", lw=1, color=BAD)
        ax.text(thr[-1], mean, f" mean = {mean:.4f}", fontsize=8.5,
                color=BAD, va="bottom", ha="right")
        ax.set_xlabel(f"IoU threshold  —  averaged over all {len(thr)}: "
                      f"{min(thr):.2f} to {max(thr):.2f}", fontsize=8.5)
        ax.set_ylabel("mAP")
        # Scale to the data. A fixed 0-1 axis flattens a weak model's curve into
        # an unreadable line along the bottom, which hides the shape that
        # actually tells you where it degrades.
        ax.set_ylim(0, max(0.05, max(mpi) * 1.35))
        ax.set_title("mAP vs IoU threshold", fontsize=10.5, color=INK)
    else:
        _blank(ax, "no per-IoU data", "mAP vs IoU threshold")

    # ── Panel 2: AP by object size ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 8:16])
    if by_size:
        labels = list(by_size)
        vals = [by_size[k].get("map50") for k in labels]
        drawn = [0 if (v is None or v != v) else v for v in vals]
        bars = ax.bar(range(len(labels)), drawn,
                      color=SIZE_COLORS[:len(labels)], width=.62)
        for i, (b, v) in enumerate(zip(bars, vals)):
            if v is None or v != v:
                ax.text(b.get_x() + b.get_width() / 2, .02, "no objects\nthis size",
                        ha="center", va="bottom", fontsize=7.5, color=MUTE)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"{k}\n{by_size[k].get('px_range', '')}px"
                            for k in labels], fontsize=8.5)
        ax.set_ylabel("mAP@0.50")
        ax.set_ylim(0, 1.08)
        ax.set_title("AP by object size  —  where the model falls off",
                     fontsize=10.5, color=INK)
    else:
        _blank(ax, "no size breakdown", "AP by object size")

    # ── Panel 3: score card ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 16:24])
    ax.axis("off")
    ax.set_title("Score card", fontsize=10.5, color=INK)
    rows = [
        ("mAP@0.50:0.95", _fmt(acc.get("map50_95"))),
        ("mAP@0.50 / @0.75", f"{_fmt(acc.get('map50'))} / {_fmt(acc.get('map75'))}"),
        ("AR@100", _fmt(acc.get("ar_100"))),
        ("best F1", f"{_fmt(curve.get('best_f1'))} @ conf "
                    f"{_fmt(curve.get('best_f1_conf'), '{:.3f}')}"),
        (f"precision @ conf {op.get('conf', '?')}", _fmt(op.get("precision"))),
        (f"recall @ conf {op.get('conf', '?')}", _fmt(op.get("recall"))),
        ("FP / empty frame",
         _fmt(fp_ef, "{:.4f}") if n_empty else "n/a (no empty frames)"),
    ]
    if lat:
        rows += [("latency mean / p99",
                  f"{lat['mean_ms']:.2f} / {lat['p99_ms']:.2f} ms"),
                 ("FPS mean / worst", f"{lat['fps_mean']:.0f} / {lat['fps_p99']:.0f}")]
        if lat.get("peak_vram_mb"):
            rows.append(("peak VRAM", f"{lat['peak_vram_mb']:.0f} MB"))
    else:
        rows.append(("latency", "not measured this route"))
    if mdl.get("params"):
        rows.append(("parameters", f"{mdl['params']:,}"))
    if mdl.get("weights_mb"):
        rows.append(("weights", f"{mdl['weights_mb']:.2f} MB"))

    tbl = ax.table(cellText=[[k, v] for k, v in rows],
                   colWidths=[.62, .38], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.42)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(.6)
        if c == 1:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("right")
        if r % 2:
            cell.set_facecolor("#F6F7F9")

    # ── Panel 4: F1 vs confidence ────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0:8])
    if d.get("conf") and d.get("f1"):
        order = np.argsort(d["conf"])
        ax.plot(np.array(d["conf"])[order], np.array(d["f1"])[order],
                lw=2, color=GOOD)
        bf, bc = curve.get("best_f1"), curve.get("best_f1_conf")
        if bc is not None:
            ax.axvline(bc, ls="--", lw=1, color=BAD)
            # Anchored in axes coordinates, not at the line. A rotated label
            # placed at x=bc spills into the neighbouring panel whenever the
            # best-F1 threshold sits near 1.0 — which it does for any model with
            # a constant score.
            ax.text(0.03, 0.94,
                    f"best F1 {_fmt(bf, '{:.3f}')} @ conf {bc:.3f}",
                    fontsize=8.5, color=BAD, va="top", ha="left",
                    transform=ax.transAxes)
        oc = op.get("conf")
        if oc is not None:
            ax.axvline(oc, ls=":", lw=1.2, color=MUTE)
            ax.text(0.03, 0.85, f"deployment conf {oc}", fontsize=8, color=MUTE,
                    va="top", ha="left", transform=ax.transAxes)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("F1")
        ax.set_xlim(0, 1)
        ax.set_title("F1 vs confidence  —  where to set your threshold",
                     fontsize=10.5, color=INK)
    else:
        _blank(ax, "no F1 curve data", "F1 vs confidence")

    # ── Panel 5: recall at fixed precision ───────────────────────────────────
    ax = fig.add_subplot(gs[3, 8:16])
    targets = [90, 95, 99]
    vals = [curve.get(f"recall_at_p{t}") or 0.0 for t in targets]
    bars = ax.bar([f"P>={t}%" for t in targets], vals, width=.55,
                  color=[GOOD if v >= .8 else WARN if v > 0 else RULE for v in vals])
    for b, v, t in zip(bars, vals, targets):
        c = curve.get(f"conf_at_p{t}")
        if v == 0:
            ax.text(b.get_x() + b.get_width() / 2, .02,
                    "never reaches\nthis precision", ha="center", va="bottom",
                    fontsize=8, color=BAD)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + .02,
                    f"{v:.3f}" + (f"\n@ conf {c:.3f}" if c else ""),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.12)
    ax.set_title("Recall at high precision  —  usable operating points",
                 fontsize=10.5, color=INK)

    # ── Panel 6: confusion matrix ────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 16:24])
    cm = np.array(op.get("confusion_matrix", []), dtype=float)
    if cm.size:
        names = list(m.get("per_class", {})) + ["background"]
        norm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8.5, rotation=30, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("Ground truth"); ax.set_ylabel("Predicted")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                        fontsize=9,
                        color="white" if norm[i, j] > .5 else INK)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
        ax.grid(False)
        ax.set_title(f"Confusion matrix @ conf {op.get('conf')}",
                     fontsize=10.5, color=INK)
    else:
        _blank(ax, "no confusion matrix", "Confusion matrix")

    # ── Panel 7: TP vs FP confidence separation ──────────────────────────────
    ax = fig.add_subplot(gs[4, 0:12])
    tpc, fpc = d.get("tp_confs", []), d.get("fp_confs", [])
    if tpc or fpc:
        lo = float(op.get("conf", 0.0))
        bins = np.linspace(lo, 1.0, 41)
        # Outlined steps, not solid overlapping bars. Filled bars drawn on top of
        # each other read as a STACK — you'd add the two heights together and get
        # the wrong total. These are two independent distributions over the same
        # axis, and the outline makes that legible.
        if fpc:
            ax.hist(fpc, bins=bins, histtype="stepfilled", alpha=.35, color=BAD)
            ax.hist(fpc, bins=bins, histtype="step", lw=1.6, color=BAD,
                    label=f"false positive (n={len(fpc):,})")
        if tpc:
            ax.hist(tpc, bins=bins, histtype="stepfilled", alpha=.35, color=GOOD)
            ax.hist(tpc, bins=bins, histtype="step", lw=1.6, color=GOOD,
                    label=f"true positive (n={len(tpc):,})")
        # Start the axis at the threshold. Running it from 0 leaves a wide empty
        # band that reads as "no detections here" when in fact everything below
        # the deployment threshold was excluded before this panel was drawn.
        ax.set_xlim(lo, 1.0)
        ax.set_xlabel(f"Confidence  (only detections at or above the deployment "
                      f"threshold of {lo:g} are shown)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_title(f"TP / FP separation at conf >= {lo:g}  —  "
                     f"overlap means no clean threshold",
                     fontsize=10.5, color=INK)
    else:
        _blank(ax, "no detections at the operating confidence",
               "TP / FP separation")

    # ── Panel 8: latency ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4, 12:24])
    if lat and lat.get("samples_ms"):
        s = np.array(lat["samples_ms"])
        ax.hist(s, bins=45, color=PRIMARY, alpha=.85)
        for k, col in (("p50_ms", GOOD), ("p95_ms", WARN), ("p99_ms", BAD)):
            ax.axvline(lat[k], ls="--", lw=1.1, color=col,
                       label=f"{k[:-3]} = {lat[k]:.2f} ms")
        ax.set_xlabel("Latency per frame (ms)  —  batch=1, incl. preprocess + NMS")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_title(f"Latency  ({lat['fps_mean']:.0f} FPS mean, "
                     f"{lat['fps_p99']:.0f} worst case)", fontsize=10.5, color=INK)
    elif lat:
        # Timings exist but the per-frame samples were stripped (metrics.json is
        # slimmed on write). Show the distribution we still have rather than
        # claiming nothing was measured — the score card would contradict it.
        _blank(ax, "", f"Latency  ({lat['fps_mean']:.0f} FPS mean)")
        lines = [
            f"mean   {lat['mean_ms']:.2f} ms",
            f"p50    {lat['p50_ms']:.2f} ms",
            f"p95    {lat['p95_ms']:.2f} ms",
            f"p99    {lat['p99_ms']:.2f} ms",
            f"min    {lat['min_ms']:.2f} ms",
            f"max    {lat['max_ms']:.2f} ms",
        ]
        ax.text(.5, .60, "\n".join(lines), ha="center", va="center",
                fontsize=11, family="monospace", color=INK,
                transform=ax.transAxes)
        ax.text(.5, .13,
                f"batch=1, incl. preprocess + NMS, over {lat.get('n_samples', '?')} frames\n"
                "(per-frame samples not retained in metrics.json —\n"
                "the histogram is in plots/latency.png)",
                ha="center", va="center", fontsize=8, color=MUTE,
                transform=ax.transAxes)
    else:
        _blank(ax,
               "Latency was not measured on this route.\n\n"
               "Accuracy above is fully comparable; speed is not.\n"
               "Run the model through evalkit on the reference GPU\n"
               "to fill this panel.",
               "Latency")

    for a in fig.get_axes():
        a.tick_params(labelsize=8.5, colors=INK)
        for sp in a.spines.values():
            sp.set_color(RULE)

    fig.text(0.045, 0.022,
             f"evalkit · {run.get('timestamp', '')} · {run.get('device', '')}"
             + (f" · {lat.get('gpu_name', '')}" if lat and lat.get("gpu_name") else "")
             + f" · conf floor {run.get('conf_floor')} · NMS IoU {run.get('nms_iou')}",
             fontsize=7.5, color=MUTE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
