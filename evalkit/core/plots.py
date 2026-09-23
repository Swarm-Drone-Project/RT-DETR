"""
plots.py — the same seven plots for every model, so runs can be compared
side by side visually as well as numerically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all(metrics: dict, latency: dict | None, out_dir: Path, name: str) -> list[Path]:
    d = metrics.get("_plot_data", {})
    if not d:
        return []
    out, made = Path(out_dir), []

    # 1. PR curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(d["recall"], d["precision"], lw=2, color="#1f77b4")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{name} — PR curve @ IoU 0.50 (AP={metrics['accuracy']['map50']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    made.append(_save(fig, out / "pr_curve.png"))

    # 2. F1 vs confidence
    fig, ax = plt.subplots(figsize=(5, 4))
    order = np.argsort(d["conf"])
    ax.plot(np.array(d["conf"])[order], np.array(d["f1"])[order], lw=2, color="#2ca02c")
    c = metrics["curve"]
    ax.axvline(c["best_f1_conf"], ls="--", color="crimson", lw=1,
               label=f"best F1={c['best_f1']:.3f} @ conf={c['best_f1_conf']:.3f}")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1")
    ax.set_title(f"{name} — F1 vs confidence"); ax.legend(fontsize=8)
    made.append(_save(fig, out / "f1_vs_conf.png"))

    # 3. mAP vs IoU
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(d["iou_thrs"], d["map_per_iou"], "o-", lw=2, color="#9467bd", ms=4)
    ax.set_xlabel("IoU threshold"); ax.set_ylabel("mAP")
    ax.set_title(f"{name} — mAP vs IoU (mAP@50:95={metrics['accuracy']['map50_95']:.3f})")
    made.append(_save(fig, out / "map_vs_iou.png"))

    # 4. AP by object size — the one that matters most for small drones
    bs = metrics.get("by_size", {})
    if bs:
        labels = list(bs)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        x = np.arange(len(labels)); w = 0.38
        ax.bar(x - w / 2, [bs[k]["map50"] for k in labels], w, label="mAP@0.50", color="#1f77b4")
        ax.bar(x + w / 2, [bs[k]["map50_95"] for k in labels], w, label="mAP@0.50:0.95", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}\n({bs[k]['px_range']}px)" for k in labels], fontsize=8)
        ax.set_ylabel("AP"); ax.set_title(f"{name} — AP by object size"); ax.legend(fontsize=8)
        made.append(_save(fig, out / "ap_by_size.png"))

    # 5. Confusion matrix
    op = metrics.get("operating_point", {})
    cm = np.array(op.get("confusion_matrix", []), dtype=float)
    if cm.size:
        names_ = list(metrics.get("per_class", {})) + ["background"]
        norm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(max(4, len(names_) * 0.9), max(3.5, len(names_) * 0.8)))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names_))); ax.set_xticklabels(names_, rotation=45, ha="right")
        ax.set_yticks(range(len(names_))); ax.set_yticklabels(names_)
        ax.set_xlabel("Ground truth"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name} — confusion matrix @ conf={op.get('conf')}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.grid(False)
        made.append(_save(fig, out / "confusion_matrix.png"))

    # 6. TP vs FP confidence separation
    tpc, fpc = d.get("tp_confs", []), d.get("fp_confs", [])
    if tpc or fpc:
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.linspace(0, 1, 41)
        if tpc:
            ax.hist(tpc, bins=bins, alpha=0.65, label=f"TP (n={len(tpc)})", color="#2ca02c")
        if fpc:
            ax.hist(fpc, bins=bins, alpha=0.65, label=f"FP (n={len(fpc)})", color="#d62728")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
        ax.set_title(f"{name} — TP / FP confidence separation"); ax.legend(fontsize=8)
        made.append(_save(fig, out / "conf_separation.png"))

    # 7. Latency
    if latency and latency.get("samples_ms"):
        s = np.array(latency["samples_ms"])
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(s, bins=50, color="#17becf", alpha=0.85)
        for k, col in (("p50_ms", "#2ca02c"), ("p95_ms", "#ff7f0e"), ("p99_ms", "crimson")):
            ax.axvline(latency[k], ls="--", lw=1, color=col,
                       label=f"{k[:-3]}={latency[k]:.2f} ms")
        ax.set_xlabel("Latency per frame (ms), batch=1")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} — latency  ({latency['fps_mean']:.1f} FPS mean)")
        ax.legend(fontsize=8)
        made.append(_save(fig, out / "latency.png"))

    return made
