#!/usr/bin/env python3
"""
run_eval.py — one script, every model, identical metrics.

    python run_eval.py --weights runs/train/student_gray/weights/best.pt \
                       --dataset /shared/drone_dataset

That is the whole command. Model format, input size, and colour mode
(grayscale vs RGB) are read out of the checkpoint — you only pass them if
auto-detection got it wrong. Scoring settings are frozen in core/protocol.py so
your numbers are directly comparable with everyone else's.

Outputs land in evalkit/results/<name>/ (wherever you run from) :

    metrics.json    all numbers, machine-readable
    summary.txt     the same numbers, human-readable
    preds.json      COCO-format predictions (rescore without touching the GPU)
    gt.json         COCO-format ground truth
    plots/*.png     the standard seven plots
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import (  # noqa: E402
    detect_format,
    get_adapter_class,
    list_formats,
    load_external,
)
from core.dashboard import generate_dashboard  # noqa: E402
from core.dataset import EvalDataset  # noqa: E402
from core.metrics import (  # noqa: E402
    compute_metrics,
    tracking_metrics,
    tracking_metrics_per_sequence,
)
from core.plots import generate_all  # noqa: E402
from core.predict import (  # noqa: E402
    measure_latency,
    run_predictions,
    run_predictions_sequential,
)
from core.sequence import group_sequences, summarise  # noqa: E402
from core.protocol import PROTOCOL, stamp  # noqa: E402

CHANNELS_BY_COLOR = {"gray": 1, "rgb": 3}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detection evaluation harness — identical metrics for every model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Scoring settings are frozen in core/protocol.py. Overriding one "
               "marks the run as not comparable.",
    )
    need = p.add_argument_group("what to evaluate")
    need.add_argument("--weights", default=None, help="Path to your model weights")
    need.add_argument("--dataset", required=True,
                      help="Dataset root (contains data.yaml + the val split)")
    need.add_argument("--predictor", default=None, metavar="FILE.py",
                      help="Your own predictor file, living in your own repo. "
                           "Use for any model evalkit doesn't natively support")
    need.add_argument("--preds", default=None, metavar="PREDS.json",
                      help="Score an existing COCO predictions file instead of "
                           "running a model. Works for ANY framework, no deps")

    auto = p.add_argument_group(
        "auto-detected from the checkpoint (pass only to override)"
    )
    auto.add_argument("--imgsz", type=int, default=None,
                      help="Inference image size. Default: the size it was trained at")
    auto.add_argument("--color", choices=("auto", "gray", "rgb"), default="auto",
                      help="Input colour mode. Default: whatever the network expects")
    auto.add_argument("--format", default=None,
                      help=f"Model format. Options: {list_formats()}")

    out = p.add_argument_group("output")
    out.add_argument("--name", default=None, help="Run name (defaults to weights folder)")
    # Anchored to this file, not the cwd, so every run lands in evalkit/results/
    # whichever directory it was launched from.
    out.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"),
                     help="Output root directory")
    out.add_argument("--split", default=PROTOCOL["split"], help="Split to evaluate")

    run = p.add_argument_group("run control")
    run.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    run.add_argument("--latency-images", type=int, default=200,
                     help="Images used for the batch=1 latency pass")
    run.add_argument("--skip-latency", action="store_true",
                     help="Accuracy only. Use when not on the reference GPU")
    run.add_argument("--skip-plots", action="store_true", help="Numbers only")
    run.add_argument("--sequence", choices=["auto", "on", "off"], default="auto",
                     help="Walk the split video-by-video in frame order, resetting "
                          "model state between videos. 'auto' follows the adapter")
    run.add_argument("--per-sequence", action="store_true",
                     help="Also break metrics down per video. Off by default: on a "
                          "large split this bloats metrics.json and the report. "
                          "Debugging aid, not part of the standard metric set")
    run.add_argument("--sequence-regex", default=None, metavar="RE",
                     help="Frame-number pattern, one capture group "
                          r"(default tries _frame_(\d+), _f(\d+), _(\d+))")
    run.add_argument("--model-name", default=None,
                     help="For --format visionkit: which plugin (yolo, yolo_fire...)")
    run.add_argument("--nc", type=int, default=None, help="Override number of classes")

    frozen = p.add_argument_group(
        "FROZEN scoring settings — changing one makes the run non-comparable"
    )
    frozen.add_argument("--conf", type=float, default=PROTOCOL["conf"],
                        help="Confidence floor for scoring")
    frozen.add_argument("--operating-conf", type=float, default=PROTOCOL["operating_conf"],
                        help="Deployment threshold for confusion matrix / FP-per-frame")
    frozen.add_argument("--nms-iou", type=float, default=PROTOCOL["nms_iou"],
                        help="NMS IoU threshold")
    frozen.add_argument("--max-det", type=int, default=PROTOCOL["max_det"],
                        help="Max detections per image")
    return p


def load_submitted_predictions(preds_path: str, dataset) -> tuple[list, dict, dict]:
    """
    Score a preds.json somebody generated in their own environment.

    This is the universal path: it works for any framework on any machine,
    because evalkit never has to import their model. The cost is that latency
    can't be measured here — see the note printed by main().
    """
    import numpy as np

    raw = json.loads(Path(preds_path).read_text())
    dets = raw.get("detections", raw) if isinstance(raw, dict) else raw
    if not isinstance(dets, list):
        sys.exit(
            f"{preds_path} should be a JSON list of detections (COCO format), "
            f"got {type(dets).__name__}."
        )
    if not dets:
        sys.exit(f"{preds_path} contains no detections.")

    required = {"image_id", "category_id", "bbox", "score"}
    missing = required - set(dets[0])
    if missing:
        sys.exit(
            f"Detections in {preds_path} are missing {sorted(missing)}.\n"
            f"Each entry must be: "
            f'{{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}}\n'
            f"bbox is top-left x, y plus width, height in ORIGINAL image pixels."
        )

    # image_id must line up with this dataset, or every box misses silently and
    # the model looks broken when really the submission is misaligned.
    valid_ids = set(dataset.image_ids)
    seen = {d["image_id"] for d in dets}
    unknown = seen - valid_ids
    if unknown:
        sys.exit(
            f"{len(unknown)} image_id(s) in {preds_path} are not in this dataset "
            f"(e.g. {sorted(unknown)[:5]}).\n"
            f"Valid ids are 1..{len(valid_ids)}, assigned by sorted filename. "
            f"Generate predictions with the same --dataset and split."
        )
    if not seen:
        sys.exit("No usable image_ids in the submission.")

    covered = len(seen & valid_ids)
    if covered < len(valid_ids):
        print(f"[evalkit] note: predictions cover {covered}/{len(valid_ids)} images "
              f"— the rest are scored as having no detections")

    # Group once rather than scanning `dets` per image — that would be
    # O(images x detections), which is minutes on a large split.
    grouped: dict[int, list] = {}
    for d in dets:
        grouped.setdefault(d["image_id"], []).append(d)

    empty = np.zeros((0, 6), dtype=np.float32)
    by_image: dict[int, np.ndarray] = {}
    for img_id in dataset.image_ids:
        rows = grouped.get(img_id)
        if not rows:
            by_image[img_id] = empty
            continue
        arr = np.zeros((len(rows), 6), dtype=np.float32)
        for i, d in enumerate(rows):
            x, y, w, h = d["bbox"]
            arr[i] = [x, y, x + w, y + h, d["score"], d["category_id"]]
        by_image[img_id] = arr

    meta = raw.get("model", {}) if isinstance(raw, dict) else {}
    info = {"params": meta.get("params"), "weights_mb": meta.get("weights_mb", 0.0)}
    return dets, by_image, info


def _run_name(weights: Path) -> str:
    """A name that identifies the training run, not the file layout.

    runs/train/student_gray/weights/best.pt -> "student_gray", since every
    checkpoint in the world is called best.pt sitting in a folder called weights.
    """
    if weights.stem in ("best", "last"):
        if weights.parent.name == "weights":
            return weights.parent.parent.name
        return weights.parent.name
    return weights.stem


def resolve_settings(adapter_cls: type, weights: Path, args) -> tuple[int, int | None, dict]:
    """
    Decide imgsz and input channels, preferring what the checkpoint knows.
    Returns (imgsz, channels_or_None, detected_dict) for logging.
    """
    detected = {}
    try:
        detected = adapter_cls.defaults_from_weights(weights) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[evalkit] could not read defaults from checkpoint: {exc}")

    if args.imgsz:
        imgsz = args.imgsz
    elif detected.get("imgsz"):
        imgsz = int(detected["imgsz"])
        print(f"[evalkit] imgsz {imgsz} (trained at this size)")
    else:
        imgsz = 640
        print("[evalkit] imgsz 640 (checkpoint did not record one)")

    if args.color != "auto":
        channels = CHANNELS_BY_COLOR[args.color]
    elif detected.get("channels"):
        channels = int(detected["channels"])
        print(f"[evalkit] {'grayscale' if channels == 1 else 'RGB'} input "
              f"({channels}-channel, from the network's first layer)")
    else:
        channels = None  # adapter decides

    return imgsz, channels, detected


def write_summary(path: Path, payload: dict) -> None:
    m, r = payload["metrics"], payload["run"]
    pr = payload.get("protocol", {})
    L = []
    add = L.append
    add("=" * 62)
    add(f"  {r['name']}")
    where = f"  {r['format']} | {r['dataset']}/{r['split']}"
    add(where + (f" | imgsz={r['imgsz']}" if r.get("imgsz") else ""))
    ch = payload["model"].get("input_channels")
    if ch:
        add(f"  input: {'grayscale' if ch == 1 else 'RGB'} ({ch}-channel)")
    if pr:
        add(f"  protocol v{pr['version']}: {pr['status']}"
            + ("" if pr["comparable"] else "  <-- NOT COMPARABLE"))
        for k, v in pr.get("deviations", {}).items():
            add(f"    {k}: protocol={v['protocol']} used={v['used']}")
    add("=" * 62)

    a = m["accuracy"]
    add("\nACCURACY")
    add(f"  mAP@0.50:0.95      {a['map50_95']:.4f}")
    add(f"  mAP@0.50           {a['map50']:.4f}")
    add(f"  mAP@0.75           {a['map75']:.4f}")
    add(f"  AR@100             {a['ar_100']:.4f}")

    add("\nAP BY OBJECT SIZE")
    for k, v in m["by_size"].items():
        # NaN means the split simply has no objects in that size band — worth
        # saying so, since a bare "nan" reads like a broken metric.
        if v["map50"] != v["map50"]:
            add(f"  {k:<8} ({v['px_range']:>7}px)  no objects this size in the split")
        else:
            add(f"  {k:<8} ({v['px_range']:>7}px)  mAP@0.50={v['map50']:.4f}   mAP@0.50:0.95={v['map50_95']:.4f}")

    c = m["curve"]
    add("\nOPERATING POINTS")
    add(f"  best F1            {c['best_f1']:.4f}  @ conf={c['best_f1_conf']:.3f}"
        f"  (P={c['best_f1_precision']:.3f} R={c['best_f1_recall']:.3f})")
    for t in (90, 95, 99):
        add(f"  recall @ P>={t}%    {c[f'recall_at_p{t}']:.4f}  @ conf={c[f'conf_at_p{t}']}")

    op = m["operating_point"]
    add(f"\nAT DEPLOYMENT CONF = {op['conf']}")
    add(f"  precision          {op['precision']:.4f}")
    add(f"  recall             {op['recall']:.4f}")
    add(f"  F1                 {op['f1']:.4f}")
    add(f"  FP per empty frame {op['fp_per_empty_frame']:.4f}"
        f"   ({op['fp_on_empty_frames']} FPs over {op['empty_frames']} empty frames)")

    if len(m["per_class"]) > 1:
        add("\nPER CLASS")
        for cls, v in m["per_class"].items():
            pc = op["per_class"].get(cls, {})
            add(f"  {cls:<16} mAP@0.50={v['map50']:.4f}  P={pc.get('precision', 0):.3f}"
                f"  R={pc.get('recall', 0):.3f}  TP={pc.get('tp', 0)} FP={pc.get('fp', 0)} FN={pc.get('fn', 0)}")

    lat = payload.get("latency")
    if lat:
        add("\nLATENCY (batch=1, end-to-end incl. preprocess + NMS)")
        add(f"  mean               {lat['mean_ms']:.2f} ms   ({lat['fps_mean']:.1f} FPS)")
        add(f"  p50 / p95 / p99    {lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} / {lat['p99_ms']:.2f} ms")
        add(f"  worst case FPS     {lat['fps_p99']:.1f}  (at p99)")
        if "peak_vram_mb" in lat:
            add(f"  peak VRAM          {lat['peak_vram_mb']:.1f} MB")

    tps = m.get("tracking_per_sequence")
    if tps:
        add("\nPER SEQUENCE (single-target)")
        add(f"  {'video':<34}{'GT':>6}{'pred':>6}{'hits':>6}{'succ':>8}{'mIoU':>7}{'FA':>5}")
        for name, t in tps.items():
            add(f"  {name[:33]:<34}{t['frames_with_gt']:>6}{t['frames_predicted']:>6}"
                f"{t['hits']:>6}{t['success_rate']:>8.3f}{t['mean_iou']:>7.3f}"
                f"{t['false_alarms']:>5}")

    mi = payload["model"]
    add("\nMODEL")
    if mi.get("params"):
        add(f"  parameters         {mi['params']:,}")
    if mi.get("weights_mb"):
        add(f"  weights file       {mi['weights_mb']:.2f} MB")
    if not lat:
        add("\n  (no latency in this run — accuracy is comparable, speed is not)")
    add("")

    path.write_text("\n".join(L))


def main() -> int:
    args = build_parser().parse_args()

    if not Path(args.dataset).is_dir():
        sys.exit(f"Dataset folder not found: {args.dataset}")
    if not (args.weights or args.preds):
        sys.exit(
            "Nothing to evaluate. Either:\n"
            "  --weights best.pt              run a model (add --predictor for your own)\n"
            "  --preds preds.json             score predictions you generated yourself"
        )

    weights = Path(args.weights) if args.weights else None
    if weights and not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    score_only = args.preds is not None
    sequences_used = None
    ran_sequential = False
    adapter_cls = None
    fmt = "submitted-predictions"

    if score_only:
        if not Path(args.preds).is_file():
            sys.exit(f"Predictions file not found: {args.preds}")
    elif args.predictor:
        try:
            adapter_cls = load_external(args.predictor)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Could not load predictor {args.predictor}:\n  {exc}")
        fmt = getattr(adapter_cls, "format_name", None) or Path(args.predictor).stem
        print(f"[evalkit] predictor: {adapter_cls.__name__} from {args.predictor}")
    else:
        try:
            fmt = args.format or detect_format(weights)
        except Exception as exc:  # noqa: BLE001
            sys.exit(
                f"Could not work out what kind of model '{weights.name}' is.\n"
                f"  {exc}\n"
                f"Options:\n"
                f"  --format {{{','.join(list_formats())}}}   if it is one of these\n"
                f"  --predictor my_predictor.py            if it is your own architecture\n"
                f"  --preds preds.json                     to just score predictions"
            )
        if fmt not in list_formats():
            sys.exit(
                f"Detected format '{fmt}' but no adapter is installed for it.\n"
                f"Available: {list_formats()}"
            )
        adapter_cls = get_adapter_class(fmt)

    name = args.name or (_run_name(weights) if weights else Path(args.preds).parent.name)
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    protocol = stamp(args)
    print(f"\n[evalkit] {name}  format={fmt}  device={args.device}")
    print(f"[evalkit] protocol v{protocol['version']} {protocol['status']}")
    if not protocol["comparable"]:
        print("[evalkit] WARNING: frozen scoring settings were changed — these "
              "numbers are NOT comparable with other runs:")
        for k, v in protocol["deviations"].items():
            print(f"           {k}: protocol={v['protocol']} used={v['used']}")

    imgsz, channels = None, None
    if not score_only:
        imgsz, channels, _detected = resolve_settings(adapter_cls, weights, args)

    try:
        dataset = EvalDataset(args.dataset, args.split)
    except FileNotFoundError as exc:
        sys.exit(
            f"{exc}\n"
            f"evalkit finds a split under {args.dataset} in one of these ways:\n"
            f"  images/{args.split}/*.jpg  + labels/{args.split}/*.txt\n"
            f"  {args.split}/images/*.jpg  + {args.split}/labels/*.txt\n"
            f"  data.yaml with  {args.split}: <folder or .txt list of images>\n"
            f"  --split path/to/list.txt   (one image path per line)"
        )
    print(f"[evalkit] dataset: {dataset.summary()}")

    dataset.save_gt(run_dir / "gt.json")

    if score_only:
        detections, raw, model_info = load_submitted_predictions(args.preds, dataset)
        print(f"[evalkit] {len(detections)} submitted detections over {len(dataset)} images")
        adapter = None
        seq_latency = None
    else:
        adapter = adapter_cls(
            weights=weights,
            imgsz=imgsz,
            conf=args.conf,
            nms_iou=args.nms_iou,
            device=args.device,
            max_det=args.max_det,
            model_name=args.model_name,
            nc=args.nc if args.nc is not None else dataset.nc,
            channels=channels,
        )
        want_seq = (args.sequence == "on") or (
            args.sequence == "auto" and getattr(adapter, "sequential", False)
        )
        # Grouping is needed to drive EXECUTION order for stateful models — a
        # tracker must see each video in frame order with a reset between. It is
        # not used for reporting unless --per-sequence is passed.
        sequences_used = group_sequences(dataset.image_paths, dataset.image_ids,
                                         args.sequence_regex)
        if want_seq:
            seqs = sequences_used
            info = summarise(seqs)
            print(f"[evalkit] sequential mode: {info['sequences']} sequence(s), "
                  f"{info['frames']} frames "
                  f"(shortest {info['shortest']}, longest {info['longest']}); "
                  f"state reset at each boundary")
            if info["sequences"] == info["frames"]:
                print("[evalkit] WARNING: every frame parsed as its own sequence — "
                      "the filenames carry no frame numbers, so a temporal model "
                      "gets no continuity. Pass --sequence-regex.")
            ran_sequential = True
            detections, raw, seq_latency = run_predictions_sequential(
                adapter, dataset, run_dir / "preds.json", seqs
            )
        else:
            detections, raw = run_predictions(adapter, dataset, run_dir / "preds.json")
            seq_latency = None
        print(f"[evalkit] {len(detections)} detections over {len(dataset)} images")
        model_info = adapter.info()

    metrics = compute_metrics(dataset, detections, raw, operating_conf=args.operating_conf)
    if "error" in metrics:
        sys.exit(f"[evalkit] {metrics['error']}")

    # How many distinct confidence values did the model actually produce? With one
    # (or a handful), every score-swept metric degenerates: pycocotools sorts by
    # score, ties fall back to insertion order, and the "PR curve" ends up
    # tracing image order rather than a confidence sweep. Record it so the report
    # can refuse to draw a curve that would be read as meaningful.
    _scores = {round(d["score"], 6) for d in detections} if detections else set()
    metrics["score_calibration"] = {
        "distinct_scores": len(_scores),
        "degenerate": len(_scores) <= 1,
    }

    metrics["tracking"] = tracking_metrics(
        dataset, raw, conf=args.operating_conf
    ) if raw else None
    # Metrics describe the split as a whole. Per-video numbers are opt-in only:
    # the final test set is large, and a table with one row per video is neither
    # readable nor a fair basis for comparing models.
    if args.per_sequence and raw and sequences_used and len(sequences_used) > 1:
        metrics["tracking_per_sequence"] = tracking_metrics_per_sequence(
            dataset, raw, sequences_used, conf=args.operating_conf
        )

    latency = None
    if score_only:
        print("[evalkit] latency not measured (scoring submitted predictions). "
              "Accuracy is comparable; speed must come from the reference GPU.")
    elif seq_latency is not None:
        # A stateful model cannot be re-run over a sample of frames without
        # corrupting its state, so timing came from the real sequential pass.
        latency = seq_latency
    elif not args.skip_latency:
        latency = measure_latency(adapter, dataset, n_images=args.latency_images)

    payload = {
        "run": {
            "name": name,
            "format": fmt,
            "weights": str(weights) if weights else None,
            "dataset": str(args.dataset),
            "split": args.split,
            "imgsz": imgsz,
            "conf_floor": args.conf,
            "operating_conf": args.operating_conf,
            "nms_iou": args.nms_iou,
            "device": args.device,
            "sequential": bool(ran_sequential),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
        },
        "protocol": protocol,
        "dataset": dataset.summary(),
        "model": model_info,
        "metrics": metrics,
        "latency": latency,
    }

    if not args.skip_plots:
        made = generate_all(metrics, latency, run_dir / "plots", name)
        print(f"[evalkit] {len(made)} plots -> {run_dir / 'plots'}")
        dash = generate_dashboard(payload, run_dir / "dashboard.png")
        if dash:
            print(f"[evalkit] dashboard -> {dash}")

    # keep plot arrays out of metrics.json; they bloat the file and nothing reads them
    slim = json.loads(json.dumps(payload))
    slim["metrics"].pop("_plot_data", None)
    if slim.get("latency"):
        slim["latency"].pop("samples_ms", None)
    (run_dir / "metrics.json").write_text(json.dumps(slim, indent=2))

    write_summary(run_dir / "summary.txt", payload)
    print((run_dir / "summary.txt").read_text())
    print(f"[evalkit] results -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
