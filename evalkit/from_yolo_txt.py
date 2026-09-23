#!/usr/bin/env python3
"""
from_yolo_txt.py — turn the label files your own detect.py already writes into
a preds.json evalkit can score.

Most people already have a working inference script. If it can dump YOLO-format
label files (virtually every YOLOv5/v7/v8 fork can, via --save-txt), you do not
need to write a predictor at all — run your script, convert, score:

    # 1. your own script. Every flag below matters — see the warning further down.
    python detect.py --weights best.pt --source <dataset>/images/val \\
                     --save-txt --save-conf \\
                     --conf-thres 0.001 --iou-thres 0.65 --max-det 300

    # 2. convert what it wrote
    python evalkit/from_yolo_txt.py --labels runs/detect/exp/labels \\
                                    --dataset /shared/drone_dataset -o preds.json

    # 3. score it
    python evalkit/run_eval.py --preds preds.json --dataset /shared/drone_dataset \\
                               --name my_model

Expected line format, one detection per line, coordinates normalised 0-1:

    <class> <cx> <cy> <w> <h> <confidence>       (--save-format 0, the default)
    <class> <x1> <y1> <x2> <y2> <confidence>     (--save-format 1, pass --boxes xyxy)

YOUR SCRIPT'S DEFAULTS ARE NOT THE PROTOCOL'S — MATCH THEM BY HAND
------------------------------------------------------------------
A label file records boxes, not the settings that produced them, so this
converter CANNOT check how you ran inference. You have to line them up yourself:

    --conf-thres 0.001    your script probably defaults to 0.25. Leave it there
                          and every detection below 0.25 is discarded before
                          evalkit sees it: the PR curve is cut off and your
                          recall and mAP read LOWER than your model deserves.
                          (This one the script can usually spot — it warns you.)

    --iou-thres 0.65      YOLOv5's detect.py defaults to 0.45. A different NMS
                          threshold means a different number of surviving boxes,
                          so your mAP moves for reasons that have nothing to do
                          with your model.

    --max-det 300         matches the protocol's cap.

Verified on this repo: with those three flags, detect.py and evalkit's own
adapter produce byte-identical detections (1617 boxes over 40 frames, worst box
difference 0.000 px). With the defaults left alone, detect.py produced 1182 —
a 27% difference in surviving boxes, from settings alone.

Latency is not measured on this route — accuracy is comparable, speed is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.dataset import EvalDataset  # noqa: E402
from core.protocol import PROTOCOL  # noqa: E402


def parse_line(parts: list[str], boxes: str, w: int, h: int) -> tuple | None:
    """One label line -> (category_id, x, y, width, height, score) in pixels."""
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    a, b, c, d = (float(v) for v in parts[1:5])
    score = float(parts[5]) if len(parts) >= 6 else None

    if boxes == "xywh":  # normalised centre x, centre y, width, height
        bw, bh = c * w, d * h
        x, y = a * w - bw / 2, b * h - bh / 2
    else:                # normalised x1, y1, x2, y2
        x, y = a * w, b * h
        bw, bh = (c - a) * w, (d - b) * h

    return cls, x, y, bw, bh, score


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert YOLO --save-txt predictions into evalkit preds.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--labels", required=True,
                    help="Folder of .txt files your detect.py wrote (runs/detect/exp/labels)")
    ap.add_argument("--dataset", required=True, help="The same dataset root you ran on")
    ap.add_argument("--split", default=PROTOCOL["split"])
    ap.add_argument("--boxes", choices=["xywh", "xyxy"], default="xywh",
                    help="xywh = YOLO default (--save-format 0); xyxy = --save-format 1")
    ap.add_argument("-o", "--out", default="preds.json")
    args = ap.parse_args()

    lbl_dir = Path(args.labels)
    if not lbl_dir.is_dir():
        sys.exit(f"Not a folder: {lbl_dir}")

    ds = EvalDataset(args.dataset, args.split)

    preds: list[dict] = []
    matched, no_conf, scores = 0, 0, []

    for img_id, img_path, meta in zip(ds.image_ids, ds.image_paths, ds.coco_gt["images"]):
        txt = lbl_dir / f"{img_path.stem}.txt"
        if not txt.exists():
            continue  # no file == no detections on that image, which is fine
        matched += 1
        w, h = meta["width"], meta["height"]

        for line in txt.read_text().strip().splitlines():
            parsed = parse_line(line.split(), args.boxes, w, h)
            if parsed is None:
                continue
            cls, x, y, bw, bh, score = parsed
            if score is None:
                no_conf += 1
                score = 1.0
            scores.append(score)
            preds.append({
                "image_id": img_id,
                "category_id": cls,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "score": round(score, 6),
            })

    if not preds:
        stems = {p.stem for p in lbl_dir.glob("*.txt")}
        sys.exit(
            f"No predictions matched.\n"
            f"  {len(stems)} .txt files in {lbl_dir}\n"
            f"  {len(ds)} images in {args.dataset}/{args.split}\n"
            f"Filenames must match the image stems (img_001.txt <- img_001.jpg). "
            f"Did you run detect.py on this exact split?"
        )

    Path(args.out).write_text(json.dumps(preds))
    print(f"{len(preds)} detections from {matched}/{len(ds)} images -> {args.out}")

    # ── the checks that save people from silently bad numbers ────────────────
    if no_conf:
        print(f"\n(!) {no_conf} lines had no confidence value, scored as 1.0.\n"
              f"    You forgot --save-conf. The PR curve and best-F1 threshold are "
              f"meaningless without it — re-run with --save-conf.")

    if scores:
        lo = min(scores)
        if lo > 0.05:
            print(f"\n(!) Lowest confidence in the file is {lo:.3f}.\n"
                  f"    That almost certainly means you ran with the default "
                  f"--conf-thres 0.25 and threw away the low-scoring tail.\n"
                  f"    Your recall and mAP will read LOWER than your model deserves. "
                  f"    Re-run detect.py with --conf-thres {PROTOCOL['conf']}.")

    extra = {p.stem for p in lbl_dir.glob("*.txt")} - {p.stem for p in ds.image_paths}
    if extra:
        print(f"\n(i) {len(extra)} .txt file(s) had no matching image and were ignored "
              f"(e.g. {sorted(extra)[:3]}).")

    print(f"\n(i) This file records boxes, not settings, so I cannot verify how you ran "
          f"inference.\n    Confirm your script used: --conf-thres {PROTOCOL['conf']} "
          f"--iou-thres {PROTOCOL['nms_iou']} --max-det {PROTOCOL['max_det']}\n"
          f"    A different NMS IoU alone changes how many boxes survive, and moves mAP.")

    print(f"\nNext:  python run_eval.py --preds {args.out} --dataset {args.dataset} "
          f"--name <your_model>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
