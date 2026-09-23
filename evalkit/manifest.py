#!/usr/bin/env python3
"""
manifest.py — list the val images with the exact image_id evalkit will score by.

Only needed if you are submitting a predictions file (`run_eval.py --preds`)
rather than letting evalkit run your model. Your predictions must use these ids,
otherwise every box lands on the wrong image and your model looks broken.

    python manifest.py --dataset /shared/drone_dataset -o manifest.json

Then, in your own code, in your own environment, with whatever framework you like:

    import json
    manifest = json.load(open("manifest.json"))

    preds = []
    for item in manifest["images"]:
        image = cv2.imread(item["path"])
        for x1, y1, x2, y2, score, cls in my_model(image):   # original pixels
            preds.append({
                "image_id": item["image_id"],
                "category_id": int(cls),
                "bbox": [x1, y1, x2 - x1, y2 - y1],          # COCO: x, y, w, h
                "score": float(score),
            })

    json.dump(preds, open("preds.json", "w"))

Send preds.json to whoever keeps the leaderboard, or score it yourself:

    python run_eval.py --preds preds.json --dataset /shared/drone_dataset --name my_model

Keep every detection down to score ~0.001 — the PR curve and the
recall-at-high-precision numbers need the low-scoring tail. Do NOT pre-filter to
your deployment threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.dataset import EvalDataset  # noqa: E402
from core.protocol import PROTOCOL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", required=True, help="Dataset root")
    ap.add_argument("--split", default=PROTOCOL["split"])
    ap.add_argument("-o", "--out", default="manifest.json")
    args = ap.parse_args()

    ds = EvalDataset(args.dataset, args.split)
    images = [
        {
            "image_id": img_id,
            "file_name": p.name,
            "path": str(p.resolve()),
            "width": im["width"],
            "height": im["height"],
        }
        for img_id, p, im in zip(ds.image_ids, ds.image_paths, ds.coco_gt["images"])
    ]

    payload = {
        "dataset": str(Path(args.dataset).resolve()),
        "split": args.split,
        "classes": ds.class_names,
        "conf_floor": PROTOCOL["conf"],
        "note": "Use these image_id values verbatim in your preds.json. "
                "bbox is [x, y, width, height] in original image pixels. "
                f"Keep detections down to score {PROTOCOL['conf']}.",
        "images": images,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f"{len(images)} images -> {args.out}")
    print(f"classes: {ds.class_names}")
    print(f"image_id runs 1..{len(images)}, assigned by sorted filename.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
