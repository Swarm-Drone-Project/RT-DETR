"""
dataset.py — load a YOLO-format val split and convert ground truth to COCO.

Finds the split, in this order:
  A) <root>/images/val/*.jpg  + <root>/labels/val/*.txt
  B) <root>/valid/images/*.jpg + <root>/valid/labels/*.txt   (Roboflow export)
  C) --split is a .txt list file of image paths (one per line)
  D) the dataset's own data.yaml, e.g. `val: splits/val.txt` or `val: images`
     — a list file or a folder, exactly as Ultralytics reads it

For C and D the label of .../images/x.jpg is .../labels/x.txt (the YOLO rule).
Relative paths in a list file are relative to the dataset root. Absolute paths
written on another machine are re-anchored to the dataset root via data.yaml's
`path:`, so a list made on the server still works from a laptop mount.

Images with no label file (or an empty one) are kept deliberately — they are
the empty-sky frames, and false positives on them are the metric that matters
most for an anti-drone system.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _image_size(path: Path) -> tuple[int, int]:
    """
    (width, height) without decoding the pixels.

    Reading the JPEG header via PIL is ~170x faster than cv2.imread, which
    matters a lot: a 29k-image split takes ~1s this way versus ~107s decoding
    every frame just to learn its shape. Falls back to a real decode if the
    header cannot be read, and cross-checks nothing — PIL and cv2 agree on
    stored dimensions for images without EXIF rotation, and rotated EXIF is
    handled by the fallback raising rather than silently disagreeing.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            # EXIF orientations 5-8 swap the axes; cv2.imread applies them, so
            # defer to cv2 in that case rather than reporting the stored size.
            exif = im.getexif() if hasattr(im, "getexif") else None
            if exif and exif.get(274, 1) >= 5:
                raise ValueError("EXIF rotation present")
            return int(w), int(h)
    except Exception:  # noqa: BLE001
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read image: {path}") from None
        return int(img.shape[1]), int(img.shape[0])


def find_data_yaml(root: Path) -> Path | None:
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        if (root / name).exists():
            return root / name
    hits = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    return hits[0] if hits else None


_EVALKIT_DIR = Path(__file__).resolve().parents[1]


def _yolo_label(img: Path) -> Path:
    """.../images/<sub>/x.jpg -> .../labels/<sub>/x.txt (last 'images' component)."""
    parts = list(img.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def _reanchor(p: str, root: Path, cfg_path: str | None) -> Path:
    """Make a path from a list file / data.yaml usable on this machine."""
    q = Path(p).expanduser()
    if not q.is_absolute():
        return root / p.removeprefix("./")
    if not q.exists() and cfg_path:
        base = cfg_path.rstrip("/") + "/"
        if p.startswith(base):
            return root / p[len(base):]
    return q


def _read_list(list_file: Path, root: Path, cfg_path: str | None) -> list[Path]:
    out = []
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(_reanchor(line, root, cfg_path))
    missing = [p for p in out if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(out)} images listed in {list_file} do not exist "
            f"(e.g. {missing[0]}). Check --dataset points at the dataset root."
        )
    return out


def resolve_split(root: Path, split: str, cfg: dict) -> tuple[list[Path], dict, str]:
    """
    Return (image_paths, labels_by_image, source) for the requested split.
    labels_by_image maps each image path to its label path.
    """
    aliases = {"val": ["val", "valid", "validation"], "test": ["test"], "train": ["train"]}
    names = aliases.get(split, [split])

    def from_dir(img_dir: Path, lbl_dir: Path | None, recursive: bool = False):
        it = img_dir.rglob("*") if recursive else img_dir.iterdir()
        imgs = [p for p in it if p.suffix.lower() in IMG_EXT]
        return imgs, {
            p: (lbl_dir / f"{p.stem}.txt" if lbl_dir else _yolo_label(p)) for p in imgs
        }

    # A / B: evalkit's own folder layouts
    for n in names:
        a_img, a_lbl = root / "images" / n, root / "labels" / n
        if a_img.is_dir():
            return (*from_dir(a_img, a_lbl), str(a_img))
        b_img, b_lbl = root / n / "images", root / n / "labels"
        if b_img.is_dir():
            return (*from_dir(b_img, b_lbl), str(b_img))

    cfg_path = cfg.get("path") if isinstance(cfg.get("path"), str) else None

    # C: --split names a list file (relative to cwd, else to the evalkit folder)
    if split.endswith(".txt"):
        for cand in (Path(split).expanduser(), _EVALKIT_DIR.parent / split, _EVALKIT_DIR / split):
            if cand.is_file():
                imgs = _read_list(cand, root, cfg_path)
                return imgs, {p: _yolo_label(p) for p in imgs}, str(cand)
        raise FileNotFoundError(f"Split list file not found: {split}")

    # D: whatever the dataset's data.yaml says the split is
    entry = next((cfg[n] for n in names if cfg.get(n)), None)
    if entry is not None:
        imgs: list[Path] = []
        for e in entry if isinstance(entry, list) else [entry]:
            target = _reanchor(str(e), root, cfg_path)
            if target.is_dir():
                imgs += from_dir(target, None, recursive=True)[0]
            elif target.is_file() and target.suffix == ".txt":
                imgs += _read_list(target, root, cfg_path)
            else:
                raise FileNotFoundError(
                    f"data.yaml says {split}: {e}, but {target} does not exist."
                )
        return imgs, {p: _yolo_label(p) for p in imgs}, f"data.yaml {split}: {entry}"

    raise FileNotFoundError(
        f"No '{split}' split under {root}. Expected images/{split}/, {split}/images/, "
        f"a '{split}:' entry in data.yaml, or --split pointing at a .txt list of images."
    )


class EvalDataset:
    """Image paths + COCO-format ground truth for one split."""

    def __init__(self, root: str | Path, split: str = "val", class_names: list | None = None):
        self.root = Path(root)
        self.split = split

        cfg = {}
        yml = find_data_yaml(self.root)
        if yml:
            with open(yml) as f:
                cfg = yaml.safe_load(f) or {}

        names = class_names or cfg.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        self.class_names = list(names) if names else None

        imgs, self._labels, self.source = resolve_split(self.root, split, cfg)
        # Sorted by filename so image ids don't depend on which folder layout
        # the same frames were reached through; dict.fromkeys drops duplicates.
        self.image_paths = sorted(dict.fromkeys(imgs), key=lambda p: (p.name, str(p)))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found for split '{split}' ({self.source})")

        if cfg.get("train") and cfg.get(split) == cfg.get("train") and split != "train":
            print(f"[evalkit] note: data.yaml gives '{split}' the same images as "
                  f"'train' — this is not a held-out split")

        self._build_coco_gt()

    # ── ground truth ──────────────────────────────────────────────────────────

    def _label_for(self, img_path: Path) -> Path:
        return self._labels[img_path]

    def _build_coco_gt(self) -> None:
        images, annotations = [], []
        ann_id = 1
        max_cls = 0
        self.n_empty = 0

        for img_id, img_path in enumerate(
            tqdm(self.image_paths, desc="Ground truth", dynamic_ncols=True,
                 disable=len(self.image_paths) < 2000), start=1
        ):
            w, h = _image_size(img_path)
            images.append(
                {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
            )

            lbl = self._label_for(img_path)
            rows = []
            if lbl.exists():
                for line in lbl.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    rows.append([float(x) for x in parts[:5]])

            if not rows:
                self.n_empty += 1

            for cls, cx, cy, bw, bh in rows:
                cls = int(cls)
                max_cls = max(max_cls, cls)
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                bw_px, bh_px = bw * w, bh * h
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x, y, bw_px, bh_px],  # COCO: xywh top-left
                        "area": bw_px * bh_px,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        if self.class_names is None:
            self.class_names = [str(i) for i in range(max_cls + 1)]
        self.nc = len(self.class_names)

        self.coco_gt = {
            "info": {"description": f"{self.root.name}/{self.split}"},
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": i, "name": str(n)} for i, n in enumerate(self.class_names)
            ],
        }
        self.image_ids = [im["id"] for im in images]

        # Index annotations by image once. gt_boxes() is called per image by the
        # confusion-matrix pass, so scanning the whole annotation list each time
        # is O(images x annotations) — on a 29k-image split that is ~850M
        # comparisons and turns a seconds-long pass into a many-minute one.
        self._gt_cache: dict[int, np.ndarray] = {}
        by_image: dict[int, list] = {}
        for a in annotations:
            by_image.setdefault(a["image_id"], []).append(a)
        for img_id, rows in by_image.items():
            arr = np.zeros((len(rows), 5), dtype=np.float32)
            for i, a in enumerate(rows):
                x, y, w, h = a["bbox"]
                arr[i] = [x, y, x + w, y + h, a["category_id"]]
            self._gt_cache[img_id] = arr

    # ── access ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self):
        """Yield (image_id, path, bgr_image)."""
        for img_id, path in zip(self.image_ids, self.image_paths):
            img = cv2.imread(str(path))
            yield img_id, path, img

    _EMPTY_GT = np.zeros((0, 5), dtype=np.float32)

    def gt_boxes(self, image_id: int) -> np.ndarray:
        """(M, 5) [x1, y1, x2, y2, class] for one image. O(1) via the index."""
        return self._gt_cache.get(image_id, self._EMPTY_GT)

    def save_gt(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.coco_gt, f)
        return path

    def summary(self) -> dict:
        return {
            "images": len(self),
            "annotations": len(self.coco_gt["annotations"]),
            "empty_images": self.n_empty,
            "classes": self.class_names,
        }
