# evalkit

One evaluation harness for every detection model on the drone project.

Whatever you trained — this repo's YOLOv5, an Ultralytics checkpoint, an ONNX
export, a from-scratch PyTorch architecture, or something running on a Jetson
that this machine can't even load — you get **the same metrics, computed by the
same code**, so the numbers are directly comparable.

**You never edit anything inside evalkit.** That folder is shared. Nobody should
be sending a pull request to it just to get their model measured.

---

## Contents

1. [Why this exists](#1-why-this-exists)
2. [Install](#2-install)
3. [Quickstart](#3-quickstart)
4. [Which route is yours](#4-which-route-is-yours)
5. [Route 1 — a format evalkit already knows](#route-1--a-format-evalkit-already-knows)
6. [Route 2 — you already have a working `detect.py`](#route-2--you-already-have-a-working-detectpy)
7. [Route 3 — your own architecture](#route-3--your-own-architecture)
8. [Route 4 — you can't install evalkit at all](#route-4--you-cant-install-evalkit-at-all)
9. [Video and sequence models](#video-and-sequence-models-trackers-motion-based-detectors)
10. [Dataset layout](#5-dataset-layout)
11. [What you get — every metric explained](#6-what-you-get--every-metric-explained)
12. [The dashboard](#7-the-dashboard)
13. [The frozen protocol](#8-the-frozen-protocol--why-numbers-stay-comparable)
14. [The leaderboard](#9-the-leaderboard)
15. [Every flag](#10-every-flag)
16. [Troubleshooting](#11-troubleshooting)
17. [Verifying the harness itself](#12-verifying-the-harness-itself)
18. [Team rules](#13-team-rules)
19. [How it works inside](#14-how-it-works-inside)

---

## 1. Why this exists

Two people evaluate "the same" model and get different mAP. Then the leaderboard
is measuring settings, not models. Three things cause it, and evalkit fixes all
three:

- **Different scoring settings.** Confidence floor, NMS IoU and threshold sets
  move mAP by more than the difference between two decent models. Measured on
  this project: changing NMS IoU from 0.45 to 0.65 changed the surviving box
  count from 1182 to 1617 on the same frames — a 27% swing from a setting.
  evalkit freezes every scoring knob and flags any run that changed one.
- **Different metric implementations.** AP here comes from `pycocotools`, not a
  hand-rolled loop, so there's no argument about whether it's right and the
  numbers compare to published results.
- **A metric definition that drifts.** "mAP@0.50:0.95" must be ten IoU
  thresholds from 0.50 to 0.95. Averaging six that stop at 0.75 and printing the
  same name inflates the number. evalkit's dashboard prints the thresholds it
  averaged, on the axis, so this can't happen quietly.

It also reports two things a generic harness won't, both of which matter more
than mAP for anti-drone work: **AP by object size** (buckets at 12/24/48 px,
because a drone at range is 10 px across) and **false positives per empty
frame**.

---

## 2. Install

```bash
pip install -r requirements.txt
```

That's `numpy`, `opencv-python`, `pycocotools`, `matplotlib`, `pyyaml`, `tqdm` —
all benign, and they coexist with any ML framework.

Per-format extras, only if you need them:

| Format | Extra |
|---|---|
| Ultralytics `.pt` | `pip install ultralytics` |
| ONNX | `pip install onnxruntime-gpu` |
| TorchScript / `.pt` auto-detect | `torch` |

Verify the install at any time:

```bash
python selftest.py        # builds a synthetic dataset, runs a fake detector, asserts sanity
```

> **Note on environments.** Run evalkit **from the environment where your model
> works.** Different people's models need conflicting packages — this very repo
> cannot have `ultralytics` installed (see its `requirements.txt`), so a single
> shared environment that imports everyone's model is not achievable. Each
> author runs from their own, and the scoring code is identical regardless.

---

## 3. Quickstart

```bash
python run_eval.py --weights runs/train/student_gray/weights/best.pt \
                   --dataset /shared/drone_dataset
```

Image size, colour mode and class names are read from the checkpoint. Results
land in `results/<name>/`:

```
results/student_gray/
├── dashboard.png     one page you can put on a slide
├── summary.txt       the numbers, human-readable (also printed to your terminal)
├── metrics.json      the numbers, machine-readable, plus full provenance
├── preds.json        COCO predictions — rescore later without a GPU
├── gt.json           COCO ground truth
└── plots/            the standard seven plots
```

Then, once several people have run it:

```bash
python compare.py results/ --csv leaderboard.csv
```

---

## 4. Which route is yours

Pick the first one that applies.

```
Is your model ONNX / TorchScript / Ultralytics / this repo's YOLOv5 / visionkit?
│
├─ YES ──────────────────────────────► Route 1: --weights. Nothing to write.
│
└─ NO
   │
   Can your script dump per-image labels WITH a confidence per box?
   │  (YOLOv5/v7 --save-txt --save-conf;  Ultralytics save_txt=True save_conf=True)
   │
   ├─ YES ───────────────────────────► Route 2: dump labels, then the converter.
   │
   └─ NO
      │
      Can you install evalkit's requirements alongside your model?
      │
      ├─ YES ────────────────────────► Route 3: a ~15-line predictor in YOUR repo.
      │
      └─ NO (Jetson, TensorFlow, MATLAB, an API…)
                                      └► Route 4: send a preds.json.
```

| Route | You write | Latency measured |
|---|---|---|
| 1 — known format | nothing | ✅ |
| 2 — your script dumps labels | nothing (a few flags) | ❌ |
| 3 — your predictor | ~15 lines, in your repo | ✅ |
| 4 — submitted JSON | nothing for you | ❌ |

Accuracy metrics are identical on all four. Only speed differs, because latency
has to be timed inside the model call on the reference GPU.

---

### Route 1 — a format evalkit already knows

| What you trained | Notes |
|---|---|
| This repo's YOLOv5 (teacher/student KD) | grayscale **and** RGB, any `yolov5*.yaml` |
| Ultralytics YOLOv5u/v8/v9/v10/v11/v12, RT-DETR | needs `ultralytics` |
| ONNX `.onnx` | with or without NMS baked in |
| TorchScript `.torchscript` | traced models |
| visionkit native plugins | add `--model-name yolo_fire` |

```bash
python run_eval.py --weights best.pt --dataset /shared/drone_dataset
```

The format is detected from the checkpoint itself. Override with `--format` if
it ever guesses wrong.

**Auto-detected from the checkpoint**, so you normally don't pass them:

- `--imgsz` — the size it was trained at. evalkit prints
  `imgsz 960 (trained at this size)` so you can see what it picked.
- `--color` — read from the first conv layer's weight shape. Prints
  `grayscale input (1-channel, from the network's first layer)`.
- class names, and parameter count.

Pass them explicitly only to override.

---

### Route 2 — your script can dump labels **with confidence**

Not "you have a `detect.py`" — the requirement is narrower than that. Your
script must be able to write, per image, one line per detection:

```
<class> <cx> <cy> <w> <h> <confidence>      normalised 0-1
```

`--save-txt --save-conf` is **YOLOv5-lineage syntax**, not a universal flag.
Check the table below for your tool before copying the command.

```bash
# 1. YOUR script — YOLOv5 / YOLOv7 syntax shown; see the translation table below
python detect.py --weights best.pt --source /shared/drone_dataset/images/val \
                 --save-txt --save-conf \
                 --conf-thres 0.001 --iou-thres 0.65 --max-det 300 \
                 --imgsz 960 --nosave

# 2. Convert what it wrote
python /path/to/evalkit/from_yolo_txt.py \
       --labels runs/detect/exp/labels \
       --dataset /shared/drone_dataset \
       -o preds.json

# 3. Score
python /path/to/evalkit/run_eval.py \
       --preds preds.json --dataset /shared/drone_dataset --name my_model
```

**A label file records boxes, not the settings that produced them**, so the
converter cannot check how you ran inference. Line these up by hand:

| Flag | If you skip it |
|---|---|
| `--save-txt` | nothing is written |
| `--save-conf` | no confidence per box; PR curve and best-F1 become meaningless. *The converter warns.* |
| `--conf-thres 0.001` | your script probably defaults to **0.25**, discarding the low-score tail; recall and mAP read **lower** than your model deserves. *The converter warns if the minimum confidence looks too high.* |
| `--iou-thres 0.65` | YOLOv5 defaults to **0.45**. Verified on this repo: 1182 surviving boxes at 0.45 versus **1617** at 0.65, same model, same frames. *The converter cannot see this — it prints a reminder every run.* |
| `--max-det 300` | matches the frozen protocol |

With all four set, `detect.py` and evalkit's own adapter produced **byte-identical
detections** on this repo — 1617 boxes over 40 frames, worst coordinate
difference 0.000 px.

If your script writes normalised `xyxy` (`--save-format 1`) instead of the YOLO
default `xywh`, add `--boxes xyxy` to the converter.

#### Flag translation by tool

| Your tool | Command | Notes |
|---|---|---|
| **YOLOv5 / YOLOv7** `detect.py` | `--save-txt --save-conf --conf-thres 0.001 --iou-thres 0.65 --max-det 300` | as shown above |
| **Ultralytics** (v8–v12, **RT-DETR**) | `yolo predict model=best.pt source=… save_txt=True save_conf=True conf=0.001 iou=0.65 max_det=300` | `key=value`, not `--flags`. Its `iou` default is **0.7**, not 0.45 |
| **VisionKit** `infer --save-txt` | ❌ **not usable** | writes no confidence, and names files `<stem>_<frame>.txt`. Use Route 1 instead: `--format visionkit --model-name rtdetr` |
| **Anything else** | — | if it can't emit confidence per box, use Route 3 or Route 4 |

#### NMS-free architectures (RT-DETR and the DETR family)

RT-DETR does **not** run NMS — it's a set-prediction model, and Ultralytics'
`RTDETRPredictor.postprocess` never calls `non_max_suppression` and never reads
`iou`. So for RT-DETR:

- **`iou=0.65` is a no-op.** Passing it is harmless but changes nothing, and you
  do not need to match it to anyone else's setting.
- **`max_det` is effectively the query budget** (RT-DETR's default is 300
  queries, which is why 300 is the protocol value).
- **`conf=0.001` still matters**, and matters more than usual: it's the only
  thing controlling how much of the tail you keep for the PR curve.

The NMS-IoU warning elsewhere in this README applies to NMS-based detectors —
YOLO and friends. It is not a universal requirement.

#### When Route 2 does not apply

Use Route 3 (your own predictor) or Route 4 (submitted JSON) if your script:

- can't write confidence per box (VisionKit's `infer` is in this category),
- writes label filenames that don't match the image stems,
- only outputs annotated images or a video,
- can't be pointed at a folder of images.

---

### Route 3 — your own architecture

Copy the template into **your** project and fill in two methods:

```bash
cp /path/to/evalkit/templates/external_predictor.py ~/my_detector/evalkit_predictor.py
# edit load() and predict()

python /path/to/evalkit/run_eval.py \
    --predictor ~/my_detector/evalkit_predictor.py \
    --weights   ~/my_detector/checkpoints/best.pt \
    --dataset   /shared/drone_dataset
```

The predictor lives in your repo, so `import my_detector` just works, and
evalkit never needs your packages installed alongside anyone else's. The
template finds evalkit automatically whether it sits inside your project,
beside it, or further up the tree.

**The entire contract is one method:**

```python
def predict(self, image_bgr: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 BGR  ->  (N, 6) [x1, y1, x2, y2, score, class_id]"""
```

Boxes in **original image pixels**. If you resized, map them back —
`scale_boxes()` handles the letterbox case. Do your own preprocessing inside
`predict()`, exactly as you do in deployment: if the harness forced one resize
strategy on everyone, a model trained at 512 with stretch-resize would look
artificially bad and you'd be comparing preprocessing instead of models.

**The three mistakes people make:**

1. **Pre-filtering to your deployment confidence.** Return everything down to
   `self.conf` (0.001). Filtering early silently caps your recall.
2. **Forgetting to map boxes back to original pixels.** If mAP is near zero but
   your own demo looks fine, this is almost always why.
3. **Misreading your own output contract.** Is objectness a real logit or a
   filler? Are class scores sigmoided already? A real example from this project:
   a head that emits a hardcoded objectness of `1.0` and *unsigmoided* class
   logits gives **zero detections** if you treat it as already-sigmoided, and
   **8400 identical garbage boxes** if you sigmoid it. Same tensor, both wrong.
   Only you know which it is — that's why the predictor is yours to write.

Optionally implement `defaults_from_weights()` so `--imgsz` and `--color` are
read from your checkpoint too, and `info()` so parameter count appears in the
report.

---

### Route 4 — you can't install evalkit at all

For a model on a Jetson, in TensorFlow, in MATLAB, behind an API — anything.

```bash
# 1. Get the exact image_id mapping (run this anywhere evalkit is installed)
python manifest.py --dataset /shared/drone_dataset -o manifest.json

# 2. Your own inference, any language, any machine -> preds.json

# 3. Score it (you, or whoever keeps the leaderboard)
python run_eval.py --preds preds.json --dataset /shared/drone_dataset --name my_model
```

`preds.json` is a flat COCO list:

```json
[
  {"image_id": 1, "category_id": 0, "bbox": [x, y, w, h], "score": 0.87}
]
```

- `bbox` is **top-left x, y plus width, height, in original image pixels** —
  not normalised, not `xyxy`.
- `image_id` must match `manifest.json` exactly. They're assigned **1..N by
  sorted filename**. Using 0-indexed ids is the single most common mistake;
  evalkit detects it and refuses to score rather than silently reporting zero.
- Keep every detection down to score ~0.001. Do **not** pre-filter.

---

### Video and sequence models (trackers, motion-based detectors)

Some models need frames in the order they were filmed — motion detection,
tracking, temporal smoothing, anything carrying state between frames. Declare it
on your adapter:

```python
class MyTracker(DetectorAdapter):
    sequential = True          # walk the split video-by-video, in frame order

    def reset(self):           # called at every video boundary, and after warmup
        self.prev_frame = None
        self.track = None
```

evalkit then groups the split back into videos from the filenames
(`DJI_..._frame_000123.jpg` -> sequence `DJI_..._V`, frame 123), walks each in
frame order, and calls `reset()` between them so a target from the end of one
clip is not carried into the start of the next. Frames sort by parsed frame
*number*, so `frame_10` follows `frame_9` rather than `frame_1`.

Latency for these models is timed inline during the real pass, because a
stateful model cannot be re-run over a random sample of frames without
corrupting its state.

Override the naming convention with `--sequence-regex` (one capture group for
the frame number), or force the mode with `--sequence on|off`. Stateless models
are completely unaffected.

**Two rules for sequence models, both learned the hard way:**

1. **Never subsample the frames.** Evaluate the whole split. Taking every Nth
   frame destroys motion estimation: on this project the same model scored
   success 0.073 / precision 0.116 / mean IoU 0.197 on every-30th-frame data
   versus **0.301 / 0.935 / 0.708** on contiguous frames. A stateless detector
   does not care; a tracker moves by 8x.
2. **Never evaluate a prefix.** A tracker may not acquire its target for
   hundreds of frames. Scoring the first 400 frames of one clip here gave 0%,
   while the model in fact detected in 214 frames of that video — starting at
   frame 661. Report the whole split, always.

**Read the tracking block, not mAP.** A single-target tracker returns one box
per frame with no calibrated score, so it has no PR curve and its mAP is
meaningless.

Every model gets the **same metric set** regardless of type — mAP *and*
single-target metrics (success rate, mean IoU, per-frame precision/recall/F1)
are computed for all of them. That is what lets a tracker and a detector sit in
the same table: `succ` and `mIoU` don't depend on score calibration, so they are
the fair basis for comparison when one of the models has no meaningful
confidence.

Metrics always describe the split **as a whole**. `--per-sequence` adds a
per-video breakdown, but it is off by default and is a debugging aid, not part
of the comparable metric set — on a large test split, one row per video is
unreadable and bloats `metrics.json`.

## 5. Dataset layout

Either convention works:

```
dataset/                          dataset/
├── images/val/*.jpg              ├── valid/images/*.jpg
├── labels/val/*.txt      or      ├── valid/labels/*.txt
└── data.yaml                     └── data.yaml
```

`data.yaml` needs at minimum:

```yaml
nc: 1
names: ['drone']
```

Without it, class names fall back to `"0"`, which makes reports unreadable.

Labels are standard YOLO format, one line per object, normalised:

```
<class> <cx> <cy> <w> <h>
```

**Keep your empty-sky frames.** Images with no label file (or an empty one) are
counted deliberately — they're what the false-alarm metric is measured on. On
the indrones test split, 47.8% of frames are empty (13,924 of 29,127), and that
makes `FP/empty frame` trustworthy.

---

## 6. What you get — every metric explained

Around 50 numbers per run. Every model gets the same set, whatever its type.

If you only look at four, look at these: **AP tiny** (detection range),
**recall @ P>=95%** (usable operating point), **FP / empty frame** (false alarms),
**p95 latency** (does it keep up).

---

### 6.1 Accuracy — computed by `pycocotools`

These come from the reference COCO implementation, not a hand-rolled loop, so
they are directly comparable to published numbers.

| Metric | What it means |
|---|---|
| **`mAP@0.50:0.95`** | The headline COCO number. Average Precision averaged over **ten** IoU thresholds — 0.50, 0.55, … 0.95. Because it includes very strict thresholds, it rewards *tight* boxes as well as finding the object. Always much lower than `mAP@0.50`; that is normal, not a bug. |
| **`mAP@0.50`** | AP at a single, forgiving threshold: a box counts if it overlaps ground truth by 50%. Read this as **"did it find the thing at all"**. |
| **`mAP@0.75`** | AP at a strict threshold. Read this as **"is the box actually tight"**. A big gap between `mAP@0.50` and `mAP@0.75` means the model locates objects but boxes them sloppily. |
| **`ar_1` / `ar_10` / `ar_100`** (AR@1 / @10 / @100) | Average Recall when the model is allowed only 1, 10, or 100 detections per image. `AR@1` is "if you had to pick one box per image, how often is it right"; `AR@100` is the practical ceiling on recall. If `AR@1` is far below `AR@100`, the model finds objects but ranks them poorly — the right answer is buried under false positives. |

**What AP actually is.** Sort every detection by confidence, then walk down that
list admitting detections one at a time. At each step compute precision and
recall; AP is the area under the resulting precision-recall curve. **This makes
AP a ranking metric** — it measures whether the model's confidences put good
detections above bad ones, not just whether it found things.

**Consequence:** a model that outputs a constant confidence has no ranking, so
its mAP collapses regardless of box quality. Measured on this project: taking
one model's predictions and flattening the scores to a constant, changing
nothing else, dropped `mAP@0.50` from **0.3075 to 0.0383**. For such models,
read the single-target metrics in 6.5 instead.

---

### 6.2 AP by object size

The same AP calculation, restricted to objects in a size band. Size is
`sqrt(width x height)` in pixels.

| Bucket | Pixel range | Why |
|---|---|---|
| `tiny` | 0–12 px | a drone at real detection range |
| `small` | 12–24 px | still distant |
| `medium` | 24–48 px | mid-range |
| `large` | 48+ px | close, or a big airframe |

Each reports `mAP@0.50` and `mAP@0.50:0.95`.

**Why these cut points.** COCO's own split is 32² / 96² px, which dumps almost
every drone into one bucket and tells you nothing. **`AP tiny` is usually the
single most decision-relevant number here**: two models with identical
`mAP@0.50` can differ enormously on it, and that difference *is* your detection
range. A bucket with no objects in the split says so rather than printing a
misleading zero.

---

### 6.3 Per class

`mAP@0.50` and `mAP@0.50:0.95` for each class separately, plus TP/FP/FN and
precision/recall/F1 per class at the operating point. Collapses to a single row
for a one-class drone dataset; matters as soon as anyone adds `bird`.

---

### 6.4 Choosing a confidence threshold

Your model outputs a score per box; you must pick a cut-off. These say where.

| Metric | What it means |
|---|---|
| **`best_f1`** | The best F1 (harmonic mean of precision and recall) achievable at any threshold. The model's best-balanced performance. |
| **`best_f1_conf`** | **The threshold that achieves it** — this is the number to put in your config if you want balance. |
| **`best_f1_precision`, `best_f1_recall`** | The precision and recall you actually get there, so you can see which side the balance falls on. |
| **`recall_at_p90`, `recall_at_p95`, `recall_at_p99`** | How much you still detect if you demand precision of at least 90 / 95 / 99%. **This is the anti-drone number**: "how many drones do I catch while being right 95% of the time". |
| **`conf_at_p90`, `conf_at_p95`, `conf_at_p99`** | The threshold that gets you there. |

**A `recall_at_p95` of 0.0000 means the model never reaches 95% precision at any
threshold** — there is no setting at which you can trust it. That is a much
stronger statement than a low mAP, and it is often the finding that decides
whether a model is deployable at all.

---

### 6.5 Single-target metrics — comparable across every model type

Computed for every model: take the **highest-scoring box on each frame**, compare
it to ground truth, ask whether it is right.

| Metric | What it means |
|---|---|
| **`success_rate`** | Of the frames that contain a drone, the fraction where the model's best box overlaps it by IoU >= 0.5. Plain-language: **"how often does it find the drone"**. |
| **`mean_iou`** | Average IoU on frames where there was both a target and a prediction. **"When it does find it, how tight is the box"**. Independent of how often it succeeds. |
| **`hits`, `misses`** | Raw counts behind `success_rate`. A miss is either no box at all, or a box that missed. |
| **`false_alarms`** | Boxes emitted on frames that contain **no** drone. |
| **`false_alarm_rate`** | `false_alarms` / number of empty frames. |
| **`precision`, `recall`, `f1`** | Per-frame versions: precision is hits over frames where it predicted anything; recall equals `success_rate`. |
| **`frames_total`, `frames_with_gt`, `frames_empty`, `frames_predicted`** | The denominators, so you can check the arithmetic yourself. |

**Why this exists.** mAP requires a score ranking. A tracker that answers "the
drone is here" with one box and no calibrated confidence has none, so its mAP is
meaningless — but it may track extremely well. These metrics need no ranking, so
they are the **fair basis for comparing a tracker against a detector**. They are
also exactly what tracking papers and validation scripts conventionally report.

---

### 6.6 Deployed behaviour at `conf = 0.25`

Everything above sweeps thresholds. This section fixes one — the threshold you
would actually ship — and reports what happens.

| Metric | What it means |
|---|---|
| **`precision`** | Of the boxes it emits at this threshold, the fraction that are real. |
| **`recall`** | Of the drones present, the fraction it caught. |
| **`f1`** | Their harmonic mean. |
| **`fp_per_empty_frame`** | **False alarms on frames containing no drone.** The single most important operational number: `0.7` means seven false alarms every ten empty frames. |
| **`empty_frames`, `fp_on_empty_frames`** | The counts behind it. |
| **`confusion_matrix`** | An (nc+1) x (nc+1) table with a **background row and column**, so misses (real drone, predicted background) and false alarms (predicted drone, real background) are both visible, not just class confusions. |

**Why FP/empty frame matters more than mAP for this project.** mAP barely
reflects false alarms. A real run here scored a perfect **1.0000 mAP** and
**1.0 false alarms per empty frame** simultaneously — by mAP it was flawless, and
in the field it would have cried wolf on every empty frame.

---

### 6.7 Speed and cost

| Metric | What it means |
|---|---|
| **`mean_ms`** | Average time for one frame, end to end. |
| **`p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`** | Percentiles. **Use `p95` or `p99`, not the mean**, when deciding whether the model holds a frame rate — the mean hides the stalls that drop frames. |
| **`min_ms`, `max_ms`, `std_ms`** | Range and spread. A `max` far above `p99` is usually first-frame warmup, not a real risk. |
| **`fps_mean`, `fps_p99`** | The same, as frame rates. `fps_p99` is your worst-case throughput. |
| **`peak_vram_mb`** | Peak GPU memory — decides whether it fits on the target board. |
| **`gpu_name`** | Recorded so the leaderboard can warn when runs were timed on different hardware. |
| **`n_samples`** | How many frames were timed. |
| **`frames_errored`** | Frames where the model raised an exception. Non-zero means the numbers are a **lower bound** and the model needs fixing. |

**How it is measured.** A separate batch=1 pass, warmed up and CUDA-synced,
timing the whole `predict()` call **including preprocessing and NMS** — that is
what the drone actually pays for. Batched throughput divided by batch size is a
different and much flattering number; evalkit does not report that.

For sequential models the timing comes from the real run instead, because a
stateful model cannot be re-run over a random sample without corrupting its
state.

---

### 6.8 Model and dataset facts

| Field | Meaning |
|---|---|
| `params` | Parameter count. Read alongside accuracy — a model 3x larger for 10% more mAP may not be the right trade. |
| `weights_mb` | Checkpoint size on disk. |
| `input_channels` | 1 = grayscale, 3 = RGB, read from the first conv layer. |
| `class_names` | What the class indices mean. |
| `images`, `annotations`, `empty_images` | Split composition. **`empty_images` is what `fp_per_empty_frame` is measured on** — if it is 0, that metric reads `n/a`. |

---

### 6.9 Provenance and comparability

Every `metrics.json` records the dataset path, split, image size, device,
timestamp, the exact scoring settings used, and:

| Field | Meaning |
|---|---|
| `protocol.version` | Which frozen ruleset scored this run. Different versions are not comparable. |
| `protocol.status` | `OK`, or `MODIFIED` if someone overrode a frozen setting. |
| `protocol.comparable` | `false` marks the run `(!)` on the leaderboard. |
| `protocol.deviations` | Exactly which settings were changed, and to what. |

This is what makes disagreements settleable: if two results conflict, the answer
is in these fields rather than in anyone's memory of how they ran it.

---

## 7. The dashboard

`dashboard.png` — six headline tiles over nine panels, plus the protocol stamp,
so a screenshot carries its own comparability warning.

Tiles are colour-coded: **AP tiny** and **FP/empty frame** go red when they're
bad, so the two numbers that decide an anti-drone model can't be skimmed past.

Panels: mAP vs IoU · AP by object size · score card · PR curve · F1 vs
confidence · recall at high precision · confusion matrix · TP/FP separation ·
latency.

Two rules it keeps:

- **Every axis says what it contains.** The mAP-vs-IoU panel prints
  `averaged over all 10: 0.50 to 0.95` on its own label.
- **No panel is ever blank.** Where something genuinely wasn't measured —
  latency on a submitted-predictions run, a size bucket with no objects,
  FP/empty on a split with no empty frames — the panel says so in words. An
  empty pair of axes, or a green `0.000` for a test that never ran, is worse
  than no panel at all.

Viewing it over SSH with no display: open it in VS Code, or use a desktop
session (`eog dashboard.png`).

---

## 8. The frozen protocol — why numbers stay comparable

Every scoring knob lives in one file, [`core/protocol.py`](core/protocol.py):

```python
PROTOCOL = {"conf": 0.001, "nms_iou": 0.65, "max_det": 300,
            "operating_conf": 0.25, "split": "val"}
VERSION = "1.0"
```

`run_eval.py` takes its defaults from there, so nobody has to remember them.

You **can** override any of them — occasionally right when debugging — but the
run is then stamped `protocol: MODIFIED` in `metrics.json` and `summary.txt`,
and `compare.py` marks it `(!)` and prints a warning. That number is no longer
on the same footing as everyone else's.

**What is not frozen:** image size, letterbox vs stretch, colour space. Those
belong to the model, not the protocol. A net trained at 960 grayscale must run
at 960 grayscale or its numbers are meaningless.

**Bumping the version:** changing any frozen value invalidates comparison with
older results, so bump `VERSION` in the same commit. Old results keep their old
stamp and the mismatch becomes visible instead of silent.

---

## 9. The leaderboard

```bash
python compare.py results/
python compare.py results/ --sort map50_95 --csv leaderboard.csv
```

```
model                  mAP@50:95  mAP@50  AP tiny  AP small  best F1  R@P95   FP/empty  p95 ms  FPS    params M
oracle_dedup           0.9802     0.9802  0.9802   0.9901    0.9899   0.9800  0.000     —       —      —
student_gray_indrones  0.0183     0.0653  0.0094   0.2596    0.2084   0.0000  0.689     4.03    257.5  2.08
```

Any run in the folder appears; nothing to configure. `--csv` writes the full
~25-column table.

It refuses to let a table quietly mislead. It warns when runs changed the frozen
settings, used **different datasets or splits**, used **mixed protocol
versions**, or measured latency on **different GPUs** (informational — accuracy
still compares).

---

## 10. Every flag

### What to evaluate

| Flag | Default | Notes |
|---|---|---|
| `--weights` | — | path to your model weights |
| `--dataset` | **required** | dataset root containing `data.yaml` and the split |
| `--predictor FILE.py` | — | your own predictor, living in your own repo (Route 3) |
| `--preds PREDS.json` | — | score an existing COCO predictions file (Route 4) |

You need either `--weights` or `--preds`.

### Auto-detected — pass only to override

| Flag | Default | Notes |
|---|---|---|
| `--imgsz` | from checkpoint | inference image size |
| `--color` | `auto` | `gray` / `rgb`; read from the first conv layer |
| `--format` | auto-detected | `onnx`, `torchscript`, `ultralytics`, `visionkit`, `yolov5`, `custom_template` |

### Output

| Flag | Default | Notes |
|---|---|---|
| `--name` | weights folder name | run name, and the leaderboard label |
| `--out` | `results` | output root |
| `--split` | `val` | split to evaluate |

### Run control

| Flag | Default | Notes |
|---|---|---|
| `--device` | `cuda` | `cuda` / `cuda:0` / `cpu` |
| `--latency-images` | `200` | more = tighter percentiles, slower |
| `--skip-latency` | off | **use when not on the reference GPU** |
| `--skip-plots` | off | numbers only, much faster |
| `--model-name` | — | for `--format visionkit`: which plugin |
| `--nc` | from dataset | override number of classes |

### Frozen scoring — changing one marks the run non-comparable

| Flag | Frozen value |
|---|---|
| `--conf` | `0.001` |
| `--operating-conf` | `0.25` |
| `--nms-iou` | `0.65` |
| `--max-det` | `300` |

### Helper scripts

```bash
python manifest.py     --dataset DS -o manifest.json     # image_id mapping (Route 4)
python from_yolo_txt.py --labels DIR --dataset DS [--split val] [--boxes xywh|xyxy] -o preds.json
python compare.py      results/ [--sort COL] [--csv FILE]
python selftest.py                                        # verify the install
```

---

## 11. Troubleshooting

**`Could not work out what kind of model 'best.pt' is`**
Pass `--format` if it's a known type, `--predictor` if it's your own
architecture, or use `--preds` to just score predictions.

**`Can't get attribute 'DetectionModel' on <module 'models.yolo' …>`**
Your checkpoint pickles a reference to its model class, and that class isn't
importable — or a *different* package with the same name is shadowing it. Run
from the directory where your model's code lives, or make sure its repo is on
`PYTHONPATH`. This is why Route 3 puts your predictor in your own repo.

**mAP is 0.0000 but my model works in my own demo**
In order of likelihood:
1. Boxes not mapped back to original image pixels.
2. Wrong output contract — sigmoid applied twice, or not at all.
3. Wrong `image_id` mapping (Route 4).
4. The model genuinely doesn't transfer to this dataset.

To tell (4) from the rest, run the **oracle control** — see the next section. If
the oracle scores 1.0000, the harness is fine and the problem is your model or
your predictor.

**`N image_id(s) … are not in this dataset`**
Your predictions use ids that don't exist here — usually 0-indexed instead of
1-indexed, or generated against a different split. Regenerate `manifest.json`
and use those ids verbatim.

**`Detections … are missing ['category_id', 'score']`**
Your `preds.json` entries need all four keys: `image_id`, `category_id`,
`bbox`, `score`.

**`… defines no DetectorAdapter subclass`**
Your predictor file must contain exactly one class inheriting
`DetectorAdapter`. Start from `templates/external_predictor.py`.

**Recall looks lower than it should**
You probably ran inference with a deployment confidence threshold. Re-run with
`--conf-thres 0.001` so the PR curve gets its tail.

**`model produced zero detections above the confidence floor`**
Nothing at all came out at 0.001, which almost always means a preprocessing or
decode mismatch, not a bad model. Print your raw output shape and score range
before assuming the model is dead.

**Latency columns are empty**
Expected on Routes 2 and 4. Latency has to be timed inside the model call on the
reference GPU.

**Slow on a large dataset**
`--skip-plots` and a smaller `--latency-images` help. Ground-truth building reads
image headers only (~1 s for 29k images), and inference is the real cost — the
full 29,127-image indrones split takes about 2m40s end to end on an RTX PRO 5000.

---

## 12. Verifying the harness itself

Don't take the numbers on faith. Two checks, both cheap:

**Self-test** — builds a synthetic dataset, runs a deliberately imperfect fake
detector through the whole pipeline, asserts the metrics are sane and every plot
renders. No GPU, no model, no data needed:

```bash
python selftest.py
```

**Oracle control** — feed the ground truth back in as predictions. A correct
harness must score 1.0000 on your own dataset. If it doesn't, the problem is
coordinate handling, not your model:

```python
import json
gt = json.load(open("results/<run>/gt.json"))
json.dump([{"image_id": a["image_id"], "category_id": a["category_id"],
            "bbox": a["bbox"], "score": 0.9} for a in gt["annotations"]],
          open("oracle.json", "w"))
```

```bash
python run_eval.py --preds oracle.json --dataset /shared/drone_dataset --name oracle
```

This is worth running once per dataset. It's how we established that a 0.0183
mAP on indrones was a genuine model failure rather than a harness bug — the
oracle scored exactly 1.0000 on the same data.

The oracle also reveals **label quality problems**. On indrones it capped at
`AR@100 = 0.9818` instead of 1.0, which located 306 duplicate annotations
(1.8%) — a ceiling no model can exceed.

---

## 13. Team rules

Four things matter more than anything in this code:

1. **Everyone uses the same frozen val split.** Same path, no re-splits, no
   added images, no label fixes mid-experiment.
2. **Latency on one machine.** Accuracy travels between machines; latency
   doesn't. Run the timing pass on the reference GPU (ideally the deployment
   target) and use `--skip-latency` elsewhere, so misleading numbers never get
   recorded.
3. **Declare your class IDs.** If your model outputs `0: drone` and someone
   else's is `0: bird, 1: drone`, say so — the mapping isn't guessable.
4. **Evaluate the whole split, never a slice.** No prefixes, no every-Nth-frame
   sampling, no `--max-frames`. It matters most for sequence models: a tracker
   may not acquire for hundreds of frames, so a prefix can report 0% for a model
   that works fine later in the clip.
5. **Pull before you run.** Every `metrics.json` records the protocol version;
   if two results disagree, check you're on the same one.

And one recommendation: **make the predictor file part of what you hand over.**

```
their_model/
├── best.pt
├── their_code/
├── requirements.txt
└── evalkit_predictor.py    ← ~15 lines, written once by the author
```

Only the author knows their preprocessing and output contract. Fifteen lines
from them saves everyone else an interrogation — and prevents a model scoring
0.0000 for the wrong reason.

---

## 14. How it works inside

Two halves that meet at a JSON file:

```
      MODEL SIDE (varies per person)          SCORING SIDE (frozen, shared)
 ┌────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ adapters/  or  --predictor     │   │  core/dataset  → gt.json             │
 │   .predict(bgr) → (N,6) boxes  │──▶│  core/metrics  → pycocotools + curves│
 │ or you generate preds.json     │   │  core/plots    → 7 PNGs              │
 └────────────────────────────────┘   │  core/dashboard→ dashboard.png       │
                    │                 └──────────────────────────────────────┘
              preds.json  ◀────── the contract ──────────┘
```

Everything left of `preds.json` may differ between teammates. Everything right
of it is identical for everyone — that's what makes the numbers comparable.

| File | Job |
|---|---|
| `run_eval.py` | entry point; routes to one of the four sources, then runs identical scoring |
| `compare.py` | leaderboard across runs, with comparability warnings |
| `manifest.py` | `image_id` mapping for Route 4 |
| `from_yolo_txt.py` | converts YOLO `--save-txt` output into `preds.json` (Route 2) |
| `selftest.py` | end-to-end verification with synthetic data |
| `core/protocol.py` | the frozen scoring settings |
| `core/dataset.py` | YOLO split → COCO ground truth; keeps empty frames |
| `core/predict.py` | inference loop and the separate batch=1 latency pass |
| `core/metrics.py` | pycocotools AP, size buckets, operating point, confusion matrix |
| `core/plots.py` | the seven standard plots |
| `core/dashboard.py` | the one-page summary |
| `adapters/base.py` | the `DetectorAdapter` contract plus `letterbox` / `scale_boxes` / `nms_numpy` |
| `adapters/*.py` | built-in formats; auto-discovered, filename becomes the `--format` value |
| `templates/external_predictor.py` | what you copy into your own repo (Route 3) |

**Why predictions are written to disk.** Inference and scoring are separate
steps, and `preds.json` is written before any metric is computed. That means
metrics recompute in seconds, a new metric added later doesn't require re-running
anyone's model, someone can run inference on a Jetson and send you just the JSON,
and when two runs disagree you can diff them box for box.

Rescore an existing run without a GPU:

```python
from core.dataset import EvalDataset
from core.metrics import compute_metrics
import json

ds = EvalDataset("/shared/drone_dataset", "val")
dets = json.load(open("results/my_model/preds.json"))
print(compute_metrics(ds, dets, {}, operating_conf=0.25)["accuracy"])
```

**Adding a new shared format** (a maintainer job, not something a new member
should need): drop an adapter in `adapters/`, starting from
`adapters/custom_template.py`. The filename becomes the `--format` value; there
is no registration step.
