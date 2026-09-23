# RT-DETR
RT-DERT v2 model used for drone detection.

Link to weights: https://drive.google.com/file/d/1IYMBePuUbDc-XlKewqYTOo0g3I8FiGzx/view?usp=drive_link

Trained on Roboflow: https://universe.roboflow.com/tracker-qjlj1/drones_new/dataset/4

---

## Evaluate with evalkit

`evalkit/` scores the model on any drone dataset with the same frozen settings
the rest of the team uses, so numbers are directly comparable between models.
It reports mAP (overall and by drone size), precision / recall / F1, false
positives per empty frame, and latency, plus plots and a one-page dashboard.

The model is the RT-DETRv2 in `rtdetrv2_pytorch/`: ResNet-18 backbone,
**grayscale (1-channel)** input, **1 class** (`drone`), 640×640.

### 1. Set up the environment

```bash
git clone https://github.com/Swarm-Drone-Project/RT-DETR.git
cd RT-DETR
git checkout evalkit-rtdetrv2          # until this branch is merged into main

python3 -m venv .venv
source .venv/bin/activate

# PyTorch first — pick the build for your CUDA version from https://pytorch.org
pip install torch torchvision
pip install -r rtdetrv2_pytorch/requirements.txt -r evalkit/requirements.txt
```

Tested with Python 3.12, torch 2.13, torchvision 0.28, numpy 2.5, opencv 5.0,
pycocotools 2.0.11.

### 2. Download the weights and place them

Download the checkpoint from the Google Drive link at the top of this file and
save it in the **repo root** as `train_robo.pth`:

```bash
# in the browser: open the link above -> Download
# or from the terminal:
pip install gdown
gdown 1IYMBePuUbDc-XlKewqYTOo0g3I8FiGzx -O train_robo.pth
```

The folder should then look like this:

```
RT-DETR/
├── train_robo.pth        <- the weights (~477 MB, not in git)
├── rtdetrv2_pytorch/     <- model code + configs (evalkit builds the model from here)
└── evalkit/
    ├── run_eval.py
    ├── adapters/rtdetrv2_adapter.py
    └── results/          <- every run's results land here
```

The weights can live anywhere — you pass the path with `--weights` — but the
commands below assume the repo root. `*.pth` is in `.gitignore`, so it will
never be committed by accident.

### 3. Prepare the dataset

evalkit reads YOLO-format datasets. Either layout works:

```
my_dataset/                         my_dataset/            (Roboflow export)
├── data.yaml                       ├── data.yaml
├── images/<split>/*.jpg            └── <split>/
└── labels/<split>/*.txt                ├── images/*.jpg
                                        └── labels/*.txt
```

- `<split>` is `val`, `test` or `train` — you choose it with `--split`.
- `data.yaml` needs at least:
  ```yaml
  nc: 1
  names: ['drone']
  ```
- Each label file has the same name as its image, one line per drone:
  `0 cx cy w h` (class id, then box centre and size, all normalised 0–1).
  Class id must be `0`, since the model has one class.
- Frames with no drone: an empty `.txt`, or no `.txt` at all. Keep them — false
  alarms on empty sky are one of the most important numbers in the report.

### 4. Run the evaluation

All commands run from the repo root with the venv active.

**Any dataset:**

```bash
python evalkit/run_eval.py \
    --format  rtdetrv2 \
    --weights train_robo.pth \
    --dataset /path/to/my_dataset \
    --split   test \
    --device  cuda \
    --name    train_robo_my_dataset
```

**Examples used for the results in this repo:**

```bash
# InDrones test split (8 held-out videos, 29,127 frames)
python evalkit/run_eval.py --format rtdetrv2 --weights train_robo.pth \
    --dataset data_set/indrones_full_test --split test \
    --device cuda --name train_robo_indrones_gpu

# Unreal Engine synthetic val set (19,505 frames)
python evalkit/run_eval.py --format rtdetrv2 --weights train_robo.pth \
    --dataset data_set/unreal_engine_dc_val --split val \
    --device cuda --name train_robo_unrealdc_val
```

**No GPU?** Use `--device cpu --skip-latency`. Accuracy numbers are the same;
latency is skipped because CPU timings are not comparable.

Useful options:

| Option | What it does |
|---|---|
| `--name` | Folder name for this run under `evalkit/results/`. Reusing a name overwrites it |
| `--split` | Which split folder to evaluate (default `val`) |
| `--skip-latency` | Accuracy only |
| `--skip-plots` | Numbers only, no plots |
| `--latency-images N` | Images used for the latency pass (default 200) |

Do **not** change `--conf`, `--operating-conf`, `--nms-iou` or `--max-det`.
These are frozen by the team protocol; changing them marks the run
`NOT COMPARABLE`.

### 5. Read the results

Every run writes to `evalkit/results/<name>/`, whichever folder you run from:

| File | Contents |
|---|---|
| `summary.txt` | All the numbers, human-readable — start here |
| `dashboard.png` | One-page visual report |
| `metrics.json` | All the numbers, machine-readable |
| `plots/*.png` | PR curve, F1 vs confidence, mAP vs IoU, AP by size, confusion matrix, … |
| `preds.json`, `gt.json` | Raw predictions / ground truth (large, kept out of git) |

To put several runs side by side:

```bash
python evalkit/compare.py evalkit/results/
```

### Results so far (`train_robo.pth`)

| Run | Dataset | mAP@50 | mAP@50:95 | FP / empty frame | Latency (GPU, batch 1) |
|---|---|---|---|---|---|
| `train_robo_indrones_gpu` | InDrones test | 0.531 | 0.287 | 2.90 | 7.7 ms (130 FPS) |
| `train_robo_indrones_full` | InDrones test (CPU) | 0.531 | 0.287 | 2.93 | — |
| `train_robo_unrealdc_val` | Unreal Engine val | 0.140 | 0.070 | 2.61 | 16.5 ms |

Full details are in each run's `summary.txt`.

### Troubleshooting

| Message | Fix |
|---|---|
| `RT-DETR code not found at …` | Run from a full clone of this repo. If `rtdetrv2_pytorch/` is somewhere else, set `RTDETR_ROOT=/path/to/rtdetrv2_pytorch` |
| `size mismatch` in `load_state_dict` | The checkpoint is not the grayscale r18 drone model. Add its config to `_CONFIG_FOR_WEIGHTS` in `evalkit/adapters/rtdetrv2_adapter.py` |
| `Weights not found` | Check the `--weights` path (step 2) |
| `No 'test' split under …` | The folder layout doesn't match step 3, or the `--split` name is wrong |
| All metrics are `nan` | The split has no labels — check `labels/<split>/` exists and the `.txt` names match the images |

More on evalkit itself (metrics, protocol, other model formats): `evalkit/README.md`.
