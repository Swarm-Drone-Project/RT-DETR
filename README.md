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

Check that PyTorch can see the GPU — this must print `True`:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, the installed torch build does not match the machine's
CUDA driver: reinstall torch using the command for your CUDA version from
https://pytorch.org.

> Every new terminal: `cd` into the repo and run `source .venv/bin/activate`
> again before any command below.

### 2. Download the weights and place them

The weights are **not in git** (too large). Download them from Google Drive:

https://drive.google.com/file/d/1IYMBePuUbDc-XlKewqYTOo0g3I8FiGzx/view?usp=drive_link

**Option A — terminal** (works only if the file is shared as "Anyone with the link"):

```bash
pip install gdown
gdown 1IYMBePuUbDc-XlKewqYTOo0g3I8FiGzx -O train_robo.pth
```

**Option B — browser** (use this if Option A fails):

1. Open the link above in a browser, signed in to a Google account that has
   access to the file.
2. Click **Download**. Google may warn that it can't scan a file this large for
   viruses — click **Download anyway**.
3. Move the downloaded file into the repo root and name it exactly
   `train_robo.pth`. For example, if it landed in `~/Downloads`:
   ```bash
   mv ~/Downloads/<downloaded-file-name>.pth ./train_robo.pth
   ```

**If gdown says `Cannot retrieve the public link of the file`**, or the browser
says **"You need access"**: the file is not shared publicly. Click
**Request access** in the browser, or ask the RT-DETR maintainers to set the
file's sharing to *Anyone with the link → Viewer*. There is nothing wrong with
your setup.

**Check you have the right file.** Run this in the repo root:

```bash
ls -l train_robo.pth        # size must be 477254729 bytes
sha256sum train_robo.pth    # must print the value below
```

```
8e66ffe0fc2b4cc3cdefffe1f4076eb181aac3bef30298029c9bb28263b3e683
```

If the size or the checksum differs, the download is incomplete or it is a
different checkpoint. Download it again, and don't compare your numbers with
the results below until it matches. A small file (a few KB) is usually a
Google Drive HTML error page, not the weights.

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

If a path contains spaces (e.g. `~/Desktop/New Folder/...`), wrap it in quotes:
`--dataset "/home/me/Desktop/New Folder/my_dataset"`.

**Examples used for the results in this repo.** These datasets are **not in
git** either. The commands assume a copy has been placed under `data_set/` in
the repo root; with your own dataset, use the "Any dataset" command above.

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
| gdown: `Cannot retrieve the public link of the file` | The Drive file is not public — use the browser download in step 2 |
| `UnpicklingError` / `invalid load key` when loading weights | The download is broken (often an HTML page saved as `.pth`) — check size and sha256 in step 2 |
| `CUDA` errors, or torch says no GPU | Check step 1's `torch.cuda.is_available()`; reinstall torch for your CUDA version |
| `No 'test' split under …` | The folder layout doesn't match step 3, or the `--split` name is wrong |
| All metrics are `nan` | The split has no labels — check `labels/<split>/` exists and the `.txt` names match the images |

More on evalkit itself (metrics, protocol, other model formats): `evalkit/README.md`.
