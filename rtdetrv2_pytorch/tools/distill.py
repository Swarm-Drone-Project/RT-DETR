"""
distill.py
──────────
Knowledge Distillation: R50 Gray Temporal (Teacher) → R18 Gray No-Temporal (Student)

Teacher: 300 queries
Student: 100 queries  ← reduced for single-class drone detection

Three distillation losses:
  Loss 1 — Backbone feature mimicry  (P3, P4, P5 via 1x1 adaptors)
  Loss 2 — Encoder feature mimicry   (post-transformer P5)
  Loss 3 — Decoder soft labels       (KL on top-K matched queries + L1 on boxes)

Usage:
    python tools/distill.py
    python tools/distill.py --resume ./output/rtdetrv2_r18vd_drone_gray1ch_distill2/checkpoint0010.pth
    python tools/distill.py --test_only
"""

import sys
import os
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

RTDETR_PATH = "/home/devananth/office/Vision/RT-DETR/rtdetrv2_pytorch"
sys.path.insert(0, RTDETR_PATH)

from src.core import YAMLConfig
from src.solver.det_engine import evaluate

# ── CONFIG ────────────────────────────────────────────────────────────────────
TEACHER_CONFIG     = f"{RTDETR_PATH}/configs/rtdetrv2/rtdetrv2_r50vd_drone_gray1ch_temporal.yml"
TEACHER_CHECKPOINT = f"{RTDETR_PATH}/output/rtdetrv2_r50vd_drone_gray1ch_temporal/best.pth"
STUDENT_CONFIG     = f"{RTDETR_PATH}/configs/rtdetrv2/rtdetrv2_r18vd_drone_gray1ch.yml"
OUTPUT_DIR         = f"{RTDETR_PATH}/output/rtdetrv2_r18vd_drone_gray1ch_distill2"

STUDENT_WARMSTART_FROM_TEACHER = True

DEVICE       = "cuda"
NUM_EPOCHS   = 30

# Teacher has 300 queries, student has 100.
# For soft-label loss we match the top-100 teacher queries to student queries
# by score so the KL loss is computed on equal-sized tensors.
TEACHER_QUERIES = 300
STUDENT_QUERIES = 100

# Loss weights
ALPHA       = 0.3    # distillation vs GT balance
W_SOFT      = 1.0    # KL divergence weight
W_BOX       = 1.0    # soft box L1 weight
W_ENC_FEAT  = 0.5    # encoder feature MSE weight
W_BB_FEAT   = 0.3    # backbone feature MSE weight
TEMPERATURE = 4.0    # softening temperature for KL

CLIP_NORM    = 0.1
LR           = 2e-5
LR_BACKBONE  = 5e-6
WEIGHT_DECAY = 1e-4
PRINT_FREQ   = 50
CKPT_FREQ    = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADAPTORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BackboneAdaptors(nn.Module):
    """
    Projects student R18 backbone channels into teacher R50 channel space.
    R18 [128, 256, 512] → R50 [512, 1024, 2048] via 1x1 convs.
    """
    def __init__(self):
        super().__init__()
        self.adapt = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128,  512,  1, bias=False),
                          nn.BatchNorm2d(512),  nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256,  1024, 1, bias=False),
                          nn.BatchNorm2d(1024), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(512,  2048, 1, bias=False),
                          nn.BatchNorm2d(2048), nn.ReLU(inplace=True)),
        ])

    def forward(self, feats):
        return [self.adapt[i](f) for i, f in enumerate(feats)]


class EncoderAdaptor(nn.Module):
    """
    Learnable 1x1 projection on student encoder P5 features [B,256,H,W].
    Gives student features room to rotate towards teacher's richer
    temporal-aware representations without hard per-dimension constraints.
    """
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE HOOKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FeatureStore:
    def __init__(self):
        self.backbone_feats = None   # list of 3 tensors
        self.encoder_feat   = None   # single [B,256,H,W] tensor at P5


def register_hooks(model, store):
    handles = []

    def bb_hook(module, inp, out):
        store.backbone_feats = [f.detach() for f in out]

    def enc_hook(module, inp, out):
        # out is list of FPN/PAN features; index 2 = P5
        store.encoder_feat = out[2].detach()

    handles.append(model.backbone.register_forward_hook(bb_hook))
    handles.append(model.encoder.register_forward_hook(enc_hook))
    return handles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOSSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def loss_backbone_feat(s_feats, t_feats, adaptors):
    """
    Loss 1 — Backbone feature MSE at all 3 FPN levels.
    student features are projected to teacher channel space first.
    """
    adapted = adaptors(s_feats)
    total   = 0.0
    for sf, tf in zip(adapted, t_feats):
        if sf.shape[2:] != tf.shape[2:]:
            sf = F.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
        total = total + F.mse_loss(sf, tf)
    return total / 3.0


def loss_encoder_feat(s_enc, t_enc, adaptor):
    """
    Loss 2 — Encoder feature MSE at P5.
    Teacher P5 features include temporal context; student learns to
    approximate this from spatial cues alone.
    """
    sp = adaptor(s_enc)
    if sp.shape[2:] != t_enc.shape[2:]:
        sp = F.interpolate(sp, size=t_enc.shape[2:], mode='bilinear', align_corners=False)
    return F.mse_loss(sp, t_enc)


def loss_soft_labels(s_out, t_out, temperature=4.0):
    """
    Loss 3 — Soft label distillation at decoder output.

    Teacher has 300 queries, student has 100.
    Strategy: take the top-100 teacher queries ranked by max class score
    and align them with all 100 student queries.

    This is correct because:
    - The teacher's top-100 are the most confident predictions
    - The student should learn from those, not from the teacher's 200
      low-confidence/background queries
    - Both tensors are now [B, 100, C] — shapes match for KL loss

    Returns (kl_loss, box_loss).
    """
    s_logits = s_out['pred_logits']    # [B, 100, C]
    t_logits = t_out['pred_logits']    # [B, 300, C]
    s_boxes  = s_out['pred_boxes']     # [B, 100, 4]
    t_boxes  = t_out['pred_boxes']     # [B, 300, 4]

    B = s_logits.shape[0]

    # Select top-100 teacher queries by max class score
    t_scores, _ = t_logits.max(dim=-1)          # [B, 300]
    top_idx     = t_scores.topk(STUDENT_QUERIES, dim=1).indices  # [B, 100]

    # Gather top-100 teacher logits and boxes
    top_idx_l = top_idx.unsqueeze(-1).expand(-1, -1, t_logits.shape[-1])  # [B,100,C]
    top_idx_b = top_idx.unsqueeze(-1).expand(-1, -1, 4)                   # [B,100,4]
    t_logits_top = torch.gather(t_logits, 1, top_idx_l)   # [B, 100, C]
    t_boxes_top  = torch.gather(t_boxes,  1, top_idx_b)   # [B, 100, 4]

    # KL divergence on flattened [B*100, C]
    BQ, C  = B * STUDENT_QUERIES, s_logits.shape[-1]
    sf     = s_logits.reshape(BQ, C)
    tf     = t_logits_top.reshape(BQ, C)

    kl_loss = F.kl_div(
        F.log_softmax(sf / temperature, dim=-1),
        F.softmax(tf   / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Box L1 between student queries and top teacher queries
    box_loss = F.l1_loss(s_boxes, t_boxes_top)

    return kl_loss, box_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_teacher(device):
    print("Loading teacher (R50 gray temporal, 300 queries)...")
    cfg   = YAMLConfig(TEACHER_CONFIG, resume=TEACHER_CHECKPOINT)
    ckpt  = torch.load(TEACHER_CHECKPOINT, map_location=device)
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state)
    model = cfg.model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Force dynamic PE and mask computation during distillation.
    # Teacher receives multi-scale batches (480-800px) from the student
    # dataloader. Precomputed values are only valid at eval_spatial_size=640.
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    print(f"  Teacher loaded (dynamic spatial size enabled).")
    return model


def load_student(device):
    print("Loading student (R18 gray no-temporal, 100 queries)...")
    cfg   = YAMLConfig(STUDENT_CONFIG)
    model = cfg.model.to(device)

    if STUDENT_WARMSTART_FROM_TEACHER:
        print("  Warm-starting encoder/decoder from teacher...")
        t_ckpt  = torch.load(TEACHER_CHECKPOINT, map_location=device)
        t_state = t_ckpt.get("ema", {}).get("module", t_ckpt.get("model", t_ckpt))
        s_state = model.state_dict()
        n = 0
        for k, v in t_state.items():
            if k.startswith('backbone'):         continue
            if 'encoder.input_proj' in k:        continue
            if 'temporal_fusion'    in k:        continue
            # Skip decoder query embeddings — different size (300 vs 100)
            if 'query_embed'        in k:        continue
            if 'tgt_embed'          in k:        continue
            if 'query_pos_embed'    in k:        continue
            if k in s_state and s_state[k].shape == v.shape:
                s_state[k] = v
                n += 1
        model.load_state_dict(s_state)
        print(f"  Warm-started {n} tensors from teacher.")

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Student params: {total:,} total  |  {train:,} trainable")
    return model, cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_one_epoch(student, teacher, criterion,
                    bb_adapt, enc_adapt,
                    t_store, s_store,
                    loader, optimizer, scaler,
                    epoch, device):
    student.train()

    logs = {k: [] for k in
            ['loss', 'loss_gt', 'loss_kl', 'loss_box_soft', 'loss_enc', 'loss_bb']}
    t0   = time.time()

    for i, (samples, targets) in enumerate(loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Teacher forward — no grad
        with torch.no_grad():
            t_out = teacher(samples, targets)

        t_bb  = t_store.backbone_feats
        t_enc = t_store.encoder_feat

        # Student forward — with grad + AMP
        with torch.amp.autocast('cuda'):
            s_out = student(samples, targets)

            s_bb  = s_store.backbone_feats
            s_enc = s_store.encoder_feat

            l_bb          = loss_backbone_feat(s_bb, t_bb, bb_adapt)
            l_enc         = loss_encoder_feat(s_enc, t_enc, enc_adapt)
            l_kl, l_bsoft = loss_soft_labels(s_out, t_out, TEMPERATURE)
            l_gt          = sum(criterion(s_out, targets).values())

            l_distill  = (W_SOFT * l_kl + W_BOX * l_bsoft
                        + W_ENC_FEAT * l_enc + W_BB_FEAT * l_bb)
            total_loss = (1.0 - ALPHA) * l_gt + ALPHA * l_distill

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(student.parameters()) +
            list(bb_adapt.parameters()) +
            list(enc_adapt.parameters()),
            CLIP_NORM
        )
        scaler.step(optimizer)
        scaler.update()

        for k, v in zip(logs.keys(),
                        [total_loss, l_gt, l_kl, l_bsoft, l_enc, l_bb]):
            logs[k].append(v.item())

        if (i + 1) % PRINT_FREQ == 0:
            n   = PRINT_FREQ
            avg = {k: sum(logs[k][-n:]) / n for k in logs}
            print(f"Epoch[{epoch}] [{i+1:4d}/{len(loader)}]  "
                  f"loss={avg['loss']:.4f}  gt={avg['loss_gt']:.4f}  "
                  f"kl={avg['loss_kl']:.4f}  box_s={avg['loss_box_soft']:.4f}  "
                  f"enc={avg['loss_enc']:.4f}  bb={avg['loss_bb']:.4f}  "
                  f"({time.time()-t0:.0f}s)")

    mean = {k: sum(v) / max(len(v), 1) for k, v in logs.items()}
    print(f"\nEpoch[{epoch}] DONE — "
          + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()) + "\n")
    return mean


def save_ckpt(epoch, student, bb_adapt, enc_adapt, optimizer, scaler, best, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = f"{outdir}/checkpoint{epoch:04d}.pth"
    torch.save({'epoch': epoch, 'model': student.state_dict(),
                'bb_adaptors': bb_adapt.state_dict(),
                'enc_adaptor': enc_adapt.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_stat': best}, path)
    print(f"Checkpoint → {path}")
    return path


def load_ckpt(path, student, bb_adapt, enc_adapt, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt['model'])
    if 'bb_adaptors' in ckpt: bb_adapt.load_state_dict(ckpt['bb_adaptors'])
    if 'enc_adaptor' in ckpt: enc_adapt.load_state_dict(ckpt['enc_adaptor'])
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        print('No optimizer state in checkpoint, starting fresh')
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    else:
        print('No scaler state in checkpoint, starting fresh')
    epoch = ckpt.get('epoch', 0)
    print(f"Resumed from {path} (epoch {epoch})")
    return epoch + 1, ckpt.get('best_stat', {'mAP': 0.0, 'epoch': -1})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',    type=str, default=None)
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    teacher             = load_teacher(device)
    student, s_cfg      = load_student(device)
    bb_adapt            = BackboneAdaptors().to(device)
    enc_adapt           = EncoderAdaptor(256).to(device)

    t_store, s_store    = FeatureStore(), FeatureStore()
    t_handles           = register_hooks(teacher, t_store)
    s_handles           = register_hooks(student, s_store)

    criterion           = s_cfg.criterion
    train_loader        = s_cfg.train_dataloader
    val_loader          = s_cfg.val_dataloader
    evaluator           = s_cfg.evaluator
    postprocessor       = s_cfg.postprocessor

    if args.test_only:
        print("Validation only...")
        stats, _ = evaluate(student, criterion, postprocessor,
                            val_loader, evaluator, device)
        print(f"Student val stats: {stats}")
        return

    # Optimizer — three param groups
    bb_params  = [p for n, p in student.named_parameters()
                  if 'backbone' in n and p.requires_grad]
    oth_params = [p for n, p in student.named_parameters()
                  if 'backbone' not in n and p.requires_grad]
    adapt_p    = list(bb_adapt.parameters()) + list(enc_adapt.parameters())

    optimizer  = torch.optim.AdamW([
        {'params': bb_params,  'lr': LR_BACKBONE},
        {'params': oth_params, 'lr': LR},
        {'params': adapt_p,    'lr': LR},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine LR with 2-epoch warmup
    def lr_fn(e):
        if e < 2: return e / 2
        return 0.5 * (1 + math.cos(math.pi * (e - 2) / max(NUM_EPOCHS - 2, 1)))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler     = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best        = {'mAP': 0.0, 'epoch': -1}

    if args.resume:
        start_epoch, best = load_ckpt(
            args.resume, student, bb_adapt, enc_adapt, optimizer, scaler, device)
        for _ in range(start_epoch):
            scheduler.step()

    print(f"\n{'='*70}")
    print(f"Distillation  R50-gray-temporal → R18-gray (100 queries)")
    print(f"Epochs: {NUM_EPOCHS}  |  ALPHA={ALPHA}  T={TEMPERATURE}")
    print(f"Weights: KL={W_SOFT}  BOX={W_BOX}  ENC={W_ENC_FEAT}  BB={W_BB_FEAT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    log = []
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"{'─'*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS-1}  lr={optimizer.param_groups[1]['lr']:.2e}")
        print(f"{'─'*60}")

        stats = train_one_epoch(
            student, teacher, criterion,
            bb_adapt, enc_adapt,
            t_store, s_store,
            train_loader, optimizer, scaler,
            epoch, device
        )
        scheduler.step()

        # Validate
        print("Validating...")
        test_stats, coco_eval = evaluate(
            student, criterion, postprocessor,
            val_loader, evaluator, device)

        mAP = 0.0
        if coco_eval and hasattr(coco_eval, 'coco_eval'):
            try:
                mAP = coco_eval.coco_eval['bbox'].stats[0]
            except Exception:
                pass
        print(f"mAP@0.50:0.95 = {mAP:.4f}")

        if mAP > best['mAP']:
            best = {'mAP': mAP, 'epoch': epoch}
            torch.save({'model': student.state_dict(),
                        'epoch': epoch, 'mAP': mAP},
                       f"{OUTPUT_DIR}/best.pth")
            print(f"  ★ New best {mAP:.4f} → {OUTPUT_DIR}/best.pth")

        if (epoch + 1) % CKPT_FREQ == 0:
            save_ckpt(epoch, student, bb_adapt, enc_adapt,
                      optimizer, scaler, best, OUTPUT_DIR)

        log.append({'epoch': epoch, 'mAP': mAP, **stats})
        with open(f"{OUTPUT_DIR}/distill.log", 'w') as f:
            json.dump(log, f, indent=2)

    for h in t_handles + s_handles:
        h.remove()

    print(f"\nDone. Best mAP: {best['mAP']:.4f} at epoch {best['epoch']}")
    print(f"Best model: {OUTPUT_DIR}/best.pth")


if __name__ == '__main__':
    main()
