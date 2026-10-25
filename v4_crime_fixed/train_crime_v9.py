"""
VideoMamba V9 — Experiment V4: fixed gnConv2D backbone + clean dataset + honest evaluation
=========================================================================================
Transfer chain : Kinetics-400 -> UCF-101 (Mamba weights) -> Violence (13 classes)
Backbone       : VideoMamba-Small + HorNet gnConv2D (order=5, kernels [3,5,7,1], GhostModule)
                 loaded from v4_crime_fixed/mamba_simple_v7.py (V6 gnConv with 2 adapter fixes)
Dataset        : dataset_new_split_clean, pre-decoded by prepare_data_v9.py

What changed vs V8 (train_crime_v8.py), and why:
  BACKBONE
   1. gnConv2D now really works per frame (V6 reshape mixed channels with time steps).
   2. Backward-scan gnConv skips the cls token at the right position (it is last there).
   3. gnConv removed from frozen layers 0-15: it was zero-init + frozen, i.e. always 0.
      Same predictions, ~181M fewer parameters, far less compute.
   4. gnConv in layers 16-23 restarts from zero-init (UCF-101 gnConv weights were learned
      with the old reshape). Mamba weights still come from the UCF-101 checkpoint.
  ANTI-OVERFITTING
   5. Augmentation is applied identically to all 8 frames of a clip (V8 drew a new random
      crop/flip/colour/erase for EVERY frame, which turns real motion into jitter).
   6. Head dropout is actually applied (V1-V3 printed DROPOUT but never passed it).
   7. Stochastic depth 0.2, weight decay without decaying norms/biases, mixup always
      pairs two different clips, label smoothing, class weights.
   8. Frozen layers run in eval mode (their BatchNorm/DropPath stay frozen too).
   9. EMA (moving average) of the weights is evaluated every epoch alongside the raw weights.
  10. Model selection on a VALIDATION split (15% of train). Test is used once, at the end.
  DATA / EVALUATION
  11. Frames come from the whole clip (V8 used only the first 300 frames).
  12. Full-width views: 3 crops (left/centre/right) x several temporal clips; V8's "TTA"
      crops all sat inside the middle 56% of a 16:9 frame.
  ENGINEERING
  13. Pre-decoded frame cache + 4 loader workers (V8 spent most of each 21-min epoch decoding).
  14. Live log, resume after a crash/walltime, measured params / MACs / latency,
      confusion matrix + per-class precision/recall/F1 for all 13 classes.

Run (on HPC, through PBS):  qsub v4_crime_fixed/run_crime_v9.sh
Test only (after training): python train_crime_v9.py --test-only
"""

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ============================================================
# STEP 1: PATHS
# ============================================================
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
VLIL_ROOT       = os.path.expanduser('~/vlil_40')
EXP_DIR         = os.path.join(VLIL_ROOT, 'v4_crime_fixed')
SPLIT_DIR       = os.path.join(EXP_DIR, 'splits')
CKPT_DIR        = os.path.join(EXP_DIR, 'checkpoints')
LOG_DIR         = os.path.join(EXP_DIR, 'logs')
RESULT_DIR      = os.path.join(EXP_DIR, 'results')
BACKBONE_FILE   = os.path.join(SCRIPT_DIR, 'mamba_simple_v7.py')
PRETRAINED_PATH = os.path.join(VLIL_ROOT, 'v1_ucf101', 'checkpoints_ucf101', 'best_ucf101_v6.pth')
BEST_PATH       = os.path.join(CKPT_DIR, 'best_model_v9.pth')
LAST_PATH       = os.path.join(CKPT_DIR, 'last_v9.pth')

# ============================================================
# STEP 2: HYPERPARAMETERS
# ============================================================
NUM_FRAMES     = 8            # frames per clip (matches the K400 f8 / UCF-101 checkpoints)
IMG_SIZE       = 224
NUM_LAYERS     = 24
UNFREEZE_LAST  = 8            # train layers 16-23 (same as V1/V3)
BATCH_SIZE     = 2            # float32 on a shared 16 GB V100
ACCUM_STEPS    = 8            # effective batch = 16
EPOCHS         = 40
WARMUP_EPOCHS  = 3
PATIENCE       = 10           # early stopping on the validation split
MIN_LR_SCALE   = 0.01

LR_BACKBONE    = 1e-5         # Mamba layers 16-23 (+ final norm)
LR_GNCONV      = 1e-4         # gnConv2D in layers 16-23 (1e-4 is the NaN-safe value from V6)
LR_HEAD        = 1e-3         # new 13-class head
WEIGHT_DECAY   = 0.05         # not applied to norms, biases, A_log, D

HEAD_DROPOUT   = 0.5          # dropout before the classifier (now really applied)
DROP_PATH      = 0.2          # stochastic depth (max rate, linearly increasing over layers)
LABEL_SMOOTH   = 0.1
MIXUP_ALPHA    = 0.3
GRAD_CLIP      = 1.0
EMA_DECAY      = 0.995        # per optimizer step

# Clip-consistent augmentation (same random draw for all 8 frames of a clip)
RRC_SCALE      = (0.25, 1.0)  # RandomResizedCrop area range over the FULL frame
RRC_RATIO      = (3 / 4, 4 / 3)
FLIP_P         = 0.5
COLOR_JITTER   = (0.3, 0.3, 0.3, 0.05)   # brightness, contrast, saturation, hue
GRAYSCALE_P    = 0.1
ERASE_P        = 0.25
ERASE_SCALE    = (0.02, 0.2)

# Evaluation views (spatial crops x temporal clips, softmax-averaged)
VAL_CLIPS, VAL_CROPS   = 2, 3
TEST_CLIPS, TEST_CROPS = 4, 3
EVAL_CHUNK     = 6            # views per forward pass

NUM_WORKERS    = 4
SEED           = 42
USE_AMP        = False        # float16 overflows in order-5 gnConv (V6 NaN issue); V100 has no bf16

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

V1_TEST_ACC, V3_TEST_ACC = 69.64, 71.43   # previous runs (best epoch chosen ON the test set)


def log(msg=''):
    print(msg, flush=True)


def config_dict():
    keys = ['NUM_FRAMES', 'IMG_SIZE', 'UNFREEZE_LAST', 'BATCH_SIZE', 'ACCUM_STEPS', 'EPOCHS',
            'WARMUP_EPOCHS', 'PATIENCE', 'LR_BACKBONE', 'LR_GNCONV', 'LR_HEAD', 'WEIGHT_DECAY',
            'HEAD_DROPOUT', 'DROP_PATH', 'LABEL_SMOOTH', 'MIXUP_ALPHA', 'GRAD_CLIP', 'EMA_DECAY',
            'RRC_SCALE', 'RRC_RATIO', 'FLIP_P', 'COLOR_JITTER', 'GRAYSCALE_P', 'ERASE_P',
            'ERASE_SCALE', 'VAL_CLIPS', 'VAL_CROPS', 'TEST_CLIPS', 'TEST_CROPS', 'SEED',
            'PRETRAINED_PATH']
    g = globals()
    return {k: g[k] for k in keys}


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                      # older PyTorch without weights_only
        return torch.load(path, map_location='cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# STEP 3: DATA (reads the frame cache written by prepare_data_v9.py)
# ============================================================
def read_split(name):
    path = os.path.join(SPLIT_DIR, f'{name}.csv')
    if not os.path.exists(path):
        sys.exit(f'  ERROR: {path} not found. Run prepare_data_v9.py (run_prepare_v9.sh) first.')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['label'] = int(r['label'])
    return rows


def read_classes():
    with open(os.path.join(SPLIT_DIR, 'classes.txt')) as f:
        return [line.rstrip('\n').split('\t', 1)[1] for line in f if line.strip()]


def _open_cache(path):
    with np.load(path) as d:
        return d['jpg'], d['offsets']


def _decode(jpg, off, i):
    return Image.open(io.BytesIO(jpg[off[i]:off[i + 1]].tobytes())).convert('RGB')


def clip_offsets(n_clips, seg):
    """Offsets inside each temporal segment for n evenly spread clips."""
    return [min(seg - 1, int((k + 0.5) * seg / n_clips)) for k in range(n_clips)]


def crop_positions(length, n_crops):
    if n_crops == 1 or length <= IMG_SIZE:
        return [(length - IMG_SIZE) // 2] * n_crops
    return [round(k * (length - IMG_SIZE) / (n_crops - 1)) for k in range(n_crops)]


class ClipDataset(Dataset):
    """
    train=True : 8 frames with segment-based temporal jitter over the whole clip,
                 ONE random augmentation applied identically to all 8 frames.
    train=False: n_clips x n_crops views (short side resized to 224, crops spread over
                 the full width), returned as a (views, C, T, H, W) tensor.
    """

    def __init__(self, rows, train, n_clips=1, n_crops=1):
        self.rows = rows
        self.train = train
        self.n_clips = n_clips
        self.n_crops = n_crops
        self.cj = T.ColorJitter(*COLOR_JITTER)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        jpg, off = _open_cache(r['cache_path'])
        n = len(off) - 1
        seg = max(1, n // NUM_FRAMES)

        if self.train:
            frame_ids = [min(s * seg + random.randrange(seg), n - 1) for s in range(NUM_FRAMES)]
            frames = [_decode(jpg, off, k) for k in frame_ids]
            return self._train_clip(frames), r['label']

        resized = {}
        views = []
        for o in clip_offsets(self.n_clips, seg):
            frames = []
            for s in range(NUM_FRAMES):
                k = min(s * seg + o, n - 1)
                if k not in resized:
                    resized[k] = TF.resize(_decode(jpg, off, k), IMG_SIZE)
                frames.append(resized[k])
            w, h = frames[0].size
            for c in range(self.n_crops):
                if w >= h:
                    top, left = (h - IMG_SIZE) // 2, crop_positions(w, self.n_crops)[c]
                else:
                    top, left = crop_positions(h, self.n_crops)[c], (w - IMG_SIZE) // 2
                x = torch.stack([TF.to_tensor(TF.crop(f, top, left, IMG_SIZE, IMG_SIZE))
                                 for f in frames], dim=1)
                views.append((x - MEAN) / STD)
        return torch.stack(views), r['label']

    def _train_clip(self, frames):
        i, j, h, w = T.RandomResizedCrop.get_params(frames[0], scale=list(RRC_SCALE),
                                                    ratio=list(RRC_RATIO))
        flip = random.random() < FLIP_P
        fn_idx, bri, con, sat, hue = T.ColorJitter.get_params(
            self.cj.brightness, self.cj.contrast, self.cj.saturation, self.cj.hue)
        gray = random.random() < GRAYSCALE_P

        out = []
        for f in frames:
            f = TF.resized_crop(f, i, j, h, w, [IMG_SIZE, IMG_SIZE])
            if flip:
                f = TF.hflip(f)
            for fid in fn_idx.tolist():
                if fid == 0 and bri is not None:
                    f = TF.adjust_brightness(f, bri)
                elif fid == 1 and con is not None:
                    f = TF.adjust_contrast(f, con)
                elif fid == 2 and sat is not None:
                    f = TF.adjust_saturation(f, sat)
                elif fid == 3 and hue is not None:
                    f = TF.adjust_hue(f, hue)
            if gray:
                f = TF.rgb_to_grayscale(f, num_output_channels=3)
            out.append(TF.to_tensor(f))
        x = (torch.stack(out, dim=1) - MEAN) / STD           # (C, T, H, W)

        if random.random() < ERASE_P:                       # same erased box in every frame
            H, W = x.shape[-2:]
            for _ in range(10):
                area = H * W * random.uniform(*ERASE_SCALE)
                ar = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
                eh, ew = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
                if 0 < eh < H and 0 < ew < W:
                    top, left = random.randint(0, H - eh), random.randint(0, W - ew)
                    x[:, :, top:top + eh, left:left + ew] = 0.0
                    break
        return x


def _worker_init(_):
    torch.set_num_threads(1)


# ============================================================
# STEP 4: MODEL (VideoMamba-Small + fixed gnConv2D from mamba_simple_v7.py)
# ============================================================
def load_backbone_module():
    sys.path.insert(0, os.path.join(VLIL_ROOT, 'VideoMamba', 'mamba'))
    sys.path.insert(0, os.path.join(VLIL_ROOT, 'VideoMamba', 'videomamba', 'video_sm'))
    import models.videomamba as vm
    spec = importlib.util.spec_from_file_location('mamba_simple_v7', BACKBONE_FILE)
    ms7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms7)
    vm.Mamba = ms7.Mamba          # create_block() looks up "Mamba" at build time
    return vm, ms7


@torch.no_grad()
def self_test_gnconv(ms7, device):
    """Proves FIX 1 and FIX 2 on the GPU before training starts."""
    g = ms7.gnConv(dim=768, order=5).to(device).eval()
    nn.init.normal_(g.proj_out.weight, std=0.02)   # non-zero output for the test only
    hw = 14 * 14
    L = 1 + NUM_FRAMES * hw
    x = torch.randn(1, 768, L, device=device)
    y1 = g(x, cls_first=True)
    x2 = x.clone()
    x2[:, :, 1 + 3 * hw:1 + 4 * hw] += 1.0           # change frame 3 only
    y2 = g(x2, cls_first=True)
    d = (y2 - y1).abs().amax(dim=1)[0]
    per_frame = [d[1 + t * hw:1 + (t + 1) * hw].max().item() for t in range(NUM_FRAMES)]
    others = max(v for t, v in enumerate(per_frame) if t != 3)
    isolated = per_frame[3] > 1e-4 and others <= 1e-6 + 1e-4 * per_frame[3]
    cls_fwd = y1[:, :, 0].abs().max().item() == 0.0
    cls_bwd = g(x.flip([-1]), cls_first=False)[:, :, -1].abs().max().item() == 0.0
    log(f'  [CHECK] gnConv per-frame isolation: {"OK" if isolated else "FAILED"} '
        f'(change in frame 3 -> max output change: frame 3 = {per_frame[3]:.3g}, other frames = {others:.3g})')
    log(f'  [CHECK] cls token untouched: forward scan {"OK" if cls_fwd else "FAILED"}, '
        f'backward scan {"OK" if cls_bwd else "FAILED"}')
    del g
    if not (isolated and cls_fwd and cls_bwd):
        sys.exit('  ERROR: backbone self-test failed — wrong mamba_simple_v7.py?')


def build_model(vm, ms7, num_classes, device):
    freeze_until = NUM_LAYERS - UNFREEZE_LAST
    model = vm.videomamba_small(
        pretrained=False, num_classes=num_classes, num_frames=NUM_FRAMES,
        fc_drop_rate=HEAD_DROPOUT, drop_path_rate=DROP_PATH,
        ssm_cfg={'gnconv_from_layer': freeze_until},
    )
    assert isinstance(model.layers[0].mixer, ms7.Mamba), 'model is not using mamba_simple_v7'
    assert model.layers[0].mixer.gnconv is None and model.layers[-1].mixer.gnconv is not None

    # ---- load UCF-101 checkpoint: all Mamba / embedding / norm weights; NOT gnConv, NOT head ----
    log(f'\n  Loading UCF-101 checkpoint: {PRETRAINED_PATH}')
    if not os.path.exists(PRETRAINED_PATH):
        sys.exit('  ERROR: UCF-101 checkpoint not found — refusing to train from random weights.')
    ckpt = torch_load(PRETRAINED_PATH)
    for key in ('model', 'state_dict', 'module'):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
    ckpt = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}
    dropped_gn = [k for k in ckpt if '.gnconv' in k]
    dropped_head = [k for k in ckpt if k.startswith('head.')]
    state = {k: v for k, v in ckpt.items() if '.gnconv' not in k and not k.startswith('head.')}
    msg = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in msg.missing_keys if '.gnconv' not in k and not k.startswith('head.')]
    log(f'  Loaded {len(state)} tensors | skipped: {len(dropped_gn)} gnConv (old reshape) + '
        f'{len(dropped_head)} head (101 classes)')
    log(f'  Missing (expected: new gnConv layers {freeze_until}-{NUM_LAYERS - 1} + new head): '
        f'{len(msg.missing_keys)} | unexpected: {len(msg.unexpected_keys)}')
    if bad_missing or msg.unexpected_keys:
        log(f'  Missing backbone keys: {bad_missing[:10]}')
        log(f'  Unexpected keys: {msg.unexpected_keys[:10]}')
        sys.exit('  ERROR: checkpoint does not match the backbone.')
    del ckpt, state

    # ---- freeze layers 0-15, train layers 16-23 + final norm + head ----
    for p in model.parameters():
        p.requires_grad = False
    for i in range(freeze_until, NUM_LAYERS):
        for p in model.layers[i].parameters():
            p.requires_grad = True
    for p in model.norm_f.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return model.to(device), freeze_until


def set_train_mode(model, freeze_until):
    model.train()
    model.patch_embed.eval()
    for i in range(freeze_until):
        model.layers[i].eval()       # frozen layers: no DropPath / BN updates


def print_param_table(model, freeze_until):
    names = ['Patch Embedding', 'Mamba SSM Layers', 'gnConv2D Modules', 'Classification Head',
             'Norm Layers']

    def category(n):
        if n.startswith('patch_embed') or n in ('cls_token', 'pos_embed', 'temporal_pos_embedding'):
            return 0
        if n.startswith('head.'):
            return 3
        if '.gnconv' in n:
            return 2
        if 'norm' in n:
            return 4
        return 1

    tot, trn = [0] * 5, [0] * 5
    for n, p in model.named_parameters():
        c = category(n)
        tot[c] += p.numel()
        if p.requires_grad:
            trn[c] += p.numel()
    one_gn = sum(p.numel() for p in model.layers[freeze_until].mixer.gnconv.parameters())
    removed = one_gn * 2 * freeze_until
    T_, R_ = sum(tot), sum(trn)

    log('')
    log(f"  {'=' * 66}")
    log('  MODEL PARAMETER BREAKDOWN (V4 / train_crime_v9)')
    log(f"  {'-' * 66}")
    log(f"  {'Component':<26s}{'Total':>13s}{'Trainable':>13s}{'Frozen':>13s}")
    log(f"  {'-' * 66}")
    for i, nm in enumerate(names):
        label = nm + (f' (L{freeze_until}-{NUM_LAYERS - 1})' if i == 2 else '')
        log(f'  {label:<26s}{tot[i]:>13,}{trn[i]:>13,}{tot[i] - trn[i]:>13,}')
    log(f"  {'-' * 66}")
    log(f"  {'TOTAL (active)':<26s}{T_:>13,}{R_:>13,}{T_ - R_:>13,}")
    log(f"  {'':26s}{'(100%)':>13s}{f'({100 * R_ / T_:.1f}%)':>13s}{f'({100 * (T_ - R_) / T_:.1f}%)':>13s}")
    log(f"  {'-' * 66}")
    log(f'  Removed: gnConv2D in frozen layers 0-{freeze_until - 1}: {removed:,} params')
    log(f'           (zero-init + frozen in V1-V3, so they always output exactly 0)')
    log(f'  V1-V3 total incl. those inactive modules: {T_ + removed:,}')
    log(f'  One gnConv2D module: {one_gn:,} params (2 per layer: forward + backward scan)')
    log(f"  {'=' * 66}")
    return {'total_active': T_, 'trainable': R_, 'frozen': T_ - R_, 'removed_inactive_gnconv': removed,
            'v1_v3_equivalent_total': T_ + removed, 'per_component_total': dict(zip(names, tot)),
            'per_component_trainable': dict(zip(names, trn))}


@torch.no_grad()
def measure_complexity(model, freeze_until, device):
    """MACs per video measured with forward hooks (+ analytic Mamba matmuls), and GPU latency."""
    model.eval()
    macs = {}

    def make_hook(name):
        def hook(m, inp, out):
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                v = out.numel() * (m.in_channels // m.groups) * int(np.prod(m.kernel_size))
            else:
                v = out.numel() * m.in_features
            if name.startswith('patch_embed'):
                k = 'Patch embedding (Conv3d)'
            elif name.startswith('head'):
                k = 'Classification head'
            elif '.gnconv' in name:
                k = f'gnConv2D (layers {freeze_until}-{NUM_LAYERS - 1})'
            elif 'conv1d' in name:
                k = 'Mamba Conv1d'
            else:
                k = 'Mamba linear projections'
            macs[k] = macs.get(k, 0) + v
        return hook

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in model.named_modules()
             if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))]
    x = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE, device=device)
    model(x)
    for h in hooks:
        h.remove()

    # in_proj / dt_proj / out_proj are applied via weight matmuls (not module calls) -> add analytically
    mx = model.layers[0].mixer
    L = 1 + NUM_FRAMES * (IMG_SIZE // 16) ** 2
    per_layer = L * (mx.d_model * 2 * mx.d_inner + mx.d_inner * mx.d_model + 2 * mx.dt_rank * mx.d_inner)
    macs['Mamba linear projections'] = macs.get('Mamba linear projections', 0) + NUM_LAYERS * per_layer
    macs['Selective scan (estimate)'] = int(NUM_LAYERS * 2 * 4.5 * L * mx.d_inner * mx.d_state)

    gn_key = f'gnConv2D (layers {freeze_until}-{NUM_LAYERS - 1})'
    removed = macs.get(gn_key, 0) / max(1, NUM_LAYERS - freeze_until) * freeze_until
    total = sum(macs.values())
    log('')
    log(f"  {'=' * 66}")
    log(f'  COMPUTE PER VIDEO ({NUM_FRAMES} x {IMG_SIZE} x {IMG_SIZE}), measured')
    log(f"  {'-' * 66}")
    for k in ['Patch embedding (Conv3d)', 'Mamba linear projections', 'Mamba Conv1d',
              'Selective scan (estimate)', gn_key, 'Classification head']:
        log(f'  {k:<40s}{macs.get(k, 0) / 1e9:>10.2f} GMACs')
    log(f"  {'-' * 66}")
    log(f"  {'TOTAL':<40s}{total / 1e9:>10.2f} GMACs  (= {2 * total / 1e9:.1f} GFLOPs)")
    log(f'  Removed inactive gnConv (layers 0-{freeze_until - 1}) would have added '
        f'{removed / 1e9:.1f} GMACs for an output of exactly 0')
    log("  (V1-V3 banner said '~56 GFLOPs'; that number was typed in, not measured)")

    latency_ms = None
    if device.type == 'cuda':
        for _ in range(2):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        latency_ms = (time.time() - t0) / 5 * 1000
        log(f'  Inference latency (batch 1, fp32, this V100, shared GPU): {latency_ms:.0f} ms / video')
    log(f"  {'=' * 66}")
    return {'macs_by_part': {k: int(v) for k, v in macs.items()}, 'total_gmacs': total / 1e9,
            'total_gflops': 2 * total / 1e9, 'removed_gnconv_gmacs': removed / 1e9,
            'latency_ms_v100_fp32_b1': latency_ms}


# ============================================================
# STEP 5: OPTIMIZER, SCHEDULE, EMA
# ============================================================
def build_optimizer(model):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = 'head' if n.startswith('head.') else ('gnconv' if '.gnconv' in n else 'backbone')
        no_decay = p.ndim <= 1 or n.endswith('.bias') or getattr(p, '_no_weight_decay', False)
        groups.setdefault((part, no_decay), []).append(p)
    base = {'backbone': LR_BACKBONE, 'gnconv': LR_GNCONV, 'head': LR_HEAD}
    param_groups = []
    for part in ('backbone', 'gnconv', 'head'):
        for nd in (False, True):
            ps = groups.get((part, nd))
            if ps:
                param_groups.append({'params': ps, 'lr': base[part], 'base_lr': base[part],
                                     'weight_decay': 0.0 if nd else WEIGHT_DECAY,
                                     'name': part + ('_no_decay' if nd else '')})
    return torch.optim.AdamW(param_groups, lr=LR_BACKBONE, betas=(0.9, 0.999))


def lr_scale(epoch_idx):
    if epoch_idx < WARMUP_EPOCHS:
        return (epoch_idx + 1) / WARMUP_EPOCHS
    progress = (epoch_idx - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return max(MIN_LR_SCALE, 0.5 * (1 + math.cos(math.pi * progress)))


def set_lr(optimizer, epoch_idx):
    s = lr_scale(epoch_idx)
    for g in optimizer.param_groups:
        g['lr'] = g['base_lr'] * s


def trainable_state(model):
    st = {n: p for n, p in model.named_parameters() if p.requires_grad}
    st.update({n: b for n, b in model.named_buffers() if b.dtype.is_floating_point})
    return st


class ModelEMA:
    """Exponential moving average of the trainable weights (and BN statistics)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.shadow = {n: t.detach().clone() for n, t in trainable_state(model).items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        cur = trainable_state(model)
        for n, s in self.shadow.items():
            s.mul_(d).add_(cur[n].detach(), alpha=1 - d)

    @contextlib.contextmanager
    def applied(self, model):
        cur = trainable_state(model)
        backup = {n: t.detach().clone() for n, t in cur.items()}
        with torch.no_grad():
            for n, t in cur.items():
                t.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, t in cur.items():
                    t.copy_(backup[n])

    def state_dict(self):
        return {'updates': self.updates, 'shadow': {n: s.cpu() for n, s in self.shadow.items()}}

    def load_state_dict(self, sd):
        self.updates = sd['updates']
        for n, s in sd['shadow'].items():
            self.shadow[n].copy_(s)


# ============================================================
# STEP 6: TRAIN / EVALUATE
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, ema, freeze_until, device, epoch, first_check):
    set_train_mode(model, freeze_until)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    loss_sum, correct, seen, skipped, bad_steps = 0.0, 0.0, 0, 0, 0
    n_batches = len(loader)
    report_every = max(1, n_batches // 4)
    t0 = time.time()

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if MIXUP_ALPHA > 0 and x.size(0) > 1:
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            perm = torch.roll(torch.arange(x.size(0), device=device), 1)   # always a different clip
            x = lam * x + (1 - lam) * x[perm]
            y_b = y[perm]
        else:
            lam, y_b = 1.0, y

        logits = model(x)
        loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y_b)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        (loss / ACCUM_STEPS).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == n_batches:
            gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            if torch.isfinite(gn):
                optimizer.step()
                ema.update(model)
            else:
                bad_steps += 1
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (lam * (pred == y).float() + (1 - lam) * (pred == y_b).float()).sum().item()
        loss_sum += loss.item() * x.size(0)
        seen += x.size(0)

        if first_check and step == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0
            log(f'  [CHECK] first training step OK (loss {loss.item():.3f}, GPU memory {mem:.1f} GB)')
        if (step + 1) % report_every == 0 or (step + 1) == n_batches:
            log(f'    epoch {epoch} [train] {100 * (step + 1) / n_batches:3.0f}% | '
                f'loss {loss_sum / max(seen, 1):.4f} | acc {100 * correct / max(seen, 1):.1f}% | '
                f'{time.time() - t0:.0f}s')

    if skipped or bad_steps:
        log(f'  [WARN] skipped {skipped} non-finite batches, {bad_steps} non-finite updates')
    return loss_sum / max(seen, 1), 100 * correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, single_view_idx=None):
    model.eval()
    probs_all, labels_all, single_all = [], [], []
    for x, y in loader:
        B, V = x.shape[:2]
        flat = x.reshape(B * V, *x.shape[2:])
        outs = [model(flat[s:s + EVAL_CHUNK].to(device, non_blocking=True)).float().softmax(-1).cpu()
                for s in range(0, B * V, EVAL_CHUNK)]
        p = torch.cat(outs).view(B, V, -1)
        probs_all.append(p.mean(1))
        labels_all.append(y)
        if single_view_idx is not None:
            single_all.append(p[:, single_view_idx])
    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all)
    preds = probs.argmax(1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t_, p_ in zip(labels.tolist(), preds.tolist()):
        cm[t_, p_] += 1
    res = {'acc': 100 * (preds == labels).float().mean().item(),
           'loss': F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels).item(),
           'probs': probs, 'labels': labels, 'preds': preds, 'cm': cm}
    if single_view_idx is not None:
        sp = torch.cat(single_all).argmax(1)
        res['single_acc'] = 100 * (sp == labels).float().mean().item()
    return res


def per_class_line(cm, classes):
    parts = []
    for i, c in enumerate(classes):
        n = cm[i].sum().item()
        parts.append(f'{c[:10]} {100 * cm[i, i].item() / n:.0f}%' if n else f'{c[:10]} -')
    return ' | '.join(parts)


def class_metrics(cm, classes):
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i].item()
        support = cm[i].sum().item()
        predicted = cm[:, i].sum().item()
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({'class': c, 'support': support, 'correct': tp,
                     'precision': prec, 'recall': rec, 'f1': f1})
    return rows


def print_confusion(cm, classes):
    short = [c[:5] for c in classes]
    log('  Confusion matrix (rows = true class, columns = predicted):')
    log('  ' + ' ' * 15 + ''.join(f'{s:>6s}' for s in short))
    for i, c in enumerate(classes):
        log(f'  {c[:14]:<15s}' + ''.join(f'{cm[i, j].item():>6d}' for j in range(len(classes))))


def save_checkpoint(path, obj):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ============================================================
# STEP 7: FINAL TEST (runs once, on the model chosen by validation)
# ============================================================
def final_test(model, classes, test_rows, device, best_info, params_info, compute_info):
    num_classes = len(classes)
    loader = DataLoader(ClipDataset(test_rows, train=False, n_clips=TEST_CLIPS, n_crops=TEST_CROPS),
                        batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                        worker_init_fn=_worker_init)
    single_idx = (TEST_CLIPS // 2) * TEST_CROPS + TEST_CROPS // 2
    log(f'\n  Evaluating on the TEST set ({len(test_rows)} videos, {TEST_CLIPS} clips x '
        f'{TEST_CROPS} crops = {TEST_CLIPS * TEST_CROPS} views per video)...')
    res = evaluate(model, loader, device, num_classes, single_view_idx=single_idx)
    metrics = class_metrics(res['cm'], classes)
    macro_f1 = 100 * float(np.mean([m['f1'] for m in metrics]))

    log('')
    log('=' * 70)
    log('FINAL TEST RESULTS — V4 (train_crime_v9)')
    log('=' * 70)
    log(f"  Model used          : epoch {best_info['epoch']} ({best_info['weights']} weights), "
        f"chosen on VALIDATION (val acc {best_info['val_acc']:.2f}%)")
    log(f"  TEST ACCURACY       : {res['acc']:.2f}%   ({TEST_CLIPS}x{TEST_CROPS} views, main number)")
    log(f"  Test acc, 1 view    : {res['single_acc']:.2f}%   (centre clip, centre crop)")
    log(f'  Test macro-F1       : {macro_f1:.2f}%')
    log(f"  Correct             : {int(res['cm'].diag().sum())}/{len(test_rows)}")
    log('')
    log(f"  {'Class':<16s}{'Correct':>10s}{'Recall':>9s}{'Precision':>11s}{'F1':>8s}")
    for m in sorted(metrics, key=lambda r: -r['recall']):
        log(f"  {m['class']:<16s}{m['correct']:>5d}/{m['support']:<4d}{100 * m['recall']:>8.1f}%"
            f"{100 * m['precision']:>10.1f}%{100 * m['f1']:>7.1f}%")
    log('')
    print_confusion(res['cm'], classes)
    log('')
    log('  COMPARISON (note: V1/V3 picked their best epoch ON the test set, so they are optimistic):')
    log(f'    V1 (K400 -> Violence)            : {V1_TEST_ACC:.2f}%')
    log(f'    V3 (K400 -> UCF-101 -> Violence) : {V3_TEST_ACC:.2f}%')
    log(f"    V4 (fixed backbone, this run)    : {res['acc']:.2f}%  (single view {res['single_acc']:.2f}%)")
    log('=' * 70)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, 'confusion_matrix_v9.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['true \\ predicted'] + classes)
        for i, c in enumerate(classes):
            w.writerow([c] + res['cm'][i].tolist())
    with open(os.path.join(RESULT_DIR, 'per_class_metrics_v9.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)
    with open(os.path.join(RESULT_DIR, 'test_predictions_v9.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'true', 'predicted', 'confidence', 'correct'])
        for r, t_, p_, pr in zip(test_rows, res['labels'].tolist(), res['preds'].tolist(), res['probs']):
            w.writerow([r['video_path'], classes[t_], classes[p_], f'{pr.max().item():.3f}', int(t_ == p_)])
    summary = {'test_acc_multiview': res['acc'], 'test_acc_single_view': res['single_acc'],
               'test_macro_f1': macro_f1, 'test_videos': len(test_rows), 'best': best_info,
               'params': params_info, 'compute': compute_info, 'per_class': metrics,
               'config': config_dict()}
    with open(os.path.join(RESULT_DIR, 'test_results_v9.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f'  Saved: {RESULT_DIR}/ (test_results_v9.json, confusion_matrix_v9.csv, '
        f'per_class_metrics_v9.csv, test_predictions_v9.csv)')


# ============================================================
# STEP 8: MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-only', action='store_true', help='skip training, test best_model_v9.pth')
    ap.add_argument('--no-resume', action='store_true', help='ignore last_v9.pth and start fresh')
    args = ap.parse_args()

    for d in (CKPT_DIR, LOG_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)
    seed_everything(SEED)

    log('=' * 70)
    log('VideoMamba V9 — Experiment V4: fixed gnConv2D + clean data + validation-based selection')
    log('=' * 70)
    log(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    if not torch.cuda.is_available():
        sys.exit('  ERROR: no GPU visible. Mamba selective scan needs CUDA — submit through PBS on a GPU node.')
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    free, total = torch.cuda.mem_get_info()
    log(f'  GPU: {torch.cuda.get_device_name(0)} | free {free / 1e9:.1f} / {total / 1e9:.1f} GB '
        f'| PyTorch {torch.__version__}')
    if free < 6e9:
        log('  [WARN] less than 6 GB free on this GPU (another user?). Training may run out of memory.')

    log('\n  ARCHITECTURE:')
    log('    Backbone:        VideoMamba-Small (24 bidirectional Mamba layers, d=384)')
    log('    Spatial module:  HorNet gnConv2D (order=5, kernels [3,5,7,1], GhostModule, zero-init)')
    log(f'                     in layers {NUM_LAYERS - UNFREEZE_LAST}-{NUM_LAYERS - 1}, applied per frame on the 14x14 grid')
    log(f'    Backbone file:   {BACKBONE_FILE}')
    log('  TRANSFER: Kinetics-400 -> UCF-101 (Mamba weights) -> Violence; gnConv starts at zero')
    log('\n  HYPERPARAMETERS:')
    log(f'    Frames/clip {NUM_FRAMES} | batch {BATCH_SIZE} x accum {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS} '
        f'| epochs {EPOCHS} (warmup {WARMUP_EPOCHS}, early stop patience {PATIENCE})')
    log(f'    LR backbone {LR_BACKBONE} | LR gnConv {LR_GNCONV} | LR head {LR_HEAD} | cosine schedule')
    log(f'    AdamW weight decay {WEIGHT_DECAY} (no decay on norms/biases/A_log/D) | grad clip {GRAD_CLIP}')
    log('  REGULARIZATION (anti-overfitting):')
    log(f'    Head dropout {HEAD_DROPOUT} (applied) | stochastic depth {DROP_PATH} | label smoothing {LABEL_SMOOTH}')
    log(f'    Mixup alpha {MIXUP_ALPHA} | EMA decay {EMA_DECAY} | frozen layers 0-{NUM_LAYERS - UNFREEZE_LAST - 1} in eval mode')
    log('  DATA AUGMENTATION (same random draw for all frames of a clip):')
    log(f'    RandomResizedCrop scale {RRC_SCALE} ratio (0.75, 1.33) over the full frame | h-flip p={FLIP_P}')
    log(f'    Colour jitter {COLOR_JITTER} | grayscale p={GRAYSCALE_P} | erasing p={ERASE_P} {ERASE_SCALE}')
    log('    Temporal: 8 segments over the whole clip, random frame inside each segment')
    log('  EVALUATION:')
    log(f'    Validation (model selection): {VAL_CLIPS} clips x {VAL_CROPS} full-width crops, raw + EMA weights')
    log(f'    Test (once, at the end):      {TEST_CLIPS} clips x {TEST_CROPS} full-width crops')
    log(f'    Precision: float32 (AMP={USE_AMP})')
    log('=' * 70)

    # ---------------- data ----------------
    classes = read_classes()
    num_classes = len(classes)
    train_rows, val_rows, test_rows = read_split('train'), read_split('val'), read_split('test')
    missing = [r['cache_path'] for r in train_rows + val_rows + test_rows if not os.path.exists(r['cache_path'])]
    if missing:
        sys.exit(f'  ERROR: {len(missing)} cached clips missing (e.g. {missing[0]}). Re-run prepare_data_v9.py.')
    log(f'\n  DATASET (from {SPLIT_DIR}):')
    log(f"  {'Class':<18s}{'Train':>7s}{'Val':>7s}{'Test':>7s}")
    counts = {}
    for name, rows in (('train', train_rows), ('val', val_rows), ('test', test_rows)):
        counts[name] = np.bincount(np.array([r['label'] for r in rows], dtype=np.int64),
                                   minlength=num_classes)
    for i, c in enumerate(classes):
        log(f"  {c:<18s}{counts['train'][i]:>7d}{counts['val'][i]:>7d}{counts['test'][i]:>7d}")
    log(f"  {'TOTAL':<18s}{len(train_rows):>7d}{len(val_rows):>7d}{len(test_rows):>7d}")

    tc = counts['train'].astype(float)
    class_weights = torch.tensor(np.clip(tc.sum() / (num_classes * np.maximum(tc, 1)), 0.5, 10.0),
                                 dtype=torch.float32)

    # ---------------- model ----------------
    vm, ms7 = load_backbone_module()
    self_test_gnconv(ms7, device)
    model, freeze_until = build_model(vm, ms7, num_classes, device)
    params_info = print_param_table(model, freeze_until)
    compute_info = measure_complexity(model, freeze_until, device)
    torch.cuda.empty_cache()

    if args.test_only:
        if not os.path.exists(BEST_PATH):
            sys.exit(f'  ERROR: {BEST_PATH} not found.')
        ck = torch_load(BEST_PATH)
        model.load_state_dict(ck['state_dict'])
        final_test(model, classes, test_rows, device, ck['best'], params_info, compute_info)
        return

    train_loader = DataLoader(ClipDataset(train_rows, train=True), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0, worker_init_fn=_worker_init)
    val_loader = DataLoader(ClipDataset(val_rows, train=False, n_clips=VAL_CLIPS, n_crops=VAL_CROPS),
                            batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=NUM_WORKERS > 0, worker_init_fn=_worker_init)

    optimizer = build_optimizer(model)
    log('\n  Optimizer groups:')
    for g in optimizer.param_groups:
        log(f"    {g['name']:<18s} {sum(p.numel() for p in g['params']):>12,} params | "
            f"lr {g['base_lr']:.0e} | wd {g['weight_decay']}")
    ema = ModelEMA(model, EMA_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=LABEL_SMOOTH)

    best = {'epoch': 0, 'weights': None, 'val_acc': -1.0, 'val_loss': float('inf'), 'train_acc': None}
    history, patience, start_epoch = [], 0, 1
    if os.path.exists(LAST_PATH) and not args.no_resume:
        ck = torch_load(LAST_PATH)
        cur = trainable_state(model)
        with torch.no_grad():
            for n, t in ck['trainable'].items():
                cur[n].copy_(t)
        optimizer.load_state_dict(ck['optimizer'])
        ema.load_state_dict(ck['ema'])
        best, history, patience = ck['best'], ck['history'], ck['patience']
        start_epoch = ck['epoch'] + 1
        random.setstate(ck['rng']['py'])
        np.random.set_state(ck['rng']['np'])
        torch.set_rng_state(ck['rng']['torch'])
        log(f"\n  RESUMED from {LAST_PATH}: continuing at epoch {start_epoch} "
            f"(best so far {best['val_acc']:.2f}% at epoch {best['epoch']})")
        del ck

    hist_path = os.path.join(LOG_DIR, 'history_v9.csv')
    hist_fields = ['epoch', 'train_loss', 'train_acc', 'val_acc_raw', 'val_loss_raw', 'val_acc_ema',
                   'val_loss_ema', 'gap', 'lr_backbone', 'lr_gnconv', 'lr_head', 'time_s', 'best_val_acc']

    log(f'\n  Training: {len(train_loader)} batches/epoch, validation on {len(val_rows)} videos')
    stopped_early = False
    for epoch in range(start_epoch, EPOCHS + 1):
        if patience >= PATIENCE:
            stopped_early = True
            break
        t0 = time.time()
        set_lr(optimizer, epoch - 1)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, ema, freeze_until,
                                          device, epoch, first_check=(epoch == start_epoch))
        raw = evaluate(model, val_loader, device, num_classes)
        with ema.applied(model):
            ema_res = evaluate(model, val_loader, device, num_classes)

        cand = max([('raw', raw), ('ema', ema_res)], key=lambda kv: (kv[1]['acc'], -kv[1]['loss']))
        improved = (cand[1]['acc'], -cand[1]['loss']) > (best['val_acc'], -best['val_loss'])
        if improved:
            best = {'epoch': epoch, 'weights': cand[0], 'val_acc': cand[1]['acc'],
                    'val_loss': cand[1]['loss'], 'train_acc': tr_acc}
            ctx = ema.applied(model) if cand[0] == 'ema' else contextlib.nullcontext()
            with ctx:
                save_checkpoint(BEST_PATH, {'state_dict': model.state_dict(), 'best': best,
                                            'classes': classes, 'config': config_dict()})
            patience = 0
        else:
            patience += 1

        lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
        elapsed = time.time() - t0
        gap = tr_acc - cand[1]['acc']
        log(f"  Epoch {epoch:2d}/{EPOCHS}: loss {tr_loss:.4f} | train {tr_acc:.1f}% | "
            f"val raw {raw['acc']:.1f}% (loss {raw['loss']:.3f}) | val EMA {ema_res['acc']:.1f}% "
            f"(loss {ema_res['loss']:.3f}) | gap {gap:.1f}% | lr {lrs.get('backbone', 0):.1e}/"
            f"{lrs.get('gnconv', 0):.1e}/{lrs.get('head', 0):.1e} | {elapsed:.0f}s")
        log(f"    val per class ({cand[0]}): {per_class_line(cand[1]['cm'], classes)}")
        if improved:
            log(f"    >>> NEW BEST: {best['val_acc']:.2f}% ({best['weights']} weights) -> {BEST_PATH}")
        else:
            log(f"    (no improvement {patience}/{PATIENCE}; best {best['val_acc']:.2f}% at epoch {best['epoch']})")
        if epoch == start_epoch:
            log(f'    peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')

        history.append({'epoch': epoch, 'train_loss': round(tr_loss, 4), 'train_acc': round(tr_acc, 2),
                        'val_acc_raw': round(raw['acc'], 2), 'val_loss_raw': round(raw['loss'], 4),
                        'val_acc_ema': round(ema_res['acc'], 2), 'val_loss_ema': round(ema_res['loss'], 4),
                        'gap': round(gap, 2), 'lr_backbone': lrs.get('backbone'),
                        'lr_gnconv': lrs.get('gnconv'), 'lr_head': lrs.get('head'),
                        'time_s': round(elapsed), 'best_val_acc': round(best['val_acc'], 2)})
        with open(hist_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hist_fields)
            w.writeheader()
            w.writerows(history)
        save_checkpoint(LAST_PATH, {
            'epoch': epoch, 'trainable': {n: t.detach().cpu() for n, t in trainable_state(model).items()},
            'optimizer': optimizer.state_dict(), 'ema': ema.state_dict(), 'best': best,
            'history': history, 'patience': patience,
            'rng': {'py': random.getstate(), 'np': np.random.get_state(), 'torch': torch.get_rng_state()}})

    if stopped_early or patience >= PATIENCE:
        log(f'\n  EARLY STOPPING: no validation improvement for {PATIENCE} epochs.')
    log('')
    log('=' * 70)
    log('TRAINING COMPLETE')
    log(f"  Best validation accuracy: {best['val_acc']:.2f}% at epoch {best['epoch']} "
        f"({best['weights']} weights, train acc then {best['train_acc']:.1f}%)")
    log(f'  History: {hist_path}')
    log('=' * 70)

    ck = torch_load(BEST_PATH)
    model.load_state_dict(ck['state_dict'])
    final_test(model, classes, test_rows, device, best, params_info, compute_info)
    log(f'\n  Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
