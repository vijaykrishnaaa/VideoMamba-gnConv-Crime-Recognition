"""
VideoMamba V8 — Sequential Transfer Learning: K400 → UCF-101 → Violence
==============================================================
EXPERIMENT: Instead of starting from generic Kinetics-400 weights,
we start from our OWN UCF-101 checkpoint that achieved 94.6% accuracy.

The hypothesis: UCF-101 gnConv2D modules already learned human action
patterns (Punch, Boxing, Fencing, etc.) which directly overlap with
violence classes. This should give the model a massive head start.

Transfer Chain: Kinetics-400 → UCF-101 (94.6%) → Custom Violence Dataset
Architecture:   VideoMamba-Small + gnConv2D (order=5, multi-scale)
Hyperparams:    IDENTICAL to V1 (8 unfrozen layers, same LRs, same aug)

Run: conda activate videomamba_env && python train_crime_v8.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# STEP 1: ADD VIDEOMAMBA AND LOCAL MAMBA TO PATH
# ============================================================
sys.path.insert(0, os.path.expanduser('~/vlil_40/VideoMamba/mamba'))
sys.path.insert(0, os.path.expanduser('~/vlil_40/VideoMamba/videomamba/video_sm'))
from models.videomamba import videomamba_small

# ============================================================
# STEP 2: CONFIGURATION
# ============================================================
# Dataset paths
DATA_ROOT    = os.path.expanduser('~/final_year_project_29/dataset_new_split')
TRAIN_DIR    = os.path.join(DATA_ROOT, 'train')
TEST_DIR     = os.path.join(DATA_ROOT, 'test')

# Checkpoint paths
# V3 CHANGE: Save to a new folder for this experiment
CKPT_DIR = os.path.expanduser('~/vlil_40/v3_crime_transfer/checkpoints')

# V3 KEY CHANGE: Use UCF-101 checkpoint instead of raw Kinetics-400!
# This checkpoint already contains K400 knowledge + UCF-101 action understanding
PRETRAINED_PATH = os.path.expanduser('~/vlil_40/v1_ucf101/checkpoints_ucf101/best_ucf101_v6.pth')

os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# STEP 3: CLASS DEFINITIONS (13 violence classes)
# ============================================================
NUM_CLASSES = 13

CLASS_NAMES = {
    0: 'Dragging',
    1: 'Hitting',
    2: 'Kicking',
    3: 'Pushing',
    4: 'abuse',
    5: 'hair pulling',
    6: 'neck grabbing',
    7: 'punch',
    8: 'robbery',
    9: 'shooting',
    10: 'slap',
    11: 'snatching',
    12: 'stabbing',
}

# ============================================================
# STEP 4: HYPERPARAMETERS (IDENTICAL TO V1)
# ============================================================
NUM_FRAMES     = 8
BATCH_SIZE     = 2          # float32 uses more VRAM
ACCUM_STEPS    = 8          # Gradient accumulation -> effective batch = 16
EPOCHS         = 50
WARMUP_EPOCHS  = 5          # Linear warmup for first 5 epochs

LR_BACKBONE    = 1e-5       # Lower LR for pretrained Mamba backbone
LR_GNCONV      = 1e-4       # Safe LR for gnConv (prevents NaN)
LR_HEAD        = 1e-3       # Highest LR for new 13-class head

UNFREEZE_LAST  = 8          # Unfreeze last 8 Mamba layers (same as V1)
DROPOUT        = 0.3
WEIGHT_DECAY   = 0.05
GRAD_CLIP      = 1.0
LABEL_SMOOTH   = 0.1
MIXUP_ALPHA    = 0.2
PATIENCE       = 20
NUM_WORKERS    = 0
TTA_ENABLED    = True
USE_AMP        = False      # AMP float16 causes NaN in gnConv2D recursive mult


# ============================================================
# STEP 5: DATASET CLASS (Folder-based — scans ALL videos)
# ============================================================
class CrimeDataset(Dataset):
    """
    Custom Movie Violence Dataset — scans train/ and test/ folders directly.

    Folder structure:
        dataset_new_split/
            train/
                abuse/         → 46 videos
                Dragging/      → 54 videos
                ...
            test/
                abuse/         → 12 videos
                ...

    NO LEAKAGE GUARANTEE:
    - Train and val are in completely separate directories
    - Class labels assigned alphabetically from folder names
    """
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.MOV'}

    def __init__(self, split_dir, class_to_idx, is_train=True):
        self.is_train = is_train

        self.video_paths = []
        self.labels = []
        skipped = 0

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in class_to_idx:
                print(f"  [WARN] Unknown class folder: {class_name}")
                continue

            label = class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1]
                if ext.lower() in {e.lower() for e in self.VIDEO_EXTS}:
                    full_path = os.path.join(class_dir, fname)
                    self.video_paths.append(full_path)
                    self.labels.append(label)
                else:
                    skipped += 1

        mode = "TRAIN" if is_train else "VAL"
        print(f"  [{mode}] Loaded {len(self.video_paths)} videos "
              f"({skipped} non-video files skipped) from {split_dir}")

        # ============================================================
        # DATA AUGMENTATION PIPELINE
        # ============================================================
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),                      # Spatial jitter
                transforms.RandomHorizontalFlip(p=0.5),          # Horizontal flip
                transforms.ColorJitter(                          # Color jitter
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.15
                ),
                transforms.RandomGrayscale(p=0.1),               # Grayscale aug
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Random erasing
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.video_paths)

    def _get_frame_indices(self, total):
        """Sample NUM_FRAMES from video with temporal jitter during training."""
        if total < NUM_FRAMES:
            indices = np.arange(total)
            indices = np.pad(indices, (0, NUM_FRAMES - total), mode='wrap')
            return indices

        if self.is_train and total > NUM_FRAMES * 2:
            # Temporal jitter: randomly sample within each segment
            seg_len = total // NUM_FRAMES
            indices = []
            for i in range(NUM_FRAMES):
                start = i * seg_len
                end = min(start + seg_len, total)
                idx = np.random.randint(start, end)
                indices.append(idx)
            return np.array(indices)
        else:
            return np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            import cv2
            cv2.setNumThreads(0)  # Prevent OpenCV thread-leak deadlocks!
            cap = cv2.VideoCapture(path)
            frames_list = []

            for _ in range(300):
                ret, frame = cap.read()
                if not ret: break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            total = len(frames_list)
            if total == 0:
                raise ValueError("No frames found")

            indices = self._get_frame_indices(total)
            frames = [frames_list[int(i)] for i in indices]

            x = torch.stack([self.transform(f) for f in frames])
            x = x.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
            return x, label

        except Exception as e:
            print(f"  [WARN] Failed to load {path}: {e}")
            x = torch.zeros(3, NUM_FRAMES, 224, 224)
            return x, label


# ============================================================
# STEP 6: TTA DATASET (3-crop averaging for evaluation)
# ============================================================
class TTADataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

        self.crop_transforms = [
            # Left crop
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Lambda(lambda img: img.crop((0, 16, 224, 240))),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # Center crop
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # Right crop
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Lambda(lambda img: img.crop((32, 16, 256, 240))),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        ]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            import cv2
            cv2.setNumThreads(0)
            cap = cv2.VideoCapture(path)
            frames_list = []

            for _ in range(300):
                ret, frame = cap.read()
                if not ret: break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            total = len(frames_list)
            if total == 0:
                raise ValueError("No frames")

            if total < NUM_FRAMES:
                indices = np.arange(total)
                indices = np.pad(indices, (0, NUM_FRAMES - total), mode='wrap')
            else:
                indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

            frames = [frames_list[int(i)] for i in indices]

            all_crops = []
            for crop_transform in self.crop_transforms:
                crop_frames = torch.stack([crop_transform(f) for f in frames])
                crop_frames = crop_frames.permute(1, 0, 2, 3)
                all_crops.append(crop_frames)

            x = torch.stack(all_crops)
            return x, label

        except Exception as e:
            print(f"  [WARN] Failed to load {path}: {e}")
            x = torch.zeros(3, 3, NUM_FRAMES, 224, 224)
            return x, label


# ============================================================
# STEP 7: MODEL BUILDER (Sequential Transfer from UCF-101)
# ============================================================
def build_model():
    print(f"\n  === VideoMamba V8 — Sequential Transfer Learning ===")
    print(f"  Transfer Chain: Kinetics-400 → UCF-101 (94.6%) → Violence (13 classes)")

    model = videomamba_small(pretrained=False, num_classes=NUM_CLASSES)
    print(f"  Model created with gnConv2D (order=5, multi-scale) in all 24 Mamba layers")

    # ============================================================
    # LOAD UCF-101 CHECKPOINT (instead of raw K400)
    # ============================================================
    print(f"\n  Loading UCF-101 pretrained weights (94.6% accuracy)...")
    print(f"  Source: {PRETRAINED_PATH}")
    if os.path.exists(PRETRAINED_PATH):
        ckpt = torch.load(PRETRAINED_PATH, map_location='cpu')

        # Remove the 101-class head (we need a 13-class head)
        keys_to_remove = [k for k in ckpt.keys() if 'head' in k]
        for k in keys_to_remove:
            del ckpt[k]

        # Count how many gnConv keys are being loaded (these have UCF-101 knowledge!)
        gnconv_keys = [k for k in ckpt.keys() if 'gnconv' in k]

        msg = model.load_state_dict(ckpt, strict=False)
        print(f"  Missing keys (head only, expected): {len(msg.missing_keys)}")
        print(f"  Unexpected keys (discarded): {len(msg.unexpected_keys)}")
        print(f"  gnConv2D keys loaded from UCF-101: {len(gnconv_keys)}")
        print(f"  >>> gnConv modules start with ACTION KNOWLEDGE (not random!)")
    else:
        print(f"  ERROR: UCF-101 checkpoint not found at {PRETRAINED_PATH}")
        print(f"  Falling back to random initialization!")

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    total_layers = len(model.layers)
    freeze_until = total_layers - UNFREEZE_LAST
    print(f"\n  Total Mamba layers: {total_layers}")
    print(f"  Freezing layers 0-{freeze_until-1} (retaining K400 + UCF-101 knowledge)")
    print(f"  Unfreezing layers {freeze_until}-{total_layers-1} (fine-tuning for violence)")

    for i in range(freeze_until, total_layers):
        for p in model.layers[i].parameters():
            p.requires_grad = True

    if hasattr(model, 'norm_f'):
        for p in model.norm_f.parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True

    # Unfreeze gnConv ONLY in the last 8 layers
    gnconv_count = 0
    for i in range(freeze_until, total_layers):
        layer = model.layers[i]
        if hasattr(layer, 'mixer'):
            mixer = layer.mixer
            if hasattr(mixer, 'gnconv'):
                for p in mixer.gnconv.parameters():
                    p.requires_grad = True
                    gnconv_count += p.numel()
            if hasattr(mixer, 'gnconv_b'):
                for p in mixer.gnconv_b.parameters():
                    p.requires_grad = True
                    gnconv_count += p.numel()
    print(f"  gnConv2D params unfrozen in last {UNFREEZE_LAST} layers: {gnconv_count:,}")

    model = model.cuda()

    # ============================================================
    # MODEL PARAMETER BREAKDOWN (Guide's request)
    # ============================================================
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - train_params

    # Detailed breakdown by component
    patch_embed_params = 0
    mamba_ssm_params = 0
    gnconv_total_params = 0
    head_params_count = 0
    norm_other_params = 0

    # Trainable breakdown
    patch_embed_train = 0
    mamba_ssm_train = 0
    gnconv_train = 0
    head_train = 0
    norm_other_train = 0

    for name, p in model.named_parameters():
        n = p.numel()
        if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
            patch_embed_params += n
            if p.requires_grad: patch_embed_train += n
        elif 'head' in name:
            head_params_count += n
            if p.requires_grad: head_train += n
        elif 'gnconv' in name:
            gnconv_total_params += n
            if p.requires_grad: gnconv_train += n
        elif 'norm' in name:
            norm_other_params += n
            if p.requires_grad: norm_other_train += n
        else:
            mamba_ssm_params += n
            if p.requires_grad: mamba_ssm_train += n

    print(f"")
    print(f"  {'='*60}")
    print(f"  MODEL PARAMETER BREAKDOWN:")
    print(f"  {'─'*60}")
    print(f"  {'Component':<25s} {'Total':>12s} {'Trainable':>12s} {'Frozen':>12s}")
    print(f"  {'─'*60}")
    print(f"  {'Patch Embedding':<25s} {patch_embed_params:>12,} {patch_embed_train:>12,} {patch_embed_params-patch_embed_train:>12,}")
    print(f"  {'Mamba SSM Layers':<25s} {mamba_ssm_params:>12,} {mamba_ssm_train:>12,} {mamba_ssm_params-mamba_ssm_train:>12,}")
    print(f"  {'gnConv2D Modules':<25s} {gnconv_total_params:>12,} {gnconv_train:>12,} {gnconv_total_params-gnconv_train:>12,}")
    print(f"  {'Classification Head':<25s} {head_params_count:>12,} {head_train:>12,} {head_params_count-head_train:>12,}")
    print(f"  {'LayerNorm / Other':<25s} {norm_other_params:>12,} {norm_other_train:>12,} {norm_other_params-norm_other_train:>12,}")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<25s} {total_params:>12,} {train_params:>12,} {frozen_params:>12,}")
    print(f"  {'':25s} {'(100%)':>12s} {f'({100*train_params/total_params:.1f}%)':>12s} {f'({100*frozen_params/total_params:.1f}%)':>12s}")
    print(f"  {'='*60}")
    print(f"  Estimated FLOPs: ~56 GFLOPs per video (8 frames × 224×224)")

    return model


# ============================================================
# STEP 8: MIXUP AUGMENTATION
# ============================================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# STEP 9: WARMUP + COSINE ANNEALING SCHEDULER
# ============================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale


# ============================================================
# STEP 10: TRAINING LOOP (float32, no AMP)
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    nan_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"  Epoch {epoch}/{EPOCHS} [TRAIN]")
    for step, (x, y) in enumerate(pbar):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        mixed_x, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)

        logits = model(mixed_x)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss = loss / ACCUM_STEPS

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            optimizer.zero_grad()
            continue

        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS * x.size(0)
        correct += (logits.argmax(1) == y_a).sum().item() * lam
        correct += (logits.argmax(1) == y_b).sum().item() * (1 - lam)
        total += x.size(0)
        pbar.set_postfix(acc=f"{100*correct/total:.1f}%", loss=f"{loss.item()*ACCUM_STEPS:.4f}")

    if nan_batches > 0:
        print(f"  [WARN] Skipped {nan_batches} NaN batches")

    return total_loss / max(total, 1), 100 * correct / max(total, 1)


# ============================================================
# STEP 11: VALIDATION
# ============================================================
def validate(model, loader, class_names=None):
    model.eval()
    correct, total = 0, 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="  [VALIDATE]"):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            for pred, label in zip(preds.cpu().numpy(), y.cpu().numpy()):
                per_class_total[label] += 1
                if pred == label:
                    per_class_correct[label] += 1
                all_preds.append(pred)
                all_labels.append(label)

    if class_names:
        class_accs = []
        for i in range(NUM_CLASSES):
            if per_class_total[i] > 0:
                acc = 100 * per_class_correct[i] / per_class_total[i]
                class_accs.append((class_names.get(i, f"class_{i}"), acc, per_class_correct[i], per_class_total[i]))

        class_accs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Top-5 best classes:")
        for name, acc, c, t in class_accs[:5]:
            print(f"    {name:30s}: {c:3d}/{t:3d} = {acc:.1f}%")

        print(f"\n  Top-5 worst classes:")
        for name, acc, c, t in class_accs[-5:]:
            print(f"    {name:30s}: {c:3d}/{t:3d} = {acc:.1f}%")

    return 100 * correct / total, all_preds, all_labels


# ============================================================
# STEP 12: VALIDATION WITH TTA (3-crop averaging)
# ============================================================
def validate_tta(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="  [VALIDATE-TTA]"):
            y = y.cuda(non_blocking=True)
            batch_size = x.size(0)
            num_crops = x.size(1)

            avg_logits = torch.zeros(batch_size, NUM_CLASSES).cuda()
            for c in range(num_crops):
                crop = x[:, c].cuda(non_blocking=True)
                logits = model(crop)
                avg_logits += logits.float()
            avg_logits /= num_crops

            preds = avg_logits.argmax(1)
            correct += (preds == y).sum().item()
            total += batch_size

    return 100 * correct / total


# ============================================================
# STEP 13: COMPUTE CLASS WEIGHTS
# ============================================================
def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * counts.astype(float) + 1e-6)
    weights = np.clip(weights, 0.5, 10.0)
    return torch.FloatTensor(weights)


# ============================================================
# STEP 14: MAIN FUNCTION
# ============================================================
def main():
    print("=" * 70)
    print("VideoMamba V8 — Sequential Transfer: K400 → UCF-101 → Violence")
    print("=" * 70)

    # ============================================================
    # COMPLETE CONFIGURATION BANNER (Guide's request)
    # ============================================================
    print(f"\n  ARCHITECTURE:")
    print(f"    Backbone:          VideoMamba-Small (24 Mamba layers)")
    print(f"    Spatial Module:    gnConv2D (HorNet, order=5, multi-scale [3,5,7,1])")
    print(f"    Classification:    Linear head ({NUM_CLASSES} classes)")

    print(f"\n  TRANSFER LEARNING:")
    print(f"    Stage 1:           Kinetics-400 pre-training (400 action classes)")
    print(f"    Stage 2:           UCF-101 fine-tuning (94.6% accuracy)")
    print(f"    Stage 3:           Custom Violence dataset (13 classes) ← CURRENT")
    print(f"    Pretrained from:   {os.path.basename(PRETRAINED_PATH)}")

    print(f"\n  DATASET:")
    print(f"    Name:              Custom Movie Violence Dataset")
    print(f"    Classes:           {NUM_CLASSES}")
    print(f"    Data root:         {DATA_ROOT}")

    print(f"\n  HYPERPARAMETERS:")
    print(f"    Frames per video:  {NUM_FRAMES}")
    print(f"    Batch size:        {BATCH_SIZE} (gradient accum: {ACCUM_STEPS}, effective: {BATCH_SIZE*ACCUM_STEPS})")
    print(f"    Epochs:            {EPOCHS} (warmup: {WARMUP_EPOCHS})")
    print(f"    LR (backbone):     {LR_BACKBONE}")
    print(f"    LR (gnConv2D):     {LR_GNCONV}")
    print(f"    LR (head):         {LR_HEAD}")
    print(f"    Scheduler:         Cosine Annealing with Linear Warmup")
    print(f"    Optimizer:         AdamW (weight_decay={WEIGHT_DECAY})")

    print(f"\n  REGULARIZATION:")
    print(f"    Dropout:           {DROPOUT}")
    print(f"    Label Smoothing:   {LABEL_SMOOTH}")
    print(f"    Mixup Alpha:       {MIXUP_ALPHA}")
    print(f"    Gradient Clipping: {GRAD_CLIP}")
    print(f"    Early Stopping:    patience={PATIENCE}")

    print(f"\n  DATA AUGMENTATION:")
    print(f"    Spatial Jitter:    RandomCrop(224) from resized 256")
    print(f"    Horizontal Flip:   p=0.5")
    print(f"    Color Jitter:      brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15")
    print(f"    Random Grayscale:  p=0.1")
    print(f"    Random Erasing:    p=0.2, scale=(0.02, 0.2)")
    print(f"    Temporal Jitter:   Segment-based random sampling (train only)")

    print(f"\n  FINE-TUNING STRATEGY:")
    print(f"    Total layers:      24 Mamba layers")
    print(f"    Frozen layers:     0-{24-UNFREEZE_LAST-1} (retain K400+UCF-101 knowledge)")
    print(f"    Unfrozen layers:   {24-UNFREEZE_LAST}-23 (adapt for violence)")
    print(f"    gnConv2D:          Unfrozen in last {UNFREEZE_LAST} layers")
    print(f"    Head:              Fully trainable (new {NUM_CLASSES}-class)")

    print(f"\n  EVALUATION:")
    print(f"    TTA:               {TTA_ENABLED} (3-crop: left, center, right)")
    print(f"    AMP (float16):     {USE_AMP} (disabled — gnConv2D NaN issue)")
    print(f"    Precision:         float32")

    print("=" * 70)

    # --- Build datasets ---
    print(f"\n  Loading Custom Violence dataset from folder structure...")

    class_folders = sorted([d for d in os.listdir(TRAIN_DIR)
                           if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    class_to_idx = {name: i for i, name in enumerate(class_folders)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    print(f"  Found {len(class_to_idx)} classes: {class_folders}")

    CLASS_NAMES.clear()
    CLASS_NAMES.update(idx_to_class)

    train_dataset = CrimeDataset(TRAIN_DIR, class_to_idx, is_train=True)
    val_dataset   = CrimeDataset(TEST_DIR, class_to_idx, is_train=False)

    # Class distribution
    train_labels = np.array(train_dataset.labels)
    val_labels = np.array(val_dataset.labels)
    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    val_counts = np.bincount(val_labels, minlength=NUM_CLASSES)

    print(f"\n  Dataset Distribution:")
    print(f"  {'Class':<20s} {'Train':>8s} {'Test':>8s}")
    print(f"  {'─'*40}")
    for i in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[i]:<20s} {train_counts[i]:>8d} {val_counts[i]:>8d}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<20s} {len(train_dataset):>8d} {len(val_dataset):>8d}")

    # Class weights
    class_weights = compute_class_weights(train_labels)
    print(f"\n  Class weights (for imbalance):")
    for i in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[i]:20s}: {class_weights[i]:.2f}")

    # --- Data loaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    if TTA_ENABLED:
        tta_dataset = TTADataset(val_dataset.video_paths, val_dataset.labels)
        tta_loader = DataLoader(
            tta_dataset, batch_size=2,
            shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True
        )

    # --- Build model ---
    model = build_model()

    # --- Optimizer with 3-tier differential learning rates ---
    backbone_params = []
    gnconv_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'head' in name:
                head_params.append(param)
            elif 'gnconv' in name:
                gnconv_params.append(param)
            else:
                backbone_params.append(param)

    print(f"\n  Optimizer groups:")
    print(f"    Backbone params: {sum(p.numel() for p in backbone_params):,} (lr={LR_BACKBONE})")
    print(f"    gnConv2D params: {sum(p.numel() for p in gnconv_params):,} (lr={LR_GNCONV})")
    print(f"    Head params:     {sum(p.numel() for p in head_params):,} (lr={LR_HEAD})")

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': gnconv_params,   'lr': LR_GNCONV},
        {'params': head_params,     'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.cuda(),
        label_smoothing=LABEL_SMOOTH
    )

    # --- Training loop ---
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    model_path = os.path.join(CKPT_DIR, 'best_crime_v8.pth')

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        scheduler.step(epoch - 1)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        val_acc, all_preds, all_labels = validate(model, val_loader, CLASS_NAMES)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        gap = train_acc - val_acc
        print(f"  Epoch {epoch}: Loss={train_loss:.4f} | Train={train_acc:.1f}% "
              f"| Val={val_acc:.1f}% | LR={current_lr:.2e} | Gap={gap:.1f}% "
              f"| Time={epoch_time:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  >>> NEW BEST! Saved to {model_path}")
        else:
            patience_counter += 1
            print(f"  (No improvement for {patience_counter}/{PATIENCE} epochs)")

        if patience_counter >= PATIENCE:
            print(f"\n  EARLY STOPPING at epoch {epoch}!")
            print(f"  No improvement for {PATIENCE} consecutive epochs.")
            break

    # --- Final evaluation ---
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Model saved: {model_path}")
    print(f"{'=' * 70}")

    print(f"\n  Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    print(f"\n--- Standard (Center Crop) ---")
    val_acc, all_preds, all_labels = validate(model, val_loader, CLASS_NAMES)
    print(f"\n  STANDARD TEST ACCURACY: {val_acc:.2f}%")

    if TTA_ENABLED:
        print(f"\n--- TTA (3-Crop Average) ---")
        tta_acc = validate_tta(model, tta_loader)
        print(f"\n  TTA TEST ACCURACY: {tta_acc:.2f}%")

    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY — V8 (Sequential Transfer: K400 → UCF-101 → Violence)")
    print(f"  Standard: {val_acc:.2f}%")
    if TTA_ENABLED:
        print(f"  TTA (3-crop): {tta_acc:.2f}%")
        print(f"  TTA Improvement: +{tta_acc - val_acc:.2f}%")
    print(f"")
    print(f"  COMPARISON:")
    print(f"    V1 (K400 only):              69.64% TTA")
    print(f"    V3 (K400 → UCF-101 → Crime): {tta_acc:.2f}% TTA")
    print(f"    Difference:                  {tta_acc - 69.64:+.2f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
