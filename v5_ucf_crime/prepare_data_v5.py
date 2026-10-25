"""
V5 data preparation — UCF-Crime (run ONCE through PBS before train_v5.py)
==========================================================================
1. Reads the official UCF-Crime "Action Recognition" split files
   (UCF_Crimes-Train-Test-Split/Action_Regnition_splits): 14 classes x 50 videos,
   4 folds, each fold = 532 train / 168 test videos (38 / 12 per class).
2. Finds every listed video under the dataset root by file name (the videos are spread
   over Anomaly-Videos-Part-1..4 and Normal_Videos_for_Event_Recognition).
3. Decodes each of the 700 videos ONCE and stores 64 frames spread over the WHOLE video
   (UCF-Crime is 320x240, kept at native size, JPEG q95) in frame_cache/.
   UCF-Crime videos are long and untrimmed (minutes, not seconds), so 64 frames instead
   of V4's 32: training samples 8 of them, testing looks at all 64.
4. For every fold writes splits/foldK/train.csv, val.csv, test.csv:
       test  = the official test list, untouched
       val   = 15% of the official train list (6 per class), stratified, fixed seed
       train = the rest of the official train list
   Near-duplicate videos are kept on the same side of train/val.
5. Crime windows: Temporal_Anomaly_Annotation.txt gives the start/end frames of the crime
   for 140 crime videos (121 of them are in our 700). For those, 16 extra frames from INSIDE
   the crime window are cached (*_event.npz). Training never uses them; train_v5.py uses
   them for a second test number ("accuracy when looking only at the crime moment").
6. Writes logs/prep_report.txt: per-class counts, video lengths, resolutions, missing /
   unreadable videos, crime-window coverage, and train/test near-duplicates per fold.

Run:  qsub run_prepare_v5.sh   (from ~/vlil_40/v5_ucf_crime)
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT        = os.path.expanduser('~/final_year_project_29/ucf_dataset')
SPLIT_SUBDIR     = os.path.join('UCF_Crimes-Train-Test-Split', 'Action_Regnition_splits')
EXP_DIR          = os.path.expanduser('~/vlil_40/v5_ucf_crime')
NUM_FOLDS        = 4
NUM_CACHE_FRAMES = 64          # frames stored per video (training samples 8 of them)
NUM_EVENT_FRAMES = 16          # frames stored from inside the annotated crime window (test only)
ANNOTATION_FILE  = 'Temporal_Anomaly_Annotation.txt'   # searched for anywhere under the dataset root
MAX_SHORT_SIDE   = 256         # bigger videos are shrunk to this; 320x240 UCF-Crime stays native
JPEG_QUALITY     = 95
VAL_FRACTION     = 0.15        # part of each fold's train list used for model selection
SEED             = 42
DUP_HAMMING      = 6           # dHash bit distance that counts as "same frame"
DUP_MIN_MATCHES  = 2           # matching frame pairs needed to call two videos duplicates
HASH_AT          = tuple(int((k + 0.5) * NUM_CACHE_FRAMES / 4) for k in range(4))   # frames 8, 24, 40, 56
MISSING_LIMIT    = 0.05        # fail the job if more than 5% of the listed videos are missing/unreadable
VIDEO_EXTS       = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_') or 'x'


def display_name(folder):
    """'Normal_Videos_event' -> 'Normal'; the 13 crime folders keep their names."""
    return 'Normal' if folder.lower().startswith('normal') else folder


# ============================================================
# VIDEO DECODING (runs inside worker processes) — same as V4
# ============================================================
def _targets(n, k):
    """k frame indices at the centres of k equal segments of an n-frame video."""
    idx = ((np.arange(k) + 0.5) * n / k).astype(np.int64)
    return np.clip(idx, 0, max(n - 1, 0))


def _grab_at(cap, targets):
    """Walk the stream once with grab(); retrieve() only the wanted frames."""
    want = {}
    for pos, t in enumerate(targets):
        want.setdefault(int(t), []).append(pos)
    last = max(want)
    out = [None] * len(targets)
    last_good = None
    i = 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                last_good = fr
            fr = fr if (ok and fr is not None) else last_good
            for pos in want[i]:
                out[pos] = fr
        i += 1
    if any(o is None for o in out):
        return None, i
    return out, i


def read_video(path, k):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError('cannot open')
    n_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if n_est <= 0 or n_est > 10_000_000:
        n_est = 0

    if n_est > 0:
        frames, n_read = _grab_at(cap, _targets(n_est, k))
        cap.release()
        if frames is not None:
            return frames, n_est, fps
        n_true = n_read                       # header over-reported the length
    else:
        n_true = 0
        while cap.grab():
            n_true += 1
        cap.release()

    if n_true == 0:
        raise IOError('no decodable frames')
    cap = cv2.VideoCapture(path)
    frames, _ = _grab_at(cap, _targets(n_true, k))
    cap.release()
    if frames is None:
        raise IOError('frames disappeared on second pass')
    return frames, n_true, fps


def dhash64(gray):
    """64-bit difference hash; None for near-uniform frames (black/fade)."""
    import cv2
    if gray.std() < 8:
        return None
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view('>u8')[0])


def file_fingerprint(path, head_bytes=4 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read(head_bytes))
    return f"{os.path.getsize(path)}_{h.hexdigest()}"


def _init_worker():
    import cv2
    cv2.setNumThreads(0)


def read_event_frames(path, segments, n_frames, m):
    """m frames spread evenly over the annotated crime window(s) [(start, end), ...]."""
    import cv2
    idx = []
    for s, e in segments:
        s = max(0, min(int(s), n_frames - 1))
        e = max(s, min(int(e), n_frames - 1))
        idx.extend(range(s, e + 1))
    if not idx:
        raise IOError('empty crime window')
    idx = np.asarray(idx, dtype=np.int64)
    targets = idx[((np.arange(m) + 0.5) * len(idx) / m).astype(np.int64)]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError('cannot open')
    frames, _ = _grab_at(cap, targets)
    cap.release()
    if frames is None:
        raise IOError('crime-window frames not decodable')
    return frames


def write_cache(cache_path, frames, meta):
    """JPEG-encode frames (shrunk only if the short side > MAX_SHORT_SIDE) into one .npz."""
    import cv2
    h0, w0 = frames[0].shape[:2]
    scale = min(1.0, MAX_SHORT_SIDE / min(h0, w0))
    new_wh = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
    bufs, offsets = [], [0]
    for fr in frames:
        if fr.shape[:2] != (h0, w0):
            fr = cv2.resize(fr, (w0, h0))
        if scale < 1:
            fr = cv2.resize(fr, new_wh, interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode('.jpg', fr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise IOError('jpeg encode failed')
        bufs.append(enc.tobytes())
        offsets.append(offsets[-1] + len(bufs[-1]))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path[:-4] + '.tmp.npz'
    np.savez(tmp, jpg=np.frombuffer(b''.join(bufs), dtype=np.uint8),
             offsets=np.asarray(offsets, dtype=np.int64), meta=np.array(json.dumps(meta)))
    os.replace(tmp, cache_path)


def process_video(job):
    """Decode one video into the cache (or reuse an existing cache file), plus its crime window."""
    import cv2
    path, cache_path = job['video_path'], job['cache_path']
    result = dict(job)
    try:
        if os.path.exists(cache_path):
            with np.load(cache_path) as d:
                meta = json.loads(str(d['meta']))
            result['cached'] = True
        else:
            frames, n_frames, fps = read_video(path, NUM_CACHE_FRAMES)
            h0, w0 = frames[0].shape[:2]
            meta = {
                'n_frames': int(n_frames), 'fps': round(fps, 3),
                'width': int(w0), 'height': int(h0),
                'duration_s': round(n_frames / fps, 2) if fps > 0 else -1.0,
                'dhash': [dhash64(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)) for i in HASH_AT],
                'fingerprint': file_fingerprint(path),
            }
            write_cache(cache_path, frames, meta)
            result['cached'] = False
        result.update(meta)
        result['ok'] = True
    except Exception as e:  # noqa: BLE001 - report every failure, never crash the job
        result['ok'] = False
        result['error'] = f'{type(e).__name__}: {e}'
        return result

    if job.get('segments'):
        ev_path = job['event_cache_path']
        try:
            if not os.path.exists(ev_path):
                frames = read_event_frames(path, job['segments'], result['n_frames'], NUM_EVENT_FRAMES)
                write_cache(ev_path, frames, {'segments': job['segments']})
            result['event_ok'] = True
        except Exception as e:  # noqa: BLE001
            result['event_ok'] = False
            result['event_error'] = f'{type(e).__name__}: {e}'
    return result


# ============================================================
# DUPLICATE DETECTION — same as V4
# ============================================================
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(items):
    """Pairs (i, j) of videos sharing >= DUP_MIN_MATCHES near-identical frames."""
    n = len(items)
    H = np.zeros((n, 4), dtype=np.uint64)
    M = np.zeros((n, 4), dtype=bool)
    for i, it in enumerate(items):
        for j, hv in enumerate((it.get('dhash') or [])[:4]):
            if hv is not None:
                H[i, j] = np.uint64(hv)
                M[i, j] = True
    pairs = []
    for i in range(n - 1):
        x = H[i][None, :, None] ^ H[i + 1:][:, None, :]               # (n-i-1, 4, 4)
        dist = _POPCOUNT[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)
        valid = M[i][None, :, None] & M[i + 1:][:, None, :]
        matches = ((dist <= DUP_HAMMING) & valid).sum(axis=(1, 2))
        for off in np.nonzero(matches >= DUP_MIN_MATCHES)[0]:
            pairs.append((i, i + 1 + int(off), int(matches[off])))
    return pairs


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


# ============================================================
# OFFICIAL SPLIT FILES
# ============================================================
def read_class_ids(split_src):
    """ClassIDs.txt lines look like 'Abuse 1'. Returns folder names ordered by id."""
    ids = []
    with open(os.path.join(split_src, 'ClassIDs.txt')) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                ids.append((int(parts[-1]), ' '.join(parts[:-1])))
    return [name for _, name in sorted(ids)]


def read_list(path):
    """Split-file lines look like 'Abuse/Abuse013_x264.mp4 ' (trailing spaces, maybe CRLF)."""
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip().replace('\\', '/')
            if s:
                folder, _, fname = s.rpartition('/')
                out.append((folder, fname))
    return out


def index_videos(root):
    """file name (lower case) -> every path under root with that name; also finds the annotation file."""
    idx, ann_path = {}, None
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == ANNOTATION_FILE and ann_path is None:
                ann_path = os.path.join(dirpath, fn)
            if fn.startswith('._') or os.path.splitext(fn)[1].lower() not in VIDEO_EXTS:
                continue
            idx.setdefault(fn.lower(), []).append(os.path.join(dirpath, fn))
    return idx, ann_path


def read_annotations(path):
    """Lines like 'Arrest001_x264.mp4  Arrest  1185  1485  -1  -1' -> {name: [(1185, 1485)]}.
    Normal videos have -1 everywhere and get no entry."""
    windows = {}
    if not path:
        return windows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            nums = [int(x) for x in parts[2:6]]
            segs = [(s, e) for s, e in (nums[0:2], nums[2:4]) if s >= 0 and e >= s]
            if segs:
                windows[parts[0].lower()] = segs
    return windows


def resolve(idx, folder, fname):
    """Pick the file for 'folder/fname'; prefer a copy inside a folder with the class name."""
    cands = sorted(idx.get(fname.lower(), []))
    if len(cands) <= 1:
        return (cands[0] if cands else None), len(cands)

    def score(p):
        parts = [x.lower() for x in p.split(os.sep)]
        s = 2 if folder.lower() in parts else 0
        if folder.lower().startswith('normal') and 'event' in p.lower():
            s += 1
        return s
    return max(cands, key=score), len(cands)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--exp-dir', default=EXP_DIR)
    ap.add_argument('--procs', type=int, default=int(os.environ.get('NCPUS', '4')))
    args = ap.parse_args()

    t0 = time.time()
    split_src = os.path.join(args.data_root, SPLIT_SUBDIR)
    cache_dir = os.path.join(args.exp_dir, 'frame_cache')
    split_dir = os.path.join(args.exp_dir, 'splits')
    log_dir = os.path.join(args.exp_dir, 'logs')
    for d in (cache_dir, split_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    print('=' * 70)
    print('V5 DATA PREPARATION — UCF-Crime frame cache + 4-fold train/val/test lists')
    print('=' * 70)
    print(f'  Dataset root : {args.data_root}')
    print(f'  Split files  : {split_src}')
    print(f'  Output       : {args.exp_dir}')
    print(f'  Workers      : {args.procs}')
    if not os.path.isdir(split_src):
        sys.exit(f'  ERROR: split folder not found: {split_src}')

    folders = read_class_ids(split_src)
    classes = [display_name(c) for c in folders]
    folder_to_idx = {c: i for i, c in enumerate(folders)}
    print(f'  Classes ({len(classes)}): {classes}')

    fold_lists = {}
    for k in range(1, NUM_FOLDS + 1):
        tr = read_list(os.path.join(split_src, f'train_{k:03d}.txt'))
        te = read_list(os.path.join(split_src, f'test_{k:03d}.txt'))
        fold_lists[k] = (tr, te)
        print(f'  Fold {k}: {len(tr)} train / {len(te)} test listed')
    entries = sorted({e for tr, te in fold_lists.values() for e in tr + te})
    unknown = sorted({f for f, _ in entries if f not in folder_to_idx})
    if unknown:
        print(f'  [WARN] split files mention folders not in ClassIDs.txt (skipped): {unknown}')
    entries = [e for e in entries if e[0] in folder_to_idx]

    print('  Indexing video files under the dataset root ...', flush=True)
    idx, ann_path = index_videos(args.data_root)
    windows = read_annotations(ann_path)
    print(f'  {sum(len(v) for v in idx.values())} video files on disk, {len(entries)} listed in the splits')
    print(f'  Temporal annotation: {ann_path or "NOT FOUND (crime-window test will be skipped)"} '
          f'| {len(windows)} crime videos with a window')

    jobs, missing, ambiguous = [], [], []
    for folder, fname in entries:
        key = f'{folder}/{fname}'
        path, n = resolve(idx, folder, fname)
        if path is None:
            missing.append(key)
            continue
        if n > 1:
            ambiguous.append((key, n, path))
        label = folder_to_idx[folder]
        stem = safe_name(os.path.splitext(fname)[0])[:60]
        cpath = os.path.join(cache_dir, f'{label:02d}_{safe_name(classes[label])}',
                             f'{stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}.npz')
        segs = windows.get(fname.lower(), [])
        jobs.append({'key': key, 'class_name': classes[label], 'label': label,
                     'video_path': path, 'cache_path': cpath, 'segments': segs,
                     'event_cache_path': cpath[:-4] + '_event.npz' if segs else '',
                     'crime_window': ';'.join(f'{s}-{e}' for s, e in segs)})
    print(f'  To decode: {len(jobs)} (with a crime window: {sum(1 for j in jobs if j["segments"])}) '
          f'| missing on disk: {len(missing)}', flush=True)

    # ---------- decode ----------
    results = []
    with mp.Pool(args.procs, initializer=_init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(process_video, jobs, chunksize=1), 1):
            results.append(r)
            if k % 25 == 0 or k == len(jobs):
                print(f'  decoded {k}/{len(jobs)}  ({time.time() - t0:.0f}s)', flush=True)
    results.sort(key=lambda r: r['key'])
    good = [r for r in results if r['ok']]
    bad = [r for r in results if not r['ok']]
    event_bad = [r for r in good if r['segments'] and not r.get('event_ok')]
    for r in good:
        if not r.get('event_ok'):
            r['event_cache_path'] = ''        # no window, or its frames could not be decoded
    by_key = {r['key']: r for r in good}
    pos = {r['key']: i for i, r in enumerate(good)}

    # ---------- duplicates ----------
    uf = UnionFind(len(good))
    fp_seen = {}
    exact_pairs = []
    for i, r in enumerate(good):
        fp = r.get('fingerprint')
        if fp in fp_seen:
            exact_pairs.append((fp_seen[fp], i))
            uf.union(fp_seen[fp], i)
        else:
            fp_seen[fp] = i
    near_pairs = near_duplicate_pairs(good)
    for i, j, _ in near_pairs:
        uf.union(i, j)
    for i, r in enumerate(good):
        r['group_id'] = uf.find(i)
    all_pairs = [(i, j, m) for i, j, m in near_pairs] + [(i, j, 'exact') for i, j in exact_pairs]

    # ---------- per fold: train / val / test lists ----------
    fields = ['split', 'class_name', 'label', 'video_path', 'cache_path', 'n_frames', 'fps',
              'width', 'height', 'duration_s', 'group_id', 'crime_window', 'event_cache_path']
    with open(os.path.join(split_dir, 'classes.txt'), 'w') as f:
        for i, c in enumerate(classes):
            f.write(f'{i}\t{c}\n')

    fold_info = {}
    for k, (tr, te) in fold_lists.items():
        rng = random.Random(SEED + k)
        tr_keys = [f'{a}/{b}' for a, b in tr if f'{a}/{b}' in by_key]
        te_keys = [f'{a}/{b}' for a, b in te if f'{a}/{b}' in by_key]
        overlap = sorted(set(tr_keys) & set(te_keys))
        train_pool = [dict(by_key[key]) for key in tr_keys]
        test_rows = [dict(by_key[key], split='test') for key in te_keys]

        groups = {}
        for r in train_pool:
            groups.setdefault(r['group_id'], []).append(r)
        group_list = sorted(groups.values(), key=lambda g: g[0]['key'])
        rng.shuffle(group_list)
        n_per_class = np.bincount(np.array([r['label'] for r in train_pool], dtype=np.int64),
                                  minlength=len(classes))
        target_val = [max(1, int(round(VAL_FRACTION * n))) if n > 1 else 0 for n in n_per_class]
        val_count = [0] * len(classes)
        for g in group_list:
            need = {}
            for r in g:
                need[r['label']] = need.get(r['label'], 0) + 1
            fits = all(val_count[c] + n <= target_val[c] for c, n in need.items())
            useful = any(val_count[c] < target_val[c] for c in need)
            split = 'val' if (fits and useful) else 'train'
            for r in g:
                r['split'] = split
            if split == 'val':
                for c, n in need.items():
                    val_count[c] += n

        rows_by_split = {'train': [r for r in train_pool if r['split'] == 'train'],
                         'val': [r for r in train_pool if r['split'] == 'val'],
                         'test': test_rows}
        fdir = os.path.join(split_dir, f'fold{k}')
        os.makedirs(fdir, exist_ok=True)
        for s, rows in rows_by_split.items():
            with open(os.path.join(fdir, f'{s}.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)

        train_side = {pos[key] for key in tr_keys}
        test_side = {pos[key] for key in te_keys}
        cross = [(good[i], good[j], m) for i, j, m in all_pairs
                 if (i in train_side and j in test_side) or (i in test_side and j in train_side)]
        fold_info[k] = {'rows': rows_by_split, 'overlap': overlap, 'cross': cross}

    # ---------- report ----------
    lines = []
    out = lines.append
    out('=' * 70)
    out('V5 DATASET REPORT — UCF-Crime, official Action Recognition split')
    out('=' * 70)
    out(f'Dataset root: {args.data_root}')
    out(f'Videos listed: {len(entries)} | found: {len(jobs)} | decoded OK: {len(good)} | '
        f'missing on disk: {len(missing)} | unreadable: {len(bad)}')
    out('')
    rows1 = fold_info[1]['rows']
    out('Fold 1 (the first run):')
    out(f"{'Class':<16s}{'Train':>8s}{'Val':>8s}{'Test':>8s}{'Total':>8s}")
    out('-' * 48)
    for i, c in enumerate(classes):
        n = [sum(1 for r in rows1[s] if r['label'] == i) for s in ('train', 'val', 'test')]
        out(f'{c:<16s}{n[0]:>8d}{n[1]:>8d}{n[2]:>8d}{sum(n):>8d}')
    out('-' * 48)
    tot = [len(rows1[s]) for s in ('train', 'val', 'test')]
    out(f"{'TOTAL':<16s}{tot[0]:>8d}{tot[1]:>8d}{tot[2]:>8d}{sum(tot):>8d}")
    out('')
    for k, info in fold_info.items():
        r = info['rows']
        out(f"Fold {k}: train {len(r['train'])} | val {len(r['val'])} | test {len(r['test'])} | "
            f"same video in train and test: {len(info['overlap'])} | "
            f"train<->test near-duplicates: {len(info['cross'])}")
    out('')

    n_win = sum(1 for r in good if r['event_cache_path'])
    out(f'Crime windows (Temporal_Anomaly_Annotation, used ONLY for the extra crime-window test):')
    out(f'  annotation file: {ann_path or "NOT FOUND"}')
    out(f'  videos of the 700 with a cached crime window: {n_win} '
        f'({NUM_EVENT_FRAMES} frames each from inside the window)')
    for k, info in fold_info.items():
        out(f"  fold {k}: test videos with a crime window: "
            f"{sum(1 for r in info['rows']['test'] if r['event_cache_path'])}")
    if event_bad:
        out(f'  crime windows that could not be decoded ({len(event_bad)}):')
        for r in event_bad:
            out(f"    {r['key']} [{r['crime_window']}] -> {r.get('event_error')}")
    out('')

    durs = np.array([r['duration_s'] for r in good if r['duration_s'] > 0])
    if len(durs):
        out(f'Video length: min {durs.min():.1f}s | median {np.median(durs):.1f}s | '
            f'mean {durs.mean():.1f}s | max {durs.max() / 60:.1f} min | '
            f'longer than 5 min: {(durs > 300).sum()} | longer than 10 min: {(durs > 600).sum()}')
        out(f'  (8 frames sampled from a {np.median(durs):.0f}s video are ~{np.median(durs) / 8:.0f}s apart; '
            f'the cache keeps {NUM_CACHE_FRAMES} frames, ~{np.median(durs) / NUM_CACHE_FRAMES:.1f}s apart)')
    for c_i, c in enumerate(classes):
        d = np.array([r['duration_s'] for r in good if r['label'] == c_i and r['duration_s'] > 0])
        if len(d):
            out(f'  {c:<16s} median {np.median(d):6.1f}s | max {d.max():7.1f}s')
    res, fps_c = {}, {}
    for r in good:
        res[f"{r['width']}x{r['height']}"] = res.get(f"{r['width']}x{r['height']}", 0) + 1
        fps_c[round(r['fps'])] = fps_c.get(round(r['fps']), 0) + 1
    out('Most common resolutions: ' + ', '.join(f'{k} ({v})' for k, v in
                                               sorted(res.items(), key=lambda kv: -kv[1])[:6]))
    out('Frame rates: ' + ', '.join(f'{k} fps ({v})' for k, v in
                                    sorted(fps_c.items(), key=lambda kv: -kv[1])[:5]))
    out('')

    if missing:
        out(f'MISSING ON DISK ({len(missing)}) — excluded from all splits:')
        for key in missing:
            out(f'  {key}')
        out('')
    if bad:
        out(f'UNREADABLE VIDEOS ({len(bad)}) — excluded from all splits:')
        for r in bad:
            out(f"  {r['video_path']}  ->  {r['error']}")
        out('')
    if ambiguous:
        out(f'File names found more than once ({len(ambiguous)}) — the copy used:')
        for key, n, path in ambiguous[:50]:
            out(f'  {key}  ({n} copies)  ->  {path}')
        out('')

    def tag(r):
        return r['key']

    for k, info in fold_info.items():
        if info['cross']:
            out(f'Fold {k}: TRAIN <-> TEST near-duplicate videos ({len(info["cross"])}):')
            out('  (same scene/camera in train and test = possible leakage; the official test list is kept)')
            for a, b, m in info['cross'][:40]:
                out(f'  {tag(a)}  ~  {tag(b)}   [matching frames: {m}]')
    conflicts = [(good[i], good[j]) for i, j, _ in all_pairs if good[i]['label'] != good[j]['label']]
    out(f'Near-duplicate videos with DIFFERENT labels: {len(conflicts)}')
    for a, b in conflicts[:50]:
        out(f'  {tag(a)}  ~  {tag(b)}')
    out('')
    out(f'Split lists written to: {split_dir}/fold1 .. fold{NUM_FOLDS}  (+ classes.txt)')
    out(f'Frame cache: {cache_dir}  (newly decoded: {sum(1 for r in good if not r["cached"])}, '
        f'reused: {sum(1 for r in good if r["cached"])})')
    out(f'Total time: {time.time() - t0:.0f}s')

    lost = len(missing) + len(bad)
    failed = (not good or lost > MISSING_LIMIT * len(entries)
              or any(not info['rows']['val'] or not info['rows']['test'] for info in fold_info.values()))
    if failed:
        out(f'PREP FAILED — {lost} of {len(entries)} listed videos missing/unreadable '
            f'(limit {MISSING_LIMIT:.0%}), or a fold has no val/test videos. Training will not start.')
    else:
        out('PREP DONE')

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(log_dir, 'prep_report.txt'), 'w') as f:
        f.write(report + '\n')
    if failed:
        sys.exit(1)          # non-zero exit -> the dependent training job will not start


if __name__ == '__main__':
    main()
