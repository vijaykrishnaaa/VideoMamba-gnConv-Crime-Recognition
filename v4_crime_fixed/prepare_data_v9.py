"""
V4 data preparation (run ONCE through PBS before train_crime_v9.py)
====================================================================
1. Scans  dataset_new_split_clean/train  and  dataset_new_split_clean/test.
2. Decodes every video ONCE and stores 32 frames spread over the WHOLE clip
   (short side 256 px, JPEG q95) in frame_cache/. Training then never touches
   the raw videos, which removes the ~1.6 s/video OpenCV decoding cost and the
   old "first 300 frames only" limit.
3. Writes the split lists (the "excel sheets"):
       splits/train.csv  splits/val.csv  splits/test.csv  splits/classes.txt
   val = 15% of the train folder, stratified per class, fixed seed. Clips that
   look like near-duplicates of each other are kept on the same side.
   test = the test folder, untouched.
4. Writes logs/prep_report.txt: per-class counts, clip lengths, resolutions,
   unreadable videos, and train/test near-duplicate clips (possible leakage).

Run:  qsub v4_crime_fixed/run_prepare_v9.sh
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
DATA_ROOT        = os.path.expanduser('~/final_year_project_29/dataset_new_split_clean')
EXP_DIR          = os.path.expanduser('~/vlil_40/v4_crime_fixed')
NUM_CACHE_FRAMES = 32          # frames stored per video (training samples 8 of them)
SHORT_SIDE       = 256         # stored frame size (short side), aspect ratio kept
JPEG_QUALITY     = 95
VAL_FRACTION     = 0.15        # part of the train folder used for model selection
SEED             = 42
DUP_HAMMING      = 6           # dHash bit distance that counts as "same frame"
DUP_MIN_MATCHES  = 2           # matching frame pairs needed to call two clips duplicates
VIDEO_EXTS       = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_') or 'x'


# ============================================================
# VIDEO DECODING (runs inside worker processes)
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


def process_video(job):
    """Decode one video into the cache (or reuse an existing cache file)."""
    import cv2
    path, cache_path = job['video_path'], job['cache_path']
    result = dict(job)
    try:
        if os.path.exists(cache_path):
            with np.load(cache_path) as d:
                meta = json.loads(str(d['meta']))
            result.update(meta)
            result['ok'] = True
            result['cached'] = True
            return result

        frames, n_frames, fps = read_video(path, NUM_CACHE_FRAMES)
        h0, w0 = frames[0].shape[:2]
        scale = SHORT_SIDE / min(h0, w0)
        new_wh = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

        bufs, offsets, hashes = [], [0], []
        for i, fr in enumerate(frames):
            if fr.shape[:2] != (h0, w0):
                fr = cv2.resize(fr, (w0, h0))
            fr = cv2.resize(fr, new_wh, interpolation=interp)
            ok, enc = cv2.imencode('.jpg', fr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise IOError('jpeg encode failed')
            bufs.append(enc.tobytes())
            offsets.append(offsets[-1] + len(bufs[-1]))
            if i in (4, 12, 20, 28):
                hashes.append(dhash64(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))

        meta = {
            'n_frames': int(n_frames), 'fps': round(fps, 3),
            'width': int(w0), 'height': int(h0),
            'duration_s': round(n_frames / fps, 2) if fps > 0 else -1.0,
            'dhash': hashes, 'fingerprint': file_fingerprint(path),
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path[:-4] + '.tmp.npz'
        np.savez(tmp, jpg=np.frombuffer(b''.join(bufs), dtype=np.uint8),
                 offsets=np.asarray(offsets, dtype=np.int64), meta=np.array(json.dumps(meta)))
        os.replace(tmp, cache_path)
        result.update(meta)
        result['ok'] = True
        result['cached'] = False
    except Exception as e:  # noqa: BLE001 - report every failure, never crash the job
        result['ok'] = False
        result['error'] = f'{type(e).__name__}: {e}'
    return result


# ============================================================
# DUPLICATE DETECTION
# ============================================================
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(items):
    """Pairs (i, j) of clips sharing >= DUP_MIN_MATCHES near-identical frames."""
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
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--exp-dir', default=EXP_DIR)
    ap.add_argument('--procs', type=int, default=int(os.environ.get('NCPUS', '4')))
    args = ap.parse_args()

    t0 = time.time()
    train_src = os.path.join(args.data_root, 'train')
    test_src = os.path.join(args.data_root, 'test')
    cache_dir = os.path.join(args.exp_dir, 'frame_cache')
    split_dir = os.path.join(args.exp_dir, 'splits')
    log_dir = os.path.join(args.exp_dir, 'logs')
    for d in (cache_dir, split_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    print('=' * 70)
    print('V4 DATA PREPARATION — frame cache + train/val/test split lists')
    print('=' * 70)
    print(f'  Dataset root : {args.data_root}')
    print(f'  Output       : {args.exp_dir}')
    print(f'  Workers      : {args.procs}')
    for d in (train_src, test_src):
        if not os.path.isdir(d):
            sys.exit(f'  ERROR: folder not found: {d}')

    # Same ordering as V1-V3 (Python's default sort), so label numbers match old runs
    classes = sorted(d for d in os.listdir(train_src) if os.path.isdir(os.path.join(train_src, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f'  Classes ({len(classes)}): {classes}')

    jobs, skipped_files = [], []
    for split_src, root in (('train', train_src), ('test', test_src)):
        for c in sorted(os.listdir(root)):
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir):
                continue
            if c not in class_to_idx:
                print(f'  [WARN] {split_src}/{c} is not a train class — skipped')
                continue
            for fname in sorted(os.listdir(cdir)):
                vpath = os.path.join(cdir, fname)
                if os.path.splitext(fname)[1].lower() not in VIDEO_EXTS:
                    skipped_files.append(vpath)
                    continue
                label = class_to_idx[c]
                rel = f'{split_src}/{c}/{fname}'
                stem = safe_name(os.path.splitext(fname)[0])[:60]
                cpath = os.path.join(cache_dir, split_src, f'{label:02d}_{safe_name(c)}',
                                     f'{stem}_{hashlib.md5(rel.encode()).hexdigest()[:8]}.npz')
                jobs.append({'source': split_src, 'class_name': c, 'label': label,
                             'video_path': vpath, 'cache_path': cpath})
    print(f'  Videos found : {len(jobs)}  (non-video files skipped: {len(skipped_files)})')

    # ---------- decode ----------
    results = []
    with mp.Pool(args.procs, initializer=_init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(process_video, jobs, chunksize=1), 1):
            results.append(r)
            if k % 50 == 0 or k == len(jobs):
                print(f'  decoded {k}/{len(jobs)}  ({time.time() - t0:.0f}s)', flush=True)
    results.sort(key=lambda r: (r['source'], r['label'], r['video_path']))
    good = [r for r in results if r['ok']]
    bad = [r for r in results if not r['ok']]

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

    # ---------- train / val split (stratified, duplicate groups kept together) ----------
    rng = random.Random(SEED)
    train_pool = [r for r in good if r['source'] == 'train']
    test_rows = [r for r in good if r['source'] == 'test']
    groups = {}
    for r in train_pool:
        groups.setdefault(r['group_id'], []).append(r)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    n_per_class = np.bincount(np.array([r['label'] for r in train_pool], dtype=np.int64),
                              minlength=len(classes))
    target_val = [max(1, int(round(VAL_FRACTION * n))) if n > 1 else 0 for n in n_per_class]
    val_count = [0] * len(classes)
    for g in group_list:
        need = {}
        for r in g:
            need[r['label']] = need.get(r['label'], 0) + 1
        fits = all(val_count[c] + k <= target_val[c] for c, k in need.items())
        useful = any(val_count[c] < target_val[c] for c in need)
        split = 'val' if (fits and useful) else 'train'
        for r in g:
            r['split'] = split
        if split == 'val':
            for c, k in need.items():
                val_count[c] += k
    for r in test_rows:
        r['split'] = 'test'

    # ---------- write split CSVs ----------
    fields = ['split', 'class_name', 'label', 'video_path', 'cache_path', 'n_frames', 'fps',
              'width', 'height', 'duration_s', 'group_id']
    rows_by_split = {s: [r for r in good if r['split'] == s] for s in ('train', 'val', 'test')}
    for s, rows in rows_by_split.items():
        with open(os.path.join(split_dir, f'{s}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(split_dir, 'classes.txt'), 'w') as f:
        for i, c in enumerate(classes):
            f.write(f'{i}\t{c}\n')

    # ---------- report ----------
    lines = []
    out = lines.append
    out('=' * 70)
    out('V4 DATASET REPORT')
    out('=' * 70)
    out(f'Dataset root: {args.data_root}')
    out(f'Videos found: {len(jobs)} | decoded OK: {len(good)} | unreadable: {len(bad)} | '
        f'non-video files skipped: {len(skipped_files)}')
    out('')
    out(f"{'Class':<18s}{'Train':>8s}{'Val':>8s}{'Test':>8s}{'Total':>8s}")
    out('-' * 50)
    for i, c in enumerate(classes):
        n = [sum(1 for r in rows_by_split[s] if r['label'] == i) for s in ('train', 'val', 'test')]
        out(f'{c:<18s}{n[0]:>8d}{n[1]:>8d}{n[2]:>8d}{sum(n):>8d}')
    out('-' * 50)
    tot = [len(rows_by_split[s]) for s in ('train', 'val', 'test')]
    out(f"{'TOTAL':<18s}{tot[0]:>8d}{tot[1]:>8d}{tot[2]:>8d}{sum(tot):>8d}")
    out('')

    for s in ('train', 'val', 'test'):
        durs = np.array([r['duration_s'] for r in rows_by_split[s] if r['duration_s'] > 0])
        frs = np.array([r['n_frames'] for r in rows_by_split[s]])
        if len(durs):
            out(f'{s:<5s} clip length: min {durs.min():.1f}s | median {np.median(durs):.1f}s | '
                f'max {durs.max():.1f}s | clips longer than 300 frames: {(frs > 300).sum()}/{len(frs)}')
    res = {}
    for r in good:
        res[f"{r['width']}x{r['height']}"] = res.get(f"{r['width']}x{r['height']}", 0) + 1
    out('Most common resolutions: ' + ', '.join(f'{k} ({v})' for k, v in
                                               sorted(res.items(), key=lambda kv: -kv[1])[:6]))
    out('')

    if bad:
        out(f'UNREADABLE VIDEOS ({len(bad)}) — excluded from all splits:')
        for r in bad:
            out(f"  {r['video_path']}  ->  {r['error']}")
        out('')

    def tag(r):
        return f"{r['source']}/{r['class_name']}/{os.path.basename(r['video_path'])}"

    cross = [(good[i], good[j], m) for i, j, m in near_pairs if good[i]['source'] != good[j]['source']]
    cross += [(good[i], good[j], 'exact') for i, j in exact_pairs if good[i]['source'] != good[j]['source']]
    out(f'TRAIN <-> TEST near-duplicate clips: {len(cross)}')
    if cross:
        out('  (same scene in both folders = possible test leakage; show these to your guide)')
        for a, b, m in cross[:200]:
            out(f'  {tag(a)}  ~  {tag(b)}   [matching frames: {m}]')
    conflicts = [(good[i], good[j]) for i, j, _ in near_pairs if good[i]['label'] != good[j]['label']]
    out(f'Near-duplicate clips with DIFFERENT labels: {len(conflicts)}')
    for a, b in conflicts[:100]:
        out(f'  {tag(a)}  ~  {tag(b)}')
    multi = sum(1 for g in groups.values() if len(g) > 1)
    out(f'Duplicate groups inside the train folder (kept on one side of train/val): {multi}')
    out('')
    out(f'Split lists written to: {split_dir}')
    out(f'Frame cache: {cache_dir}  (newly decoded: {sum(1 for r in good if not r["cached"])}, '
        f'reused: {sum(1 for r in good if r["cached"])})')
    out(f'Total time: {time.time() - t0:.0f}s')
    out('PREP DONE' if good else 'PREP FAILED — no readable videos')

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(log_dir, 'prep_report.txt'), 'w') as f:
        f.write(report + '\n')
    if not good or not rows_by_split['val'] or not rows_by_split['test']:
        sys.exit(1)          # non-zero exit -> the dependent training job will not start


if __name__ == '__main__':
    main()
