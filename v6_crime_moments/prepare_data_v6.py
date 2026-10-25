"""
V6 data preparation — UCF-Crime CRIME MOMENTS (run ONCE through PBS before train_v6.py)
=======================================================================================
Task: "a crime is happening right now — which one?"  13 crime classes (no Normal).

1. Videos: every crime video under the dataset root (Anomaly-Videos-Part-1..4), 950 in total.
2. Split (official UCF-Crime anomaly-detection split):
       test  = the 140 crime videos of Temporal_Anomaly_Annotation.txt (the official test
               videos), cut to their OFFICIAL crime window (start/end frames given by the authors)
       train = all other crime videos (~810), cut to the crime window found in the UCA
               annotations (Yuan et al., CVPR 2024: start/end times of every event in 1,854
               UCF-Crime videos). UCA describes every event in plain sentences without saying
               which one is the crime, so the crime sentences are picked with a keyword list per
               class (CRIME_WORDS below); their time spans are merged and padded by 1 s.
               If no sentence matches (or the video is not in UCA): --unmatched whole (default)
               uses the whole video, --unmatched skip leaves it out.
       val   = 15% of train, stratified per class, fixed seed (picks the best epoch).
3. Decodes each video ONCE and stores 32 frames spread evenly over its crime window
   (320x240, JPEG q95) in frame_cache/. Training takes 8 of them; the test looks at all 32.
4. Writes splits/train.csv, val.csv, test.csv, classes.txt, crime_windows.csv (every window
   with the sentences that produced it, for spot-checking) and logs/prep_report.txt
   (counts, window lengths, how well the keyword windows match the official windows on the
   test videos, unreadable videos, near-duplicates).

Run:  qsub run_prepare_v6.sh   (from ~/vlil_40/v6_crime_moments)
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
EXP_DIR          = os.path.expanduser('~/vlil_40/v6_crime_moments')
UCA_DIR          = os.path.join(EXP_DIR, 'uca_annotations')   # UCFCrime_Train/Val/Test.json
ANNOTATION_FILE  = 'Temporal_Anomaly_Annotation.txt'           # searched for under DATA_ROOT
NUM_CACHE_FRAMES = 32          # frames stored per video, spread over the crime window
MAX_SHORT_SIDE   = 256         # bigger videos are shrunk to this; 320x240 UCF-Crime stays native
JPEG_QUALITY     = 95
MERGE_GAP_S      = 3.0         # crime sentences closer than this are merged into one window
PAD_S            = 1.0         # seconds added before and after each UCA crime window
VAL_FRACTION     = 0.15
SEED             = 42
DUP_HAMMING      = 6
DUP_MIN_MATCHES  = 3           # V5 used 2: too many false alarms on dark / similar CCTV frames
HASH_AT          = tuple(int((k + 0.5) * NUM_CACHE_FRAMES / 4) for k in range(4))   # 4, 12, 20, 28
VIDEO_EXTS       = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}

CLASSES = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting',
           'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

# Words that mark a UCA sentence as describing the crime (matched on lower-case text).
# Checked on 2026-10-17: 816 of 944 UCA crime videos get a window; on the 124 official test
# videos the keyword windows cover 70% of the official crime window on average (IoU 0.35).
CRIME_WORDS = {
    'Abuse':         r'\b(hit|beat|kick|slap|punch|push|pull|drag|throw|threw|thrown|chok|strangl|attack|abus|bull|strik|struck|whip|grab|pinch|smash|knock|fell|fall|stomp|trampl|bit)',
    'Arrest':        r'\b(arrest|handcuff|cuff|police|officer|cop|detain|subdu|restrain|pinn?ed|pin\b|tackl|press|held down|hold down|controll|search)',
    'Arson':         r'\b(fire|burn|flame|ignit|lighter|light|gasoline|petrol|smok|pour|blaz|torch|match)',
    'Assault':       r'\b(hit|beat|kick|punch|attack|assault|slap|push|knock|stab|fight|strik|struck|throw|threw|grab|chok|stomp|fell|fall)',
    'Burglary':      r'\b(break|broke|smash|pry|pried|climb|crowbar|rummag|ransack|steal|stole|theft|burglar|sneak|forc|kick|shatter|search|took|take)',
    'Explosion':     r'\b(explo|blast|burst|fire|flame|smok|bomb|flash|light)',
    'Fighting':      r'\b(fight|punch|hit|kick|beat|wrestl|brawl|scuffl|push|attack|throw|threw|grab|slap|strik|struck|knock|fell|fall|chas|tussl)',
    'RoadAccidents': r'\b(hit|crash|collid|collision|knock|overturn|roll|accident|struck|strik|ran over|run over|rear|flip|slam|bump|smash|fell|fall|flew|thrown)',
    'Robbery':       r'\b(rob|gun|knife|pistol|weapon|threat|point|money|cash|grab|steal|stole|snatch|mask|dagger|machete|hold up|held up)',
    'Shooting':      r'\b(shoot|shot|gun|fire|firing|pistol|rifle|bullet|weapon)',
    'Shoplifting':   r'\b(steal|stole|pocket|hid|hide|conceal|slip|stuff|tuck|shoplift|put .* (in|into) (his|her|the) (bag|pocket|clothes|pants|coat|jacket))',
    'Stealing':      r'\b(steal|stole|took|take|grab|snatch|carr|drove away|drive away|ran away|rode away|theft|thief|pick|unlock|pry|cut)',
    'Vandalism':     r'\b(smash|break|broke|kick|throw|threw|damag|destroy|hit|spray|graffiti|vandal|shatter|knock|overturn|tore|tear|punch|beat|scratch)',
}
NAME_RE = re.compile(r'^(' + '|'.join(CLASSES) + r')\d+_x264$', re.IGNORECASE)


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_') or 'x'


# ============================================================
# CRIME WINDOWS
# ============================================================
def merge(ivs, gap=0.0):
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def uca_window(entry, cls):
    """Crime window (list of [start_s, end_s]) from the UCA sentences, or ([], []) if none match."""
    pat = re.compile(CRIME_WORDS[cls])
    ivs, hits = [], []
    for (s, e), sent in zip(entry.get('timestamps', []), entry.get('sentences', [])):
        s, e = float(s), float(e)
        if e > s and pat.search(sent.lower()):
            ivs.append((s, e))
            hits.append(sent.strip())
    if not ivs:
        return [], []
    win = merge(ivs, MERGE_GAP_S)
    dur = float(entry.get('duration') or 0) or None
    padded = [[max(0.0, s - PAD_S), (min(dur, e + PAD_S) if dur else e + PAD_S)] for s, e in win]
    return merge(padded), hits


def total(ivs):
    return sum(e - s for s, e in ivs)


def overlap(a, b):
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


# ============================================================
# VIDEO DECODING (runs inside worker processes)
# ============================================================
def _even(idx, k):
    idx = np.asarray(idx, dtype=np.int64)
    return idx[((np.arange(k) + 0.5) * len(idx) / k).astype(np.int64)]


def _targets(n, k, segs_f):
    """k frame indices spread over the window segments (frame numbers), or the whole video."""
    idx = []
    for s, e in segs_f or []:
        s = max(0, min(int(s), n - 1))
        e = max(s, min(int(e), n - 1))
        idx.extend(range(s, e + 1))
    if not idx:
        idx = range(max(n, 1))
    return _even(list(idx), k)


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


def read_window(path, job, k):
    """Returns frames, n_frames, fps, window used (in frames)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError('cannot open')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if n <= 0 or n > 10_000_000:
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        cap = cv2.VideoCapture(path)
    f = fps if fps > 0 else 30.0
    if job.get('window_f'):
        segs = job['window_f']
    elif job.get('window_s'):
        segs = [[s * f, e * f] for s, e in job['window_s']]
    else:
        segs = None
    frames, n_read = _grab_at(cap, _targets(n, k, segs))
    cap.release()
    if frames is None:                        # header over-reported the length: retry with the real one
        if n_read == 0:
            raise IOError('no decodable frames')
        n = n_read
        cap = cv2.VideoCapture(path)
        frames, _ = _grab_at(cap, _targets(n, k, segs))
        cap.release()
        if frames is None:
            raise IOError('frames disappeared on second pass')
    used = [[max(0, min(int(s), n - 1)), max(0, min(int(e), n - 1))] for s, e in (segs or [[0, n - 1]])]
    return frames, n, fps, used


def dhash64(gray):
    import cv2
    if gray.std() < 8:
        return None
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view('>u8')[0])


def _init_worker():
    import cv2
    cv2.setNumThreads(0)


def process_video(job):
    import cv2
    result = dict(job)
    try:
        if os.path.exists(job['cache_path']):
            with np.load(job['cache_path']) as d:
                meta = json.loads(str(d['meta']))
            result.update(meta)
            result['ok'], result['cached'] = True, True
            return result

        frames, n, fps, used = read_window(job['video_path'], job, NUM_CACHE_FRAMES)
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
        f = fps if fps > 0 else 30.0
        meta = {
            'n_frames': int(n), 'fps': round(fps, 3), 'width': int(w0), 'height': int(h0),
            'duration_s': round(n / f, 2),
            'window_frames': used, 'window_s': round(sum(e - s + 1 for s, e in used) / f, 2),
            'dhash': [dhash64(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)) for i in HASH_AT],
        }
        os.makedirs(os.path.dirname(job['cache_path']), exist_ok=True)
        tmp = job['cache_path'][:-4] + '.tmp.npz'
        np.savez(tmp, jpg=np.frombuffer(b''.join(bufs), dtype=np.uint8),
                 offsets=np.asarray(offsets, dtype=np.int64), meta=np.array(json.dumps(meta)))
        os.replace(tmp, job['cache_path'])
        result.update(meta)
        result['ok'], result['cached'] = True, False
    except Exception as e:  # noqa: BLE001 - report every failure, never crash the job
        result['ok'] = False
        result['error'] = f'{type(e).__name__}: {e}'
    return result


# ============================================================
# DUPLICATE DETECTION (report + keep groups on one side of train/val)
# ============================================================
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(items):
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
        x = H[i][None, :, None] ^ H[i + 1:][:, None, :]
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
    ap.add_argument('--uca-dir', default=UCA_DIR)
    ap.add_argument('--unmatched', choices=['whole', 'skip'], default='whole',
                    help='train videos without a UCA crime window: use the whole video, or leave out')
    ap.add_argument('--procs', type=int, default=int(os.environ.get('NCPUS', '4')))
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = os.path.join(args.exp_dir, 'frame_cache')
    split_dir = os.path.join(args.exp_dir, 'splits')
    log_dir = os.path.join(args.exp_dir, 'logs')
    for d in (cache_dir, split_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    print('=' * 70)
    print('V6 DATA PREPARATION — UCF-Crime crime moments (13 classes)')
    print('=' * 70)
    print(f'  Dataset root : {args.data_root}')
    print(f'  UCA folder   : {args.uca_dir}')
    print(f'  Output       : {args.exp_dir}')
    print(f'  Unmatched    : {args.unmatched} | workers: {args.procs}')

    # ---------- videos on disk + official annotation file ----------
    videos, ann_path = {}, None
    for dirpath, _, filenames in os.walk(args.data_root):
        for fn in filenames:
            if fn == ANNOTATION_FILE and ann_path is None:
                ann_path = os.path.join(dirpath, fn)
            stem, ext = os.path.splitext(fn)
            if fn.startswith('._') or ext.lower() not in VIDEO_EXTS:
                continue
            m = NAME_RE.match(stem)
            if not m:
                continue
            cls = next(c for c in CLASSES if c.lower() == m.group(1).lower())
            key = stem.lower()
            path = os.path.join(dirpath, fn)
            # same file stored twice: prefer the copy inside a folder named after the class
            if key not in videos or cls.lower() in [p.lower() for p in dirpath.split(os.sep)]:
                videos[key] = {'stem': stem, 'cls': cls, 'path': path}
    print(f'  Crime videos on disk: {len(videos)}')
    if not ann_path:
        sys.exit(f'  ERROR: {ANNOTATION_FILE} not found under {args.data_root}')
    print(f'  Official test annotation: {ann_path}')

    official = {}
    with open(ann_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            nums = [int(x) for x in p[2:6]]
            segs = [[s, e] for s, e in (nums[0:2], nums[2:4]) if s >= 0 and e >= s]
            if segs:
                official[os.path.splitext(p[0])[0].lower()] = segs

    # ---------- UCA ----------
    uca = {}
    for fn in ('UCFCrime_Train.json', 'UCFCrime_Val.json', 'UCFCrime_Test.json'):
        path = os.path.join(args.uca_dir, fn)
        if not os.path.exists(path):
            sys.exit(f'  ERROR: UCA file missing: {path}')
        with open(path) as f:
            for k, v in json.load(f).items():
                uca[k.lower()] = v
    print(f'  UCA videos: {len(uca)}')

    # ---------- jobs ----------
    jobs, skipped, missing_test = [], [], []
    for key in sorted(official):
        if key not in videos:
            missing_test.append(key)
    for key, v in sorted(videos.items()):
        cls, label = v['cls'], CLASSES.index(v['cls'])
        job = {'key': v['stem'], 'class_name': cls, 'label': label, 'video_path': v['path'],
               'window_f': None, 'window_s': None, 'window_source': '', 'crime_sentences': '',
               'uca_window_s': None}
        if key in uca:
            win_s, hits = uca_window(uca[key], cls)
            job['uca_window_s'] = win_s or None
        else:
            win_s, hits = [], []
        if key in official:
            job['source'] = 'test'
            job['window_f'] = official[key]
            job['window_source'] = 'official'
        else:
            job['source'] = 'train'
            if win_s:
                job['window_s'] = win_s
                job['window_source'] = 'uca'
                job['crime_sentences'] = ' || '.join(h[:160] for h in hits[:4])
            elif args.unmatched == 'whole':
                job['window_source'] = 'whole (not in UCA)' if key not in uca else 'whole (no crime sentence)'
            else:
                skipped.append((v['stem'], 'not in UCA' if key not in uca else 'no crime sentence'))
                continue
        job['cache_path'] = os.path.join(cache_dir, f'{label:02d}_{cls}',
                                         f"{safe_name(v['stem'])}_{hashlib.md5(v['path'].encode()).hexdigest()[:8]}.npz")
        jobs.append(job)
    print(f"  To decode: {len(jobs)} (test {sum(j['source'] == 'test' for j in jobs)}, "
          f"train pool {sum(j['source'] == 'train' for j in jobs)}) | skipped: {len(skipped)}", flush=True)

    # ---------- decode ----------
    results = []
    with mp.Pool(args.procs, initializer=_init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(process_video, jobs, chunksize=1), 1):
            results.append(r)
            if k % 50 == 0 or k == len(jobs):
                print(f'  decoded {k}/{len(jobs)}  ({time.time() - t0:.0f}s)', flush=True)
    results.sort(key=lambda r: (r['source'], r['label'], r['key']))
    good = [r for r in results if r['ok']]
    bad = [r for r in results if not r['ok']]

    # ---------- duplicates ----------
    near_pairs = near_duplicate_pairs(good)
    uf = UnionFind(len(good))
    for i, j, _ in near_pairs:
        uf.union(i, j)
    for i, r in enumerate(good):
        r['group_id'] = uf.find(i)

    # ---------- train / val split ----------
    rng = random.Random(SEED)
    train_pool = [r for r in good if r['source'] == 'train']
    test_rows = [r for r in good if r['source'] == 'test']
    groups = {}
    for r in train_pool:
        groups.setdefault(r['group_id'], []).append(r)
    group_list = sorted(groups.values(), key=lambda g: g[0]['key'])
    rng.shuffle(group_list)
    n_per_class = np.bincount(np.array([r['label'] for r in train_pool], dtype=np.int64), minlength=len(CLASSES))
    target_val = [max(1, int(round(VAL_FRACTION * n))) if n > 1 else 0 for n in n_per_class]
    val_count = [0] * len(CLASSES)
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
    for r in test_rows:
        r['split'] = 'test'
    rows_by_split = {s: [r for r in good if r.get('split') == s] for s in ('train', 'val', 'test')}

    fields = ['split', 'class_name', 'label', 'video_path', 'cache_path', 'window_source', 'window_s',
              'duration_s', 'n_frames', 'fps', 'width', 'height', 'group_id']
    for s, rows in rows_by_split.items():
        with open(os.path.join(split_dir, f'{s}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(split_dir, 'classes.txt'), 'w') as f:
        for i, c in enumerate(CLASSES):
            f.write(f'{i}\t{c}\n')
    with open(os.path.join(split_dir, 'crime_windows.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'class', 'video', 'window_source', 'window_frames', 'window_s', 'duration_s',
                    'crime_sentences'])
        for s in ('train', 'val', 'test'):
            for r in rows_by_split[s]:
                w.writerow([s, r['class_name'], r['key'], r['window_source'], r['window_frames'],
                            r['window_s'], r['duration_s'], r['crime_sentences']])

    # ---------- report ----------
    lines = []
    out = lines.append
    out('=' * 70)
    out('V6 DATASET REPORT — UCF-Crime crime moments')
    out('=' * 70)
    out(f'Crime videos on disk: {len(videos)} | decoded OK: {len(good)} | unreadable: {len(bad)} | '
        f'skipped (--unmatched skip): {len(skipped)}')
    if missing_test:
        out(f'Official test videos NOT found on disk ({len(missing_test)}): {missing_test}')
    out('')
    out(f"{'Class':<15s}{'Train':>7s}{'Val':>6s}{'Test':>6s}   {'train windows: UCA / whole':<28s}"
        f"{'median window (s)':>18s}")
    out('-' * 82)
    for i, c in enumerate(CLASSES):
        n = [sum(1 for r in rows_by_split[s] if r['label'] == i) for s in ('train', 'val', 'test')]
        pool = [r for r in train_pool if r['label'] == i]
        n_uca = sum(1 for r in pool if r['window_source'] == 'uca')
        wins = [r['window_s'] for r in pool if r['window_source'] == 'uca']
        twin = [r['window_s'] for r in test_rows if r['label'] == i]
        med = f"{np.median(wins):.1f} / {np.median(twin):.1f}" if wins and twin else '-'
        out(f'{c:<15s}{n[0]:>7d}{n[1]:>6d}{n[2]:>6d}   {n_uca:>6d} / {len(pool) - n_uca:<19d}{med:>18s}')
    out('-' * 82)
    out(f"{'TOTAL':<15s}{len(rows_by_split['train']):>7d}{len(rows_by_split['val']):>6d}"
        f"{len(rows_by_split['test']):>6d}")
    out('(median window: train UCA windows / test official windows)')
    out('')

    durs = np.array([r['duration_s'] for r in train_pool])
    wins = np.array([r['window_s'] for r in train_pool if r['window_source'] == 'uca'])
    if len(durs) and len(wins):
        out(f'Train videos: median length {np.median(durs):.1f}s -> median crime window {np.median(wins):.1f}s '
            f'(the model now sees ~{np.median(wins) / np.median(durs):.0%} of the video, around the crime)')
    tw = np.array([r['window_s'] for r in test_rows])
    if len(tw):
        out(f'Test windows (official): median {np.median(tw):.1f}s | min {tw.min():.1f}s | max {tw.max():.1f}s')

    # keyword windows vs official windows on the test videos (quality check of the train windows)
    ious, covs = [], []
    for r in test_rows:
        if r.get('uca_window_s') and r['fps'] > 0:
            off = [[s / r['fps'], e / r['fps']] for s, e in r['window_frames']]
            inter = overlap(r['uca_window_s'], off)
            union = total(r['uca_window_s']) + total(off) - inter
            ious.append(inter / max(union, 1e-6))
            covs.append(inter / max(total(off), 1e-6))
    if ious:
        out(f'Quality check on {len(ious)} test videos (keyword window from UCA vs official window): '
            f'mean IoU {np.mean(ious):.2f}, the keyword window covers {np.mean(covs):.0%} of the official crime')
    out('')
    out('One example per class (train, UCA crime sentences used):')
    for c in CLASSES:
        ex = next((r for r in train_pool if r['class_name'] == c and r['window_source'] == 'uca'), None)
        if ex:
            out(f"  {c:<14s} {ex['key']} [{ex['window_s']}s of {ex['duration_s']}s]: {ex['crime_sentences'][:200]}")
    out('')
    if bad:
        out(f'UNREADABLE VIDEOS ({len(bad)}) — excluded:')
        for r in bad:
            out(f"  {r['video_path']}  ->  {r['error']}")
        out('')
    if skipped:
        out(f'SKIPPED train videos ({len(skipped)}): ' + ', '.join(f'{k} ({why})' for k, why in skipped[:60]))
        out('')
    cross = [(good[i], good[j], m) for i, j, m in near_pairs
             if {good[i]['split'], good[j]['split']} >= {'test'} and good[i]['split'] != good[j]['split']]
    out(f'TRAIN/VAL <-> TEST near-duplicate videos (>= {DUP_MIN_MATCHES} matching frames): {len(cross)}')
    for a, b, m in cross[:40]:
        out(f"  {a['split']}/{a['key']}  ~  {b['split']}/{b['key']}   [matching frames: {m}]")
    out('')
    out(f'Split lists: {split_dir} (train.csv, val.csv, test.csv, classes.txt, crime_windows.csv)')
    out(f'Frame cache: {cache_dir}  (newly decoded: {sum(1 for r in good if not r["cached"])}, '
        f'reused: {sum(1 for r in good if r["cached"])})')
    out(f'Total time: {time.time() - t0:.0f}s')
    failed = (not rows_by_split['train'] or not rows_by_split['val'] or len(rows_by_split['test']) < 100)
    out('PREP FAILED — not enough train/val/test videos. Training will not start.' if failed else 'PREP DONE')

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(log_dir, 'prep_report.txt'), 'w') as f:
        f.write(report + '\n')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
