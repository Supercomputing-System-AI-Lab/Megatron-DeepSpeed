#!/usr/bin/env python3
"""mem_peaks.py: the memory profiler's per-rank peak of one attempt, read from the files the attempt left behind, folded per pipeline
stage and written next to the model's prediction (the OOM rule of 2026-10-10: "the memory profiler runs on every attempt and its
per-rank peak is written into recovery.log next to the model's prediction").

WHERE THE NUMBERS COME FROM (all GiB; the busiest rank of a stage is what an out of memory is decided by):
  1. <attempt>/memsnap/peak_rank<r>.txt   the memsnap shim's peak writer (bin/memsnap/memsnap_boot.py): every 30 s each rank rewrites
                                            its never-reset torch.cuda.max_memory_allocated / max_memory_reserved; the last file holds the
                                            peak as of the last write, so for a rank that died it is at most 30 s old
  2. <attempt>/ranks/rank_<r>.log           Megatron's own line "[Rank r] (after N iterations) memory (GB) | allocated: .. | max allocated: ..
                                            | reserved: .. | max reserved: .." (GB there means 1024^3), printed once per attempt by every rank
                                            after the first logged iteration; and the shim's "[memsnap] rank r at exit: max allocated X GiB,
                                            max reserved Y GiB" line at a normal exit (the detectors' os._exit skips it)
  3. <attempt>/exit/r<r>.json               the exit record of a detector: 'hbm_gb' = torch.cuda.memory_allocated at the failure (1e9 bytes),
                                            and for an out of memory the driver's own sentence "X GiB is allocated by PyTorch"
With the shim on, the max_memory_* counters are never reset (DeepSpeed's see_memory_usage and the TIMING resets no longer touch them),
so every source is the peak since the process started. Without the shim, the rank log's "max allocated" may be a peak since the last
reset and is labelled so.
Usage: mem_peaks.py --attempt-dir attempts/attempt_0 --world 512 --pp 2 [--out data/mem_peaks_a0.tsv] [--pred "0:39.7/50.7/59.5;1:41.2/52.4/61.0"]
Prints one sentence, then one machine line PEAK_GIB=..;PEAK_RANK=..;PEAK_STAGE=..;PEAK_SRC=..;STAGES=s:gib:rank:src,...;N_RANKS=..
"""
import argparse, glob, json, os, re, sys

GIB = 1024 ** 3


def read_rank_log(path):
    """(max allocated GiB, max reserved GiB or None, source) from the rank log, or None. The log is filtered with grep first: a 512-rank
    attempt with NCCL_DEBUG=INFO leaves gigabytes of rank logs, and grep reads them at the disk's speed while Python would not."""
    best = None; best_res = None; src = None; shim = False
    try:
        import subprocess
        data = subprocess.run(['grep', '-a', '-E', r'max allocated: [0-9.]+|\[memsnap\]|memsnap_boot', path], capture_output=True, text=True, errors='replace').stdout
    except Exception:
        try:
            with open(path, 'rb') as fh:
                data = fh.read().decode('utf-8', 'replace')
        except OSError:
            return None
    for m in re.finditer(r'max allocated: ([0-9.]+)', data):
        v = float(m.group(1))
        if best is None or v > best:
            best = v; src = 'rank log "max allocated"'
    for m in re.finditer(r'\[memsnap\] rank \d+ at exit: max allocated ([0-9.]+) GiB, max reserved ([0-9.]+) GiB', data):
        v = float(m.group(1)); shim = True
        if best is None or v >= best:
            best = v; best_res = float(m.group(2)); src = 'memsnap exit line'
    if '[memsnap]' in data or 'memsnap_boot' in data:
        shim = True
    if best is None:
        return None
    return best, best_res, src + (' (never reset: the memsnap shim)' if shim else ' (reset by the trainer between layers: a lower bound)')


def read_peak_file(path):
    try:
        v = open(path).read().split()
        return float(v[0]), float(v[1]), 'memsnap peak file'
    except Exception:
        return None


def read_exit(path):
    try:
        r = json.load(open(path))
    except Exception:
        return None
    ev = r.get('evidence', ''); hb = None
    m = re.search(r'hbm_gb=([0-9.]+)', ev)
    if m:
        hb = float(m.group(1)) * 1e9 / GIB
    m2 = re.search(r'([0-9.]+) GiB is allocated by PyTorch', ev)
    if m2:
        hb = max(hb or 0, float(m2.group(1)))
    if hb is None:
        return None
    return hb, None, f"exit record (code {r.get('code')}, {r.get('class')}, step {r.get('step')}: the allocated memory at the failure)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt-dir', required=True); ap.add_argument('--world', type=int, required=True); ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--out', default=None); ap.add_argument('--pred', default='', help='the model\'s per-stage prediction "s:alloc/vram/line;..." (GiB)')
    a = ap.parse_args()
    per_stage = max(1, a.world // max(1, a.pp))
    peaks = {}   # rank -> (alloc, reserved, src)
    for f in glob.glob(os.path.join(a.attempt_dir, 'memsnap', 'peak_rank*.txt')):
        r = int(re.search(r'peak_rank(\d+)', f).group(1)); v = read_peak_file(f)
        if v: peaks[r] = v
    for f in glob.glob(os.path.join(a.attempt_dir, 'ranks', 'rank_*.log')):
        r = int(re.search(r'rank_(\d+)\.log', f).group(1)); v = read_rank_log(f)
        if v and (r not in peaks or v[0] > peaks[r][0]): peaks[r] = v
    for f in glob.glob(os.path.join(a.attempt_dir, 'exit', 'r*.json')):
        r = int(re.search(r'r(\d+)\.json', f).group(1)); v = read_exit(f)
        if v and (r not in peaks or v[0] > peaks[r][0]): peaks[r] = v
    pred = {}
    for item in [x for x in a.pred.split(';') if x.strip()]:
        try:
            s, rest = item.split(':'); al, vr, ln = rest.split('/'); pred[int(s)] = (float(al), float(vr), float(ln))
        except Exception:
            pass
    if not peaks:
        print(f'no memory peak was recorded in {a.attempt_dir} (no memsnap peak file, no rank-log memory line, no exit record with a memory figure).')
        print('PEAK_GIB=-;PEAK_RANK=-;PEAK_STAGE=-;PEAK_SRC=-;STAGES=;N_RANKS=0')
        if a.out:
            with open(a.out, 'w') as fh:
                fh.write('# mem_peaks: no data\nrank\tstage\tmax_allocated_gib\tmax_reserved_gib\tsource\n')
        return
    stages = {}
    for r, (al, res, src) in peaks.items():
        s = r // per_stage
        if s not in stages or al > stages[s][0]:
            stages[s] = (al, r, src, res)
    br = max(peaks, key=lambda r: peaks[r][0]); bal, bres, bsrc = peaks[br]
    parts = []
    for s in sorted(stages):
        al, r, src, res = stages[s]
        p = pred.get(s)
        parts.append(f"stage {s}: {al:.1f} GiB allocated on rank {r}" + (f" ({res:.1f} reserved)" if res else '') + (f", the model predicted {p[0]:.1f} allocated / {p[2]:.1f} out-of-memory line" if p else '') + f" [{src}]")
    sent = (f"the measured peak of this attempt is {bal:.1f} GiB allocated on rank {br} (stage {br // per_stage}); per stage: " + '; '.join(parts)
            + f"; {len(peaks)} of {a.world} ranks reported a figure.")
    print(sent)
    print(f"PEAK_GIB={bal:.1f};PEAK_RANK={br};PEAK_STAGE={br // per_stage};PEAK_SRC={bsrc.replace(';', ',')};STAGES=" +
          ','.join(f"{s}:{stages[s][0]:.1f}:{stages[s][1]}:{stages[s][2].split(' (')[0]}" for s in sorted(stages)) + f";N_RANKS={len(peaks)}")
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write(f'# mem_peaks of {a.attempt_dir}: the per-rank peak (GiB) and its source; the model prediction per stage: {a.pred or "-"}\n')
            fh.write('rank\tstage\tmax_allocated_gib\tmax_reserved_gib\tsource\n')
            for r in sorted(peaks):
                al, res, src = peaks[r]
                fh.write(f'{r}\t{r // per_stage}\t{al:.3f}\t{res if res is None else round(res, 3)}\t{src}\n')


if __name__ == '__main__':
    main()
