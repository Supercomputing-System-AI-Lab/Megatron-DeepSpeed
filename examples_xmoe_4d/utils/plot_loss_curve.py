#!/usr/bin/env python3
"""
plot_loss_curve.py — training-loss comparison for `run_exp_training.sh loss_validate`.

Reads the merged logs of the two loss_validate runs (X-MoE-4D and X-MoE), clips both
curves to the LEAST COMMON number of steps so the comparison is apples-to-apples, and
writes a PNG (plus a CSV of the plotted data).

Run from scripts-frontier/ after BOTH loss_validate jobs have finished:

    python3 ../utils/plot_loss_curve.py

Auto-discovery: loss_validate jobs are the ones whose job dir contains an
`a-xmoe-lv-*.o` file (RUN_TYPE="lv"). The newest such job is picked per framework.
Override with explicit logs:

    python3 ../utils/plot_loss_curve.py --xmoe4d <path/full_run.log> --xmoe <path/full_run.log>
"""

import argparse
import csv
import glob
import os
import re
import sys

# Loss lines look like:
#  iteration        1/    1000 | consumed samples: ... | lm loss: 1.131777E+01 | ... | TFLOPs: 35.17 |
ITER_LOSS_RE = re.compile(r"iteration\s+(\d+)\s*/\s*\d+\s*\|.*?lm loss:\s*([0-9.eE+-]+)")

FRAMEWORKS = {
    "xmoe4d": {"tag": "X-MOE-4D", "label": "X-MoE-4D", "color": "#1f77b4"},
    "xmoe": {"tag": "X-MOE", "label": "X-MoE", "color": "#d62728"},
}


def _read_gbs(log_path):
    """Pull the actual global batch size out of a merged log (driver echoes it)."""
    try:
        with open(log_path, "r", errors="replace") as fh:
            for line in fh:
                m = re.search(r"GLOBAL_BATCH_SIZE=(\d+)", line)
                if m:
                    return m.group(1)
                m = re.search(r"global batch size:\s*(\d+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def parse_losses(log_path):
    """Return {iteration: lm_loss} from a merged/rank log.

    A merged full_run.log can contain the same iteration twice (rank_0 and the last
    rank are both appended), so we key by iteration number to de-duplicate.
    """
    losses = {}
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            m = ITER_LOSS_RE.search(line)
            if m:
                losses[int(m.group(1))] = float(m.group(2))
    return losses


def _rank_logs_newest_last(job_dir):
    """rank_*.log for a job, ordered by RANK NUMBER (not lexically: rank_9 < rank_15)."""
    out = []
    for f in glob.glob(os.path.join(job_dir, "rank_*.log")):
        m = re.search(r"rank_(\d+)\.log$", f)
        if m:
            out.append((int(m.group(1)), f))
    return [f for _, f in sorted(out)]


def _candidate_logs(job_dir):
    """Logs worth trying for this job, best first.

    full_run.log is best (the merged log). Failing that, prefer the LAST rank:
    with PP>1 Megatron emits the `iteration ... lm loss:` lines only on the final
    pipeline stage, so rank_0.log is empty of losses and falling back to it -- as
    this function used to -- can never succeed for a pipeline-parallel run.
    """
    cands = []
    full = os.path.join(job_dir, "full_run.log")
    if os.path.isfile(full):
        cands.append(full)
    ranks = _rank_logs_newest_last(job_dir)
    if ranks:
        cands.append(ranks[-1])     # last pipeline stage
        if ranks[0] != ranks[-1]:
            cands.append(ranks[0])
    return cands


def find_log(job_root, tag):
    """Newest loss_validate (RUN_TYPE='lv') job for `tag` THAT ACTUALLY HAS LOSSES.

    Picking purely by mtime used to hand back the newest job dir even when the run
    had crashed or been killed before emitting a single iteration -- so one aborted
    run masked every good one behind it. Every candidate is now parsed first and
    skipped unless it yields loss lines.
    """
    candidates = []
    for job_dir in glob.glob(os.path.join(job_root, "job_*")):
        if not os.path.isdir(job_dir):
            continue
        # loss_validate runs are the ones with an a-xmoe-lv-*.o inside
        if not glob.glob(os.path.join(job_dir, "a-xmoe-lv-*.o")):
            continue
        # Underscore-delimited on BOTH sides on purpose. The job dir is
        # job_<id>_pp<PP>_ep<EP>_<MOE_TYPE>_ckpt_..., and the two tags are
        # "X-MOE-4D" and "X-MOE" -- one a strict prefix of the other. The
        # trailing "_" is what keeps _X-MOE_ from matching _X-MOE-4D_, so do
        # not relax this to a bare `tag in basename`.
        if f"_{tag}_" not in os.path.basename(job_dir):
            continue
        for log in _candidate_logs(job_dir):
            if parse_losses(log):
                candidates.append((os.path.getmtime(log), log))
                break
        else:
            print(f"  (skipping {os.path.basename(job_dir)}: no loss lines — "
                  f"crashed or killed before the first iteration?)")
    if not candidates:
        return None
    return max(candidates)[1]  # newest by mtime, among those with real data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xmoe4d", help="path to the X-MoE-4D run's full_run.log")
    ap.add_argument("--xmoe", help="path to the X-MoE run's full_run.log")
    ap.add_argument("--logs-dir", default="logs", help="where the job_* dirs live (default: logs)")
    ap.add_argument("--outdir", default="results", help="output folder (default: results)")
    ap.add_argument("--name", default="loss_validate_curve", help="output basename")
    args = ap.parse_args()

    # ---- locate the two logs ------------------------------------------------
    paths = {"xmoe4d": args.xmoe4d, "xmoe": args.xmoe}
    for key, meta in FRAMEWORKS.items():
        if paths[key] is None:
            paths[key] = find_log(args.logs_dir, meta["tag"])
        if paths[key] is None:
            sys.exit(
                f"ERROR: no finished loss_validate run found for {meta['label']} "
                f"({meta['tag']}) under '{args.logs_dir}/'.\n"
                f"       Run './run_exp_training.sh loss_validate' and wait for both jobs, "
                f"or pass --{key} <path/full_run.log>."
            )
        print(f"{meta['label']:6s} log: {paths[key]}")

    # The two legs must come from the SAME loss_validate invocation to be comparable:
    # the curve depends on the global batch, and auto-discovery picks each framework's
    # newest job independently -- so a re-run of one leg alone silently pairs runs with
    # different GBS (or different node counts). Compare and refuse to be quiet about it.
    gbs_seen = {k: _read_gbs(paths[k]) for k in FRAMEWORKS}
    if len({v for v in gbs_seen.values() if v}) > 1:
        # Built with plain concatenation, not nested-quote f-strings: those only parse
        # on Python 3.12+, and this util also gets run under the system interpreter.
        detail = ", ".join(FRAMEWORKS[k]["label"] + "=" + str(v) for k, v in gbs_seen.items())
        print("\nWARNING: the two runs used DIFFERENT global batch sizes (" + detail + ")."
              "\n         These curves are NOT comparable. Re-run both legs together, or pass"
              "\n         --xmoe4d/--xmoe explicitly to select a matched pair.\n")

    # ---- parse --------------------------------------------------------------
    series = {}
    for key, meta in FRAMEWORKS.items():
        losses = parse_losses(paths[key])
        if not losses:
            sys.exit(f"ERROR: no 'lm loss' lines parsed from {paths[key]}")
        series[key] = losses
        print(f"{meta['label']:6s} steps parsed: {len(losses)} (max iteration {max(losses)})")

    # ---- clip to the LEAST COMMON steps -------------------------------------
    common = sorted(set(series["xmoe4d"]) & set(series["xmoe"]))
    if not common:
        sys.exit("ERROR: the two runs share no common iteration numbers.")
    last = common[-1]
    steps = [i for i in common if i <= last]
    print(f"\nLeast common steps: {len(steps)} (iterations {steps[0]}..{steps[-1]})")

    # ---- plot ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / login node
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("ERROR: matplotlib is not installed in this environment.\n"
                 "       pip install matplotlib   (or use the CSV this script still writes)")

    os.makedirs(args.outdir, exist_ok=True)
    png = os.path.join(args.outdir, f"{args.name}.png")
    csv_path = os.path.join(args.outdir, f"{args.name}.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, meta in FRAMEWORKS.items():
        ys = [series[key][i] for i in steps]
        ax.plot(steps, ys, label=meta["label"], color=meta["color"], linewidth=1.6)

    ax.set_xlabel("Training step")
    ax.set_ylabel("LM loss")
    # Read GBS from the log rather than hardcoding it: run_exp_training.sh takes
    # GBS_LOSS from the environment (GBS_LOSS=64 for a smoke run), so a literal 320
    # silently mislabels every non-default plot.
    gbs = _read_gbs(paths["xmoe4d"]) or "?"   # paths[], not args[] -- args is None when auto-discovered
    ax.set_title(f"Training loss: X-MoE-4D vs X-MoE (10B, GBS={gbs}, {len(steps)} common steps)")
    ax.legend()
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "xmoe_4d_lm_loss", "xmoe_lm_loss"])
        for i in steps:
            w.writerow([i, series["xmoe4d"][i], series["xmoe"][i]])

    # final losses at the common horizon — the number you actually quote
    fe, fx = series["xmoe4d"][steps[-1]], series["xmoe"][steps[-1]]
    print(f"\nfinal lm loss @ step {steps[-1]}:  X-MoE-4D={fe:.6f}   X-MoE={fx:.6f}   (diff {fe - fx:+.6f})")
    print(f"saved: {png}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
