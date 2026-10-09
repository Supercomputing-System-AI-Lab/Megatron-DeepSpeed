#!/usr/bin/env python3
"""
plot_loss_curve.py — training-loss comparison for `run_exp_training.sh loss_validate`.

Reads the merged logs of the two loss_validate runs (ELMoE-3D and X-MoE), clips both
curves to the LEAST COMMON number of steps so the comparison is apples-to-apples, and
writes a PNG (plus a CSV of the plotted data).

Run from scripts-frontier/ after BOTH loss_validate jobs have finished:

    python3 ../utils/plot_loss_curve.py

Auto-discovery: loss_validate jobs are the ones whose job dir contains an
`a-xmoe-lv-*.o` file (RUN_TYPE="lv"). The newest such job is picked per framework.
Override with explicit logs:

    python3 ../utils/plot_loss_curve.py --elmoe <path/full_run.log> --xmoe <path/full_run.log>
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
    "elmoe": {"tag": "ELMOE-3D", "label": "ELMoE", "color": "#1f77b4"},
    "xmoe": {"tag": "X-MOE", "label": "X-MoE", "color": "#d62728"},
}


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


def find_log(job_root, tag):
    """Newest loss_validate (RUN_TYPE='lv') job dir whose name carries `tag`."""
    candidates = []
    for job_dir in glob.glob(os.path.join(job_root, "job_*")):
        if not os.path.isdir(job_dir):
            continue
        # loss_validate runs are the ones with an a-xmoe-lv-*.o inside
        if not glob.glob(os.path.join(job_dir, "a-xmoe-lv-*.o")):
            continue
        if f"_{tag}_" not in os.path.basename(job_dir):
            continue
        for name in ("full_run.log", "rank_0.log"):
            log = os.path.join(job_dir, name)
            if os.path.isfile(log):
                candidates.append((os.path.getmtime(log), log))
                break
    if not candidates:
        return None
    return max(candidates)[1]  # newest by mtime


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elmoe", help="path to the ELMoE run's full_run.log")
    ap.add_argument("--xmoe", help="path to the X-MoE run's full_run.log")
    ap.add_argument("--logs-dir", default="logs", help="where the job_* dirs live (default: logs)")
    ap.add_argument("--outdir", default="results", help="output folder (default: results)")
    ap.add_argument("--name", default="loss_validate_curve", help="output basename")
    args = ap.parse_args()

    # ---- locate the two logs ------------------------------------------------
    paths = {"elmoe": args.elmoe, "xmoe": args.xmoe}
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

    # ---- parse --------------------------------------------------------------
    series = {}
    for key, meta in FRAMEWORKS.items():
        losses = parse_losses(paths[key])
        if not losses:
            sys.exit(f"ERROR: no 'lm loss' lines parsed from {paths[key]}")
        series[key] = losses
        print(f"{meta['label']:6s} steps parsed: {len(losses)} (max iteration {max(losses)})")

    # ---- clip to the LEAST COMMON steps -------------------------------------
    common = sorted(set(series["elmoe"]) & set(series["xmoe"]))
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
    ax.set_title(f"Training loss: ELMoE vs X-MoE (10B, GBS=320, {len(steps)} common steps)")
    ax.legend()
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "elmoe_lm_loss", "xmoe_lm_loss"])
        for i in steps:
            w.writerow([i, series["elmoe"][i], series["xmoe"][i]])

    # final losses at the common horizon — the number you actually quote
    fe, fx = series["elmoe"][steps[-1]], series["xmoe"][steps[-1]]
    print(f"\nfinal lm loss @ step {steps[-1]}:  ELMoE={fe:.6f}   X-MoE={fx:.6f}   (diff {fe - fx:+.6f})")
    print(f"saved: {png}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
