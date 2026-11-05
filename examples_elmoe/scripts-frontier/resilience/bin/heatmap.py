#!/usr/bin/env python3
"""heatmap.png of one beat (design 3.7), from <job>/data/agg_last.json: one row per node, one column per local rank (8);
colour = the age of the rank's last heartbeat line (white = fresh, dark red = 30 s and older); the letter = the phase
(f forward, b backward, a all_to_all, s save, S save_barrier, p p2p, i init, l load, t step); a black frame = a rank the
minority or stillness rule named. Drawn ONLY when an alarm fires (the aggregator), at the end of each attempt (the job script),
and on demand:  python heatmap.py --job-dir <job folder> [--why <text>]   (the user, 2026-10-09 13:55: not every minute)."""
import argparse, json, os, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(job, why=''):
    t = json.load(open(os.path.join(job, 'data', 'agg_last.json')))
    ranks = {int(r): d for r, d in t['ranks'].items()}
    nodes = []
    for r in sorted(ranks):
        if ranks[r]['node'] not in nodes:
            nodes.append(ranks[r]['node'])
    G = 8
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.4 * len(nodes)), dpi=110)
    fig.patch.set_facecolor('white')
    import numpy as np
    A = np.full((len(nodes), G), np.nan)
    named = set(t.get('minority', [])) | set(t.get('still', []))
    letters = {'forward': 'f', 'backward': 'b', 'all_to_all': 'a', 'save': 's', 'save_barrier': 'S', 'p2p': 'p', 'init': 'i',
               'load': 'l', 'step': 't'}
    for r, d in ranks.items():
        i, j = nodes.index(d['node']), r % G
        A[i, j] = min(float(d['age']), 30.0)
        ax.text(j, i, f"{r}\n{letters.get(d['phase'], '?')}", ha='center', va='center', fontsize=6)
        if r in named:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2, ec='black'))
    ax.imshow(A, cmap='Reds', vmin=0, vmax=30, aspect='auto')
    ax.set_yticks(range(len(nodes))); ax.set_yticklabels(nodes, fontsize=7)
    ax.set_xticks(range(G)); ax.set_xticklabels([f'local {k}' for k in range(G)], fontsize=6)
    ax.set_title(f"attempt {t['attempt']}, step {t['step']}, beat {time.strftime('%H:%M:%S', time.localtime(t['time']))}"
                 + (f" ({why})" if why else '') + "\ncolour = age of the last heartbeat line (0-30 s); frame = a rank an alarm named",
                 fontsize=7)
    fig.tight_layout()
    out = os.path.join(job, 'heatmap.png')
    fig.savefig(out + '.tmp.png'); os.replace(out + '.tmp.png', out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', required=True)
    ap.add_argument('--why', default='on demand')
    a = ap.parse_args()
    main(a.job_dir, a.why)
