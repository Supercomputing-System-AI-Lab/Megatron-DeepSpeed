#!/usr/bin/env python3
"""routing_view.py (2026-10-06): the figures of the routing record written by deepspeed/moe/route_log.py (XMOE_ROUTE_LOG=1).

Input: a job folder (its routing/ subfolder) or the routing/ folder itself: rank<r>.bin + rank<r>.json per rank.
What it computes: per MoE layer and per training step, the number of tokens each expert received, summed over the micro-batches of the
step and over the ranks (default: all ranks = every token of the global batch counted once; --ranks a,b,c restricts the sum, e.g. one
expert-parallel group). The load share = count / (mean count over the experts of that step): 1 is perfect balance.
Output (in --out, default <routing dir>/figs):
  routing_L<i>.png   per layer: the heatmap experts x steps of the load share, and the trend of the hottest expert's share against the
                     capacity factor (from the sidecar when token dropping was on, else --capacity)
  routing.html       every layer's figure plus a table (steps, mean and max imbalance, the three hottest experts)
Usage: routing_view.py JOB_OR_ROUTING_DIR [--layer i] [--all] [--ranks 0,1,2] [--capacity 1.25] [--out DIR]
"""
import argparse, base64, glob, io, json, os, re, sys
import numpy as np

import importlib.util
_RL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'deepspeed', 'moe', 'route_log.py'))
_spec = importlib.util.spec_from_file_location('route_log', _RL); route_log = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(route_log)
# loaded by path on purpose: importing the deepspeed package needs ROCm on the path (its op builders probe CUDA_HOME at import), the reader needs only numpy + torch


def load_dir(d, ranks=None):
    files = sorted(glob.glob(os.path.join(d, 'rank*.bin')), key=lambda p: int(re.search(r'rank(\d+)\.bin', p).group(1)))
    if ranks is not None:
        files = [p for p in files if int(re.search(r'rank(\d+)\.bin', p).group(1)) in ranks]
    if not files:
        sys.exit(f'no rank*.bin under {d}')
    meta = {}
    per_step = {}      # step -> [n_gates, E] summed over micro-batches and ranks
    for p in files:
        r = int(re.search(r'rank(\d+)\.bin', p).group(1))
        side = p[:-4] + '.json'
        if os.path.exists(side):
            meta[r] = json.load(open(side))
        for step, arr in route_log.read(p):
            s = arr.sum(axis=1).astype(np.int64)     # over the micro-batches
            per_step[step] = per_step.get(step, 0) + s
    steps = sorted(per_step)
    cube = np.stack([per_step[s] for s in steps])    # [steps, n_gates, E]
    return steps, cube, meta, [int(re.search(r'rank(\d+)\.bin', p).group(1)) for p in files]


def fig_layer(li, steps, mat, cap, title_extra=''):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    mean = mat.mean(axis=1, keepdims=True); share = np.where(mean > 0, mat / np.maximum(mean, 1e-9), 0.0)
    E = mat.shape[1]
    fig = plt.figure(figsize=(10.8, 6.2), dpi=100); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.4)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(share.T, aspect='auto', cmap='magma', vmin=0, vmax=max(2.0, min(4.0, float(share.max()))), interpolation='nearest',
                   extent=(steps[0] - 0.5, steps[-1] + 0.5, E - 0.5, -0.5))
    ax.set_ylabel('expert id'); ax.set_xlabel('training step')
    ax.set_title(f'MoE layer {li}{title_extra}: load share per expert (count / mean count of the step; 1 = perfect balance), {len(steps)} steps', fontsize=10, loc='left')
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label('load / mean load')
    ax2 = fig.add_subplot(gs[1])
    mx = share.max(axis=1); ax2.plot(steps, mx, color='#b2182b', lw=1.4, label='hottest expert: load / mean')
    ax2.plot(steps, share.std(axis=1), color='#2166ac', lw=1.0, label='std of the share over the experts')
    if cap:
        ax2.axhline(cap, color='#333', ls='--', lw=0.8); ax2.text(steps[-1], cap, f' capacity factor {cap}', ha='right', va='bottom', fontsize=8.5)
    ax2.set_xlim(steps[0] - 0.5, steps[-1] + 0.5); ax2.set_ylim(0, max(float(mx.max()) * 1.15, (cap or 0) * 1.15, 1.0))
    ax2.set_xlabel('training step'); ax2.set_ylabel('imbalance'); ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5, frameon=False)
    b = io.BytesIO(); fig.savefig(b, format='png', bbox_inches='tight'); plt.close(fig)
    hot = np.argsort(-mat.mean(axis=0))[:3]
    return b.getvalue(), dict(layer=li, steps=len(steps), mean_imbalance=float(mx.mean()), max_imbalance=float(mx.max()), hottest=[int(h) for h in hot],
                              share_of_hottest=float(mat[:, hot[0]].sum() / max(mat.sum(), 1)) * E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--layer', type=int); ap.add_argument('--all', action='store_true'); ap.add_argument('--ranks'); ap.add_argument('--capacity', type=float); ap.add_argument('--out')
    a = ap.parse_args()
    d = a.dir if os.path.isdir(os.path.join(a.dir, 'routing')) is False else os.path.join(a.dir, 'routing')
    ranks = {int(x) for x in a.ranks.split(',')} if a.ranks else None
    steps, cube, meta, used = load_dir(d, ranks)
    m0 = next(iter(meta.values()), {})
    cap = a.capacity or (m0.get('capacity_factor') if m0.get('drop_tokens') else None)
    out = a.out or os.path.join(d, 'figs'); os.makedirs(out, exist_ok=True)
    layers = range(cube.shape[1]) if a.all or a.layer is None else [a.layer]
    extra = f' (sum over ranks {used[0]}..{used[-1]}, {len(used)} ranks; {m0.get("experts", cube.shape[2])} experts, top-{m0.get("k", "?")})'
    rows, imgs = [], []
    for li in layers:
        png, row = fig_layer(li, steps, cube[:, li, :], cap, extra)
        open(os.path.join(out, f'routing_L{li}.png'), 'wb').write(png); rows.append(row); imgs.append((li, png))
        print(f"layer {li}: {row['steps']} steps, mean imbalance {row['mean_imbalance']:.2f}, max {row['max_imbalance']:.2f}, hottest experts {row['hottest']}")
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Routing record</title><style>html,body{background:#fff;color:#111;font:15px/1.45 system-ui,sans-serif;margin:24px}'
            'table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}th{background:#f3f3f3}img{width:1080px;max-width:100%}</style></head><body>',
            f'<h1>Routing record: {os.path.abspath(d)}</h1><p>Per MoE layer and training step, the tokens each expert received, summed over micro-batches and over ranks {used}. '
            f'Load share = count / mean count of the step. Capacity factor {cap if cap else "n/a (dropless)"}.</p>',
            '<table><tr><th>layer</th><th>steps</th><th>mean imbalance (hottest / mean)</th><th>max imbalance</th><th>hottest experts</th></tr>']
    for r in rows:
        html.append(f"<tr><td>{r['layer']}</td><td>{r['steps']}</td><td>{r['mean_imbalance']:.2f}</td><td>{r['max_imbalance']:.2f}</td><td>{r['hottest']}</td></tr>")
    html.append('</table>')
    for li, png in imgs:
        html.append(f'<h2>MoE layer {li}</h2><img src="data:image/png;base64,{base64.b64encode(png).decode()}">')
    html.append('</body></html>')
    open(os.path.join(out, 'routing.html'), 'w').write('\n'.join(html))
    print('wrote', os.path.join(out, 'routing.html'))


if __name__ == '__main__':
    main()
