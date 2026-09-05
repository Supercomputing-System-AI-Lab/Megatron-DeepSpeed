#!/usr/bin/env python
"""Average (soup) N universal-checkpoint atom dirs into a new universal dir.

    python soup_ucp_atoms.py --out checkpoint/lc16k-idoc-anneal/global_soup_universal \
        checkpoint/lc16k-idoc-anneal/global_step930_universal \
        checkpoint/lc16k-idoc-anneal/global_step940_universal \
        checkpoint/lc16k-idoc-anneal/global_step950_universal

Copies the LAST input dir wholesale (model states, metadata), then overwrites every
zero/<param>/fp32.pt with the elementwise mean across all inputs (OLMo 3 soups the
last 3 extension checkpoints; Llama 3.1 Polyak-averages the anneal). Optimizer
moment atoms (exp_avg*) are taken from the last checkpoint unaveraged - the soup is
a weights artifact for eval/SFT, not a training-resume point.
"""
import argparse, os, shutil, sys
os.environ.setdefault('DS_ACCELERATOR', 'cpu')
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('inputs', nargs='+')
a = ap.parse_args()
assert len(a.inputs) >= 2, 'need >=2 checkpoints to soup'
for d in a.inputs:
    assert os.path.isdir(os.path.join(d, 'zero')), f'not a universal dir: {d}'
assert not os.path.exists(a.out), f'{a.out} exists'

print(f'copying base {a.inputs[-1]} -> {a.out}')
shutil.copytree(a.inputs[-1], a.out)
params = sorted(os.listdir(os.path.join(a.inputs[-1], 'zero')))
print(f'averaging {len(params)} params across {len(a.inputs)} checkpoints')
n_avg = n_missing = 0
for i, p in enumerate(params):
    fps = [os.path.join(d, 'zero', p, 'fp32.pt') for d in a.inputs]
    if not all(os.path.exists(f) for f in fps):
        n_missing += 1
        continue
    tens = [torch.load(f, map_location='cpu', weights_only=False) for f in fps]
    vals = [t['param'] if isinstance(t, dict) and 'param' in t else t for t in tens]
    mean = torch.stack([v.float() for v in vals]).mean(0)
    outp = os.path.join(a.out, 'zero', p, 'fp32.pt')
    base = torch.load(outp, map_location='cpu', weights_only=False)
    if isinstance(base, dict) and 'param' in base:
        base['param'] = mean.to(vals[-1].dtype)
        torch.save(base, outp)
    else:
        torch.save(mean.to(vals[-1].dtype), outp)
    n_avg += 1
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(params)}')
print(f'done: averaged {n_avg}, skipped(missing-in-some) {n_missing}')
print('sanity: load one atom back'); torch.load(os.path.join(a.out, 'zero', params[0], 'fp32.pt'), map_location='cpu', weights_only=False)
print(f'SOUP READY: {a.out}')
