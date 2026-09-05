#!/usr/bin/env python
"""Rigorously verify a Universal Checkpoint against the DeepSpeed checkpoint it came from.

WHY THIS EXISTS
---------------
A same-topology round trip through UCP proves nothing: a wrong merge and a wrong split cancel
exactly when TP is unchanged. And `load_hp_checkpoint_state`'s only guard is
`full_numel == tp_world_size * slice_numel`, which a wrong `cat_dim` satisfies. So neither the
converter nor a resume-at-same-TP can detect a transposed merge.

This script closes that hole by reconstructing every full parameter **independently** -- from
first-principles knowledge of how Megatron shards each layer type, never from the converter's
own pattern lists -- and comparing against the atom the converter produced. If the published
patterns are wrong for any parameter class, the two disagree.

The expert check is independent in a second way: the atoms come from the ZeRO optimizer shards,
while `layer_*_expert_*_model_states.pt` is written by a completely separate code path
(`engine.py:_save_moe_checkpoint`). Agreement means the global-expert-id mapping is consistent
end to end.

USAGE
-----
    # after conversion
    python verify_ucp_atoms.py --src .../global_step90 --universal .../global_step90_universal

    # before conversion (inventory + replication checks only)
    python verify_ucp_atoms.py --src .../global_step90

    --max-experts N   sample N expert atoms instead of all 3456 (default 64; 0 = all)
    -v                list every parameter, not just failures
"""
import argparse
import glob
import io
import os
import pickle
import re
import sys
from collections import defaultdict

import torch

# --------------------------------------------------------------------------------------------
# Loading. The mp_rank files contain client objects (argparse.Namespace, LR scheduler, RNG
# state) whose classes live in megatron/, which needs the full GPU stack importable. We only
# ever read tensors, so resolve unknown classes to inert placeholders instead. Read-only.
# --------------------------------------------------------------------------------------------
_ALLOW = ('torch', 'collections', 'numpy', '_codecs', 'builtins', 'argparse', '__builtin__')


class _Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module.split('.')[0] in _ALLOW:
            return super().find_class(module, name)
        return type(name, (object, ), {'__setstate__': lambda s, st: None,
                                       '__init__': lambda s, *a, **k: None})


class _PickleModule:
    Unpickler = _Unpickler

    @staticmethod
    def load(f, **kw):
        return _Unpickler(f, **kw).load()

    @staticmethod
    def loads(b, **kw):
        return _Unpickler(io.BytesIO(b), **kw).load()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_PickleModule)


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        name = f'{prefix}{k}'
        if isinstance(v, torch.Tensor):
            out[name] = v
        elif isinstance(v, dict):
            out.update(flatten(v, name + '.'))
    return out


# --------------------------------------------------------------------------------------------
# First-principles sharding rules. Derived from the Megatron layer definitions, NOT from the
# checkpoint's published pattern lists -- that independence is the whole point.
# --------------------------------------------------------------------------------------------
LAYER = r'language_model\.encoder\.layers\.\d+\.'
RULES = [
    # (regex, rule, human label)
    (r'language_model\.embedding\.word_embeddings\.weight$', 'vocab', 'vocab embedding'),
    (r'language_model\.output_layer\.weight$', 'vocab', 'vocab output layer'),
    (LAYER + r'self_attention\.query_key_value\.weight$', 'cat0', 'qkv (column parallel)'),
    (LAYER + r'self_attention\.dense\.weight$', 'cat1', 'attn dense (row parallel)'),
    (LAYER + r'mlp\.dense_4h_to_h\.weight$', 'cat1', 'mlp 4h_to_h (row parallel)'),
    (LAYER + r'mlp\.shared_experts\.shared_mlp\.dense_4h_to_h\.weight$', 'cat1',
     'shared-expert 4h_to_h (row parallel)'),
    (LAYER + r'mlp\.dense_h_to_4h\.weight$', 'swiglu', 'mlp h_to_4h (swiglu, 2 sub-params)'),
    (LAYER + r'mlp\.shared_experts\.shared_mlp\.dense_h_to_4h\.weight$', 'swiglu',
     'shared-expert h_to_4h (swiglu, 2 sub-params)'),
    (LAYER + r'input_layernorm\.weight$', 'same', 'input layernorm (replicated)'),
    (LAYER + r'post_attention_layernorm\.weight$', 'same', 'post-attn layernorm (replicated)'),
    (r'language_model\.encoder\.final_layernorm\.weight$', 'same', 'final layernorm (replicated)'),
    (LAYER + r'mlp\.deepspeed_moe\.gate\.wg\.weight$', 'avg', 'MoE router (replicated, averaged)'),
]
EXPERT_RE = re.compile(LAYER + r'mlp\.deepspeed_moe\.experts\.deepspeed_experts\.\d+\.')


def classify(name):
    for pat, rule, label in RULES:
        if re.match(pat, name):
            return rule, label
    if EXPERT_RE.match(name):
        return 'expert', 'routed expert (not TP-sharded, etp=False)'
    return None, None


def merge(rule, slices):
    """Reconstruct the full tensor from per-TP-rank slices, from first principles."""
    if rule in ('same', 'avg'):
        # 'same' takes rank 0 (they must already be identical -- checked separately);
        # 'avg' is the mean, matching what the converter is asked to do.
        return slices[0] if rule == 'same' else torch.stack([s.float() for s in slices]).mean(0)
    if rule in ('cat0', 'vocab'):
        return torch.cat(slices, dim=0)
    if rule == 'cat1':
        return torch.cat(slices, dim=1)
    if rule == 'swiglu':
        # Sharded layout is [gate_r ; up_r] per rank; canonical is [gate_all ; up_all].
        halves = [torch.chunk(s, 2, dim=0) for s in slices]
        return torch.cat([torch.cat([h[0] for h in halves], dim=0),
                          torch.cat([h[1] for h in halves], dim=0)], dim=0)
    raise ValueError(rule)


def report(tag, results, verbose):
    """results: list of (name, ok, maxdev, note)"""
    bad = [r for r in results if not r[1]]
    n = len(results)
    status = 'PASS' if not bad else f'*** FAIL ({len(bad)}/{n}) ***'
    worst = max((r[2] for r in results), default=0.0)
    print(f'  {tag:46} {n:5d} checked   max dev {worst:.3e}   {status}')
    for name, ok, dev, note in results:
        if verbose or not ok:
            print(f'      {"ok " if ok else "BAD"} {name}  dev={dev:.3e}  {note}')
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='the DeepSpeed checkpoint (global_step<N>)')
    ap.add_argument('--universal', default=None, help='the converted folder (global_step<N>_universal)')
    ap.add_argument('--max-experts', type=int, default=64, help='0 = check all expert atoms')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    failures = 0
    print(f'\nsource    : {args.src}')
    print(f'universal : {args.universal or "(not given -- pre-conversion checks only)"}\n')

    # ---------------------------------------------------------------- phase 1: source topology
    mp_files = sorted(glob.glob(os.path.join(args.src, 'mp_rank_*_model_states.pt')))
    assert mp_files, f'no mp_rank_*_model_states.pt in {args.src}'
    tp = len(mp_files)
    print(f'=== PHASE 1  source checkpoint ({tp} TP rank{"s" if tp > 1 else ""}) ===')

    sd0 = load(mp_files[0])
    uci = sd0.get('universal_checkpoint_info', {})
    orig_vocab = uci.get('original_vocab_size')
    slices = [flatten(load(f)['module']) for f in mp_files]
    names = sorted(slices[0])
    print(f'  {len(names)} non-expert parameters per rank; original_vocab_size={orig_vocab}')

    unclassified = [n for n in names if classify(n)[0] is None]
    if unclassified:
        failures += 1
        print(f'  *** {len(unclassified)} parameter(s) match no reconstruction rule -- the rule '
              f'table is incomplete, results below are NOT exhaustive:')
        for n in unclassified[:10]:
            print(f'        {n}')
    else:
        print('  all parameters are covered by a reconstruction rule')

    # replicated params must be bit-identical across TP; router is expected to drift
    if tp > 1:
        print('\n=== PHASE 2  replication check (what merge_tp_slices asserts) ===')
        buckets = defaultdict(list)
        for n in names:
            rule, label = classify(n)
            if rule in ('same', 'avg'):
                t0 = slices[0][n].float()
                dev = max(((t0 - slices[r][n].float()).abs().max().item()
                           / max(t0.abs().max().item(), 1e-12)) for r in range(1, tp))
                buckets[label].append((n, dev == 0.0 if rule == 'same' else True, dev,
                                       'bit-identical' if dev == 0 else 'drifts across TP ranks'))
        for label, res in buckets.items():
            failures += report(label, res, args.verbose)

    if not args.universal:
        print(f'\n{"=" * 78}\npre-conversion checks done. failures={failures}\n{"=" * 78}')
        return 1 if failures else 0

    # ------------------------------------------- phase 3: atoms vs independent reconstruction
    zdir = os.path.join(args.universal, 'zero')
    assert os.path.isdir(zdir), f'no zero/ in {args.universal}'
    atom_names = sorted(d for d in os.listdir(zdir) if os.path.isdir(os.path.join(zdir, d)))
    print(f'\n=== PHASE 3  {len(atom_names)} atoms vs independent reconstruction ===')

    non_expert = [n for n in atom_names if not EXPERT_RE.match(n)]
    expert = [n for n in atom_names if EXPERT_RE.match(n)]
    print(f'  {len(non_expert)} non-expert, {len(expert)} expert')

    missing = set(names) - set(atom_names)
    if missing:
        failures += 1
        print(f'  *** {len(missing)} source parameter(s) have no atom, e.g. {sorted(missing)[:3]}')

    buckets = defaultdict(list)
    for n in non_expert:
        rule, label = classify(n)
        if rule is None or rule == 'expert':
            continue
        if n not in slices[0]:
            buckets[label].append((n, False, float('inf'), 'atom has no source parameter'))
            continue
        fp32 = os.path.join(zdir, n, 'fp32.pt')
        if not os.path.isfile(fp32):
            buckets[label].append((n, False, float('inf'), 'missing fp32.pt'))
            continue
        atom = load(fp32)['param'].float()
        recon = merge(rule, [slices[r][n] for r in range(tp)]).float()
        if rule == 'vocab' and orig_vocab:
            # UCP stores vocab atoms stripped of Megatron's padding rows
            recon = recon[:orig_vocab]
        if atom.shape != recon.shape:
            buckets[label].append((n, False, float('inf'),
                                   f'shape atom{tuple(atom.shape)} != recon{tuple(recon.shape)}'))
            continue
        # atom is the fp32 master; the mp_rank copy is that master cast to bf16
        dev = (atom.bfloat16().float() - recon).abs().max().item()
        scale = max(recon.abs().max().item(), 1e-12)
        rel = dev / scale
        # exact for concatenations; averaging in fp32 vs bf16 differs by ~1 bf16 ulp
        tol = 0.0 if rule != 'avg' else 2 ** -8
        buckets[label].append((n, rel <= tol, rel, f'shape={tuple(atom.shape)}'))
    for label, res in buckets.items():
        failures += report(label, res, args.verbose)

    # ------------------------------------- phase 4: expert atoms vs the per-expert weight files
    print(f'\n=== PHASE 4  expert atoms vs layer_*_expert_*_model_states.pt (independent path) ===')
    first_k = (uci.get('moe') or {}).get('first_k_dense_replace', 0)
    sample = expert if args.max_experts == 0 else expert[::max(1, len(expert) // args.max_experts)]
    res = []
    for n in sample:
        m = re.match(r'language_model\.encoder\.layers\.(\d+)\..*deepspeed_experts\.(\d+)\.', n)
        layer, gid = int(m.group(1)), int(m.group(2))
        moe_layer = layer - first_k          # file id is the MoE-layer index, not the layer index
        cand = glob.glob(os.path.join(args.src, f'layer_{moe_layer}_expert_{gid}_mp_rank_*_model_states.pt'))
        if not cand:
            res.append((n, False, float('inf'), f'no weight file layer_{moe_layer}_expert_{gid}_*'))
            continue
        w = load(cand[0])
        if n not in w:
            res.append((n, False, float('inf'), f'{os.path.basename(cand[0])} lacks this key'))
            continue
        atom = load(os.path.join(zdir, n, 'fp32.pt'))['param'].float()
        ref = w[n].float()
        if atom.shape != ref.shape:
            res.append((n, False, float('inf'),
                        f'shape atom{tuple(atom.shape)} != file{tuple(ref.shape)}'))
            continue
        rel = (atom.bfloat16().float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)
        res.append((n, rel == 0.0, rel, f'layer_{moe_layer}_expert_{gid} shape={tuple(atom.shape)}'))
    failures += report(f'routed experts (sampled {len(sample)}/{len(expert)})', res, args.verbose)

    print(f'\n{"=" * 78}')
    print('ALL CHECKS PASSED' if not failures else f'*** {failures} CHECK GROUP(S) FAILED ***')
    print(f'{"=" * 78}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
