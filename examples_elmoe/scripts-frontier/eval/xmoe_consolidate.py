#!/usr/bin/env python
# xmoe_consolidate.py — flatten an X-MoE DeepSpeed checkpoint into one .pt file
#
# Reads:
#   <input>/mp_rank_00_model_states.pt
#     └── 'module' -> 'language_model' -> nested dict with:
#           - embedding
#           - per-layer attention / layernorm
#           - per-layer mlp.deepspeed_moe.gate.wg (router weights)
#           - final_layernorm
#   <input>/layer_<L>_expert_<E>_mp_rank_<MP>_model_states.pt   (typically 1536 files
#     for 24 layers x 64 experts; each is already a flat dict with 4 keys for the
#     expert's FFN up/down projections.)
#
# Writes:
#   <output> — one .pt file holding a flat state_dict keyed by global expert
#              index. evaluate_v04.py loads this and does the per-rank
#              global->local expert remap at load time.
#
# Run from a LOGIN NODE (no GPU needed; just file I/O). Takes a few minutes
# the first time per checkpoint.

import argparse
import glob
import os
import re
import time

import torch

# PyTorch >= 2.6 defaults to weights_only=True, which rejects argparse.Namespace
# in the saved checkpoint metadata. The checkpoint is trusted (we wrote it), so
# flip the default for this process only.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def flatten(d, prefix=''):
    """Walk a (possibly partially-nested) state dict, returning flat dotted-path entries.

    The X-MoE shared file has mixed nesting: shared['module']['language_model'] has
    'embedding' (further nested) and 'encoder' (already dotted-flat under 'encoder').
    Recursing handles both consistently.
    """
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif torch.is_tensor(v):
            out[key] = v
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input_folder', required=True,
                        help='global_step<N>/ directory of an X-MoE DeepSpeed checkpoint')
    parser.add_argument('--output_path', required=True,
                        help='Destination .pt file for the consolidated flat state_dict')
    parser.add_argument('--dtype', default='keep',
                        choices=['keep', 'bfloat16', 'float16', 'float32'],
                        help='Cast all float tensors to this dtype before saving (default: keep as-is)')
    args = parser.parse_args()

    in_dir = args.input_folder
    out_path = args.output_path
    assert os.path.isdir(in_dir), f'Not a directory: {in_dir}'
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Shared (non-expert) params from mp_rank_00_model_states.pt
    # ------------------------------------------------------------------------
    shared_path = os.path.join(in_dir, 'mp_rank_00_model_states.pt')
    assert os.path.exists(shared_path), f'Missing shared file: {shared_path}'
    print(f'[1/3] Reading shared params from {shared_path}')
    t0 = time.time()
    shared = torch.load(shared_path, map_location='cpu')
    assert 'module' in shared, f'Expected shared["module"], got keys: {list(shared.keys())[:10]}'

    flat_sd = flatten(shared['module'])
    n_gates = sum(1 for k in flat_sd if k.endswith('.deepspeed_moe.gate.wg.weight'))
    print(f'  -> {len(flat_sd)} flat tensor keys ({time.time() - t0:.1f}s)')
    print(f'  -> {n_gates} MoE router (gate.wg) tensors')

    # ------------------------------------------------------------------------
    # 2. Expert params, one file per (layer, global_expert_idx)
    # ------------------------------------------------------------------------
    expert_glob = os.path.join(in_dir, 'layer_*_expert_*_mp_rank_*_model_states.pt')
    expert_files = sorted(glob.glob(expert_glob))
    assert expert_files, f'No expert files matching {expert_glob}'
    print(f'[2/3] Merging {len(expert_files)} expert files...')
    t0 = time.time()

    expert_name_re = re.compile(r'^layer_(\d+)_expert_(\d+)_mp_rank_(\d+)_model_states\.pt$')
    bad = 0
    for i, ef in enumerate(expert_files):
        if not expert_name_re.match(os.path.basename(ef)):
            print(f'  WARN: unexpected filename, skipping: {os.path.basename(ef)}')
            bad += 1
            continue
        sd = torch.load(ef, map_location='cpu')
        if not isinstance(sd, dict):
            print(f'  WARN: not a dict, skipping: {os.path.basename(ef)}')
            bad += 1
            continue
        # Expert files are already flat dicts with full dotted keys
        for k, v in sd.items():
            if torch.is_tensor(v):
                flat_sd[k] = v
        if (i + 1) % 200 == 0 or i == len(expert_files) - 1:
            print(f'  [{i + 1:>5}/{len(expert_files)}] {len(flat_sd)} keys; '
                  f'elapsed {time.time() - t0:.1f}s')
    print(f'  -> done: {len(flat_sd)} flat keys ({time.time() - t0:.1f}s); skipped {bad}')

    # ------------------------------------------------------------------------
    # 3. Optional dtype cast
    # ------------------------------------------------------------------------
    if args.dtype != 'keep':
        target_dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }[args.dtype]
        print(f'[3/3] Casting all float tensors to {target_dtype}')
        t0 = time.time()
        flat_sd = {
            k: (v.to(target_dtype) if v.is_floating_point() else v)
            for k, v in flat_sd.items()
        }
        print(f'  -> done ({time.time() - t0:.1f}s)')

    # ------------------------------------------------------------------------
    # Sanity summary
    # ------------------------------------------------------------------------
    expert_keys = [k for k in flat_sd if '.experts.deepspeed_experts.' in k]
    attn_keys = [k for k in flat_sd if '.self_attention.' in k]
    emb_keys = [k for k in flat_sd if '.embedding.' in k]
    gate_keys = [k for k in flat_sd if k.endswith('.deepspeed_moe.gate.wg.weight')]
    ln_keys = [k for k in flat_sd if '.layernorm.' in k or '.final_layernorm.' in k]
    print('\nFinal stats:')
    print(f'  Total flat keys      : {len(flat_sd)}')
    print(f'  Expert tensors       : {len(expert_keys)} (expect num_layers * num_experts * 4)')
    print(f'  MoE gate tensors     : {len(gate_keys)} (expect num_layers)')
    print(f'  Self-attention       : {len(attn_keys)}')
    print(f'  Embedding            : {len(emb_keys)}')
    print(f'  LayerNorm            : {len(ln_keys)}')

    # ------------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------------
    t0 = time.time()
    print(f'\nSaving consolidated state_dict to {out_path}')
    torch.save(flat_sd, out_path)
    sz_gb = os.path.getsize(out_path) / 1e9
    print(f'  -> {sz_gb:.2f} GB in {time.time() - t0:.1f}s')

    print('\nFirst 5 keys:')
    for k in list(flat_sd.keys())[:5]:
        print(f'  {k}')
    print('Last 5 keys:')
    for k in list(flat_sd.keys())[-5:]:
        print(f'  {k}')


if __name__ == '__main__':
    main()
