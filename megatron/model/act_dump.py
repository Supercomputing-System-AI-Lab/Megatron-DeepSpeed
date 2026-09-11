# coding=utf-8
"""The activation dump (2026-09-11; the sequence-parallel forward question of the V3 enablement): with XMOE_ACT_DUMP=<dir> in the
environment, every rank saves the output of the pipe embedding, of each transformer layer's attention, of its MLP (the MoE block
or the dense MLP) and of the layer itself, for the first XMOE_ACT_DUMP_CALLS (default 1) forward calls of each module -- the
first micro-batch of the first step -- as bf16 CPU tensors <dir>/rank<r>_<name>_call<k>.pt (embed, L03.attn, L03.mlp, L03.out,
...). Under Megatron's sequence parallelism a rank's tensor is its sequence shard; the TP ranks' shards concatenated along dim 0
give the full [s, b, h] that bin/compare_act_dump.py sets against the run without SP, layer by layer, from the same weights and
the same micro-batch. Off without the variable; forward hooks only, nothing in the graph changes."""
import os

import torch


def attach(model):
    d = os.getenv('XMOE_ACT_DUMP')
    if not d:
        return
    n = int(os.getenv('XMOE_ACT_DUMP_CALLS', '1') or 1)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    os.makedirs(d, exist_ok=True)
    counts = {}

    def save(name, out):
        k = counts.get(name, 0)
        if k >= n:
            return
        counts[name] = k + 1
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            torch.save(t.detach().to(torch.bfloat16).cpu(), f'{d}/rank{rank}_{name}_call{k}.pt')

    def hook_for(name):
        return lambda mod, inp, out: save(name, out)

    from megatron.model.language_model import EmbeddingPipe
    from megatron.model.transformer import ParallelTransformerLayerPipe
    attached = 0
    for mod in model.modules():
        if isinstance(mod, EmbeddingPipe):
            mod.register_forward_hook(hook_for('embed'))
            attached += 1
        elif isinstance(mod, ParallelTransformerLayerPipe):
            L = int(getattr(mod, 'layer_number', 0))
            mod.register_forward_hook(hook_for(f'L{L:02d}.out'))
            if hasattr(mod, 'self_attention'):
                mod.self_attention.register_forward_hook(hook_for(f'L{L:02d}.attn'))
            if hasattr(mod, 'mlp'):
                mod.mlp.register_forward_hook(hook_for(f'L{L:02d}.mlp'))
            attached += 1
    if rank == 0:
        print(f'[act_dump] {attached} modules dumped to {d} for the first {n} call(s) each', flush=True)
