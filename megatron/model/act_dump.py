# coding=utf-8
"""The activation dump (2026-09-11; the sequence-parallel forward question of the V3 enablement): with XMOE_ACT_DUMP=<dir> in the
environment, every rank saves the output of the pipe embedding, of each transformer layer's attention, of its MLP (the MoE block
or the dense MLP) and of the layer itself, for the first XMOE_ACT_DUMP_CALLS (default 1) forward calls of each module -- the
first micro-batch of the first step -- as bf16 CPU tensors <dir>/rank<r>_<name>_call<k>.pt (embed, L03.attn, L03.mlp, L03.out,
...). Under Megatron's sequence parallelism a rank's tensor is its sequence shard; the TP ranks' shards concatenated along dim 0
give the full [s, b, h] that bin/compare_act_dump.py sets against the run without SP, layer by layer, from the same weights and
the same micro-batch. Since 2026-09-12 also the pipe's final norm ('final_norm') and its head ('logits', cut to
XMOE_ACT_DUMP_LOGITS_LEN per call), for the comparison against HuggingFace's modules (bin/compare_hf_xmoe.py). Off without the
variable; forward hooks only, nothing in the graph changes."""
import os

import torch

# [2026-09-12] the micro-batch index of the forward in flight, set by the fixed-token rider (pretrain_gpt_deepspeed._fixed_tokens) on
# the stages that load the batch; with it a module's output is saved once per micro-batch, so the recompute of activation
# checkpointing (the same hooks firing again during the backward) is not counted as a new call. None: count the calls as before.
MICRO_BATCH_INDEX = None


def attach(model):
    d = os.getenv('XMOE_ACT_DUMP')
    if not d:
        return
    n = int(os.getenv('XMOE_ACT_DUMP_CALLS', '1') or 1)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if os.getenv('XMOE_ACT_DUMP_DP0', '') == '1':
        # [2026-09-12] only the data-parallel rank 0 of each stage / TP shard writes (the replicas hold the same values)
        try:
            from megatron.core import parallel_state as ps
            if ps.get_data_parallel_rank() != 0:
                return
        except Exception:
            pass
    os.makedirs(d, exist_ok=True)
    counts = {}

    # [2026-09-12, the HF comparison] the logits are cut to XMOE_ACT_DUMP_LOGITS_LEN (one length per call, comma-separated: the
    # prompts' lengths) along the sequence, so a 163,840-wide vocabulary is saved only where the prompt is
    lens = [int(x) for x in os.getenv('XMOE_ACT_DUMP_LOGITS_LEN', '').split(',') if x.strip()]

    saved = set()

    def save(name, out, cut=False):
        if MICRO_BATCH_INDEX is not None:
            k = MICRO_BATCH_INDEX
            if k >= n or (name, k) in saved:
                return
            saved.add((name, k))
        else:
            k = counts.get(name, 0)
            if k >= n:
                return
            counts[name] = k + 1
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            if cut and k < len(lens):
                t = t[:lens[k]]
            torch.save(t.detach().to(torch.bfloat16).cpu(), f'{d}/rank{rank}_{name}_call{k}.pt')

    def hook_for(name, cut=False):
        return lambda mod, inp, out: save(name, out, cut)

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
    # the pipe's final norm and its head (the top-level children named by their spec index): 'final_norm', 'logits' [s, b, v]
    import megatron.model.gpt_model as gm
    head_types = tuple(t for t in (getattr(gm, c, None) for c in ('LMHeadPipe', 'LMHeadChunkedPipe', 'LMHeadMTPPipe')) if t is not None)
    norm_names = ('MixedFusedRMSNorm', 'RMSNorm', 'LayerNorm', 'FusedLayerNorm', 'MixedFusedLayerNorm')
    for name, mod in model.named_modules():
        last = name.rsplit('.', 1)[-1]
        if not last.isdigit():
            continue
        if head_types and isinstance(mod, head_types):
            mod.register_forward_hook(hook_for('logits', cut=True))
            attached += 1
        elif type(mod).__name__ in norm_names:
            mod.register_forward_hook(hook_for('final_norm'))
            attached += 1
    if rank == 0:
        print(f'[act_dump] {attached} modules dumped to {d} for the first {n} call(s) each', flush=True)
