#!/usr/bin/env python
# coding=utf-8
"""Framework side: dump per-layer hidden states + final logits for fixed token ids.

Reuses evaluate_v04.load_ds_checkpoint_and_setup_megatron (the SAME proven load path the eval
uses), so it must be launched with the SAME megatron args + 8-rank srun as the eval (EP=8).
Registers forward hooks on the embedding and each encoder layer; aligns 1:1 with HF's
output_hidden_states in compare.py.

LAUNCH: copy the eval's `python ... evaluate_v04.py <megatron_options> <data_options>
<deepspeed_options>` line from the rendered eval SLURM, swap the script to this file, drop the
eval-only flags (--task_list/--results_path/--correctness_tests/--lm-eval-response-cache/
--padding-side/--limit), and append:  --dump-out fw_acts.pt  [--dump-ids 100000,549,...]
Run only rank 0's dump (all ranks compute; rank 0 writes).
"""
import argparse, os, sys, torch

REPO = "/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE"
sys.path.insert(0, os.path.join(REPO, "tasks/eval_harness"))
sys.path.insert(0, REPO)

BOS = 100000

def dump_args(parser):
    g = parser.add_argument_group("dump")
    g.add_argument("--dump-out", required=True)
    g.add_argument("--dump-ids", default=None, help="comma-separated ids; MUST match hf_dump")
    g.add_argument("--dump-with-bos", action="store_true")
    return parser

def find_encoder_layers(engine):
    """Robustly reach the ParallelTransformer layer list under DeepSpeed/FP16 wrappers."""
    m = engine.module
    for _ in range(4):  # peel possible module wrappers
        if hasattr(m, "language_model"):
            break
        m = getattr(m, "module", m)
    lm = m.language_model
    enc = lm.encoder
    return lm, enc, enc.layers

def main():
    from evaluate_v04 import load_ds_checkpoint_and_setup_megatron
    from megatron import get_args, get_tokenizer
    from megatron.utils import get_ltor_masks_and_position_ids

    engine = load_ds_checkpoint_and_setup_megatron(extra_args_provider=dump_args)
    args = get_args()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    lm, enc, layers = find_encoder_layers(engine)

    # ---- hooks: embedding output + each layer output ----
    acts = {}
    def cap(name):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            acts[name] = h.detach().float().cpu()
        return hook
    lm.embedding.register_forward_hook(cap("emb"))
    for i, layer in enumerate(layers):
        layer.register_forward_hook(cap(f"L{i}"))

    # ---- fixed input ----
    if args.dump_ids:
        ids = [int(x) for x in args.dump_ids.split(",")]
    else:
        ids = [BOS, 549, 6077, 280, 7239, 317] if args.dump_with_bos else [549, 6077, 280, 7239, 317]
    if rank == 0:
        print(f"[fw_dump] input ids ({len(ids)}): {ids}")

    device = torch.cuda.current_device()
    tokens = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
    tokzr = get_tokenizer()
    attn, _lossm, pos = get_ltor_masks_and_position_ids(
        tokens, tokzr.eod, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss)

    args.curriculum_learning_legacy = 1   # so RoPE uses curriculum_seqlen, not seq_length=4096
    engine.eval()
    with torch.no_grad():
        out = engine(tokens, pos, attn, curriculum_seqlen=len(ids))
    logits = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().cpu()
    logits = logits[0] if logits.dim() == 3 else logits   # [seq, vocab]

    if rank == 0:
        # order hidden states: emb, L0, L1, ... to match HF [emb, L1, L2, ...]
        ordered = [acts["emb"][0]] + [acts[f"L{i}"][0] for i in range(len(layers))]
        torch.save({"ids": ids, "hidden": ordered, "logits": logits,
                    "argmax": logits.argmax(-1)}, args.dump_out)
        print(f"[fw_dump] saved {len(ordered)} hidden states (each {tuple(ordered[0].shape)}) "
              f"+ logits {tuple(logits.shape)} -> {args.dump_out}")
        print(f"[fw_dump] next-token argmax: {logits.argmax(-1).tolist()}")

if __name__ == "__main__":
    main()
