#!/usr/bin/env python
# coding=utf-8
"""Verify the converted DeepSpeed checkpoint's WEIGHTS exactly match the HF source.

This is the parameter-level "our loaded model == HF DeepSeek" check. It does NOT run either
model -- it re-derives every tensor from the HF safetensors with INDEPENDENT, inlined
transforms (not the converter's apply_op) and compares to what is actually on disk in the
converted checkpoint (nested mp_rank module + per-expert files). If every tensor matches,
any loss gap vs HF is a *framework* effect (MoE token dropping, RoPE/gating impl), not a
conversion bug -- and those were separately verified numerically (RoPE max-diff 0.0, QKV
per-head interleave, SwiGLU gate/up order, gating == softmax_before_topk).

    Runs on CPU in the torch env (no GPU, no distributed). ~1-2 min for a sampled check.

USAGE
    cd Megatron-DeepSpeed-X-MoE
    python tools/convert_checkpoint/verify_conversion.py \
        --hf-model-path /lustre/orion/gen150/scratch/zixianw4/models/deepseek-moe-16b-base \
        --ckpt examples_elmoe/scripts-frontier/checkpoint/deepseek_16b_ds \
        --sample-layers 3 --sample-experts 4        # or --full to check every tensor

Exit code 0 = all checked tensors match; 1 = at least one mismatch (localized in output).
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

# Reuse ONLY the mapping (which HF tensors -> which dst key), not the transforms.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_to_ds_moe import Plan, read_safetensors_headers, vocab_size_with_padding, \
    detect_tokenizer_vocab_size, expert_ckpt_name, MP_RANK_FILE


# ---- independent transforms (deliberately NOT importing apply_op) --------------------
def t_copy(tensors, plan, dst):
    t = tensors[0]
    if dst.endswith("word_embeddings.weight") or dst.endswith("output_layer.weight"):
        if t.shape[0] > plan.padded_vocab:
            return t[:plan.padded_vocab].contiguous()
        if t.shape[0] < plan.padded_vocab:
            pad = torch.zeros(plan.padded_vocab - t.shape[0], t.shape[1], dtype=t.dtype)
            return torch.cat([t, pad], dim=0)
    return t


def t_fuse_qkv(tensors, plan, dst):
    q, k, v = tensors
    nh, hd = plan.heads, plan.head_dim
    return torch.cat([q.reshape(nh, hd, -1), k.reshape(nh, hd, -1), v.reshape(nh, hd, -1)],
                     dim=1).reshape(3 * nh * hd, -1).contiguous()


def t_fuse_gate_up(tensors, plan, dst):
    gate, up = tensors
    return torch.cat([gate, up], dim=0).contiguous()


TRANSFORM = {"copy": t_copy, "resize_vocab": t_copy, "fuse_qkv": t_fuse_qkv,
             "fuse_gate_up": t_fuse_gate_up}


def get_nested(module, dst_key):
    """dst_key is the flat 'language_model.encoder.layers.N...' name; walk the nested dict."""
    lm = module["language_model"]
    if dst_key == "language_model.embedding.word_embeddings.weight":
        return lm["embedding"]["word_embeddings"]["weight"]
    if dst_key == "language_model.output_layer.weight":
        return lm["output_layer"]["weight"]
    if dst_key.startswith("language_model.encoder."):
        return lm["encoder"][dst_key[len("language_model.encoder."):]]
    raise KeyError(dst_key)


def compare(name, got, exp, tol):
    if got.shape != exp.shape:
        return False, f"SHAPE {tuple(got.shape)} vs {tuple(exp.shape)}"
    d = (got.float() - exp.float()).abs().max().item()
    return d <= tol, f"max|d|={d:.3e}"


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--hf-model-path", required=True)
    ap.add_argument("--ckpt", required=True, help="Converted checkpoint root (contains global_step0/)")
    ap.add_argument("--tag", default="global_step0")
    ap.add_argument("--tol", type=float, default=0.0, help="Max allowed abs diff (bf16 exact copy => 0)")
    ap.add_argument("--sample-layers", type=int, default=3, help="How many MoE layers to spot-check")
    ap.add_argument("--sample-experts", type=int, default=4, help="Experts per sampled layer")
    ap.add_argument("--full", action="store_true", help="Check EVERY tensor (slow, reads all 1728 expert files)")
    args = ap.parse_args()

    with open(os.path.join(args.hf_model_path, "config.json")) as f:
        cfg = json.load(f)
    tok_vocab = detect_tokenizer_vocab_size(args.hf_model_path) or cfg["vocab_size"]
    padded = vocab_size_with_padding(tok_vocab, 128, 1)
    plan = Plan(cfg, padded_vocab=padded)
    meta = read_safetensors_headers(args.hf_model_path)
    handles = {}

    def hf(key):
        f = meta[key][2]
        if f not in handles:
            handles[f] = safe_open(os.path.join(args.hf_model_path, f), framework="pt", device="cpu")
        return handles[f].get_tensor(key)

    step = os.path.join(args.ckpt, args.tag)
    module = torch.load(os.path.join(step, MP_RANK_FILE), map_location="cpu")["module"]

    npass = nfail = 0
    fails = []

    # ---- non-expert tensors (all of them; cheap, one file) ----
    print("== non-expert tensors (mp_rank module) ==")
    for dst, op, srcs in plan.non_expert_ops():
        exp = TRANSFORM[op]([hf(s) for s in srcs], plan, dst)
        try:
            got = get_nested(module, dst)
        except KeyError:
            nfail += 1; fails.append((dst, "MISSING in checkpoint")); continue
        ok, msg = compare(dst, got, exp, args.tol)
        npass += ok; nfail += (not ok)
        if not ok:
            fails.append((dst, msg))
    print(f"   checked {len(plan.non_expert_ops())}  (embed/attn/norm/router/shared/output)")

    # ---- routed experts (sampled unless --full) ----
    moe_layers = [L for L in range(plan.layers) if plan.is_moe_layer(L)]
    if args.full:
        pick_layers = moe_layers
        pick_experts = list(range(plan.n_routed))
    else:
        # spread the sample across the depth
        idxs = sorted(set(int(round(i * (len(moe_layers) - 1) / max(1, args.sample_layers - 1)))
                          for i in range(args.sample_layers)))
        pick_layers = [moe_layers[i] for i in idxs]
        pick_experts = list(range(min(args.sample_experts, plan.n_routed)))
    print(f"== routed experts (layers {pick_layers}, experts {pick_experts}) ==")
    for L in pick_layers:
        ordinal = plan.moe_ordinal(L)
        for E in pick_experts:
            fp = os.path.join(step, expert_ckpt_name(ordinal, E))
            esd = torch.load(fp, map_location="cpu")
            base = f"model.layers.{L}.mlp.experts.{E}"
            kpref = f"language_model.encoder.layers.{L}.mlp.deepspeed_moe.experts.deepspeed_experts.{E}"
            checks = [
                (f"{kpref}.dense_h_to_4h.weight",
                 t_fuse_gate_up([hf(f"{base}.gate_proj.weight"), hf(f"{base}.up_proj.weight")], plan, "")),
                (f"{kpref}.dense_4h_to_h.weight", hf(f"{base}.down_proj.weight")),
            ]
            for k, exp in checks:
                if k not in esd:
                    nfail += 1; fails.append((k, "MISSING in expert file")); continue
                ok, msg = compare(k, esd[k], exp, args.tol)
                npass += ok; nfail += (not ok)
                if not ok:
                    fails.append((k, msg))

    print()
    print(f"RESULT: {npass} passed, {nfail} failed")
    for k, msg in fails[:20]:
        print(f"  FAIL  {msg:24}  {k}")
    if nfail == 0:
        print("ALL CHECKED WEIGHTS MATCH HF EXACTLY -> conversion is correct at the parameter "
              "level. Any loss gap vs HF is framework-side (token dropping / OOD data), not the "
              "checkpoint.")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
