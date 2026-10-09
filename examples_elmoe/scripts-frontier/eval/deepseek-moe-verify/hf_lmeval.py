#!/usr/bin/env python
# coding=utf-8
"""Run HF deepseek-moe-16b through the SAME lm-eval tasks as our framework, for a controlled
harness-vs-harness / model-vs-model comparison.

WHY
---
Our framework scores arc_easy at 76.2 (acc) / 73.1 (acc_norm) 0-shot, but the DeepSeekMoE paper
reports 68.1. Three things could cause a ~5-8 pt gap: (a) our forward, (b) our lm-eval wrapper
(padding / BOS / tokenization), or (c) lm-eval's methodology differing from DeepSeek's internal
harness. This script isolates them by running the HF weights through lm-eval's OWN HFLM wrapper on
the identical task YAML (same prompt template, same acc/acc_norm computation). Compare:

    paper (DeepSeek harness, HF weights)     = 68.1        <- given
    THIS (lm-eval HFLM,      HF weights)     = ?           <- run this
    ours  (lm-eval XMoELM,   our framework)  = 76.2 / 73.1 <- registry

    THIS ~ ours  -> our stack agrees with standard lm-eval; the gap vs paper is harness methodology.
    THIS ~ paper -> lm-eval-on-HF matches paper but we don't -> difference is in OUR stack; then do
                    the per-doc drill-down (`--log-samples` here + the same on our path).

MC scoring is deterministic argmax over choice log-likelihoods -- there is NO temperature/sampling
involved, so decoding params are irrelevant here.

DEPENDENCY: lm-eval's HFLM needs `accelerate` (not in the env). Install once (no GPU needed):
    pip install accelerate            # into the eval env, or `pip install --user accelerate`

USAGE (GPU node; ~5 min model load + ~15 min arc_easy):
    export HF_DATASETS_CACHE=<eval>/cache/hf_datasets_cache
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    srun -N1 -n1 --gpus-per-node=1 python hf_lmeval.py \
        --model-path ../../../../../../models/deepseek-moe-16b-base \
        --tasks arc_easy,arc_challenge,piqa --num-fewshot 0 \
        --out hf_lmeval_arc.json --log-samples
Only 1 GCD is needed (16B bf16 ~ 32 GB fits in 64 GB); more GPUs do not help a single-process HFLM.
"""

import argparse
import json
import os
import sys
import time

# Local model + cached datasets only; never hit the Hub (compute node has no direct network).
for _k in ('HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ.setdefault(_k, '1')
# Point datasets at the EVAL cache (where prefetch_datasets.py stored them), so offline mode finds
# them regardless of the caller's shell. This file lives in .../eval/deepseek-moe-verify, and the
# cache is at .../eval/cache/hf_datasets_cache -> one dir up.
_EVAL_CACHE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            os.pardir, 'cache', 'hf_datasets_cache'))
os.environ.setdefault('HF_DATASETS_CACHE', _EVAL_CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True, help='local deepseek-moe-16b-base dir')
    ap.add_argument('--tasks', default='arc_easy', help='comma lm-eval task names')
    ap.add_argument('--num-fewshot', type=int, default=0)
    ap.add_argument('--batch-size', default='auto', help='int or "auto" (lm-eval tunes it)')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--limit', type=int, default=None, help='cap docs/task (debug only)')
    ap.add_argument('--out', default='hf_lmeval.json')
    ap.add_argument('--log-samples', action='store_true',
                    help='also dump per-doc choice log-likelihoods to <out>.samples.json for the '
                         '1:1 comparison against our framework')
    args = ap.parse_args()

    try:
        import torch
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    except ModuleNotFoundError as e:
        sys.exit(f"import failed ({e}). Run under transformers 4.36.2 (the groupgemm env) which "
                 f"natively runs the DeepSeek remote code; add lm_eval there with:\n"
                 f"  <groupgemm python> -m pip install -e "
                 f"/lustre/orion/gen150/scratch/zixianw4/lm-evaluation-harness 'transformers==4.36.2'")

    # IMPORTANT: transformers 5.x REMOVED APIs the DeepSeek 2024-era remote code needs
    # (is_torch_fx_available, DynamicCache.from_legacy_cache), so the forward only runs correctly
    # under transformers 4.36.2 -- the same env that produced the verified hf_dump reference. We
    # construct the model EXACTLY as hf_dump.py does (shim + rope-scaling guard + eager attn + bf16)
    # and hand the instance to HFLM, so lm-eval only contributes its SCORING, not model loading --
    # removing every model-construction variable from the comparison.
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, 'is_torch_fx_available'):
        _iu.is_torch_fx_available = lambda: False
    import transformers
    if not transformers.__version__.startswith('4.36'):
        print(f"[hf_lmeval] WARNING: transformers {transformers.__version__} != 4.36.x. The DeepSeek "
              f"forward is only verified under 4.36.2; results here may be invalid.", flush=True)

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    bs = args.batch_size if args.batch_size == 'auto' else int(args.batch_size)
    dtype = getattr(torch, args.dtype)
    print(f"[hf_lmeval] model={args.model_path}\n[hf_lmeval] tasks={tasks} "
          f"fewshot={args.num_fewshot} batch_size={bs} dtype={args.dtype} "
          f"transformers={transformers.__version__}", flush=True)

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if isinstance(getattr(cfg, 'rope_scaling', None), dict) and 'type' not in cfg.rope_scaling:
        cfg.rope_scaling = None
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=cfg, trust_remote_code=True,
        torch_dtype=dtype, attn_implementation='eager').to('cuda').eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
    print(f"[hf_lmeval] model loaded (eager attn, {args.dtype}) in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    res = simple_evaluate(model=lm, tasks=tasks, num_fewshot=args.num_fewshot,
                          limit=args.limit, log_samples=args.log_samples,
                          bootstrap_iters=0)
    print(f"[hf_lmeval] eval done in {time.time() - t0:.1f}s", flush=True)

    # Print the headline metrics the way the paper reports them.
    print("\n=== results (HF via lm-eval) ===")
    for task, m in sorted(res.get('results', {}).items()):
        acc = m.get('acc,none'); accn = m.get('acc_norm,none')
        parts = []
        if acc is not None:
            parts.append(f"acc={acc * 100:.1f}")
        if accn is not None:
            parts.append(f"acc_norm={accn * 100:.1f}")
        print(f"  {task:16s} {'  '.join(parts) or m}")

    samples = res.pop('samples', None)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n[hf_lmeval] wrote {args.out}")
    if args.log_samples and samples is not None:
        with open(args.out + '.samples.json', 'w') as f:
            json.dump(samples, f, default=str)
        print(f"[hf_lmeval] wrote per-doc samples -> {args.out}.samples.json")


if __name__ == '__main__':
    main()
