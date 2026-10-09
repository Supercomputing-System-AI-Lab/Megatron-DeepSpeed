#!/usr/bin/env python
# coding=utf-8
"""Does the MISSING BOS token explain the framework's ~4.8 vs HF's ~2.4?

Clean A/B on the SAME tokens: read the framework's exact training tokens from its .bin (which
have NO BOS), and run the HF reference model two ways --
    (1) no-BOS   : the window as-is (exactly what the framework trains on)
    (2) with-BOS : the same window with BOS (id 100000) prepended at position 0
-- and compare mean LM loss. Same weights, same tokens; the only difference is the leading BOS,
so this isolates the attention-sink / sequence-start effect.

    with-BOS ~= 2.4 and no-BOS ~= 4.8  -> the missing BOS IS the gap. Fix: prepend BOS in the
                                          data pipeline so each sequence starts with 100000.
    both ~= 2.4                        -> BOS is not it; something else about the .bin.
    both ~= 4.8                        -> the framework FORWARD differs -> logit-dump hook next.

Run on a GPU node. Needs the same (older) transformers you used for verify_hf_reference.py.

USAGE
    cd Megatron-DeepSpeed-X-MoE
    srun -N1 -n1 -c7 --gpus-per-node=8 python tools/convert_checkpoint/test_bos_effect.py \
        --hf-model-path ../../models/deepseek-moe-16b-base \
        --bin examples_elmoe/data/oscar-deepseek_text_document \
        --seq-len 4095 --num-windows 8
"""
import argparse
import os
import statistics
import sys

import torch

BOS_ID = 100000   # DeepSeek begin-of-sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model-path", required=True)
    ap.add_argument("--bin", required=True, help="Megatron .bin prefix (framework's exact tokens)")
    ap.add_argument("--seq-len", type=int, default=4095,
                    help="Real tokens per window. Default 4095 so [BOS]+window == 4096 <= max_position.")
    ap.add_argument("--num-windows", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False
    cfg = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if isinstance(getattr(cfg, "rope_scaling", None), dict) and "type" not in cfg.rope_scaling:
        cfg.rope_scaling = None
    print("Loading HF model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    dev = next(model.parameters()).device

    sys.path.insert(0, os.getcwd())
    from megatron.data.indexed_dataset import make_dataset, infer_dataset_impl
    ds = make_dataset(args.bin, infer_dataset_impl(args.bin))
    need = args.seq_len * args.num_windows + 1
    stream = []
    for i in range(len(ds)):
        stream.extend(int(x) for x in ds[i])
        if len(stream) >= need:
            break
    windows = [stream[w * args.seq_len:(w + 1) * args.seq_len] for w in range(args.num_windows)]
    windows = [w for w in windows if len(w) == args.seq_len]
    print(f"  {len(windows)} windows of {args.seq_len} real tokens (first ids: {windows[0][:6]})")
    assert windows[0][0] != BOS_ID, "window already starts with BOS -- .bin unexpectedly has BOS"

    def lm_loss(seq):
        ii = torch.tensor(seq[:-1], device=dev).unsqueeze(0)
        lab = torch.tensor(seq[1:], device=dev).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=ii).logits.float()
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), lab.view(-1)).item()

    no_bos, with_bos = [], []
    for i, w in enumerate(windows):
        a = lm_loss(w)
        b = lm_loss([BOS_ID] + w)
        no_bos.append(a); with_bos.append(b)
        print(f"  window {i}:  no-BOS {a:.4f}   with-BOS {b:.4f}   delta {a-b:+.4f}")

    mn, mb = statistics.mean(no_bos), statistics.mean(with_bos)
    print()
    print(f"MEAN  no-BOS (framework's data) : {mn:.4f}   (ppl {torch.exp(torch.tensor(mn)).item():.1f})")
    print(f"MEAN  with-BOS (prepend 100000) : {mb:.4f}   (ppl {torch.exp(torch.tensor(mb)).item():.1f})")
    print(f"MEAN  delta (no-BOS - with-BOS) : {mn - mb:+.4f}")
    print()
    print("READ:")
    print("  - no-BOS ~= your ~4.8 AND with-BOS ~= 2.4  => the missing BOS IS the gap.")
    print("    Fix: prepend BOS in preprocessing so each sequence starts with 100000. Forward is fine.")
    print("  - both ~= 2.4  => BOS isn't it; the framework FORWARD differs (build the logit hook).")


if __name__ == "__main__":
    main()
