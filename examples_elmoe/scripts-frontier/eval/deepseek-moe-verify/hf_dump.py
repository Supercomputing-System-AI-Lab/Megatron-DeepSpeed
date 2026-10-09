#!/usr/bin/env python
# coding=utf-8
"""HF DeepSeek reference: dump per-layer hidden states + final logits for fixed token ids.

output_hidden_states=True gives hidden_states as a tuple of length (num_layers+1):
  [0] = embedding output, [i] = output of layer i-1. We save all of them + logits so
  fw_dump.py's per-layer hooks can be aligned 1:1 in compare.py.

USAGE (GPU node):
  srun -N1 -n1 --gpus-per-node=8 python hf_dump.py \
     --hf-model-path ../../../../models/deepseek-moe-16b-base --out hf_acts.pt --with-bos
"""
import argparse, os, torch

# The model dir is fully local (config + modeling_deepseek.py + safetensors + tokenizer). On a
# Frontier compute node outbound HTTP is firewalled unless the OLCF proxy is exported, so a bare
# `from_pretrained(..., trust_remote_code=True)` HANGS on a hub lookup. Force offline by default
# (all files are local); override with HF_HUB_OFFLINE=0 if you ever point --hf-model-path at a repo id.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BOS = 100000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is Paris. The sun rises in the east.")
    ap.add_argument("--with-bos", action="store_true", help="prepend BOS (100000), matching training intent")
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"],
                    help="HF attention backend. 'eager' upcasts softmax to fp32 (high-precision "
                         "reference). Use 'sdpa'/'flash_attention_2' to match the framework's "
                         "bf16 Flash regime and isolate eager-vs-flash precision from a real bug.")
    ap.add_argument("--ids", default=None, help="comma-separated token ids to force EXACT same input as fw_dump")
    ap.add_argument("--ids-file", default=None, help="file with comma-separated ids (from wikitext_ids.py --out-ids); overrides --ids/--prompt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False
    cfg = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if isinstance(getattr(cfg, "rope_scaling", None), dict) and "type" not in cfg.rope_scaling:
        cfg.rope_scaling = None
    tok = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl).to("cuda").eval()
    print(f"HF attention implementation: {args.attn_impl}")

    if args.ids_file:
        with open(args.ids_file) as f:
            ids = [int(x) for x in f.read().strip().split(",") if x.strip()]
    elif args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        ids = tok.encode(args.prompt)
        if args.with_bos and ids and ids[0] != BOS:
            ids = [BOS] + ids
    print(f"input ids ({len(ids)}): {ids}")

    dev = next(model.parameters()).device
    inp = torch.tensor(ids, device=dev).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=inp, output_hidden_states=True, use_cache=False)
    hidden = [h[0].float().cpu() for h in out.hidden_states]   # list len L+1, each [seq, h]
    logits = out.logits[0].float().cpu()                       # [seq, vocab]
    torch.save({"ids": ids, "hidden": hidden, "logits": logits,
                "argmax": logits.argmax(-1)}, args.out)
    print(f"saved {len(hidden)} hidden states (each {tuple(hidden[0].shape)}) + logits "
          f"{tuple(logits.shape)} -> {args.out}")
    print(f"next-token argmax: {logits.argmax(-1).tolist()}")

if __name__ == "__main__":
    main()
