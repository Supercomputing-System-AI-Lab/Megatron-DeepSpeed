#!/usr/bin/env python
# coding=utf-8
"""Run the HF DeepSeek-MoE reference to (1) report its LM loss on the SAME oscar data your
run trains on, and (2) dump reference logits for a fixed prompt so you can diff them against
the framework's output.

WHY: your framework run starts at loss ~4.8. To know whether that's "correct but confounded"
(OOD data + MoE token dropping) or an actual conversion bug, compare against HF's own forward
on the same tokens. HF does DENSE top-k with NO token dropping, so:
    - HF loss ~= framework loss  -> conversion + framework math agree; 4.8 is just the data.
    - HF loss << framework loss  -> the gap is token dropping (raise --moe-eval-capacity-factor
                                    and eval, or compare in eval mode) OR a conversion bug --
                                    run verify_conversion.py to rule the checkpoint in/out.

Needs: transformers, safetensors, and the model's remote code (modeling_deepseek.py,
configuration_deepseek.py) already sitting in --hf-model-path. 16B in bf16 ~= 32GB: use a GPU
node (--device cuda) or a big-RAM login node (--device cpu, slow).

USAGE
    python tools/convert_checkpoint/verify_hf_reference.py \
        --hf-model-path /lustre/orion/gen150/scratch/zixianw4/models/deepseek-moe-16b-base \
        --oscar examples_elmoe/data/oscar-1GB.jsonl \
        --seq-len 4096 --num-windows 8 --device cuda \
        --dump-logits ref_logits.pt
"""
import argparse
import json
import os

import torch


def build_windows(oscar_path, tokenizer, seq_len, num_windows):
    """Mirror the training pipeline: concatenate tokenized docs, chunk into seq_len blocks.

    Matches tools/preprocess_data.py (HFTokenizer, no --append-eod in the prep we ran), so the
    loss is comparable to the framework's. Reads only enough docs to fill num_windows blocks.
    """
    need = seq_len * num_windows + 1
    ids = []
    with open(oscar_path) as f:
        for line in f:
            txt = json.loads(line).get("text", "")
            if not txt:
                continue
            ids.extend(tokenizer.encode(txt))
            if len(ids) >= need:
                break
    windows = []
    for w in range(num_windows):
        chunk = ids[w * seq_len:(w + 1) * seq_len + 1]
        if len(chunk) == seq_len + 1:
            windows.append(chunk)
    return windows


def build_windows_from_bin(bin_prefix, seq_len, num_windows):
    """Read the framework's EXACT training tokens from the Megatron .bin (no re-tokenizing).

    This feeds HF the identical token stream the framework trains on -- same ids, no BOS,
    packed the same way -- so HF's loss on THIS isolates data (BOS/packing) from the forward:
        HF-on-.bin ~= 2.4  -> data is fine, the framework FORWARD differs (build the logit hook)
        HF-on-.bin ~= 4.8  -> it's the data prep (missing BOS / no EOS / packing), not the forward
    """
    import sys, os
    sys.path.insert(0, os.getcwd())
    from megatron.data.indexed_dataset import make_dataset, infer_dataset_impl
    ds = make_dataset(bin_prefix, infer_dataset_impl(bin_prefix))
    need = seq_len * num_windows + 1
    ids = []
    for i in range(len(ds)):
        ids.extend(int(x) for x in ds[i])
        if len(ids) >= need:
            break
    windows = []
    for w in range(num_windows):
        chunk = ids[w * seq_len:(w + 1) * seq_len + 1]
        if len(chunk) == seq_len + 1:
            windows.append(chunk)
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model-path", required=True)
    ap.add_argument("--oscar", help="oscar-1GB.jsonl (re-tokenized WITH BOS). Omit if using --bin.")
    ap.add_argument("--bin", default=None,
                    help="Megatron .bin prefix (framework's EXACT tokens, no BOS). Takes "
                    "precedence over --oscar. Use this to feed HF the same tokens the run trains on.")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--num-windows", type=int, default=8, help="How many seq_len blocks to average loss over")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dump-logits", default=None, help="Save logits for a fixed prompt to this .pt")
    ap.add_argument("--prompt", default="The capital of France is",
                    help="Fixed prompt for the logit dump (kept short so no MoE capacity overflow)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # DeepSeek's 2023 remote code (modeling_deepseek.py) imports symbols that newer
    # transformers moved/removed. They are only capability probes, unused on the forward
    # path -- shim any that are missing so trust_remote_code can exec the module. `from X
    # import Y` reads Y as an attribute of the already-loaded module X, so setting the
    # attribute here (before from_pretrained triggers the remote import) is enough.
    import transformers.utils.import_utils as _iu
    for _name, _default in [("is_torch_fx_available", lambda: False),
                            ("is_torch_fx_proxy", lambda *a, **k: False)]:
        if not hasattr(_iu, _name):
            setattr(_iu, _name, _default)

    dtype = torch.bfloat16

    print(f"Loading HF model (trust_remote_code) from {args.hf_model_path} ...")
    from transformers import AutoConfig
    tok = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)

    # config.json has rope_scaling: null (standard RoPE, theta 10000). Newer transformers
    # auto-populates config.rope_scaling with a default dict keyed 'rope_type', but the 2023
    # remote code's _init_rope reads ['type'] -> KeyError. Force it back to None so the model
    # takes its standard-RoPE branch (which is what null means).
    config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if isinstance(getattr(config, "rope_scaling", None), dict) and "type" not in config.rope_scaling:
        config.rope_scaling = None

    # No device_map=auto -> avoids the `accelerate` dependency. Loads to CPU (~32GB bf16),
    # then .to(device). The whole 16B fits one MI250X GCD (64GB), so a single-GPU move works.
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path, config=config, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager")   # avoid flash-attn/sdpa paths (ROCm + old remote code)
    model = model.to(args.device)
    model.eval()
    dev = next(model.parameters()).device
    print(f"  model on {dev}")

    # ---- (1) reference loss on the same data ----
    if args.bin:
        print(f"Reading framework tokens from {args.bin} (.bin, no BOS) into "
              f"{args.num_windows} x {args.seq_len} windows ...")
        windows = build_windows_from_bin(args.bin, args.seq_len, args.num_windows)
    else:
        assert args.oscar, "pass --oscar or --bin"
        print(f"Tokenizing oscar into {args.num_windows} x {args.seq_len} windows (WITH BOS) ...")
        windows = build_windows(args.oscar, tok, args.seq_len, args.num_windows)
    print(f"  got {len(windows)} full windows")

    losses = []
    with torch.no_grad():
        for i, chunk in enumerate(windows):
            ids = torch.tensor(chunk[:-1], device=dev).unsqueeze(0)
            labels = torch.tensor(chunk[1:], device=dev).unsqueeze(0)
            out = model(input_ids=ids)
            logits = out.logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1))
            losses.append(loss.item())
            print(f"  window {i}: loss {loss.item():.4f}")
    mean = sum(losses) / len(losses)
    print()
    print(f"HF REFERENCE mean LM loss on oscar (seq_len={args.seq_len}): {mean:.4f}  "
          f"(perplexity {torch.exp(torch.tensor(mean)).item():.1f})")
    print(f"  -> compare to your framework's step-1 loss (~4.8).")
    print(f"     close  => conversion correct; the gap you see is token dropping + OOD data.")
    print(f"     HF much lower => raise --moe-eval-capacity-factor / eval mode, and run")
    print(f"                      verify_conversion.py to rule the checkpoint in/out.")

    # ---- (2) fixed-prompt logits for a framework-side diff ----
    if args.dump_logits:
        enc = tok(args.prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[0].float().cpu()  # [seq, vocab]
        torch.save({"prompt": args.prompt,
                    "input_ids": enc["input_ids"][0].cpu(),
                    "logits": logits,
                    "argmax": logits.argmax(-1)}, args.dump_logits)
        print(f"\nSaved reference logits for prompt {args.prompt!r} -> {args.dump_logits}")
        print(f"  input_ids: {enc['input_ids'][0].tolist()}")
        print(f"  next-token argmax per position: {logits.argmax(-1).tolist()}")
        print("  (Have the framework forward the SAME input_ids in eval mode and compare "
              "logits[:, :100015] -- ignore the padded tail. Match on argmax/top-k over real "
              "tokens = correct.)")


if __name__ == "__main__":
    main()
