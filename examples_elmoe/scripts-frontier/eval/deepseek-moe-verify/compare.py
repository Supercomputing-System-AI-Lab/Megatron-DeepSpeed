#!/usr/bin/env python
# coding=utf-8
"""Compare HF vs framework per-layer hidden states + logits. Localizes the first divergence.

  python compare.py --hf hf_acts.pt --fw fw_acts.pt

Reads {ids, hidden:[emb,L1,...], logits, argmax} from each. Prints, per aligned layer, the
cosine similarity and max abs diff of the hidden state, then the final-logit agreement. The
FIRST layer whose cosine drops well below 1.0 is where the framework forward diverges from HF
-> that layer's sublayer (attention for the dense/all layers; MoE for layers 1..27) is the bug.
"""
import argparse, torch

def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--fw", required=True)
    ap.add_argument("--tol", type=float, default=0.99, help="cosine below this flags divergence")
    args = ap.parse_args()

    hf = torch.load(args.hf, map_location="cpu")
    fw = torch.load(args.fw, map_location="cpu")

    if hf["ids"] != fw["ids"]:
        print("!! INPUT IDS DIFFER — comparison invalid. Use identical --ids/--dump-ids.")
        print(f"   hf: {hf['ids'][:12]}...\n   fw: {fw['ids'][:12]}...")
        return 1

    Hh, Hf = hf["hidden"], fw["hidden"]
    n = min(len(Hh), len(Hf))
    print(f"aligning {n} hidden states (hf has {len(Hh)}, fw has {len(Hf)})")
    print(f"{'layer':<8}{'cosine':>12}{'max|d|':>14}{'  flag'}")
    first_bad = None
    for i in range(n):
        c = cos(Hh[i], Hf[i])
        d = (Hh[i].float() - Hf[i].float()).abs().max().item()
        name = "emb" if i == 0 else f"L{i-1}"
        flag = "" if c >= args.tol else "  <-- DIVERGES"
        if c < args.tol and first_bad is None:
            first_bad = name
        print(f"{name:<8}{c:>12.5f}{d:>14.4e}{flag}")

    # logits (HF vocab 102400 vs framework padded 100096 -> compare the common range)
    V = min(hf["logits"].shape[-1], fw["logits"].shape[-1])
    hl, fl = hf["logits"][..., :V], fw["logits"][..., :V]
    lc = cos(hl, fl)
    top1 = (hl.argmax(-1) == fl.argmax(-1)).float().mean().item()
    print(f"\nlogits cosine        : {lc:.5f}")
    print(f"next-token top-1 match: {top1*100:.1f}%")
    print(f"hf argmax: {hl.argmax(-1).tolist()}")
    print(f"fw argmax: {fl.argmax(-1).tolist()}")

    print()
    if first_bad is None and lc >= args.tol:
        print("VERDICT: forward MATCHES HF at every layer -> attention + MoE + swiglu + shared "
              "expert + output are all correct. Any eval gap is data (BOS) / tokenization.")
    else:
        loc = first_bad or "logits/output"
        kind = ("dense-MLP or attention" if loc in ("emb", "L0") else
                "attention or MoE (router/experts/swiglu/shared)")
        print(f"VERDICT: first divergence at {loc} -> the bug is in that layer's {kind}. "
              f"Everything BEFORE {loc} matches, so it's isolated there.")

if __name__ == "__main__":
    main()
