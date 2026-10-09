#!/usr/bin/env python
# coding=utf-8
"""Op-by-op HF-vs-framework compare. Finds the FIRST operation that diverges, per layer.

  python sublayer_compare.py --hf hf_sublayers1k.pt --fw fw_sublayers1k.pt

Both files are {f"L{li}.{op}": [S, feature]}. For each layer, ops are printed in CAUSAL order
(ln1 -> q,k,v -> q_rot,k_rot -> ctx -> attn_out -> ln2 -> router -> mlp_out) with global cosine +
position-band cosine + max|d|. The first op whose cosine drops (esp. at high positions) is the bug:
  ln1 off        -> layer input already wrong (upstream)
  q/k/v off      -> QKV projection or head layout
  q_rot/k_rot off (q/k ok) -> RoPE
  ctx off (q_rot/k_rot ok) -> attention kernel/softmax
  attn_out off (ctx ok)    -> o_proj
  mlp_out off (ln2 ok)     -> MLP / MoE
"""
import argparse, torch

ORDER = ["ln1", "q", "k", "v", "q_rot", "k_rot", "ctx", "attn_out", "ln2", "router", "mlp_out"]


def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--fw", required=True)
    ap.add_argument("--tol", type=float, default=0.99)
    args = ap.parse_args()
    hf = torch.load(args.hf, map_location="cpu", weights_only=False)
    fw = torch.load(args.fw, map_location="cpu", weights_only=False)

    layers = sorted({int(k.split(".")[0][1:]) for k in hf if k.startswith("L")})
    for li in layers:
        print(f"\n================ Layer {li} ================")
        print(f"{'op':<10}{'cosine':>10}{'max|d|':>12}   position-bands (cos)")
        first_bad = None
        for op in ORDER:
            key = f"L{li}.{op}"
            if key not in hf or key not in fw:
                if key in hf or key in fw:
                    print(f"{op:<10}   (only in {'hf' if key in hf else 'fw'}; skipped)")
                continue
            a, b = hf[key], fw[key]
            if a.shape != b.shape:
                if a.numel() == b.numel():
                    b = b.reshape(a.shape)
                else:
                    print(f"{op:<10}   SHAPE MISMATCH hf{tuple(a.shape)} fw{tuple(b.shape)} — skipped")
                    continue
            c = cos(a, b)
            d = (a.float() - b.float()).abs().max().item()
            S = a.shape[0]
            edges = [0, 16, 256, 1024, S]
            bands = [(edges[j], min(edges[j + 1], S)) for j in range(len(edges) - 1) if edges[j] < S]
            pb = "  ".join(f"[{lo}:{hi}]={cos(a[lo:hi], b[lo:hi]):.4f}" for lo, hi in bands)
            flag = "" if c >= args.tol else "  <-- DIVERGES"
            if c < args.tol and first_bad is None:
                first_bad = op
            print(f"{op:<10}{c:>10.5f}{d:>12.4e}   {pb}{flag}")
        print(f"  => first divergent op in L{li}: {first_bad or 'none (all match)'}")


if __name__ == "__main__":
    main()
