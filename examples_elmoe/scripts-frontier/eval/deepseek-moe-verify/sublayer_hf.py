#!/usr/bin/env python
# coding=utf-8
"""HF DeepSeek: capture OP-LEVEL tensors per layer for op-by-op HF-vs-framework localization.

Captures, for each --layers index, all normalized to [seq, feature] on CPU (batch dropped, heads
flattened) so they align 1:1 with the framework's sublayer dump (evaluate_v04 --dump-sublayers):
  ln1      = input_layernorm output                    [S, H]
  q,k,v    = q_proj/k_proj/v_proj output (pre-RoPE)     [S, H]
  q_rot,k_rot = q,k after apply_rotary_pos_emb          [S, H]  (heads flattened)
  ctx      = o_proj INPUT (attention context)           [S, H]
  attn_out = o_proj output                              [S, H]
  ln2      = post_attention_layernorm output            [S, H]
  mlp_out  = mlp / MoE output                           [S, H]

USAGE (GPU node):
  srun -N1 -n1 --gpus-per-node=8 python sublayer_hf.py \
     --hf-model-path ../../../../../../models/deepseek-moe-16b-base \
     --ids-file ids_1024.txt --layers 0,1 --out hf_sublayers1k.pt
"""
import argparse, os, sys, torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model-path", required=True)
    ap.add_argument("--ids-file", required=True, help="comma ids (from wikitext_ids.py --out-ids)")
    ap.add_argument("--layers", default="0,1", help="comma layer indices to capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    args = ap.parse_args()
    layer_ids = [int(x) for x in args.layers.split(",")]

    from transformers import AutoModelForCausalLM, AutoConfig
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False
    cfg = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if isinstance(getattr(cfg, "rope_scaling", None), dict) and "type" not in cfg.rope_scaling:
        cfg.rope_scaling = None
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl).to("cuda").eval()

    with open(args.ids_file) as f:
        ids = [int(x) for x in f.read().strip().split(",") if x.strip()]
    print(f"HF attn={args.attn_impl}; ids={len(ids)}; layers={layer_ids}")

    core = model.model                                   # DeepseekModel
    acts = {}

    def to_sh(t):                                        # [B,S,H] -> [S,H]
        return t.detach().float().cpu()[0]

    def out_hook(key):
        def h(_m, _i, o):
            acts[key] = to_sh(o[0] if isinstance(o, (tuple, list)) else o)
        return h

    def pre_hook(key):
        def h(_m, inp):
            acts[key] = to_sh(inp[0])
        return h

    for li in layer_ids:
        L = core.layers[li]
        L.input_layernorm.register_forward_hook(out_hook(f"L{li}.ln1"))
        L.self_attn.q_proj.register_forward_hook(out_hook(f"L{li}.q"))
        L.self_attn.k_proj.register_forward_hook(out_hook(f"L{li}.k"))
        L.self_attn.v_proj.register_forward_hook(out_hook(f"L{li}.v"))
        L.self_attn.o_proj.register_forward_pre_hook(pre_hook(f"L{li}.ctx"))
        L.self_attn.o_proj.register_forward_hook(out_hook(f"L{li}.attn_out"))
        L.post_attention_layernorm.register_forward_hook(out_hook(f"L{li}.ln2"))
        L.mlp.register_forward_hook(out_hook(f"L{li}.mlp_out"))

    # Patch module-level apply_rotary_pos_emb to capture rotated q/k (one call per layer, in order).
    attn_mod = sys.modules[type(core.layers[layer_ids[0]].self_attn).__module__]
    _orig = attn_mod.apply_rotary_pos_emb
    rope = {}
    ctr = [0]

    def _wrap(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        qe, ke = _orig(q, k, cos, sin, position_ids, unsqueeze_dim)
        # [B, nH, S, hd] -> [S, nH*hd]
        def flat(x):
            x = x.detach().float().cpu()[0]              # [nH, S, hd]
            return x.permute(1, 0, 2).reshape(x.shape[1], -1)
        rope[ctr[0]] = (flat(qe), flat(ke))
        ctr[0] += 1
        return qe, ke
    attn_mod.apply_rotary_pos_emb = _wrap

    inp = torch.tensor(ids, device="cuda").unsqueeze(0)
    with torch.no_grad():
        model(input_ids=inp, use_cache=False)
    attn_mod.apply_rotary_pos_emb = _orig

    # Every layer calls apply_rotary_pos_emb exactly once, in layer order, so call index == layer.
    for li in layer_ids:
        if li in rope:
            acts[f"L{li}.q_rot"], acts[f"L{li}.k_rot"] = rope[li]

    torch.save(acts, args.out)
    print(f"saved {len(acts)} op tensors ({sorted(acts.keys())}) -> {args.out}")
    for kk in sorted(acts):
        print(f"  {kk:16s} {tuple(acts[kk].shape)}")


if __name__ == "__main__":
    main()
