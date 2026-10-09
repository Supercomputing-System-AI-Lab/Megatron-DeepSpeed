"""deepseek-moe-verify dump/compare tooling.

Forward a fixed token sequence, capture per-layer (or per-op) tensors, and either save
them or print an inline HF-vs-framework divergence report -- then exit before lm-eval.
Driven by the --dump-* flags; see tasks_args in main.py. Megatron imports stay
function-local so importing this module never drags model code in.
"""
import torch


def _print_dim_breakdown(name, a, b):
    """Per-layer top divergent feature dims: is the gap concentrated in a few large-magnitude
    "massive-activation" dims (=> bf16 rounding, benign) or spread structurally (=> real bug)?"""
    a2, b2 = a.float().reshape(-1, a.shape[-1]), b.float().reshape(-1, a.shape[-1])
    diff = (a2 - b2).abs()                     # [S, D]
    per_dim = diff.max(0).values               # [D]
    k = min(5, per_dim.numel())
    top = per_dim.topk(k)
    tot = (diff ** 2).sum().item()
    share1 = (diff[:, top.indices[0]] ** 2).sum().item() / max(tot, 1e-9)
    print(f"[compare] --- {name} breakdown (top-{k} divergent feature dims) ---", flush=True)
    print(f"[compare]   {'dim':>6}{'|d|max':>12}{'|hf|max':>12}{'|fw|max':>12}", flush=True)
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        hfm = a2[:, idx].abs().max().item()
        fwm = b2[:, idx].abs().max().item()
        print(f"[compare]   {idx:>6}{val:>12.3f}{hfm:>12.2f}{fwm:>12.2f}", flush=True)
    print(f"[compare]   top-1 dim carries {share1 * 100:.1f}% of the total squared diff "
          f"(high % + large |hf|/|fw| => massive-activation bf16 rounding, benign; "
          f"low % spread across dims => structural).", flush=True)


def _print_hf_fw_compare(ref, fw_hidden, fw_logits, tol=0.99, names=None, breakdown_every=10):
    """deepseek-moe-verify: inline HF-vs-framework per-layer divergence report (rank 0 only).

    ref        = {ids, hidden:[emb,L0_out,...,norm(L27_out)], logits} saved by hf_dump.py.
    fw_hidden  = this framework's states in the SAME 1:1 order (last entry is post-final-norm too).
    fw_logits  = this framework's [seq, vocab] logits.
    names      = optional display labels per aligned state (defaults to emb / L{i-1}).
    Prints per-layer cosine + max|diff|, a top-divergent-dim breakdown for L0, L10, L20, ... (every
    `breakdown_every` layers) AND for the first flagged layer, then final-logit cosine + next-token
    top-1 agreement + a verdict. Mirrors compare.py so the verdict is inline in rank_0.log.
    """
    def cos(a, b):
        # float64, NOT float32: the logits are [seq, vocab] ~ 4e8 elements, and a float32
        # dot-product vs norm over that many heavy-tailed values disagree by ~15% (they use
        # different reduction orders), which makes the "cosine" exceed 1 -- a self-comparison
        # cos(logits, logits) reads 1.17 in float32, 1.0 in float64. float64 is exact at any size.
        a = a.flatten().double(); b = b.flatten().double()
        return (a @ b / (a.norm() * b.norm() + 1e-9)).item()

    def label(i):
        if names is not None and i < len(names):
            return names[i]
        return 'emb' if i == 0 else f'L{i-1}'

    Hh, Hf = ref['hidden'], fw_hidden
    n = min(len(Hh), len(Hf))
    print('\n' + '=' * 64, flush=True)
    print('[compare] HF (transformers) vs framework, per aligned layer', flush=True)
    print(f"[compare] aligning {n} states (hf {len(Hh)}, fw {len(Hf)}); flag cosine < {tol}", flush=True)
    print(f"[compare] {'layer':<10}{'cosine':>12}{'max|d|':>14}  flag", flush=True)
    first_bad = None
    store = {}                                 # i -> (name, a, b), for the breakdowns
    for i in range(n):
        a, b = Hh[i], Hf[i]
        if a.shape != b.shape and a.numel() == b.numel():
            b = b.reshape(a.shape)
        c = cos(a, b)
        d = (a.float() - b.float()).abs().max().item()
        name = label(i)
        store[i] = (name, a, b)
        bad = c < tol
        if bad and first_bad is None:
            first_bad = i
        print(f"[compare] {name:<10}{c:>12.5f}{d:>14.4e}  {'<-- DIVERGES' if bad else ''}", flush=True)

    # Breakdown for L0, L10, L20, ... (layer j is stored at index j+1) plus the first flagged layer.
    sel = {i for i in range(1, n) if (i - 1) % breakdown_every == 0}
    if first_bad is not None:
        sel.add(first_bad)
    for i in sorted(sel):
        _print_dim_breakdown(*store[i])

    # Position-band cosine: localizes whether the divergence is at HIGH positions (=> RoPE /
    # attention position-dependence, the long-context signature) vs uniform across positions.
    # Only meaningful for long sequences. L0 (dense, index 1) isolates attention (no MoE).
    S = store[1][1].shape[0] if n > 1 and store[1][1].dim() >= 1 else 0
    if S > 256:
        edges = [0, 16, 64, 256, 1024, S]
        bands = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1) if edges[j] < S]
        band_layers = sorted({1, max(1, n // 2), n - 1})   # L0 (dense), a middle layer, last
        print("[compare] --- position-band cosine (early match + late drop => RoPE/attention "
              "position bug) ---", flush=True)
        for i in band_layers:
            nm, a, b = store[i]
            cells = []
            for lo, hi in bands:
                hi = min(hi, S)
                cells.append(f"[{lo}:{hi}]={cos(a[lo:hi], b[lo:hi]):.4f}")
            print(f"[compare]   {nm:<10} " + "  ".join(cells), flush=True)

    # HF vocab (102400) vs framework padded vocab (100096): compare the common range only.
    V = min(ref['logits'].shape[-1], fw_logits.shape[-1])
    hl, fl = ref['logits'][..., :V], fw_logits[..., :V]
    lc = cos(hl, fl)
    top1 = (hl.argmax(-1) == fl.argmax(-1)).float().mean().item()
    print(f"[compare] logits cosine        : {lc:.5f}", flush=True)
    print(f"[compare] next-token top-1 match: {top1 * 100:.1f}%", flush=True)
    print(f"[compare] hf argmax: {hl.argmax(-1).tolist()}", flush=True)
    print(f"[compare] fw argmax: {fl.argmax(-1).tolist()}", flush=True)
    if first_bad is None and lc >= tol:
        print('[compare] VERDICT: forward MATCHES HF at every layer -> attention + MoE + swiglu + '
              'shared-expert + output all correct; any eval gap is data (BOS) / tokenization.',
              flush=True)
    else:
        loc = store[first_bad][0] if first_bad is not None else 'logits/output'
        kind = ('dense-MLP or attention' if loc in ('emb', 'L0')
                else 'attention or MoE (router / experts / swiglu / shared)')
        print(f"[compare] VERDICT: FIRST divergence at {loc} -> the bug is in that layer's {kind}. "
              f"Everything before {loc} matches HF, so it is isolated there.", flush=True)
    print('=' * 64, flush=True)


def _fw_capture_sublayers(lmod, layer_ids):
    """deepseek-moe-verify: register op-level hooks on the given framework layers and patch the
    rotary fn to capture rotated q/k. Returns (acts, rope_calls, restore). All tensors kept as-is;
    reshaping to [S, feature] happens at save time (needs np/head_dim). Rotary calls arrive in
    (q, k) order per layer, in layer order, so rope_calls[2*Li]=q_rot, [2*Li+1]=k_rot."""
    import megatron.model.transformer as _tf
    acts, handles = {}, []

    def sq(t):                                   # [S,B,...]->[S,...] (B=1)
        t = t.detach().float().cpu()
        return t.squeeze(1) if (t.dim() >= 3 and t.shape[1] == 1) else t

    def out_hook(key):
        def h(_m, _i, o):
            acts[key] = sq(o[0] if isinstance(o, (tuple, list)) else o)
        return h

    def pre_hook(key):
        def h(_m, inp):
            acts[key] = sq(inp[0])
        return h

    layers = lmod.encoder.layers
    for li in layer_ids:
        L = layers[li]; A = L.self_attention
        handles.append(L.input_layernorm.register_forward_hook(out_hook(f"L{li}.ln1")))
        handles.append(A.query_key_value.register_forward_hook(out_hook(f"L{li}.qkv")))
        handles.append(A.dense.register_forward_pre_hook(pre_hook(f"L{li}.ctx")))
        handles.append(A.dense.register_forward_hook(out_hook(f"L{li}.attn_out")))
        handles.append(L.post_attention_layernorm.register_forward_hook(out_hook(f"L{li}.ln2")))
        handles.append(L.mlp.register_forward_hook(out_hook(f"L{li}.mlp_out")))
        moe = getattr(L.mlp, 'deepspeed_moe', None)   # MoE layers: capture router logits (gate.wg)
        if moe is not None and hasattr(moe, 'gate') and hasattr(moe.gate, 'wg'):
            handles.append(moe.gate.wg.register_forward_hook(out_hook(f"L{li}.router")))

    rope_calls = []
    _orig = _tf.apply_rotary_pos_emb
    def _wrap(t, freqs):
        o = _orig(t, freqs)
        rope_calls.append(o.detach().float().cpu())   # [S,B,np,hn]
        return o
    _tf.apply_rotary_pos_emb = _wrap

    def restore():
        _tf.apply_rotary_pos_emb = _orig
        for h in handles:
            h.remove()
    return acts, rope_calls, restore


def _fw_save_sublayers(acts, rope_calls, layer_ids, n_head, head_dim, content_len, out_path):
    """Reshape captured framework ops to [S, feature] (heads flattened, B dropped) and save."""
    H = n_head * head_dim
    out = {}
    for li in layer_ids:
        pre = f"L{li}"
        if f"{pre}.qkv" in acts:                 # [S, np*3*hn] -> q,k,v each [S, np*hn]
            qkv = acts[f"{pre}.qkv"][:content_len]
            qkv = qkv.reshape(qkv.shape[0], n_head, 3 * head_dim)
            out[f"{pre}.q"] = qkv[:, :, 0 * head_dim:1 * head_dim].reshape(-1, H)
            out[f"{pre}.k"] = qkv[:, :, 1 * head_dim:2 * head_dim].reshape(-1, H)
            out[f"{pre}.v"] = qkv[:, :, 2 * head_dim:3 * head_dim].reshape(-1, H)
        for nm in ['ln1', 'ctx', 'attn_out', 'ln2', 'mlp_out', 'router']:
            if f"{pre}.{nm}" in acts:
                out[f"{pre}.{nm}"] = acts[f"{pre}.{nm}"][:content_len]
        if len(rope_calls) > 2 * li + 1:         # rotated q/k: [S,B,np,hn] -> [S, np*hn]
            for j, nm in ((2 * li, 'q_rot'), (2 * li + 1, 'k_rot')):
                r = rope_calls[j]
                r = r.squeeze(1) if (r.dim() == 4 and r.shape[1] == 1) else r
                out[f"{pre}.{nm}"] = r.reshape(r.shape[0], -1)[:content_len]
    torch.save(out, out_path)
    print(f"[sublayers] saved {len(out)} op tensors ({sorted(out.keys())}) -> {out_path}", flush=True)


def _dump_activations_and_exit(lm, model, args):
    """deepseek-moe-verify: forward fixed ids, capture per-layer hidden + logits (rank 0), then
    optionally save them (--dump-activations) and/or print an inline HF-vs-framework divergence
    report (--dump-compare-ref). Always exits before lm-eval (caller returns)."""
    from megatron.utils import get_ltor_masks_and_position_ids
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    m = model.module
    for _ in range(4):                       # peel FP16/DeepSpeed wrappers to reach GPTModel
        if hasattr(m, 'language_model'):
            break
        m = getattr(m, 'module', m)
    lmod = m.language_model
    layers = lmod.encoder.layers

    acts = {}
    def cap(name):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            acts[name] = h.detach().float().cpu().squeeze()   # [s,b,h] or [b,s,h] -> [seq,hidden]
        return hook
    lmod.embedding.register_forward_hook(cap('emb'))
    for i, layer in enumerate(layers):
        layer.register_forward_hook(cap(f'L{i}'))
    # HF's LAST hidden_states entry is POST the final RMSNorm (modeling_deepseek.py:1233-1237),
    # while a raw layer-27 hook is PRE-norm -> the two are different tensors and the last row would
    # show a spurious "divergence". Capture the final_layernorm output too so the last position is
    # compared apples-to-apples (post-norm vs post-norm). Falls back to raw L27 if not present.
    enc = lmod.encoder
    if getattr(enc, 'final_layernorm', None) is not None:
        enc.final_layernorm.register_forward_hook(cap('final'))

    # ---- resolve token ids. Priority: explicit --dump-ids > ids stored in the HF ref > default.
    # Taking ids from the ref guarantees HF and the framework forward IDENTICAL tokens (no copy-paste).
    ref = None
    if getattr(args, 'dump_compare_ref', None):
        ref = torch.load(args.dump_compare_ref, map_location='cpu', weights_only=False)
    if getattr(args, 'dump_ids', None):
        ids = [int(x) for x in args.dump_ids.split(',')]
    elif ref is not None and 'ids' in ref:
        ids = [int(x) for x in ref['ids']]
    else:
        ids = [100000, 549, 6077, 280, 7239, 317]
    content_len = len(ids)
    # --dump-pad N: forward the content ids RIGHT-PADDED to N with the eod pad token, then compare
    # only positions [:content_len]. This puts the forward in the SAME regime as the real eval
    # (every window padded to seq_length) so a padding/varlen-only bug is reproduced here.
    pad_to = getattr(args, 'dump_pad', 0) or 0
    pad_id = getattr(args, 'dump_pad_id', 0)     # 0 = the eval's real right-pad fill (torch.zeros)
    if pad_to and pad_to > content_len:
        fwd_ids = ids + [pad_id] * (pad_to - content_len)
    else:
        fwd_ids = ids
    if rank == 0:
        note = (f" (right-padded to {len(fwd_ids)} with pad id {pad_id} [eval uses 0]; "
                f"comparing content [:{content_len}])") if len(fwd_ids) != content_len else ""
        print(f"[dump] ids ({content_len}): {ids[:24]}{'...' if content_len > 24 else ''}{note}", flush=True)
    dev = torch.cuda.current_device()
    tokens = torch.tensor(fwd_ids, device=dev, dtype=torch.long).unsqueeze(0)
    attn, _lossm, pos = get_ltor_masks_and_position_ids(
        tokens, lm.tokenizer.eod, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss)
    # RoPE is sized from args.curriculum_seqlen ONLY when curriculum_learning_legacy is on
    # (gpt_model.py sets args.curriculum_seqlen from the passed curriculum_seqlen;
    #  language_model.py then does rotary_pos_emb(args.curriculum_seqlen)). Without this it
    #  defaults to the full args.seq_length (4096) and mismatches a short dump input. Setting
    #  these two only affects this one forward-and-exit dump; the normal eval path is untouched.
    # deepseek-moe-verify: op-level capture (ln1/qkv/rope/ctx/attn_out/ln2/mlp/router) for the
    # requested layers, to localize WHICH op first diverges from HF (sublayer_hf.py + compare).
    sub_ids = None
    if getattr(args, 'dump_sublayers', None):
        sub_ids = [int(x) for x in args.dump_sublayers.split(',') if x.strip() != '']
        sub_acts, sub_rope, sub_restore = _fw_capture_sublayers(lmod, sub_ids)

    args.curriculum_learning_legacy = 1
    model.eval()
    with torch.no_grad():
        out = model(tokens, pos, attn, curriculum_seqlen=len(fwd_ids))
    logits = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().cpu().squeeze()
    logits = logits[:content_len]                         # drop pad positions

    if sub_ids is not None:
        sub_restore()
        if rank == 0:
            out_path = (getattr(args, 'dump_sublayers_out', None)
                        or (args.dump_activations + '.sub.pt' if getattr(args, 'dump_activations', None)
                            else 'fw_sublayers.pt'))
            n_head = args.num_attention_heads            # TP=1 -> per-partition == full
            head_dim = args.hidden_size // args.num_attention_heads
            _fw_save_sublayers(sub_acts, sub_rope, sub_ids, n_head, head_dim, content_len, out_path)
    if rank == 0:
        # Hooks captured [len(fwd_ids), D]; slice back to the content positions for the compare.
        # Align to HF's [emb, out(L0), ..., out(L26), norm(out(L27))]: the last entry is the
        # final_layernorm output (post-norm), NOT the raw layer-27 residual (see hook comment).
        last_key = 'final' if 'final' in acts else f'L{len(layers) - 1}'
        ordered = ([acts['emb'][:content_len]]
                   + [acts[f'L{i}'][:content_len] for i in range(len(layers) - 1)]
                   + [acts[last_key][:content_len]])
        names = ['emb'] + [f'L{i}' for i in range(len(layers) - 1)] + [
            f'L{len(layers) - 1}+norm' if last_key == 'final' else f'L{len(layers) - 1}']
        print(f"[dump] argmax: {logits.argmax(-1).tolist()}", flush=True)
        if getattr(args, 'dump_activations', None):
            torch.save({'ids': ids, 'hidden': ordered, 'logits': logits,
                        'argmax': logits.argmax(-1)}, args.dump_activations)
            print(f"[dump] saved {len(ordered)} hidden (each {tuple(ordered[0].shape)}) + logits "
                  f"{tuple(logits.shape)} -> {args.dump_activations}", flush=True)
        if ref is not None:
            if ref.get('ids') is not None and [int(x) for x in ref['ids']] != ids:
                print(f"[compare] !! WARNING: ref ids {list(ref['ids'])[:8]}... != forwarded ids "
                      f"{ids[:8]}... — comparison is INVALID (omit --dump-ids to use the ref's ids).",
                      flush=True)
            _print_hf_fw_compare(ref, ordered, logits, names=names)
