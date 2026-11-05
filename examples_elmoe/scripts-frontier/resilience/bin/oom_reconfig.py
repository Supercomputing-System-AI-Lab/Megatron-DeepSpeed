#!/usr/bin/env python3
"""oom_reconfig.py: my three-layer rule upon a GPU out of memory (2026-10-10 01:45), as a function from the current layout to the
next layout and the reason, plus the memory gate that runs before the job's first attempt.

THE RULE, in my words. d = TP x U is the model-parallel degree of the dense part; W = the world size; dp_moe = W / (PP x EP);
dp_non_moe = W / (PP x d); activation checkpointing stays on; the tokens per step never change (so the gradient accumulation
GA = GBS / (mbs x dp_non_moe) follows the layout). Upon exit code 11 (gpu_oom) the loop takes ONE rung of a fixed ladder, never a sweep:
  step 0  if mbs > 1: mbs 1 with the accumulation doubled; the same save loads (no conversion).
  step 1  align d to EP (TP x U = EP) with the TP/U split of the fewest communication bytes read from the layout budget table
          (the bytes code of records/2026-10/1005_2026_LAYOUT_BUDGETS.html, copied below), among the splits that fit the memory.
  step 2  double EP and d together (dp_moe and dp_non_moe both halve). When d cannot double because the attention heads bound it
          (Ulysses needs heads / TP divisible by U: the 16B's 16 heads cap d at 16), EP doubles alone and the sentence says so.
  step 3  double PP, keep EP and d.
  after the last rung: STOP and mail; the model does not fit this machine at these tokens per step.
Every layout change (steps 1-3) loads the UNIVERSAL checkpoint of 'latest' (the UCP workers convert it); step 0 loads the same save.
THE MEMORY GATE before the run: the memory model (utils/v3_memory_model.py, unchanged) predicts every stage's peak; the layout
heatmap's colour is read from the published rows (records/2026-10/1005_2026_LAYOUT_BUDGETS_rows.json) when the cell is drawn there,
else computed by the same colour rule at the run's own tokens per step (a cell the page could not split over its replicas reads
"batch" there and has no colour); the out-of-memory line is allocated + 16 GiB (the largest allocator slack ever measured) + the
non-torch part, against the 64 GiB of a GCD. A red layout, or a layout above the line, gets a WARNING and the run proceeds; a
yellow layout or one within 4 GiB of the line is said to be at the margin. Never a refusal.
WHAT IS COPIED: groups_of_rank0 / n_local / frac_a2a / frac_ring / comm_step / fits and the per-model colour rules come verbatim from
assistant_deep_dive/2026-08-31-pagegen/make_layout_budgets.py (2026-10-05), because the sandbox tree has no copy of that page
generator and a job must not import matplotlib-side page code from the production tree; the memory model is imported from this
tree's utils/ (identical to production, checked 2026-10-10).
CLI (the loop calls these):
  oom_reconfig.py gate  --model DeepSeek_16B --nodes 64 --pp 2 --ep 16 --tp 1 --ulysses 1 --mbs 1 --gbs 256 --seq 131072 --chunked-ce 2048 --capacity 1.0 [--sp]
  oom_reconfig.py next  --model ... (the INITIAL layout) --taken 0 --oom-step 1 --oom-stage 1 --oom-rank 300 --oom-hbm 63.8 [--out-block FILE] [--shell]
  oom_reconfig.py ladder --model ... (the initial layout): the whole ladder, one line per rung
  oom_reconfig.py --selftest
"""
import argparse, json, math, os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS = os.environ.get('RES_UTILS') or os.path.normpath(os.path.join(HERE, '..', '..', '..', 'utils'))
if not os.path.exists(os.path.join(UTILS, 'v3_memory_model.py')):   # the mock tests copy bin/ elsewhere: fall back to this tree's utils
    UTILS = '/lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience/Megatron-DeepSpeed-X-MoE/examples_elmoe/utils'
sys.path.insert(0, UTILS)
from v3_memory_model import MemoryModel, Shapes, GiB, load_registry, non_torch_vram, SLACK_WORST   # noqa: E402

HEATMAP_JSON = os.environ.get('RES_HEATMAP_JSON', '/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/records/2026-10/1005_2026_LAYOUT_BUDGETS_rows.json')
GCD_GIB = 64.0            # the HBM of one MI250X GCD
MARGIN_GIB = 4.0          # "at the margin" = within this of the line
B16 = 2
REG = load_registry()

# the per-model colour rules of the layout heatmap (make_layout_budgets.py MODELS, 2026-10-05): the 16B is coloured by the allocated
# peak with the thresholds its measured runs set; the 671B by the VRAM estimate; any other model by the VRAM rule (validated on the 671B)
SPECS = {
    '16B': dict(rule='alloc', lo=50, hi=55, cce=2048, cap=1.0, key='16B',
                rule_text='green = allocated peak under 50 GiB, yellow = 50-55 (the zone the 32k production run occupied at 54.4), red = above 55 (the 128k probe without sequence parallelism failed at 58.3)'),
    'V3': dict(rule='vram', lo=60, hi=64, cce=1024, cap=1.25, key='V3',
               rule_text='green = VRAM estimate under 60 GiB, yellow = 60-64 (at the margin), red = above 64 (the rule validated on the 671B runs)'),
}


def spec_of(model):
    if model.startswith('DeepSeek_16B'):
        return SPECS['16B']
    if model.startswith('DeepSeek_V3'):
        return SPECS['V3']
    return dict(SPECS['V3'], key=None, rule_text=SPECS['V3']['rule_text'] + ' (the general rule; this model has no heatmap page)')


# ----------------------------------------------------------------------------------------------- copied from make_layout_budgets.py
def groups_of_rank0(pp, tp, P, R, ep, world):
    """The member ranks of rank 0's groups. Rank order: megatron/core/parallel_state.py (the 2026-09-17 Ulysses enablement):
    rank = ((p x R + d) x P + c) x TP + t, TP innermost. DeepSpeed's expert groups: X-MoE does not apply TP to the experts, so
    deepspeed/moe/layer.py selects the no-mpu builder _create_expert_and_data_parallel (deepspeed/utils/groups.py line 120): EP
    CONSECUTIVE ranks of the stage, whatever TP is; the expert-data group is the ranks at stride EP."""
    def rank_of(p, d, c, t): return ((p * R + d) * P + c) * tp + t
    stage = list(range(world // pp))
    return dict(tp=[rank_of(0, 0, 0, t) for t in range(tp)], ulysses=[rank_of(0, 0, c, 0) for c in range(P)],
                dp_dense=[rank_of(0, d, c, 0) for d in range(R) for c in range(P)], ep=stage[:ep], dp_expert=stage[::ep],
                pp=[rank_of(q, 0, 0, 0) for q in range(pp)])


def n_local(members): return sum(1 for r in members if r // 8 == 0)
def frac_a2a(n, nl): return 0.0 if n <= 1 else (n - nl) / (n - 1)         # all-to-all: the share of my peers that sit on other nodes
def frac_ring(n, nl): return 0.0 if nl >= n else 1.0 / nl                 # ring all-gather / reduce-scatter: one hop in nl crosses a node


def comm_step(spec, r, st):
    """bytes one GPU moves per training step (forward + backward) for one layout, total and across nodes, by term (the MegaScale-MoE
    term frame of the plan page, re-normalised per step; see the page's part 3 for the derivation of every line)."""
    S = spec['S']; tp, P, ep, pp, R = r['tp'], r['P'], r['ep'], r['pp'], r['R']; s = r['seq'] * r.get('mbs', 1)
    div = tp * P; t_slice = s / P; t_rank = s / div; h, H, k = S.h, S.heads, S.k
    g = groups_of_rank0(pp, tp, P, R, ep, r['world']); nl = {q: n_local(v) for q, v in g.items()}
    tp_attn = 2 * (tp - 1) / tp * t_slice * h * B16 if tp > 1 else 0.0
    if S.v3:
        uly_a2a = (P - 1) / P * t_slice * (H / tp) * (S.d_qk + S.d_v) * B16 if P > 1 else 0.0
        uly_ag = (P - 1) / P * s * (S.kv_lora + S.d_rope) * B16 if P > 1 else 0.0
    else:
        uly_a2a = (P - 1) / P * t_slice * (H / tp) * (h / H) * 4 * B16 if P > 1 else 0.0; uly_ag = 0.0
    ep_t = 2 * k * t_rank * h * (ep - 1) / ep * B16
    tp_mlp = 2 * (tp - 1) / tp * t_slice * h * B16 if tp > 1 else 0.0
    layers = st['layers']; n_lay = len(layers); n_moe = sum(1 for l in layers if S.is_moe(l)); mb = r['mb']
    f_u_a2a, f_u_ag, f_ep = frac_a2a(P, nl['ulysses']), frac_ring(P, nl['ulysses']), frac_a2a(ep, nl['ep'])
    terms = {}
    terms['tp'] = (2 * mb * n_lay * (tp_attn + tp_mlp), 0.0)
    terms['ulysses'] = (2 * mb * n_lay * (uly_a2a + uly_ag), 2 * mb * n_lay * (uly_a2a * f_u_a2a + uly_ag * f_u_ag))
    terms['ep'] = (2 * mb * n_moe * ep_t, 2 * mb * n_moe * ep_t * f_ep)
    pp_b = 2 * t_rank * h * 4 if pp > 1 else 0.0
    terms['pp'] = (mb * pp_b, mb * pp_b * (0.0 if nl['pp'] >= 2 else 1.0))
    g_d, g_e = P * R, r['dp_moe']
    dpd = 2 * (g_d - 1) / g_d * st['params_dense'] * B16; dpe = 2 * (g_e - 1) / g_e * st['params_expert'] * B16
    terms['dp'] = (dpd + dpe, dpd * frac_ring(g_d, nl['dp_dense']) + dpe * frac_ring(g_e, nl['dp_expert']))
    total = sum(v[0] for v in terms.values()); inter = sum(v[1] for v in terms.values())
    return dict(total=total, inter=inter, terms=terms, nl=nl, span=dict(ulysses=max(1, math.ceil(tp * P / 8)), ep=max(1, math.ceil(tp * ep / 8))))


def fits(spec, r):
    v = r['vram'] if spec['rule'] == 'vram' else r['alloc']
    lo, hi = spec['lo'], spec['hi']
    return 'green' if v <= lo else ('yellow' if v <= hi else 'red')
# ----------------------------------------------------------------------------------------------- end of the copied code


class Ctx:
    """everything a decision needs that is not the layout: the model entry, the budget, the sequence, the tokens per step"""
    def __init__(self, model, nodes, seq, gbs, chunked_ce=None, capacity=None, ckpt='full', gpus_per_node=8, heatmap=HEATMAP_JSON, intra_doc=False):
        self.model = model; self.entry = REG[model]; self.nodes = nodes; self.world = nodes * gpus_per_node; self.seq = seq; self.gbs = gbs
        self.spec = dict(spec_of(model)); self.spec['S'] = Shapes(self.entry)
        self.cce = chunked_ce if chunked_ce is not None else self.spec['cce']
        self.cap = capacity if capacity is not None else self.spec['cap']
        self.ckpt = ckpt
        self.intra_doc = bool(intra_doc)   # [2026-10-13] --intra-document-attention in the run: Ulysses cannot be chosen (megatron/arguments.py)
        self.heads = self.entry['heads']; self.experts = self.entry['ep']; self.layers = self.entry['layers']
        self.heatmap = None; self.heatmap_note = 'no heatmap file'
        try:
            if self.spec.get('key') and os.path.exists(heatmap):
                self.heatmap = json.load(open(heatmap)).get(self.spec['key'], [])
                self.heatmap_note = f'{heatmap} ({len(self.heatmap)} rows of the {self.spec["key"]} grids)'
        except Exception as e:                      # the page is read-only reference; the rule never depends on it
            self.heatmap = None; self.heatmap_note = f'the heatmap file could not be read ({e})'


def L(pp, ep, tp, u, mbs):
    return dict(pp=int(pp), ep=int(ep), tp=int(tp), u=int(u), mbs=int(mbs), d=int(tp) * int(u))


def lay_text(l, ctx=None):
    t = f"PP{l['pp']} EP{l['ep']} TP{l['tp']} x U{l['u']} (d = {l['d']}, mbs {l['mbs']}"
    if ctx is not None:
        dv = derived(l, ctx)
        t += f", dp_moe {dv['dp_moe']}, dp_non_moe {dv['dp_non_moe']}, GA {dv['ga']}"
    return t + ')'


def derived(l, ctx):
    per_stage = ctx.world // l['pp']
    dp_moe = per_stage // l['ep']; dp_non_moe = per_stage // l['d']
    ga = ctx.gbs // (l['mbs'] * dp_non_moe) if dp_non_moe and ctx.gbs % (l['mbs'] * dp_non_moe) == 0 else None
    return dict(dp_moe=dp_moe, dp_non_moe=dp_non_moe, ga=ga, per_stage=per_stage)


def infeasible(l, ctx):
    """None when the layout can be launched on this budget, else the sentence that says why not."""
    W, pp, ep, tp, u, d = ctx.world, l['pp'], l['ep'], l['tp'], l['u'], l['d']
    if pp < 1 or pp > ctx.layers or W % pp: return f'PP {pp} does not divide the {W} ranks or exceeds the {ctx.layers} layers'
    per_stage = W // pp
    if ep < 1 or per_stage % ep or ep > ctx.experts: return f'EP {ep} does not divide the {per_stage} ranks of a stage or exceeds the {ctx.experts} experts'
    if tp not in (1, 2, 4, 8): return f'TP {tp} is not one of 1, 2, 4, 8 (TP stays inside the node)'
    if ctx.heads % tp: return f'TP {tp} does not divide the {ctx.heads} attention heads'
    if tp > 1 and u > 1 and not ctx.entry.get('v3'):
        # [2026-10-10 12:45, sandbox job 5647530 (O2)] TP > 1 needs Megatron's companion sequence parallelism, and this tree composes it with
        # Ulysses only for the DeepSeek-V3 latent attention (megatron/initialize.py raises "not compatible with Megatron-LM's sequence parallel"
        # for any other attention): the rule chose TP2 x U4 for the 16B and every rank died at start-up, three attempts in a row
        return f'TP {tp} x Ulysses {u}: this model\'s attention cannot combine tensor parallelism with Ulysses (only the DeepSeek-V3 latent attention can; megatron/initialize.py)'
    if u > 1 and getattr(ctx, 'intra_doc', False):
        # [2026-10-13, sandbox, the 16B data-quality runs] intra-document attention builds its document boundaries on the full sequence before
        # get_batch slices it per Ulysses rank: megatron/arguments.py asserts ds_sequence_parallel_size == 1 under --intra-document-attention
        # (line 482, the standard attention; line 681 for the latent attention). Without this check the rule chose TP1 x U16 for the 16B at
        # TP8 EP8 with intra-document attention ON, a layout every rank would reject at start-up.
        return f'Ulysses {u}: this run uses intra-document attention, which megatron/arguments.py does not allow with Ulysses (the document boundaries would be cut per rank)'
    if u < 1 or (ctx.heads // tp) % u: return f'Ulysses {u} does not divide the {ctx.heads // tp} heads of a tensor-parallel rank (the heads bound d at {ctx.heads})'
    if per_stage % d: return f'd {d} does not divide the {per_stage} ranks of a stage'
    if ctx.seq % (u * tp): return f'the sequence {ctx.seq} is not divisible by U {u} x TP {tp}'
    if l['mbs'] < 1: return 'mbs below 1'
    dv = derived(l, ctx)
    if dv['ga'] is None: return f'the tokens per step ({ctx.gbs} sequences) cannot be split over dp_non_moe {dv["dp_non_moe"]} at mbs {l["mbs"]}'
    return None


def predict(l, ctx):
    """the memory model's per-stage prediction of the layout: allocated, reserved, VRAM estimate, the out-of-memory line (GiB)"""
    mm = MemoryModel(ctx.model, ctx.nodes, l['pp'], l['ep'], l['mbs'], ctx.gbs, ctx.seq, ctx.ckpt, tp=l['tp'], sp=l['tp'] > 1,
                     chunked_ce=ctx.cce, seq_div=l['u'], capacity=ctx.cap, mask=False, registry=REG)
    # mask=False: the pipeline path under flash attention builds no [s, s] mask (pretrain_gpt_deepspeed.py, the 2026-09-19 change:
    # a one-element placeholder per micro-batch and no constant), and the test runs without intra-document attention
    rows = []
    for s in range(l['pp']):
        r = mm.stage(s)
        rows.append(dict(stage=s, layers=f"{r['layers'][0]}-{r['layers'][-1]}" if r['layers'] else '-', inflight=r['inflight'],
                         alloc=r['allocated_peak'] / GiB, reserved=r['reserved_est'] / GiB, vram=r['vram_est'] / GiB, line=r['vram_worst'] / GiB,
                         params_dense=r['params_dense'], params_expert=r['params_expert'], layers_list=r['layers']))
    heaviest = max(rows, key=lambda r: r['vram'])
    return dict(rows=rows, heaviest=heaviest, micro_batches=mm.micro_batches, non_torch=non_torch_vram(ctx.world) / GiB, slack=SLACK_WORST / GiB)


def bytes_of(l, ctx, pred=None):
    """the bytes one GPU moves per step for the layout at the run's own tokens per step (the copied comm_step on the busiest stage)"""
    pred = pred or predict(l, ctx)
    dv = derived(l, ctx)
    S = ctx.spec['S']
    cst = max(pred['rows'], key=lambda q: (sum(1 for x in q['layers_list'] if S.is_moe(x)), len(q['layers_list'])))
    st = dict(layers=cst['layers_list'], params_dense=cst['params_dense'], params_expert=cst['params_expert'])
    r = dict(tp=l['tp'], P=l['u'], ep=l['ep'], pp=l['pp'], R=dv['dp_non_moe'], seq=ctx.seq, mbs=l['mbs'], world=ctx.world, mb=dv['ga'], dp_moe=dv['dp_moe'])
    c = comm_step(ctx.spec, r, st)
    return dict(total=c['total'] / GiB, inter=c['inter'] / GiB, terms={k: (v[0] / GiB, v[1] / GiB) for k, v in c['terms'].items()}, stage=cst['stage'])


def heatmap_cell(l, ctx):
    """the published heatmap row of this layout at this budget and sequence, when the page drew it (status ok); else None"""
    if not ctx.heatmap:
        return None
    for x in ctx.heatmap:
        if (x['nodes'] == ctx.nodes and x['seq'] == ctx.seq and x.get('mbs', 1) == l['mbs'] and x['pp'] == l['pp'] and x['ep'] == l['ep']
                and x['tp'] == l['tp'] and x['P'] == l['u']):
            return x
    return None


def colour_of(l, ctx, pred=None):
    """(colour, source sentence): the heatmap cell's colour when drawn, else the same rule applied to the model at the run's GBS"""
    cell = heatmap_cell(l, ctx)
    pred = pred or predict(l, ctx)
    hv = pred['heaviest']
    computed = fits(ctx.spec, dict(alloc=hv['alloc'], vram=hv['vram']))
    if cell is not None and cell.get('status') == 'ok':
        return cell['fit'], (f"the layout heatmap draws this cell {cell['fit']} ({cell['alloc']:.1f} allocated / {cell['vram']:.1f} VRAM GiB at its {cell['gbs']} sequences per step); "
                             f"at this run's {ctx.gbs} sequences per step the same rule gives {computed} ({hv['alloc']:.1f} / {hv['vram']:.1f})"), cell
    if cell is not None:
        why = 'the tokens per step cannot be split over that many replicas' if cell.get('status') == 'batch' else f"status {cell.get('status')}"
        return computed, (f"the layout heatmap has NO colour for this cell (it reads '{cell.get('status')}' there: {why}), so the colour is computed by the page's rule "
                          f"at this run's {ctx.gbs} sequences per step: {computed} ({hv['alloc']:.1f} allocated / {hv['vram']:.1f} VRAM GiB)"), cell
    return computed, (f"the layout heatmap has no cell for this budget and sequence, so the colour is computed by the page's rule at this run's "
                      f"{ctx.gbs} sequences per step: {computed} ({hv['alloc']:.1f} allocated / {hv['vram']:.1f} VRAM GiB)"), None


def gate(l, ctx):
    """the memory gate's verdict for the first attempt: the sentences for gate.log and the numbers for the record"""
    bad = infeasible(l, ctx)
    if bad:
        return dict(ok=False, warning=f"WARNING (the memory gate): the layout {lay_text(l)} cannot be launched on {ctx.nodes} nodes: {bad}; the run proceeds and the launch will say so.", lines=[])
    pred = predict(l, ctx); colour, src, cell = colour_of(l, ctx, pred); hv = pred['heaviest']
    by = bytes_of(l, ctx, pred)
    lines = [f"the memory gate (the memory model utils/v3_memory_model.py and the layout heatmap {os.path.basename(HEATMAP_JSON)}): layout {lay_text(l, ctx)} on {ctx.nodes} nodes at {ctx.seq // 1024}k, "
             f"{ctx.gbs} sequences per step ({ctx.gbs * ctx.seq / 1e6:.1f}M tokens), chunked CE {ctx.cce}, capacity {ctx.cap}, checkpointing {ctx.ckpt}.",
             f"  the colour rule of this model: {ctx.spec['rule_text']}.",
             f"  the colour: {src}.",
             f"  the model's peak per stage (GiB per GCD; VRAM = allocated + max(18 pct, 7 GiB) slack + {pred['non_torch']:.1f} non-torch; the out-of-memory line = allocated + {pred['slack']:.0f} GiB slack + non-torch, against {GCD_GIB:.0f}):"]
    for r in pred['rows']:
        lines.append(f"    stage {r['stage']} (layers {r['layers']}, {r['inflight']} in flight): allocated {r['alloc']:.1f}, reserved {r['reserved']:.1f}, VRAM {r['vram']:.1f}, out-of-memory line {r['line']:.1f}")
    lines.append(f"  the bytes one GPU moves per step at this layout: {by['total']:.1f} GiB total, {by['inter']:.1f} across nodes (stage {by['stage']}; terms tp {by['terms']['tp'][0]:.1f}, ulysses {by['terms']['ulysses'][0]:.1f}, ep {by['terms']['ep'][0]:.1f}, pp {by['terms']['pp'][0]:.1f}, dp {by['terms']['dp'][0]:.1f}).")
    worst_line = max(r['line'] for r in pred['rows']); ws = max(pred['rows'], key=lambda r: r['line'])['stage']
    red = colour == 'red' or worst_line > GCD_GIB
    margin = (not red) and (colour == 'yellow' or worst_line > GCD_GIB - MARGIN_GIB)
    if red:
        warning = (f"WARNING (the memory gate): the layout heatmap marks {lay_text(l)} at {ctx.seq // 1024}k {colour} (predicted peak {hv['alloc']:.1f} GiB allocated / {hv['vram']:.1f} VRAM on stage {hv['stage']}; "
                   f"the out-of-memory line {worst_line:.1f} GiB on stage {ws} against the {GCD_GIB:.0f} GiB of a GCD); the run proceeds, because the run itself is the test and the OOM rule handles a failure.")
    elif margin:
        warning = (f"WARNING (the memory gate): the layout {lay_text(l)} at {ctx.seq // 1024}k is AT THE MARGIN ({colour} by the heatmap rule: predicted peak {hv['alloc']:.1f} GiB allocated / {hv['vram']:.1f} VRAM on stage {hv['stage']}; "
                   f"the out-of-memory line {worst_line:.1f} GiB on stage {ws} is within {MARGIN_GIB:.0f} GiB of the {GCD_GIB:.0f} GiB of a GCD, and the model's known miss on the 16B at 128k is 6.4 GiB); the run proceeds.")
    else:
        warning = None
        lines.append(f"  verdict: the layout is {colour}; the out-of-memory line {worst_line:.1f} GiB (stage {ws}) is under the {GCD_GIB:.0f} GiB of a GCD by {GCD_GIB - worst_line:.1f} GiB; no warning.")
    return dict(ok=True, colour=colour, warning=warning, lines=lines, pred=pred, bytes=by, heaviest=hv, line=worst_line, line_stage=ws, cell=cell)


def split_choice(d, base, ctx):
    """the TP x U split of d with the fewest bytes among the splits that fit (the heatmap colour not red); returns (layout, sentence, table)"""
    cands = []
    for tp in (1, 2, 4, 8):
        if d % tp: continue
        l = dict(base, tp=tp, u=d // tp, d=d)
        if infeasible(l, ctx): continue
        pred = predict(l, ctx); col, _, _ = colour_of(l, ctx, pred); by = bytes_of(l, ctx, pred)
        cands.append(dict(l=l, colour=col, total=by['total'], inter=by['inter'], alloc=pred['heaviest']['alloc'], vram=pred['heaviest']['vram']))
    if not cands:
        return None, (f'no TP x U split of d = {d} can be launched (the heads, the stage size or the sequence do not allow it'
                      + ('; intra-document attention excludes every Ulysses split' if getattr(ctx, 'intra_doc', False) else '') + ')'), []
    fitting = [c for c in cands if c['colour'] != 'red'] or cands
    # the bytes across nodes decide when the splits differ by more than 1 GiB there (a byte over Slingshot costs about ten times a byte over
    # xGMI); at a fixed d the splits share the EP and Ulysses spans, so their across-nodes bytes tie within rounding and the TOTAL decides
    fitting.sort(key=lambda c: (round(c['inter']), c['total']))
    best = fitting[0]
    table = ', '.join(f"TP{c['l']['tp']} x U{c['l']['u']} {c['total']:.1f} GiB per step ({c['inter']:.1f} across nodes; {c['colour']})" for c in sorted(cands, key=lambda c: c['l']['tp']))
    runner = fitting[1] if len(fitting) > 1 else None
    sent = (f"the split with the fewest bytes: TP{best['l']['tp']} x U{best['l']['u']} at {best['total']:.1f} GiB per step"
            + (f" against {runner['total']:.1f} for TP{runner['l']['tp']} x U{runner['l']['u']}" if runner else '')
            + (f" (the splits marked red by the memory rule were not eligible)" if len(fitting) < len(cands) else '')
            + f"; all splits: {table}")
    return best['l'], sent, cands


def ladder(initial, ctx):
    """the rungs of the fixed ladder from the initial layout: [(step, layout, reason)]; a rung the rule cannot take is skipped with its sentence in 'skipped'"""
    rungs = []; skipped = []
    cur = dict(initial)
    # step 0: micro-batch 1
    if cur['mbs'] > 1:
        new = dict(cur, mbs=1)
        rungs.append(dict(step=0, layout=new, why=f"step 0: micro-batch {cur['mbs']} -> 1 with the accumulation x {cur['mbs']} (the tokens per step unchanged); the same save loads, no conversion"))
        cur = new
    else:
        skipped.append('step 0 is not needed: the micro-batch is already 1')
    # step 1: align d to EP
    if cur['d'] < cur['ep']:
        new, sent, _ = split_choice(cur['ep'], cur, ctx)
        if new is None:
            skipped.append(f'step 1 cannot align d to EP {cur["ep"]}: {sent}')
        else:
            rungs.append(dict(step=1, layout=new, why=f"step 1: d aligned to EP: d {cur['d']} -> {new['d']} (= EP {cur['ep']}); {sent}"))
            cur = new
    elif cur['d'] == cur['ep']:
        skipped.append(f"step 1 is not needed: d {cur['d']} already equals EP {cur['ep']}")
    else:
        skipped.append(f"step 1 is skipped: d {cur['d']} already exceeds EP {cur['ep']}, so aligning would lower d and raise the activations")
    # step 2: double EP and d together
    ep2 = cur['ep'] * 2; d2 = cur['d'] * 2
    ep_ok = infeasible(dict(cur, ep=ep2), ctx) is None
    d_split, d_sent, _ = split_choice(d2, dict(cur, ep=ep2 if ep_ok else cur['ep']), ctx)
    d_ok = d_split is not None and infeasible(d_split, ctx) is None
    if ep_ok and d_ok:
        new = dict(d_split, ep=ep2)
        rungs.append(dict(step=2, layout=new, why=f"step 2: EP and d doubled together: EP {cur['ep']} -> {ep2}, d {cur['d']} -> {d2} (dp_moe and dp_non_moe both halve); {d_sent}"))
        cur = new
    elif ep_ok:
        new = dict(cur, ep=ep2)
        rungs.append(dict(step=2, layout=new, why=f"step 2: EP doubled alone: EP {cur['ep']} -> {ep2} (dp_moe halves, the expert state per GPU halves); d cannot double because {d_sent if d_split is None else 'the ' + str(ctx.heads) + ' attention heads bound d at ' + str(ctx.heads)}"))
        cur = new
    elif d_ok:
        new = d_split
        rungs.append(dict(step=2, layout=new, why=f"step 2: d doubled alone: d {cur['d']} -> {d2} (dp_non_moe halves); EP cannot double: {infeasible(dict(cur, ep=ep2), ctx)}; {d_sent}"))
        cur = new
    else:
        skipped.append(f"step 2 cannot double EP ({infeasible(dict(cur, ep=ep2), ctx)}) nor d ({d_sent})")
    # step 3: double PP
    new = dict(cur, pp=cur['pp'] * 2)
    bad = infeasible(new, ctx)
    if bad is None:
        rungs.append(dict(step=3, layout=new, why=f"step 3: PP doubled: PP {cur['pp']} -> {new['pp']}, EP {cur['ep']} and d {cur['d']} kept (half the layers per stage)"))
        cur = new
    else:
        skipped.append(f'step 3 cannot double PP: {bad}')
    return rungs, skipped


def decide(initial, taken, ctx, oom=None):
    """the decision upon the (taken + 1)-th OOM of the job: the next rung or the stop; returns a dict with the prose block"""
    rungs, skipped = ladder(initial, ctx)
    oom = oom or {}
    head = (f"upon the out of memory at step {oom.get('step', '?')} on stage {oom.get('stage', '?')} (rank {oom.get('rank', '?')}, HBM {oom.get('hbm', '?')} GB allocated at the failure)"
            f" in attempt {oom.get('attempt', '?')}, with {taken} re-configuration(s) already taken in this job")
    if taken >= len(rungs):
        text = (f"{head}, the ladder is exhausted ({len(rungs)} rung(s): {'; '.join(r['why'].split(':')[0] for r in rungs) or 'none'}"
                f"{'; skipped: ' + '; '.join(skipped) if skipped else ''}): the rule STOPS the run and sends mail, because the model does not fit this machine at these tokens per step.")
        return dict(stop=True, text=text, rungs=rungs, skipped=skipped)
    r = rungs[taken]; new = r['layout']
    prev = initial if taken == 0 else rungs[taken - 1]['layout']
    pred = predict(new, ctx); by = bytes_of(new, ctx, pred); col, src, _ = colour_of(new, ctx, pred)
    pr = ', '.join(f"stage {q['stage']} {q['alloc']:.1f} allocated / {q['vram']:.1f} VRAM / line {q['line']:.1f}" for q in pred['rows'])
    loads = 'the same save loads (no conversion)' if r['step'] == 0 else "the universal checkpoint of 'latest' loads (the layout changed)"
    text = (f"{head}, the rule takes {r['why']}: {lay_text(prev)} -> {lay_text(new, ctx)}; {loads}; "
            f"the model predicts for the new layout: {pr} GiB ({col} by the heatmap rule: {src}); the bytes per step {by['total']:.1f} GiB ({by['inter']:.1f} across nodes)."
            + (f" Not taken on the way: {'; '.join(skipped)}." if skipped else ''))
    return dict(stop=False, step=r['step'], layout=new, prev=prev, text=text, pred=pred, bytes=by, colour=col, rungs=rungs, skipped=skipped, needs_universal=r['step'] != 0)


def ep_clipped(l, ctx):
    """the layout with EP lowered to the largest power of two that divides the ranks of a stage, when EP no longer does (fewer nodes)"""
    per_stage = ctx.world // l['pp'] if l['pp'] and ctx.world % l['pp'] == 0 else 0
    ep = l['ep']
    while ep > 1 and (per_stage == 0 or per_stage % ep):
        ep //= 2
    return dict(l, ep=ep) if ep != l['ep'] else None


def fits_now(l, ctx, pred=None):
    """(True/False, pred, colour): the layout fits when its colour is not red and the VRAM estimate of its heaviest stage is within a GCD"""
    pred = pred or predict(l, ctx)
    col, _, _ = colour_of(l, ctx, pred)
    return (col != 'red' and pred['heaviest']['vram'] <= GCD_GIB), pred, col


def shrink(initial, good, ctx_kw):
    """[2026-10-11, the user's rule] no spare node is left: the largest node count M <= good at which the CURRENT layout (EP clipped to
    the stage when it no longer divides it), or failing that a layout of the out-of-memory fallback order computed at M, can be launched
    and fits the memory model. Returns a dict (nodes, layout, why, pred, colour, tried) or None when no node count fits. Never a sweep
    over layouts: at each M the candidates are the current layout and the fallback order's layouts, in that order."""
    tried = []
    for m in range(int(good), 0, -1):
        ctx = Ctx(nodes=m, **ctx_kw)
        cands = [('the same configuration on fewer nodes', initial)]
        c = ep_clipped(initial, ctx)
        if c is not None:
            cands.append((f"the same configuration with EP {initial['ep']} -> {c['ep']} (EP must divide the {ctx.world // initial['pp']} ranks of a stage)", c))
        base = c if (c is not None and infeasible(initial, ctx)) else initial
        for r in ladder(base, ctx)[0]:
            cands.append((r['why'], r['layout']))
        for why, l in cands:
            bad = infeasible(l, ctx)
            if bad:
                continue
            ok, pred, col = fits_now(l, ctx)
            tried.append((m, lay_text(l), col, round(pred['heaviest']['vram'], 1)))
            if ok:
                return dict(nodes=m, layout=l, why=why, pred=pred, colour=col, ctx=ctx, tried=tried)
    return dict(nodes=0, tried=tried)


# ----------------------------------------------------------------------------------------------- the CLI
def _ctx_from(a):
    return Ctx(a.model, a.nodes, a.seq, a.gbs, chunked_ce=a.chunked_ce, capacity=a.capacity, ckpt=a.ckpt, intra_doc=a.intra_doc)


def _layout_from(a):
    return L(a.pp, a.ep, a.tp, a.ulysses, a.mbs)


def shell_assign(d):
    return '\n'.join(f'{k}={shlex.quote(str(v))}' for k, v in d.items())


def main():
    ap = argparse.ArgumentParser(description='the OOM re-configuration rule and the memory gate')
    ap.add_argument('cmd', nargs='?', default='gate', choices=['gate', 'next', 'ladder', 'shrink'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--model'); ap.add_argument('--nodes', type=int); ap.add_argument('--seq', type=int); ap.add_argument('--gbs', type=int)
    ap.add_argument('--pp', type=int, default=1); ap.add_argument('--ep', type=int, default=8); ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--ulysses', type=int, default=1); ap.add_argument('--mbs', type=int, default=1)
    ap.add_argument('--chunked-ce', type=int, default=None); ap.add_argument('--capacity', type=float, default=None)
    ap.add_argument('--ckpt', default='full', choices=['full', 'none'])
    ap.add_argument('--intra-doc', action='store_true', help='[2026-10-13] the run uses --intra-document-attention: no Ulysses split (U must be 1)')
    ap.add_argument('--taken', type=int, default=0, help='re-configurations already taken in this job (the rung index)')
    ap.add_argument('--oom-step', default='?'); ap.add_argument('--oom-stage', default='?'); ap.add_argument('--oom-rank', default='?'); ap.add_argument('--oom-hbm', default='?'); ap.add_argument('--oom-attempt', default='?')
    ap.add_argument('--out-block', default=None, help='write the prose block here'); ap.add_argument('--shell', action='store_true', help='print shell assignments for the loop')
    ap.add_argument('--json', default=None)
    ap.add_argument('--good', type=int, default=None, help='shrink: the healthy nodes left for training')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.model and a.nodes and a.seq and a.gbs):
        ap.error('--model --nodes --seq --gbs are required')
    ctx = _ctx_from(a); l = _layout_from(a)
    if a.cmd == 'gate':
        g = gate(l, ctx)
        for s in g['lines']:
            print(s)
        if g['warning']:
            print(g['warning'])
        if a.shell:
            print(shell_assign(dict(MG_COLOUR=g.get('colour', '-'), MG_WARNING=g['warning'] or '', MG_LINE=f"{g.get('line', 0):.1f}", MG_ALLOC=f"{g['heaviest']['alloc']:.1f}" if g.get('heaviest') else '-',
                                    MG_VRAM=f"{g['heaviest']['vram']:.1f}" if g.get('heaviest') else '-', MG_STAGE=g['heaviest']['stage'] if g.get('heaviest') else '-',
                                    MG_BYTES=f"{g['bytes']['total']:.1f}" if g.get('bytes') else '-', MG_INTER=f"{g['bytes']['inter']:.1f}" if g.get('bytes') else '-',
                                    MG_PRED=';'.join(f"{r['stage']}:{r['alloc']:.1f}/{r['vram']:.1f}/{r['line']:.1f}" for r in g['pred']['rows']) if g.get('pred') else '-')))
        if a.json:
            json.dump({k: v for k, v in g.items() if k in ('colour', 'warning', 'lines', 'line', 'line_stage')} | dict(pred=g['pred']['rows'] if g.get('pred') else None, bytes=g.get('bytes')), open(a.json, 'w'), indent=1, default=str)
        return
    if a.cmd == 'ladder':
        rungs, skipped = ladder(l, ctx)
        print(f'the ladder from {lay_text(l, ctx)} on {ctx.nodes} nodes at {ctx.seq // 1024}k, {ctx.gbs} sequences per step:')
        for r in rungs:
            p = predict(r['layout'], ctx); b = bytes_of(r['layout'], ctx, p)
            print(f"  rung {rungs.index(r)} = {r['why']} -> {lay_text(r['layout'], ctx)}: heaviest stage {p['heaviest']['stage']} {p['heaviest']['alloc']:.1f} allocated / {p['heaviest']['vram']:.1f} VRAM / line {p['heaviest']['line']:.1f} GiB; bytes {b['total']:.1f} / {b['inter']:.1f} GiB")
        for s in skipped:
            print(f'  skipped: {s}')
        print('  after the last rung: STOP and mail')
        return
    if a.cmd == 'shrink':
        kw = dict(model=a.model, seq=a.seq, gbs=a.gbs, chunked_ce=a.chunked_ce, capacity=a.capacity, ckpt=a.ckpt, intra_doc=a.intra_doc)
        r = shrink(l, a.good if a.good is not None else a.nodes, kw)
        good = a.good if a.good is not None else a.nodes
        if r['nodes'] == 0:
            text = (f"no spare node is left and {good} healthy nodes remain for the {a.nodes} the run was planned on: the memory model finds NO configuration "
                    f"that can be launched on {good} or fewer nodes and fits (the current one, {lay_text(l)}, and the out-of-memory fallback order were tried "
                    f"at every node count; {len(r['tried'])} of them could be launched, none fits); the run stops and the user is notified.")
            if a.shell:
                print(shell_assign(dict(SHRINK_NEXT=0, SHRINK_TEXT=text)))
            else:
                print(text)
            return
        n = r['layout']; ctx = r['ctx']; dv = derived(n, ctx); hv = r['pred']['heaviest']
        same = all(n[k] == l[k] for k in ('pp', 'ep', 'tp', 'u', 'mbs'))
        text = (f"no spare node is left and {good} healthy nodes remain for the {a.nodes} the run was planned on: the memory model's largest fitting choice is "
                f"{r['nodes']} nodes ({r['nodes'] * 8} GPUs) with {lay_text(n, ctx)}, which is {r['why']}; the heaviest stage {hv['stage']} is predicted at "
                f"{hv['alloc']:.1f} GiB allocated / {hv['vram']:.1f} GiB VRAM ({r['colour']} by the memory rule); the tokens per step are unchanged "
                f"({a.gbs} sequences, GA {dv['ga']}); the restart loads the universal checkpoint because the data-parallel width "
                f"{'changes' if same else 'and the parallel layout change'}; {good - r['nodes']} healthy node(s) stay idle as spares.")
        if a.shell:
            print(shell_assign(dict(SHRINK_NEXT=1, SHRINK_TEXT=text, SHRINK_NODES=r['nodes'], NEW_PP=n['pp'], NEW_EP=n['ep'], NEW_TP=n['tp'], NEW_U=n['u'],
                                    NEW_MBS=n['mbs'], NEW_D=n['d'], NEW_GA=dv['ga'], NEW_LAYOUT=lay_text(n, ctx), OLD_LAYOUT=lay_text(l),
                                    NEW_PRED=';'.join(f"{q['stage']}:{q['alloc']:.1f}/{q['vram']:.1f}/{q['line']:.1f}" for q in r['pred']['rows']),
                                    NEW_COLOUR=r['colour'], NEW_VRAM=f"{hv['vram']:.1f}")))
        else:
            print(text)
            for t in r['tried']:
                print(f'  tried: {t[0]} nodes, {t[1]}: {t[2]}, VRAM {t[3]} GiB')
        return
    if a.cmd == 'next':
        d = decide(l, a.taken, ctx, oom=dict(step=a.oom_step, stage=a.oom_stage, rank=a.oom_rank, hbm=a.oom_hbm, attempt=a.oom_attempt))
        if a.out_block:
            open(a.out_block, 'w').write(d['text'] + '\n')
        if a.shell:
            if d['stop']:
                print(shell_assign(dict(OOM_NEXT=0, OOM_TEXT=d['text'])))
            else:
                n = d['layout']; dv = derived(n, ctx); hv = d['pred']['heaviest']
                print(shell_assign(dict(OOM_NEXT=1, OOM_STEP=d['step'], OOM_TEXT=d['text'], OOM_NEEDS_UNIVERSAL=1 if d['needs_universal'] else 0,
                                        NEW_PP=n['pp'], NEW_EP=n['ep'], NEW_TP=n['tp'], NEW_U=n['u'], NEW_MBS=n['mbs'], NEW_D=n['d'], NEW_SP=1 if n['tp'] > 1 else 0,
                                        NEW_DP_MOE=dv['dp_moe'], NEW_DP_NON_MOE=dv['dp_non_moe'], NEW_GA=dv['ga'], NEW_LAYOUT=lay_text(n, ctx), OLD_LAYOUT=lay_text(d['prev'], ctx),
                                        NEW_PRED=';'.join(f"{r['stage']}:{r['alloc']:.1f}/{r['vram']:.1f}/{r['line']:.1f}" for r in d['pred']['rows']),
                                        NEW_ALLOC=f"{hv['alloc']:.1f}", NEW_VRAM=f"{hv['vram']:.1f}", NEW_LINE=f"{max(r['line'] for r in d['pred']['rows']):.1f}", NEW_COLOUR=d['colour'],
                                        NEW_BYTES=f"{d['bytes']['total']:.1f}", NEW_INTER=f"{d['bytes']['inter']:.1f}")))
        else:
            print(d['text'])
        return


# ----------------------------------------------------------------------------------------------- the self-test (tests/test_oom_reconfig.py calls these too)
def selftest():
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(('PASS ' if cond else 'FAIL ') + msg)
        ok = ok and bool(cond)
    # 1. the 16B test case: 64 nodes, 128k, PP2 EP16 TP1 U1 mbs 1, 256 sequences per step
    ctx = Ctx('DeepSeek_16B', 64, 131072, 256, chunked_ce=2048, capacity=1.0)
    l0 = L(2, 16, 1, 1, 1)
    g = gate(l0, ctx)
    check(g['colour'] == 'red' and g['warning'] and 'WARNING' in g['warning'], f"16B PP2 EP16 d1: the gate warns red ({g.get('colour')}, line {g.get('line', 0):.1f})")
    rungs, skipped = ladder(l0, ctx)
    check([r['step'] for r in rungs] == [1, 2, 3], f'16B ladder steps {[r["step"] for r in rungs]} (step 0 skipped at mbs 1)')
    check(rungs[0]['layout']['d'] == 16 and rungs[0]['layout']['tp'] == 1 and rungs[0]['layout']['u'] == 16, f"16B step 1 = {lay_text(rungs[0]['layout'])} (the fewest bytes among the splits of 16: TP1 x U16)")
    check(rungs[1]['layout']['ep'] == 32 and rungs[1]['layout']['d'] == 16, f"16B step 2 = {lay_text(rungs[1]['layout'])} (EP doubles alone: the 16 heads bound d at 16)")
    check(rungs[2]['layout']['pp'] == 4 and rungs[2]['layout']['ep'] == 32 and rungs[2]['layout']['d'] == 16, f"16B step 3 = {lay_text(rungs[2]['layout'])}")
    d3 = decide(l0, 3, ctx); check(d3['stop'] and 'STOPS' in d3['text'], 'the fourth OOM stops with mail')
    d0 = decide(l0, 0, ctx, dict(step=1, stage=1, rank=300, hbm=63.8, attempt=0)); check(not d0['stop'] and d0['step'] == 1 and d0['needs_universal'], 'the first OOM takes step 1 and needs the universal')
    # 2. the 16B d2 layout as I was asked to read it: the heatmap has no colour (the 'batch' cell) and the model says it fits
    ctx2 = Ctx('DeepSeek_16B', 64, 131072, 128, chunked_ce=2048, capacity=1.0)
    c2, src2, cell2 = colour_of(L(2, 16, 1, 2, 1), ctx2)
    check(cell2 is not None and cell2.get('status') == 'batch' and c2 == 'green', f"16B PP2 EP16 TP1 U2: the heatmap cell reads '{cell2.get('status') if cell2 else None}', the computed colour is {c2}")
    # 3. the user's example chain on a model whose heads allow d 32 and whose TP1 column is red: the 671B at 256 nodes, 128k, mbs 2
    ctx3 = Ctx('DeepSeek_V3_671B', 256, 131072, 480, chunked_ce=1024, capacity=1.25)
    l3 = L(8, 32, 2, 16, 2)
    r3, s3 = ladder(l3, ctx3)
    check([r['step'] for r in r3] == [0, 2, 3], f'671B production ladder steps {[r["step"] for r in r3]} (step 1 skipped: d = EP = 32 already)')
    check(r3[0]['layout']['mbs'] == 1 and r3[0]['layout']['d'] == 32, f"671B step 0 = {lay_text(r3[0]['layout'])}")
    check(r3[1]['layout']['ep'] == 64 and r3[1]['layout']['d'] == 64, f"671B step 2 = {lay_text(r3[1]['layout'])} (EP and d doubled together)")
    check(r3[2]['layout']['pp'] == 16 and r3[2]['layout']['ep'] == 64 and r3[2]['layout']['d'] == 64, f"671B step 3 = {lay_text(r3[2]['layout'])}")
    # the split choice where the TP1 column is red: PP8 EP32 d32 on the 671B grid picks TP2 x U16 (TP1 x U32 is red there)
    sp, sent, cands = split_choice(32, L(8, 32, 1, 1, 2), ctx3)
    check(sp['tp'] == 2 and sp['u'] == 16, f"671B d32 split at mbs 2 = TP{sp['tp']} x U{sp['u']} (TP1 x U32 is red on the heatmap at micro-batch 2, so TP2 x U16 wins by bytes)")
    sp1, _, _ = split_choice(32, L(8, 32, 1, 1, 1), ctx3)
    check(sp1['tp'] == 1 and sp1['u'] == 32, f"671B d32 split at mbs 1 = TP{sp1['tp']} x U{sp1['u']} (every split fits at micro-batch 1, so the fewest bytes win)")
    # 4. the example chain PP2 EP16 d2 -> PP2 EP16 TP2 x U8 -> PP2 EP32 d32 -> PP4 EP32 d32, as a shape: a 128-head model on 64 nodes with the heatmap off
    ctx4 = Ctx('DeepSeek_V3_671B', 64, 131072, 128, chunked_ce=1024, capacity=1.25, heatmap='/nonexistent')
    r4, s4 = ladder(L(2, 16, 1, 2, 1), ctx4)
    shape = [(r['layout']['pp'], r['layout']['ep'], r['layout']['d']) for r in r4]
    check(shape == [(2, 16, 16), (2, 32, 32), (4, 32, 32)], f'the example chain shape PP/EP/d = {shape}')
    # 5. feasibility sentences
    check(infeasible(L(2, 16, 2, 16, 1), ctx) is not None, 'the 16B cannot take TP2 x U16 (16 heads): ' + str(infeasible(L(2, 16, 2, 16, 1), ctx)))
    check(infeasible(L(2, 128, 1, 1, 1), ctx) is not None, 'EP 128 exceeds the 64 experts of the 16B')
    # 6. [2026-10-11] shrink when no spare is left: the 16B on 64 nodes at 128k, 256 sequences per step, PP2 EP16 TP1 x U16, 63 good nodes:
    #    the tokens per step (256 sequences) must split over dp_non_moe = 8M / (2 x 16) = M / 4, so M must be 4 x a power of two: 32 nodes
    kw = dict(model='DeepSeek_16B', seq=131072, gbs=256, chunked_ce=2048, capacity=1.0, ckpt='full')
    r6 = shrink(L(2, 16, 1, 16, 1), 63, kw)
    check(r6['nodes'] == 32 and r6['layout']['ep'] == 16 and r6['layout']['d'] == 16, f"16B shrink from 64 to 63 good nodes -> {r6['nodes']} nodes, {lay_text(r6['layout']) if r6.get('layout') else '-'}")
    # 7. a 2-node 16B run (PP1 EP16 TP1 x U1, 4k, 32 sequences) with one good node left: EP 16 cannot divide 8 ranks, so EP is clipped to 8
    kw7 = dict(model='DeepSeek_16B', seq=4096, gbs=32, chunked_ce=None, capacity=1.0, ckpt='full')
    r7 = shrink(L(1, 16, 1, 1, 1), 1, kw7)
    check(r7['nodes'] in (0, 1), f"16B 2 nodes -> 1 good node: {r7['nodes']} nodes{', ' + lay_text(r7['layout']) if r7.get('layout') else ' (none fits)'}")
    # 8. nothing fits: the 671B at 256k on 2 nodes
    kw8 = dict(model='DeepSeek_V3_671B', seq=262144, gbs=64, chunked_ce=1024, capacity=1.25, ckpt='full')
    r8 = shrink(L(8, 64, 2, 32, 1), 2, kw8)
    check(r8['nodes'] == 0, f"671B at 256k with 2 good nodes: no configuration fits ({len(r8['tried'])} launchable tried)")
    # 9. [2026-10-13] the 16B at 256k on 72 nodes, PP1 EP8 TP8 x U1 mbs 1, 72 sequences per step, WITH intra-document attention: no Ulysses split
    #    may be chosen; step 2 doubles EP alone (d 16 has no launchable split: TP16 leaves the node, TP1 x U16 / TP2 x U8 need Ulysses), step 3 PP 2
    ctx9 = Ctx('DeepSeek_16B_32k', 72, 262144, 72, chunked_ce=2048, capacity=1.0, intra_doc=True)
    r9, s9 = ladder(L(1, 8, 8, 1, 1), ctx9)
    shape9 = [(r['step'], r['layout']['pp'], r['layout']['ep'], r['layout']['tp'], r['layout']['u']) for r in r9]
    check(shape9 == [(2, 1, 16, 8, 1), (3, 2, 16, 8, 1)], f'16B 256k intra-doc ladder (step, PP, EP, TP, U) = {shape9}')
    check(all(r['layout']['u'] == 1 for r in r9), 'no rung of the intra-document ladder uses Ulysses')
    ctx9b = Ctx('DeepSeek_16B_32k', 72, 262144, 72, chunked_ce=2048, capacity=1.0)
    r9b, _ = ladder(L(1, 8, 8, 1, 1), ctx9b)
    check(r9b[0]['layout']['u'] == 16, f"without intra-document attention the same ladder still takes {lay_text(r9b[0]['layout'])} (unchanged behaviour)")
    return 0 if ok else 1


if __name__ == '__main__':
    main()
