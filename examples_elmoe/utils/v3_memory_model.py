#!/usr/bin/env python3
"""v3_memory_model.py -- the analytical GPU-memory model of DeepSeek-V3-style models (and the older MHA + FFN MoE shapes) on the
X-MoE training path: ZeRO-1 (DeepSpeedZeroOptimizer over FusedAdam, bf16 weights and gradients), pipeline (1F1B), expert and
tensor parallelism, activation checkpointing (full, none, or the first n layers), the fp32 residual stream, flash attention,
the grouped-GEMM experts without token dropping, the chunked cross entropy.  Every term is a formula on the registry entry's
shapes; the three calibration constants (the non-torch VRAM, the allocator's slack, the ZeRO bucket) are measured (2026-09-14,
runs M1 5485255, W2 5483732, P64 5483075) and named as such.

Usage (the same for the user and the assistant):
  python3 v3_memory_model.py --model V3S_16B --nodes 1 --pp 1 --ep 8 --mbs 1 --gbs 2048 --seq 4096 --ckpt full
  python3 v3_memory_model.py --model DeepSeek_V3_671B --nodes 64 --pp 32 --ep 8 --mbs 1 --gbs 4096 --seq 4096 --ckpt full
  python3 v3_memory_model.py --model DeepSeek_V3_671B --nodes 64 --pp 32 --ep 8 --mbs 1 --gbs 4096 --seq 16384 --ckpt full --json out.json
  python3 v3_memory_model.py --validate            # against the measured points in v3_memory_measured.json
From Python: MemoryModel(model='V3S_16B', nodes=1, pp=1, ep=8, mbs=1, gbs=2048, seq=4096, ckpt='full').stage(0)

Where the older planner (deepspeed/moe/planner/memory_calculator.py, utils/memory_calculator_v2.py) differs, and why its numbers
were off for these models: it assumes a 4h^2 attention (the latent attention is q_a/q_b/kv_a/kv_b/o with lora ranks and a 192-wide
head), a two-matrix FFN expert (SwiGLU is three), fp16 residuals (ours are fp32: the checkpoint input per layer is 4 bytes per element),
no shared experts and no dense first layers, an fp32-per-dp optimizer for the embedding that ignores dp, a 'drop factor' on the
expert tokens (no tokens are dropped here), a flat 7 GB 'system overhead', and it never separates the allocated peak, the
allocator's reserved pool and the VRAM the node monitor sees (three different numbers, see the page)."""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
GiB = 1024 ** 3
BF16, FP32 = 2, 4

# ---- the measured constants (bytes); each names its measurement ----
NON_TORCH_VRAM_1NODE = 2.7 * GiB  # rocm-smi VRAM minus the allocator's reserved pool on one node (W2: 51.8 - 49.1 GB): the HIP context, RCCL, the kernel images
NON_TORCH_VRAM_512 = 3.8 * GiB    # the same at 512 ranks (P64 rank 0: 35.2 - 32.0, rank 511: 37.8 - 33.5 GB): more communicators, more RCCL buffers
def non_torch_vram(world):
    """interpolated in log2(world) between the two measurements; a measured constant, not a formula"""
    lo, hi = math.log2(8), math.log2(512); x = min(max(math.log2(max(world, 8)), lo), hi)
    return NON_TORCH_VRAM_1NODE + (NON_TORCH_VRAM_512 - NON_TORCH_VRAM_1NODE) * (x - lo) / (hi - lo)
ALLOC_SLACK = 0.18                # reserved / allocated - 1 on a typical rank without expandable segments: M1 60.2 / 51.3 = 0.173 (block recompute), R16f 52.6 / 44.1 = 0.195 (every layer checkpointed); the caching allocator's fragmentation
SLACK_FLOOR = 10 * GiB            # the busiest rank of a stage: R671b (512 ranks, every rank's never-reset maximum) reserved minus allocated 8.4-15.7 GiB per stage at the worst rank while the allocated peak differs by 0.7 GiB across a stage's 16 ranks
SLACK_WORST = 16 * GiB            # the largest slack seen (R671b stage 24: 15.7 GiB); the out-of-memory line is allocated + this + the non-torch part
ZERO_BUCKET = 5e8                 # DeepSpeed reduce_bucket_size elements (the job config); two bf16 buckets live at the backward's peak (M1 frames :2068/:2075)
ATTN_LSE = 4                      # flash attention keeps the log-sum-exp per query per head in fp32


def load_registry():
    sys.path.insert(0, HERE)
    import model_registry
    for name in ('MODEL_REGISTRY', 'MODELS', 'REGISTRY', 'registry'):
        if hasattr(model_registry, name):
            return getattr(model_registry, name)
    # fall back: the first dict of dicts with an 'h' field
    for v in vars(model_registry).values():
        if isinstance(v, dict) and v and all(isinstance(x, dict) and 'h' in x for x in list(v.values())[:3]):
            return v
    raise RuntimeError('model registry dict not found')


class Shapes:
    """the per-layer parameter and activation shapes of one model entry"""
    def __init__(self, entry, tp=1):
        e = entry; self.e = e
        self.h = e['h']; self.layers = e['layers']; self.heads = e['heads']; self.E = e['ep']; self.k = e['k']
        self.I = e['ffn']; self.shared = e.get('shared', 0); self.first_dense = e.get('first_k_dense', 0)
        self.dense_I = e.get('dense_ffn', 4 * self.h); self.vocab = e['vocab']; self.untied = e.get('untie_embeddings', True)
        self.swiglu = e.get('swiglu', False); self.v3 = e.get('v3'); self.tp = tp
        if self.v3:
            v = self.v3; self.q_lora = v.get('q_lora_rank'); self.kv_lora = v['kv_lora_rank']
            self.d_nope, self.d_rope, self.d_v = v['qk_nope_head_dim'], v['qk_rope_head_dim'], v['v_head_dim']
            self.d_qk = self.d_nope + self.d_rope
        self.expert_interval = e.get('expert_interval', 1)

    def is_moe(self, layer):
        return layer >= self.first_dense and ((layer - self.first_dense) % self.expert_interval == 0 if self.expert_interval > 1 else True)

    # ---- parameters (elements) ----
    def attn_params(self):
        h, H = self.h, self.heads
        if self.v3:
            n = 0
            if self.q_lora:
                n += h * self.q_lora + self.q_lora + self.q_lora * H * self.d_qk
            else:
                n += h * H * self.d_qk
            n += h * (self.kv_lora + self.d_rope) + self.kv_lora + self.kv_lora * H * (self.d_nope + self.d_v)
            n += H * self.d_v * h
            return n
        return 4 * h * h                              # the MHA of the older shapes

    def mlp_matrices(self, inter):
        return (3 if self.swiglu else 2) * self.h * inter

    def dense_mlp_params(self):
        return self.mlp_matrices(self.dense_I)

    def expert_params(self):                          # one routed expert
        return self.mlp_matrices(self.I)

    def shared_params(self):
        return self.mlp_matrices(self.I * self.shared) if self.shared else 0

    def router_params(self):
        return self.h * self.E + self.E                # the gate weight and the correction bias (a buffer, no optimizer state)

    def norms_params(self):
        n = 2 * self.h
        if self.v3:
            n += (self.q_lora or 0) + self.kv_lora
        return n

    # ---- activations kept for the backward, per token, per layer, when the layer is NOT checkpointed (bytes) ----
    def attn_act_per_token(self):
        """the M1 frames: the two norm outputs (bf16), the latent projections and their norms, q/k/v padded to the query width
        for the flash kernel plus its output and log-sum-exp, the o_proj input, and the residual add in fp32"""
        h, H = self.h, self.heads
        b = h * BF16                                              # input norm output
        if self.v3:
            b += (self.q_lora or 0) * BF16 * 2                    # q_a out + its norm
            b += (self.kv_lora + self.d_rope) * BF16 * 2          # kv_a out + its norm
            b += H * self.d_qk * BF16                             # q_b out
            b += H * (self.d_nope + self.d_v) * BF16              # kv_b out
            b += 3 * H * self.d_qk * BF16                         # q, k, v (v padded to the query width) for the flash kernel
            b += H * self.d_qk * BF16 + H * ATTN_LSE              # flash output (padded) + the log-sum-exp
            b += H * self.d_v * BF16                              # the o_proj input
        else:
            b += 3 * h * BF16 + h * BF16 + H * ATTN_LSE + h * BF16
        b += h * FP32                                             # the residual add (fp32 residual stream)
        return b

    def moe_act_per_token(self):
        """post-attention norm output, the router scores (fp32), the dispatched tokens (k copies, bf16) for the expert GEMMs,
        the all-to-all buffer, the SwiGLU inputs of the routed and shared experts, the combine output (fp32), the residual add"""
        h, k, I = self.h, self.k, self.I
        b = h * BF16                                              # norm output
        b += self.E * FP32 * 2                                    # router logits and scores
        b += k * h * BF16                                         # the dispatched tokens (expert GEMM inputs)
        b += k * h * BF16                                         # the all-to-all output buffer
        b += k * (2 if self.swiglu else 1) * I * BF16             # SwiGLU inputs of the routed experts
        b += k * I * BF16                                         # the expert down-projection input
        if self.shared:
            b += (2 if self.swiglu else 1) * I * self.shared * BF16 + I * self.shared * BF16
        b += h * FP32                                             # the combine output in fp32
        b += h * FP32                                             # the residual add
        return b

    def dense_mlp_act_per_token(self):
        return self.h * BF16 + (2 if self.swiglu else 1) * self.dense_I * BF16 + self.dense_I * BF16 + self.h * FP32


class MemoryModel:
    def __init__(self, model, nodes, pp, ep, mbs, gbs, seq, ckpt='full', tp=1, gpus_per_node=8, zero=1, partition=None,
                 recompute_layers=None, chunked_ce=None, registry=None):
        reg = registry or load_registry()
        self.name = model; self.entry = reg[model]; self.S = Shapes(self.entry, tp)
        self.nodes, self.pp, self.ep, self.tp, self.mbs, self.gbs, self.seq = nodes, pp, ep, tp, mbs, gbs, seq
        self.ckpt = ckpt; self.zero = zero; self.chunked_ce = chunked_ce
        self.world = nodes * gpus_per_node
        assert self.world % (pp * tp) == 0 and self.world % (pp * ep) == 0, (self.world, pp, tp, ep)
        self.dp_non_moe = self.world // (pp * tp)
        self.dp_moe = self.world // (pp * ep)
        assert gbs % (mbs * self.dp_non_moe) == 0, f'GBS {gbs} is not a multiple of mbs {mbs} x dp_non_moe {self.dp_non_moe}'
        self.micro_batches = gbs // (mbs * self.dp_non_moe)       # per pipeline (the gradient-accumulation steps)
        self.partition = partition or self.balanced_partition()
        self.recompute_layers = set(recompute_layers or [])
        self.T = mbs * seq                                          # tokens per micro-batch per stage

    def balanced_partition(self):
        """Megatron's partition_balanced over the transformer layers: the first (layers mod pp) stages carry one more layer"""
        base, extra = divmod(self.S.layers, self.pp)
        return [base + (1 if i < extra else 0) for i in range(self.pp)]

    def stage_layers(self, s):
        start = sum(self.partition[:s]); return list(range(start, start + self.partition[s]))

    def layer_ckpt(self, layer):
        if self.ckpt == 'full': return True
        if self.ckpt == 'none': return False
        if self.ckpt == 'block': return layer in self.recompute_layers
        raise ValueError(self.ckpt)

    # ---- one stage ----
    def stage(self, s):
        S = self.S; T = self.T; out = {}
        layers = self.stage_layers(s)
        # parameters per rank of this stage: dense (sharded by tp), experts (by ep and tp), router (replicated), norms
        dense = 0; expert = 0
        for l in layers:
            dense += S.attn_params() / self.tp + S.norms_params()
            if S.is_moe(l):
                expert += S.E / self.ep * S.expert_params() / self.tp
                dense += S.shared_params() / self.tp + S.router_params()
            else:
                dense += S.dense_mlp_params() / self.tp
        emb = S.vocab * S.h / self.tp if s == 0 else 0
        head = (S.vocab * S.h / self.tp if S.untied else 0) if s == self.pp - 1 else 0
        if not S.untied and s == self.pp - 1 and self.pp > 1:
            head = S.vocab * S.h / self.tp                        # the tied weight is duplicated on the last stage
        dense += emb + head + (S.h if s == self.pp - 1 else 0)   # final norm
        out['params_dense'] = dense; out['params_expert'] = expert
        # ZeRO-1 over DeepSpeedZeroOptimizer: bf16 weights + bf16 gradients on every rank; fp32 master + Adam m, v over the group
        out['weights_bf16'] = (dense + expert) * BF16
        out['grads_bf16'] = (dense + expert) * BF16
        out['master_fp32'] = dense * FP32 / self.dp_non_moe + expert * FP32 / self.dp_moe
        out['adam_fp32'] = 2 * (dense * FP32 / self.dp_non_moe + expert * FP32 / self.dp_moe)
        has_moe = any(S.is_moe(l) for l in layers)
        out['zero_buckets'] = 2 * ZERO_BUCKET * BF16 * (2 if (has_moe and self.dp_moe > 1) else 1)   # the expert gradients get their own reduce buckets when dp_moe > 1 (none needed at dp_moe 1: M1 had two buckets)
        # activations: in-flight micro-batches under 1F1B = min(pp - s, micro_batches)
        inflight = max(2, min(self.pp - s, self.micro_batches)) if self.pp > 1 else 1   # deepspeed/runtime/pipe/schedule.py TrainSchedule.num_pipe_buffers
        per_layer_full = []; per_layer_ckpt = []
        for l in layers:
            full = S.attn_act_per_token() + (S.moe_act_per_token() if S.is_moe(l) else S.dense_mlp_act_per_token())
            per_layer_full.append(full * T)
            per_layer_ckpt.append(self.layer_ckpt(l))
        kept = sum((S.h * FP32 * T) if c else f for f, c in zip(per_layer_full, per_layer_ckpt))   # a checkpointed layer keeps its fp32 input
        if self.pp > 1:
            kept += S.h * FP32 * T                                # the stage's output stays in pipe_buffers['outputs'] until its backward
        out['_act_kept_per_microbatch'] = kept                    # informational (underscored keys are not summed)
        out['act_kept_inflight'] = kept * inflight
        # the recompute of one checkpointed layer during its backward: its full activations once, plus the layer's gradients
        recompute = max((f for f, c in zip(per_layer_full, per_layer_ckpt) if c), default=0)
        out['act_recompute_transient'] = recompute
        # the backward of that layer: the gradients of its largest activations exist beside them (the all-to-all backward buffer
        # and the dispatched-token gradient, the SwiGLU input gradients of the routed and shared experts, the combine and residual
        # gradients in fp32); a dense layer's counterpart is its MLP intermediate gradient
        moe_layers = [l for l in layers if S.is_moe(l)]
        if moe_layers:
            bwd = T * (2 * S.k * S.h * BF16 + S.k * (2 if S.swiglu else 1) * S.I * BF16 + (2 if S.swiglu else 1) * S.I * S.shared * BF16 + 2 * S.h * FP32)
        else:
            bwd = T * ((2 if S.swiglu else 1) * S.dense_I * BF16 + 2 * S.h * FP32)
        out['act_backward_transient'] = bwd
        # the head: the logits and their gradient in fp32 (chunked or not) and the embedding's output
        if s == self.pp - 1:
            chunk = min(self.chunked_ce, T) if self.chunked_ce else T
            out['logits_fp32'] = 2 * chunk * S.vocab / self.tp * FP32
        else:
            out['logits_fp32'] = 0
        out['embedding_act'] = (T * S.h * BF16) if s == 0 else 0
        # pipeline send/receive buffers (the fp32 residual crosses the stages) and the expert all-to-all send buffer
        out['pipe_buffers'] = (2 * T * S.h * FP32) if self.pp > 1 else 0
        out['a2a_send'] = T * S.k * S.h * BF16 if any(S.is_moe(l) for l in layers) else 0
        alloc = sum(v for k, v in out.items() if not k.startswith('_') and k not in ('params_dense', 'params_expert'))
        out['allocated_peak'] = alloc
        out['reserved_est'] = alloc + max(ALLOC_SLACK * alloc, SLACK_FLOOR)       # the busiest rank of the stage (the allocated peak is the same on every rank of a stage)
        out['vram_est'] = out['reserved_est'] + non_torch_vram(self.world)
        out['vram_worst'] = alloc + SLACK_WORST + non_torch_vram(self.world)       # the out-of-memory line: the largest allocator slack ever measured on top of the allocated peak
        out['layers'] = layers; out['inflight'] = inflight
        return out

    def table(self):
        rows = []
        for s in range(self.pp):
            r = self.stage(s)
            rows.append(dict(stage=s, layers=f"{r['layers'][0]}-{r['layers'][-1]}" if r['layers'] else '-', inflight=r['inflight'],
                             params_B=(r['params_dense'] + r['params_expert']) / 1e9,
                             state_GiB=(r['weights_bf16'] + r['grads_bf16'] + r['master_fp32'] + r['adam_fp32']) / GiB,
                             act_GiB=(r['act_kept_inflight'] + r['act_recompute_transient'] + r['act_backward_transient'] + r['logits_fp32'] + r['embedding_act']) / GiB,
                             buffers_GiB=(r['zero_buckets'] + r['pipe_buffers'] + r['a2a_send']) / GiB,
                             allocated_GiB=r['allocated_peak'] / GiB, reserved_GiB=r['reserved_est'] / GiB, vram_GiB=r['vram_est'] / GiB, vram_worst_GiB=r['vram_worst'] / GiB))
        return rows

    def query_stage_memory(self, stage_id, num_layers, num_ckpt_layers, micro_batch_size):
        """the older planner's query: the total GiB of a stage of num_layers layers (num_ckpt_layers of them checkpointed) at a micro-batch"""
        saved = (self.partition, self.mbs, self.T, self.ckpt, self.recompute_layers)
        try:
            parts = [0] * self.pp; parts[stage_id] = num_layers
            rest = self.S.layers - num_layers
            for i in range(self.pp):
                if i != stage_id and rest > 0:
                    take = min(rest, max(1, rest // max(1, self.pp - 1))); parts[i] = take; rest -= take
            self.partition = parts; self.mbs = micro_batch_size; self.T = micro_batch_size * self.seq
            start = sum(parts[:stage_id]); self.ckpt = 'block'; self.recompute_layers = set(range(start, start + num_ckpt_layers))
            return self.stage(stage_id)['vram_est'] / GiB
        finally:
            self.partition, self.mbs, self.T, self.ckpt, self.recompute_layers = saved


def fmt_table(rows):
    head = f"{'stage':>5} {'layers':>8} {'inflight':>8} {'params B':>9} {'state':>7} {'act':>7} {'buffers':>8} {'alloc':>7} {'reserved':>9} {'VRAM':>7} {'worst':>7}  (GiB per GCD; worst = allocated + the largest slack + non-torch, the out-of-memory line)"
    lines = [head]
    for r in rows:
        lines.append(f"{r['stage']:>5} {r['layers']:>8} {r['inflight']:>8} {r['params_B']:>9.2f} {r['state_GiB']:>7.1f} {r['act_GiB']:>7.1f} {r['buffers_GiB']:>8.1f} {r['allocated_GiB']:>7.1f} {r['reserved_GiB']:>9.1f} {r['vram_GiB']:>7.1f} {r['vram_worst_GiB']:>7.1f}")
    return '\n'.join(lines)


def validate(measured_path):
    """every measured point: the model's prediction beside the measurement, per stage"""
    M = json.load(open(measured_path)); worst = 0; results = []
    for p in M['points']:
        mm = MemoryModel(**p['layout'])
        print(f"== {p['label']}: {p['what']}")
        for s, meas in p['measured'].items():
            r = mm.stage(int(s)); pred = r['vram_est'] / GiB if 'vram_GiB' in meas else r['allocated_peak'] / GiB
            key = 'vram_GiB' if 'vram_GiB' in meas else 'allocated_GiB'
            ref = meas[key]
            err = 100 * (pred - ref) / ref; worst = max(worst, abs(err))
            results.append(dict(label=p['label'], stage=int(s), predicted=pred, measured=ref, key=key, err_pct=err, alloc=r['allocated_peak'] / GiB, state=(r['weights_bf16'] + r['grads_bf16'] + r['master_fp32'] + r['adam_fp32']) / GiB, act=(r['act_kept_inflight'] + r['act_recompute_transient'] + r['act_backward_transient'] + r['logits_fp32']) / GiB, components={c: dict(predicted=r[c] / GiB, measured=v) for c, v in meas.get('components_GiB', {}).items() if c in r}))
            print(f"   stage {s:>2}: predicted {pred:6.1f} GiB  measured {ref:6.1f} GiB ({key})  {err:+5.1f} %   [alloc {r['allocated_peak']/GiB:.1f}, state {(r['weights_bf16']+r['grads_bf16']+r['master_fp32']+r['adam_fp32'])/GiB:.1f}, act {(r['act_kept_inflight']+r['act_recompute_transient']+r['act_backward_transient']+r['logits_fp32'])/GiB:.1f}]")
            for comp, val in meas.get('components_GiB', {}).items():
                pv = r.get(comp)
                if pv is not None:
                    print(f"      {comp:>24s}: predicted {pv/GiB:6.2f}  measured {val:6.2f}  {100*(pv/GiB-val)/val:+5.1f} %")
    print(f"worst error over the points: {worst:.1f} %")
    json.dump(dict(points=results, worst_pct=worst), open(os.path.join(HERE, 'v3_memory_validation.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model'); ap.add_argument('--nodes', type=int); ap.add_argument('--pp', type=int, default=1); ap.add_argument('--ep', type=int, default=8)
    ap.add_argument('--tp', type=int, default=1); ap.add_argument('--mbs', type=int, default=1); ap.add_argument('--gbs', type=int); ap.add_argument('--seq', type=int)
    ap.add_argument('--ckpt', default='full', choices=['full', 'none', 'block']); ap.add_argument('--recompute-layers', default='', help='block: e.g. 0-8')
    ap.add_argument('--chunked-ce', type=int, default=None); ap.add_argument('--partition', default=None, help='layers per stage, e.g. 2,2,...,1')
    ap.add_argument('--json', default=None); ap.add_argument('--validate', action='store_true'); ap.add_argument('--measured', default=os.path.join(HERE, 'v3_memory_measured.json'))
    a = ap.parse_args()
    if a.validate:
        validate(a.measured); sys.exit(0)
    rl = None
    if a.recompute_layers:
        lo, hi = a.recompute_layers.split('-'); rl = range(int(lo), int(hi) + 1)
    part = [int(x) for x in a.partition.split(',')] if a.partition else None
    mm = MemoryModel(a.model, a.nodes, a.pp, a.ep, a.mbs, a.gbs, a.seq, a.ckpt, tp=a.tp, partition=part, recompute_layers=rl, chunked_ce=a.chunked_ce)
    print(f"{a.model}: {a.nodes} nodes, PP {a.pp} x EP {a.ep} x TP {a.tp}, dp_non_moe {mm.dp_non_moe}, dp_moe {mm.dp_moe}, mbs {a.mbs}, GBS {a.gbs} -> {mm.micro_batches} micro-batches per pipeline, seq {a.seq}, ckpt {a.ckpt}")
    rows = mm.table(); print(fmt_table(rows))
    print(f"heaviest stage: {max(rows, key=lambda r: r['vram_GiB'])['stage']} at {max(r['vram_GiB'] for r in rows):.1f} GiB VRAM, {max(r['vram_worst_GiB'] for r in rows):.1f} at the worst slack (64 GiB per GCD)")
    if a.json:
        json.dump(dict(layout=vars(a), rows=rows, stages=[mm.stage(s) for s in range(a.pp)]), open(a.json, 'w'), indent=1, default=list); print('wrote', a.json)
