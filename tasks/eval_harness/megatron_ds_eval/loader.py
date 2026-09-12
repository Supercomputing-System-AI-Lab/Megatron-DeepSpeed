"""Checkpoint loading + Megatron/DeepSpeed setup for the eval harness.

Everything between "parse the CLI" and "return a model with the right bytes in it":
argument merge with the checkpoint's stored args, initialize_megatron, the three
weight-load paths (universal checkpoint / consolidated flat .pt / plain DeepSpeed),
and the preconditions that make a silent random-init impossible.
"""
import os
import re

import torch

import megatron
from megatron import print_rank_0
from megatron.training import setup_model_and_optimizer
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron

from pretrain_gpt import model_provider
from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint

# ----------------------------------------------------------------------------
# PyTorch >= 2.6 changed torch.load's default to weights_only=True for
# security. Megatron-DeepSpeed checkpoints carry an `argparse.Namespace`
# (the training args), which the strict default refuses. The checkpoint is
# trusted (we wrote it), so flip the default back. Affects only this Python
# process — no global pytorch/torch.load behavior changed for other users.
# ----------------------------------------------------------------------------
_original_torch_load = torch.load
def _torch_load_allow_pickle(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_allow_pickle


# ---------------------------------------------------------------------------
# Helpers copied verbatim from tasks/eval_harness/evaluate.py.
# The model build/checkpoint load is the verified path used in training-
# adjacent eval since BLOOM; we only swap the lm-eval interface around it.
# ---------------------------------------------------------------------------

def override_args(args, override_args_, skip_keys, skip_if_specified_keys):
    # Imported/converted checkpoints (e.g. the HF->DS DeepSeek converter) carry no
    # stored 'args' Namespace -- get_args() returns None. Nothing to merge: the eval
    # already receives the full model config on the CLI (megatron_options + ARCH_FLAGS),
    # so fall through to the CLI args unchanged.
    if override_args_ is None:
        return
    for k, v in vars(override_args_).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


def _resolve_universal_zero_dir(load_dir):
    """Resolve --load (the selector/parent dir) to the universal checkpoint's zero/ directory.

    Raises with an actionable message instead of letting the engine's silent
    warn-and-random-init path run (engine.py:2870 only WARNS on a missing tag file).
    """
    tag_file = os.path.join(load_dir, 'latest_universal')
    if not os.path.isfile(tag_file):
        raise FileNotFoundError(
            f'--universal-checkpoint given but {load_dir}/latest_universal is missing. '
            '(--load takes the directory CONTAINING <tag>_universal plus latest_universal; '
            'the eval template builds a private selector dir for exactly this.)')
    tag = open(tag_file).read().strip()
    zero_dir = os.path.join(load_dir, tag, 'zero')
    if not os.path.isdir(zero_dir) or not os.listdir(zero_dir):
        raise FileNotFoundError(
            f'{zero_dir} is missing or empty -- latest_universal points at "{tag}" but that '
            'conversion is absent/incomplete. Re-run ds_to_universal.')
    return zero_dir


def _atom_name_candidates(name, num_layers):
    """Twin names under which this parameter's atom may be stored: BOTH directions
    of the GPTModel <-> pipe-spec naming bijection.

    Forward (GPTModel name -> pipe-spec twins) serves a no-PP eval reading a
    pipe-class-written checkpoint -- the only direction the old
    _pipe_name_candidates offered. Reverse (pipe-spec name -> GPTModel twins) is
    the PP fix: under pipeline parallelism the engine's param_names come from
    GPTModelPipe, whose modules are registered by GLOBAL spec index
    (deepspeed pipe/module.py: name = str(local_idx + self._local_start)), while
    a no-PP-trained checkpoint's atoms are GPTModel-canonical
    (language_model.encoder.layers.N....). With only the forward direction,
    EVERY stage parameter missed: job 5354152 (pp=2, 16 spec entries on stage 0)
    lost all 514 stage params, probe D at pp=4 (9 entries) lost 145 -- the miss
    count tracks the stage's spec-entry count exactly because the mapping, not
    the weights, was absent. The bijection itself lives in
    megatron/model/gpt_model.py (pipe_name_candidates / gptmodel_name_candidates,
    the same pair universal_checkpoint_name_aliases declares) so loader and
    model cannot drift apart; imported lazily because this module is imported
    before megatron is initialised.
    """
    from megatron.model.gpt_model import pipe_name_candidates, gptmodel_name_candidates
    cands = list(pipe_name_candidates(name, num_layers))
    for c in gptmodel_name_candidates(name, num_layers):
        if c not in cands:
            cands.append(c)
    return cands


def _tp_slice_spec(full_shape, target_shape, meta, tp_rank, tp_size):
    """How to cut THIS rank's tensor-parallel shard out of a full universal atom.

    Pure arithmetic (stdlib only, login-node testable). The decision comes from
    the CHECKPOINT'S OWN partition metadata, written by ds_to_universal from the
    model's universal_checkpoint_info declarations and normalised by the caller
    into `meta`:
        vocab            VOCAB_TENSOR  -- stored UNPADDED (100015 true rows)
        cat_dim          CAT_DIM       -- 0 column-split, 1 row-parallel
        n_sub            PARAM_N_SUB_PARAMS -- gated-MLP twin sub-params
        sub_param_shape  SUB_PARAM_SHAPE    -- refused (never written here)
    This mirrors deepspeed/checkpoint/universal_checkpoint.py's
    load_hp_checkpoint_state, the reference implementation training uses.

    VOCAB RULE (binding; OVERNIGHT_STATE 05:00): pad the unpadded vocab atom to
    target_rows * tp_size rows of EXACT ZEROS **BEFORE** slicing -- 100015 pads
    to 100352 at TP4 (4 x 25088), THEN rank r takes rows [r*25088, (r+1)*25088).
    Slicing before padding hands rank 3 rows [75264, 100015) plus a short read
    -- every rank's embedding rows silently misaligned. The pad rows are exact
    zeros so a dummy row scores an exactly-zero logit under
    --disable-bias-linear (and the adapter slices logits to [:vocab_size]
    anyway).

    Returns one of:
        ('copy',)                          shapes equal: replicated/averaged atom
        ('reshape_copy',)                  same numel, different shape: view_as
        ('vocab_pad_slice', pad_to, lo, hi)  zero-pad dim0 to pad_to, rows [lo,hi)
        ('slice', dim, lo, hi)             plain CAT_DIM chunk, this rank's range
        ('subparams', dim, [(lo, hi), ...])  per-sub-param ranges, cat along dim
        ('mismatch', reason)               refuse loudly; caller records bad_shape
    """
    full = tuple(int(x) for x in full_shape)
    target = tuple(int(x) for x in target_shape)
    tp_rank, tp_size = int(tp_rank), int(tp_size)
    if tp_size < 1:
        tp_size = 1
    if not 0 <= tp_rank < tp_size:
        return ('mismatch', f'tp_rank {tp_rank} outside tp_size {tp_size}')
    if full == target:
        # Replicated (or TP-averaged) parameter: the atom already IS the local
        # tensor -- this is also every expert weight (experts are never
        # tensor-split in this codepath) and the whole model at tp=1.
        return ('copy',)
    if meta.get('vocab'):
        if len(full) != len(target) or full[1:] != target[1:]:
            return ('mismatch', 'vocab atom trailing dims differ from param')
        pad_to = target[0] * tp_size
        if full[0] > pad_to:
            return ('mismatch',
                    f'vocab atom rows {full[0]} exceed padded target {pad_to}')
        return ('vocab_pad_slice', pad_to, tp_rank * target[0],
                (tp_rank + 1) * target[0])
    if meta.get('sub_param_shape') is not None:
        return ('mismatch',
                'SUB_PARAM_SHAPE atom: this model\'s universal_checkpoint_info '
                'never declares PARAMETER_WITH_SUB_PARAMS; refusing to guess')
    dim = meta.get('cat_dim')
    n_sub = int(meta.get('n_sub') or 1)
    if dim is None:
        # No partition metadata: the converter only omits CAT_DIM for averaged
        # (replicated) parameters, whose shapes matched above. Same-numel
        # reshapes stay legal (today's view_as fallback); anything else is a
        # real mismatch and must stay LOUD.
        if _numel(full) == _numel(target):
            return ('reshape_copy',)
        return ('mismatch', 'no partition metadata (cat_dim absent) and sizes differ')
    dim = int(dim)
    if len(full) != len(target) or not 0 <= dim < len(full):
        return ('mismatch', f'cat_dim {dim} invalid for atom rank {len(full)}')
    if any(full[i] != target[i] for i in range(len(full)) if i != dim) \
            or full[dim] != target[dim] * tp_size:
        if _numel(full) == _numel(target):
            return ('reshape_copy',)
        return ('mismatch', f'atom dim {dim} is {full[dim]}, want target '
                            f'{target[dim]} x tp {tp_size}')
    if n_sub <= 1:
        per = full[dim] // tp_size
        return ('slice', dim, tp_rank * per, (tp_rank + 1) * per)
    if full[dim] % (n_sub * tp_size):
        return ('mismatch', f'{n_sub} sub-params x tp {tp_size} do not divide '
                            f'atom dim {full[dim]}')
    sub = full[dim] // n_sub
    per = sub // tp_size
    return ('subparams', dim,
            [(s * sub + tp_rank * per, s * sub + (tp_rank + 1) * per)
             for s in range(n_sub)])


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _load_from_universal_atoms(model, zero_dir):
    """Copy the universal checkpoint's fp32 master weights into the module, by NAME.

    WHY NOT engine.load_checkpoint(): the engine only takes its universal branch when the
    DEEPSPEED CONFIG carries checkpoint.load_universal=true (Megatron's --universal-checkpoint
    flag never reaches it -- job 5259354 proved this: `load_universal_checkpoint .... False`,
    "Unable to find latest file", random weights, sciq 0.2). And even wired up, the engine's
    restoration runs through the ZeRO/bf16 optimizer machinery -- the eval config is ZeRO
    stage 0 and needs no optimizer at all.

    A universal checkpoint is one directory per PARAMETER NAME with the fp32 master in
    fp32.pt. The engine's param_names (deepspeed engine.py:274,293) already maps every local
    param -- experts included, with GLOBAL expert indices -- to exactly these names. So the
    weights-only load is: for each param, read its atom, shard it to this rank by the
    checkpoint's own partition metadata (_tp_slice_spec), cast, copy. No optimizer.
    One param resident at a time.

    TP (job 5354153, "114 shape mismatches"): atoms are stored FULL; each rank takes
    its slice along the atom's own CAT_DIM (0 column-split, 1 row-parallel), with the
    vocab padded to tp_size x target rows BEFORE slicing. PP (job 5354152, "514
    params without atoms"): a pipe-built model's param names resolve to the
    checkpoint's GPTModel-canonical atom names through _atom_name_candidates. A cell
    with TP>1 AND PP>1 together is mandatory in validation -- the PP name failure
    sits upstream and HIDES the TP failure (probe E, job 5354557: TP4, zero shape
    mismatches, because every name missed first).
    """
    param_names = getattr(model, 'param_names', None)
    if not param_names:
        raise RuntimeError('engine has no param_names mapping; cannot address atoms by name')

    # Vocabulary parameters are stored UNPADDED in the universal checkpoint (ds_to_universal
    # strips Megatron's pad-to-divisible-by rows precisely so the checkpoint is TP-degree
    # free: 100015 true rows vs 100096 padded at TP=1). The conversion marks exactly those
    # atoms with VOCAB_TENSOR in the payload dict -- the checkpoint's own declaration, the
    # same field DeepSpeed's universal loader consults (universal_checkpoint.py:66). Job
    # 5260409 taught the hard way that name/pattern matching here is fragile: the model's
    # universal_checkpoint_info patterns are REGEXES, which a substring test silently fails.
    try:
        from deepspeed.checkpoint.constants import (VOCAB_TENSOR, CAT_DIM,
                                                    PARAM_N_SUB_PARAMS,
                                                    SUB_PARAM_SHAPE)
    except Exception:
        VOCAB_TENSOR, CAT_DIM = 'vocab_tensor', 'cat_dim'
        PARAM_N_SUB_PARAMS, SUB_PARAM_SHAPE = ('param_n_sub_params',
                                               'sub_param_shape')

    # This rank's tensor-parallel coordinates. mpu is initialised by the time the
    # atoms load (initialize_megatron ran in load_ds_checkpoint_and_setup_megatron).
    from megatron.core import mpu as _mpu
    tp_size = int(_mpu.get_tensor_model_parallel_world_size())
    tp_rank = int(_mpu.get_tensor_model_parallel_rank())

    loaded, nbytes = 0, 0
    missing, bad_shape = [], []
    with torch.no_grad():
        num_layers = megatron.get_args().num_layers
        for param, name in param_names.items():
            fp32_path = os.path.join(zero_dir, name, 'fp32.pt')
            if not os.path.isfile(fp32_path):
                # namespace fallback, BOTH directions (see _atom_name_candidates):
                # a no-PP eval reading a pipe-written checkpoint tries the global
                # spec-index twins; a PP eval reading a no-PP-written checkpoint
                # tries the GPTModel-canonical twins (job 5354152's 514 misses).
                for twin in _atom_name_candidates(name, num_layers):
                    alt = os.path.join(zero_dir, twin, 'fp32.pt')
                    if os.path.isfile(alt):
                        fp32_path = alt
                        break
                else:
                    missing.append(name)
                    continue
            payload = torch.load(fp32_path, map_location='cpu')
            if isinstance(payload, dict):
                t = payload['param']
                meta = {'vocab': bool(payload.get(VOCAB_TENSOR, False)),
                        'cat_dim': payload.get(CAT_DIM),
                        'n_sub': payload.get(PARAM_N_SUB_PARAMS, 1),
                        'sub_param_shape': payload.get(SUB_PARAM_SHAPE)}
            else:
                t = payload
                meta = {'vocab': False, 'cat_dim': None, 'n_sub': 1,
                        'sub_param_shape': None}
            # Shard by the checkpoint's OWN partition metadata (_tp_slice_spec's
            # docstring carries the rules; at tp=1 every branch reduces to the
            # old copy/pad/view_as behaviour value-for-value, which is what the
            # fingerprint gate re-verifies). The vocab branch pads to
            # tp_size x target rows of EXACT ZEROS BEFORE slicing -- with
            # --disable-bias-linear a zero row scores an exactly-zero logit, and
            # the adapter slices logits to [:vocab_size] before log_softmax
            # anyway, so dummy rows never enter a score.
            spec = _tp_slice_spec(tuple(t.shape), tuple(param.shape), meta,
                                  tp_rank, tp_size)
            kind = spec[0]
            if kind == 'mismatch':
                bad_shape.append((name, tuple(t.shape), tuple(param.shape),
                                  spec[1]))
                continue
            if kind == 'copy':
                param.data.copy_(t.to(dtype=param.dtype, device=param.device))
            elif kind == 'reshape_copy':
                param.data.copy_(t.view_as(param).to(dtype=param.dtype,
                                                     device=param.device))
            elif kind == 'vocab_pad_slice':
                _, _pad_to, lo, hi = spec
                shard = torch.zeros((hi - lo,) + tuple(t.shape[1:]),
                                    dtype=t.dtype)
                n_real = min(max(t.shape[0] - lo, 0), hi - lo)
                if n_real > 0:
                    shard[:n_real].copy_(t[lo:lo + n_real])
                param.data.copy_(shard.to(dtype=param.dtype,
                                          device=param.device))
            elif kind == 'slice':
                _, dim, lo, hi = spec
                param.data.copy_(t.narrow(dim, lo, hi - lo)
                                  .to(dtype=param.dtype, device=param.device))
            else:                                       # 'subparams'
                _, dim, ranges = spec
                shard = torch.cat([t.narrow(dim, lo, hi - lo)
                                   for lo, hi in ranges], dim=dim)
                param.data.copy_(shard.to(dtype=param.dtype,
                                          device=param.device))
            loaded += 1
            nbytes += t.numel() * t.element_size()

    if missing or bad_shape:
        raise RuntimeError(
            f'universal atom load FAILED at tp_rank {tp_rank}/{tp_size}: '
            f'{len(missing)} params without atoms (first: {missing[:5]}), '
            f'{len(bad_shape)} shape/partition mismatches (first: '
            f'{bad_shape[:3]}; entries are (name, atom_shape, param_shape, '
            f'reason)). Refusing to evaluate a partially loaded model.')
    print_rank_0(f'[load] universal atoms: {loaded} params loaded by name from {zero_dir} '
                 f'({nbytes / 2**30:.1f} GiB fp32 read, cast to model dtype)')
    # [2026-09-12] the persistent BUFFERS (the DeepSeek-V3 router's balancing bias expert_bias, added to the sigmoid
    # scores for the top-k choice): a universal checkpoint written by ds_to_universal.extract_buffers or by the HF import
    # carries them as zero/<name>/buffer.pt under the module's own names; without them the router selects its experts
    # with a zero bias and the released model's routing is not reproduced. Restored by name (with the same namespace
    # twins as the parameters), replicated (never sliced); a buffer without an atom is reported, not refused (a model
    # without such buffers has none to restore).
    module = getattr(model, 'module', model)
    n_buf, no_atom = 0, []
    with torch.no_grad():
        for bname, buf in module.named_buffers():
            cands = [bname] + list(_atom_name_candidates(bname, num_layers))
            path = next((os.path.join(zero_dir, c, 'buffer.pt') for c in cands
                         if os.path.isfile(os.path.join(zero_dir, c, 'buffer.pt'))), None)
            if path is None:
                if bname.endswith('expert_bias'):
                    no_atom.append(bname)
                continue
            payload = torch.load(path, map_location='cpu')
            value = payload['buffer'] if isinstance(payload, dict) else payload
            if tuple(value.shape) != tuple(buf.shape):
                raise RuntimeError(f'buffer atom {path} has shape {tuple(value.shape)}, the model\'s {bname} {tuple(buf.shape)}')
            buf.copy_(value.to(dtype=buf.dtype, device=buf.device))
            n_buf += 1
    if n_buf or no_atom:
        print_rank_0(f'[load] universal buffer atoms: {n_buf} buffers restored by name'
                     + (f'; {len(no_atom)} router-bias buffers WITHOUT an atom (first: {no_atom[:3]}) stay at their initial value'
                        if no_atom else ''))


def load_ds_checkpoint_and_setup_megatron(extra_args_provider):
    """Parse Megatron args, build the model, and load a DeepSpeed checkpoint."""
    # Modern megatron-deepspeed defines parse_args in megatron.arguments, not
    # the (BigScience-era) megatron.global_vars._parse_args. We also need to
    # import megatron.initialize so we can monkey-patch the *imported* name
    # inside that module (initialize.py did `from megatron.arguments import
    # parse_args` at load time, so patching megatron.arguments.parse_args
    # alone would not affect initialize_megatron's already-bound reference).
    from megatron.arguments import parse_args as _parse_args
    import megatron.initialize as _mi

    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None
    args = _parse_args(extra_args_provider=extra_args_provider)

    # The engine LOADS WEIGHTS from args.load, but the training-args merge below must read
    # from the ORIGINAL global_step directory. Under --universal-checkpoint, args.load is the
    # per-job .eval_ucp selector (a symlink + latest_universal, no mp_rank/zero files), and
    # DeepSpeedCheckpoint on it dies with ZeroDivisionError (dp_degree = len(zero_files)//...).
    # --ckpt-step-dir carries the real step dir; the pre-UCP layouts still fall back to
    # args.load. If neither yields readable stored args, proceed with CLI args only -- the
    # exact behavior weights-only imports (e.g. the HF->DS DeepSeek conversion) always had.
    _args_src = getattr(args, 'ckpt_step_dir', None) or args.load
    try:
        ds_checkpoint = DeepSpeedCheckpoint(
            _args_src,
            tp_degree=args.tensor_model_parallel_size,
            pp_degree=args.pipeline_model_parallel_size,
            no_pp=args.no_pipeline_parallel,
        )
        cp_args = ds_checkpoint.get_args()
    except Exception as _e:
        print(f'[loader] WARN: cannot read stored training args from {_args_src} ({_e!r}); '
              'proceeding with CLI args only. Correct for weights-only imports; for a trained '
              'checkpoint pass --ckpt-step-dir pointing at its global_step<N> directory.',
              flush=True)
        cp_args = None

    skip_keys = [
        'world_size', 'rank', 'local_rank', 'device_count',
        'micro_batch_size', 'global_batch_size', 'batch_size',
        'tensorboard_dir', 'deepspeed', 'deepspeed_config',
        'data_parallel_size', 'pipeline_model_parallel_size',
        'tensor_model_parallel_size', 'moe_expert_parallel_size',
        'moe_token_dropping', 'load', 'rampup_batch_size',
        'iteration', 'inference', 'random_ltd',
        # A TRAINED checkpoint stores finetune=False, which would stomp the eval CLI's
        # --finetune and silently change which load branch megatron/checkpointing.py takes
        # (observed in job 5259354: `finetune ... False` despite the flag). CLI must win.
        'finetune',
        # Same stomp, different flag: a run trained WITHOUT --universal-checkpoint stores
        # universal_checkpoint=False, which overrides the eval CLI's --universal-checkpoint
        # and silently routes the load down the plain-DS branch (observed 2026-08-15 on the
        # SFT checkpoints: `--load .../.eval_ucp/<tag> has no latest file`). Older
        # checkpoints happened to store True, which is why this never surfaced before.
        # The weight source is an EVAL-time decision; the CLI must win.
        'universal_checkpoint',
        # KV decode (Phase 5): _checkpointed_forward's signature takes no inference_params
        # (transformer.py:2078) -- it DROPS the kwarg silently while the full-length rotary
        # still flows in, a guaranteed shape mismatch. A trained checkpoint stores
        # checkpoint_activations=True; eval must never inherit it (eval does no backward).
        'checkpoint_activations', 'recompute_granularity',
        # Same stomp class again (2026-08-30, jobs 5380809-17 + canary 5382089): a checkpoint
        # trained with TP-companion --sequence-parallel stores sequence_parallel=True. The eval
        # harness feeds UNSCATTERED full-length batches -- there is no scatter at the embedding
        # like training's dropout-region entry -- so ColumnParallelLinear's SP all-gather
        # multiplies the sequence axis by TP (hellaswag q512 -> 4096 vs rotary 512,
        # rotary_pos_embedding.py:71). SP is a training-time activation-sharding optimization;
        # weights are identical without it. Eval must ALWAYS run with it off, and the CLI never
        # passes it, so protecting the parse-time False is sufficient. Its validate_args
        # companion async_tensor_model_parallel_allreduce is auto-derived and must not be
        # inherited either.
        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
        # Same stomp class (2026-08-30, job 5382739): the checkpoint's stored rotary_base
        # silently overrides the eval CLI's --rotary-base, so the submit-side rotary guard
        # (ROTARY_BASE_BY_RUN + the template FATAL check) was cosmetic whenever the
        # checkpoint stores the value -- and rotary counterfactuals (evaluating a model AT
        # a different base than it trained with) were impossible. The CLI is ALWAYS set
        # explicitly (the template refuses to run without it), so protecting it is safe:
        # in production the declared value equals the stored value anyway.
        'rotary_base',
        # Megatron's validate_args has three "auto-derive then assert
        # clean state" pairs. The checkpoint's saved args carry the
        # post-derivation values, which trip the assertions when merged
        # back in. We strip all three pairs:
        #   - num_layers <-> {encoder,decoder}_num_layers (args.py:254-262)
        #   - seq_length <-> {encoder,decoder}_seq_length (args.py:288-293)
        #   - weight_decay_incr_style=constant <-> {start,end}_weight_decay (316-323)
        'encoder_num_layers', 'decoder_num_layers',
        'encoder_seq_length', 'decoder_seq_length',
        'start_weight_decay', 'end_weight_decay',
        # seq_length / max_position_embeddings MUST be controllable from the
        # eval CLI so we can shrink the inference seq_length (e.g. from the
        # 4096 training value down to 2048 for short-content tasks via the
        # 8th field of the EVAL_MAP value). Without these in skip_keys,
        # override_args copies the checkpoint's training-time 4096 back into
        # args, the model is built with seq_length=4096, the eval adapter
        # uses lm.max_length=4096, and the runtime cost stays at the 4096
        # value regardless of what --seq-length was on the CLI. Verified by
        # diagnostic logs: bash SEQ_LEN was overridden to 2048 but Python's
        # args.seq_length was 4096 → mask shape (1,1,4096,4096) in attn-setup.
        'seq_length', 'max_position_embeddings',
        # Fusion flags we MUST control from the eval CLI: the training
        # checkpoint stored bias_gelu_fusion=True, but on PyTorch+ROCm the
        # torch.jit bias-gelu fuser emits CUDA `__nv_bfloat16` source that
        # HIP's compiler rejects. We pass --no-bias-gelu-fusion at eval and
        # need that to survive the override. bias_dropout_fusion same idea
        # (no-op at eval since dropout=0, but stays consistent).
        #
        # Notes on what is NOT skipped any more (was a wrong fix before):
        #   - use_flash_attn_v1/v2/triton: the training checkpoint has Flash
        #     v2 on; we WANT Flash on at eval too. With my FlashSelfAttention
        #     edit (varlen strip/restitch when an attention_mask is passed),
        #     Flash correctly handles left-padding. Forcing Flash off at eval
        #     instead routes to CoreAttention → FusedScaleMaskSoftmax, which
        #     on this ROCm node fails at `import scaled_masked_softmax_cuda`
        #     (libamdhip64.so.6 missing on the compute node LD_LIBRARY_PATH).
        #   - masked_softmax_fusion: only matters when CoreAttention runs;
        #     with Flash on it's dead code. Keep training's value.
        'bias_gelu_fusion', 'bias_dropout_fusion',
    ]
    skip_if_specified = ['merge_file', 'vocab_file']

    if args.eval_fp32:
        cp_args.fp16 = False
        cp_args.bf16 = False
        cp_args.params_dtype = torch.float32

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # Patch the parse_args reference that initialize_megatron sees, so its
    # internal call returns our merged args instead of re-reading CLI. Do
    # NOT pre-set _GLOBAL_ARGS — set_global_variables() asserts it is None.
    _mi.parse_args = lambda *_args, **kwarg: args

    initialize_megatron()
    torch.distributed.barrier()

    override_args(args, cp_args, skip_keys, skip_if_specified)
    _print_args("eval arguments (post-merge)", args)

    # ----------------------------------------------------------------------
    # Diagnostic: print fusion flags AS THEY WILL BE USED for model build.
    # FusedScaleMaskSoftmax reads args.masked_softmax_fusion AT CONSTRUCTION
    # TIME (transformer.py:310). If the override_args call above replaced
    # our --no-masked-softmax-fusion=False with the checkpoint's True, the
    # fused kernel will silently drop our custom mask. The eval ran
    # 4614641/4614672 both produced bit-identical degenerate logits from
    # exactly this trap. Make the resolved value visible from rank_0.log.
    print_rank_0(f">>> [args-resolved] masked_softmax_fusion = "
                 f"{getattr(args, 'masked_softmax_fusion', '<unset>')} "
                 f"(False = our mask is honored; True = fused kernel drops it "
                 f"when attn_mask_type==causal)")
    print_rank_0(f">>> [args-resolved] bias_gelu_fusion       = "
                 f"{getattr(args, 'bias_gelu_fusion', '<unset>')}  "
                 f"(False expected for ROCm)")
    print_rank_0(f">>> [args-resolved] use_flash_attn_*       = v1="
                 f"{getattr(args, 'use_flash_attn_v1', None)} "
                 f"v2={getattr(args, 'use_flash_attn_v2', None)} "
                 f"triton={getattr(args, 'use_flash_attn_triton', None)} "
                 f"(all None/False expected; Flash path bypasses our mask too)")
    print_rank_0(f">>> [args-resolved] seq_length             = "
                 f"{getattr(args, 'seq_length', '<unset>')}  "
                 f"max_position_embeddings = "
                 f"{getattr(args, 'max_position_embeddings', '<unset>')}  "
                 f"(eval-CLI value should survive checkpoint-arg override; "
                 f"if these don't match what --seq-length set, seq_length / "
                 f"max_position_embeddings are missing from skip_keys)")

    if not args.deepspeed:
        raise NotImplementedError(
            "evaluate_v04.py requires --deepspeed --deepspeed_config <path>. "
            "Non-deepspeed path is not supported in this adapter."
        )

    import deepspeed
    deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = (
        lambda self, ckpt_file_name: None
    )

    # Mirror what megatron.training.pretrain() does between initialize_megatron
    # and setup_model_and_optimizer: load the deepspeed_config JSON into a
    # dict on args.deepspeed_config_dict and clear args.deepspeed_config to
    # None. setup_model_and_optimizer passes both to deepspeed.initialize,
    # which asserts that only one of (args.deepspeed_config, config kwarg)
    # is non-None.
    if args.deepspeed_config is not None:
        import json as _json
        with open(args.deepspeed_config, 'r', encoding='utf-8') as _f:
            args.deepspeed_config_dict = _json.load(_f)
        args.deepspeed_config = None

    cp_path = args.load
    # Weight-source selection. The engine's own load runs ONLY for a plain DS checkpoint
    # (args.load left intact). Both other paths null args.load and place the weights
    # themselves after the model is built:
    #   consolidated -- flat .pt, global->local expert remap (below)
    #   universal    -- per-parameter atoms addressed by name (_load_from_universal_atoms;
    #                   the engine's universal branch is config-gated and optimizer-coupled,
    #                   see that function's docstring for why it is not used)
    _ucp_zero_dir = None
    if args.consolidated_load is not None:
        args.load = None
    elif getattr(args, 'universal_checkpoint', False):
        _ucp_zero_dir = _resolve_universal_zero_dir(args.load)
        print_rank_0(f'[load] universal checkpoint resolved: {_ucp_zero_dir}')
        args.load = None

    # engine.py:2894 calls optimizer._restore_from_bit16_weights() whenever the ZeRO load is
    # SKIPPED. It indexes bf16_partitioned_groups[partition_id], built for the CURRENT layout, and
    # raises IndexError on non-zero ranks after a DP change.
    #
    # Patched on the CLASS, not the instance: the optimizer is constructed inside
    # setup_model_and_optimizer, which is also where load_checkpoint runs, so an instance patch
    # would be too late. Skipped on the universal path only because there the engine load never
    # runs at all (args.load was nulled above) -- there is nothing to patch.
    if not getattr(args, 'universal_checkpoint', False):
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as _DSZ
        _DSZ._restore_from_bit16_weights = lambda self: None
        if hasattr(_DSZ, 'refresh_fp32_params'):
            _DSZ.refresh_fp32_params = lambda self: None
        print_rank_0('[load] patched _restore_from_bit16_weights (non-universal path)')
    # Plain-DS path only (consolidated and universal nulled args.load above). The engine
    # consults exactly the `latest` file here -- job 5259354 showed it IGNORES
    # latest_universal unless its config carries checkpoint.load_universal -- and
    # megatron/checkpointing.py:557 only WARNS when the tag cannot be resolved, leaving the
    # model randomly initialised and the eval reporting chance accuracy. Fail loudly instead.
    if args.load is not None:
        if not os.path.isfile(os.path.join(args.load, 'latest')):
            raise FileNotFoundError(
                f'--load {args.load} has no `latest` file, so the DeepSpeed engine would '
                'silently leave the model randomly initialised. (--load takes the PARENT of '
                'the global_step* directory. For a universal checkpoint pass '
                '--universal-checkpoint; the engine load is bypassed there.)')
        print_rank_0(f'[load] tag file: {args.load}/latest -> '
                     f'{open(os.path.join(args.load, "latest")).read().strip()}')

    # Modern Megatron-DeepSpeed requires model_type. GPT is decoder-only;
    # pretrain_gpt_deepspeed.py uses ModelType.encoder_or_decoder.
    model, _, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    model = model[0]
    zero_enabled = model._config.zero_enabled
    model._config.zero_enabled = False

    if args.consolidated_load is not None:
        # ---- Consolidated-checkpoint path -----------------------------------
        # Bypass DeepSpeed engine load_checkpoint entirely. The consolidator
        # script (eval/xmoe_consolidate.py) merged the 1 shared + 1536 expert
        # files into a single flat state_dict keyed by GLOBAL expert index.
        # Here we remap global -> local expert index for THIS rank's EP slice,
        # then call model.module.load_state_dict directly.
        import re as _re
        from deepspeed.utils import groups as _ds_groups
        print_rank_0(f'[consolidated_load] reading {args.consolidated_load}')
        flat_sd = torch.load(args.consolidated_load, map_location='cpu')
        print_rank_0(f'[consolidated_load] {len(flat_sd)} flat keys loaded')

        ep_size = max(1, args.moe_expert_parallel_size)
        num_global = args.num_experts[0] if isinstance(args.num_experts, (list, tuple)) else args.num_experts
        num_local = num_global // ep_size

        # Determine this rank's EP rank. The deepspeed group naming uses the
        # number of LOCAL experts (e.g. 'ep_size_8' for 8 local per rank), so
        # we ask groups for the rank within that named group.
        try:
            ep_group_name = _ds_groups._get_max_expert_size_name()
            ep_rank = _ds_groups._get_expert_parallel_rank(ep_group_name)
        except Exception as _e:
            # Fallback: assume EP world-rank == global-rank when EP == world.
            ep_rank = torch.distributed.get_rank() % ep_size
            print_rank_0(f'[consolidated_load] groups lookup failed ({_e}); '
                         f'using fallback ep_rank={ep_rank}')

        my_lo = ep_rank * num_local
        my_hi = my_lo + num_local
        print_rank_0(f'[consolidated_load] this rank: ep_rank={ep_rank}, '
                     f'global experts [{my_lo}..{my_hi - 1}] -> local [0..{num_local - 1}]')

        expert_re = _re.compile(r'(\.experts\.deepspeed_experts\.)(\d+)(\.)')
        local_sd = {}
        dropped_expert_keys = 0
        for k, v in flat_sd.items():
            m = expert_re.search(k)
            if m is None:
                # Non-expert key (embedding, attention, layernorm, MoE gate, etc.)
                # — same on every rank.
                local_sd[k] = v
                continue
            global_idx = int(m.group(2))
            if not (my_lo <= global_idx < my_hi):
                dropped_expert_keys += 1
                continue
            local_idx = global_idx - my_lo
            new_k = expert_re.sub(lambda mm: f'{mm.group(1)}{local_idx}{mm.group(3)}', k, count=1)
            local_sd[new_k] = v

        print_rank_0(f'[consolidated_load] remapped to {len(local_sd)} local keys; '
                     f'dropped {dropped_expert_keys} expert keys not on this rank')

        # Bypass Megatron's GPTModel.load_state_dict override (gpt_model.py:202).
        # That override:
        #   (a) expects a hierarchical state_dict (top-level 'language_model'
        #       key wrapping a nested dict, plus a 'moe_state_dict' field), and
        #   (b) returns None instead of an IncompatibleKeys tuple.
        # Our consolidated state_dict is FLAT (dotted module-path keys, the way
        # state_dict() naturally emits them). Calling torch.nn.Module.load_state_dict
        # directly on model.module bypasses the override and uses PyTorch's
        # standard flat-key load, which is what our format needs.
        result = torch.nn.Module.load_state_dict(model.module, local_sd, strict=False)
        if result is not None:
            missing, unexpected = result
            print_rank_0(f'[consolidated_load] load_state_dict: '
                         f'{len(missing)} missing, {len(unexpected)} unexpected')
            if missing[:5]:
                print_rank_0(f'[consolidated_load] first missing keys: {missing[:5]}')
            if unexpected[:5]:
                print_rank_0(f'[consolidated_load] first unexpected keys: {unexpected[:5]}')
        else:
            print_rank_0('[consolidated_load] load_state_dict returned None '
                         '(unexpected for nn.Module base; proceeding)')
    elif _ucp_zero_dir is not None:
        # ---- Universal-checkpoint path: atoms by name, no engine involvement ----
        _load_from_universal_atoms(model, _ucp_zero_dir)
    else:
        # ---- Plain DS checkpoint: the engine already loaded during setup ----
        # args.load was left intact, so setup_model_and_optimizer ->
        # megatron.checkpointing.load_checkpoint -> engine.load_checkpoint() ran,
        # resolving the tag from the `latest` file in --load.
        print_rank_0(f'[load] weights loaded by the engine from {cp_path}')

    model._config.zero_enabled = zero_enabled

    if args.eval_fp32:
        model = model.float()

    torch.distributed.barrier()
    return model
