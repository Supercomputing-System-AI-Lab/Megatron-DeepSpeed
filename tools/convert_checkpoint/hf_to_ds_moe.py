#!/usr/bin/env python
# coding=utf-8
"""Convert a HuggingFace DeepSeek-MoE checkpoint into a Megatron-DeepSpeed MoE checkpoint.

This is the *import* direction (HF -> DeepSpeed). The other scripts in this folder
(deepspeed_to_megatron.py, deepspeed_to_transformers.py) only export, and only handle
dense GPT-2, so they cannot be reused here.

    Verified against: deepseek-ai/deepseek-moe-16b-base
    Target model:     GPTModel (i.e. runs launched with --no-pipeline-parallel), TP=1

--------------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------------
Step 1 -- dry run (no torch required, reads only safetensors headers). Validates that
          every expected key exists with the expected shape and prints what would be
          written. ALWAYS run this first:

    cd Megatron-DeepSpeed-X-MoE
    python tools/convert_checkpoint/hf_to_ds_moe.py \
        --hf-model-path /lustre/orion/gen150/scratch/zixianw4/models/deepseek-moe-16b-base \
        --output-folder  /lustre/orion/gen150/scratch/zixianw4/ckpt/deepseek_16b_ds \
        --dry-run

Step 2 -- real conversion (needs torch + safetensors; run inside the ROCm env, on a
          compute node or a login node with enough RAM -- peak RSS is one tensor at a
          time, but writes ~31GB):

    python tools/convert_checkpoint/hf_to_ds_moe.py \
        --hf-model-path /lustre/orion/gen150/scratch/zixianw4/models/deepseek-moe-16b-base \
        --output-folder  /lustre/orion/gen150/scratch/zixianw4/ckpt/deepseek_16b_ds

Step 3 -- launch. The DeepSeek_16B entry in examples_elmoe/utils/model_registry.py already
          emits the matching ARCH_FLAGS. Point the run at the output folder and load
          weights only (no optimizer/RNG state exists in a converted checkpoint):

    --load /lustre/orion/gen150/scratch/zixianw4/ckpt/deepseek_16b_ds \
    --finetune --no-load-optim --no-load-rng

          The run MUST use --no-pipeline-parallel (GPTModel naming) and TP=1, which is
          what the current frontier templates already do.

--------------------------------------------------------------------------------------
WHY EACH TRANSFORM EXISTS  (all verified against this repo, not assumed)
--------------------------------------------------------------------------------------
1. RoPE -- NO permutation.
   megatron/model/rotary_pos_embedding.py::_rotate_half does
       rearrange(x, '... (j d) -> ... j d', j=2) -> x1=x[:D/2], x2=x[D/2:] -> cat(-x2, x1)
   which is bit-identical to HF's rotate_half, and emb=cat((freqs,freqs)) with base 10000
   matches DeepSeek's rope_theta=10000. So q/k need no reordering. (The famous Llama
   conversion permute() is NOT needed here -- that one exists because the *original Meta*
   weights are interleaved; HF DeepSeek weights are already half-split.)

2. QKV -- per-head interleave REQUIRED.
   megatron/model/transformer.py reshapes [sq,b,(np*3*hn)] -> [sq,b,np,3*hn] then splits
   into 3 along the last dim. So the fused matrix is [h0_q,h0_k,h0_v, h1_q,h1_k,h1_v, ...],
   NOT [all_q; all_k; all_v]. Getting this wrong loads cleanly and silently scrambles
   attention.

3. SwiGLU -- concat [gate; up], gate FIRST.
   megatron/model/transformer.py:
       def swiglu(x): x = torch.chunk(x, 2, dim=-1); return F.silu(x[0]) * x[1]
   HF: down_proj(silu(gate_proj(x)) * up_proj(x))  =>  x[0]=gate, x[1]=up.
   --swiglu sets gated_linear_unit=True which doubles dense_h_to_4h's output width to 2*ffn.

4. Shared experts -- direct copy, NO fusing.
   HF stores mlp.shared_experts.* already fused as ONE MLP of width
   n_shared * moe_intermediate_size = 2*1408 = 2816. transformer.py builds the shared
   expert as ParallelMLP with _shared_cfg.ffn_hidden_size = ffn_hidden_size * n_shared
   = 2816. Identical -- so gate/up still need the [gate;up] concat, but no per-expert merge.

5. Expert file index is the MoE-MODULE ORDINAL, not the transformer layer index.
   Both deepspeed/runtime/engine.py::_save_moe_checkpoint and ::load_moe_state_dict walk
   `for n_module, module in model.named_modules(): if isinstance(module, MoE): ...` with a
   `moe_layer_id` counter. DeepSeek has first_k_dense_replace=1, so transformer layer L
   (1..27) is MoE ordinal L-1 (0..26). The FILE is layer_{L-1}_expert_{E}..., while the
   KEY inside references layers.{L}. These differ -- for an all-MoE model they coincide,
   which is exactly why this is easy to get wrong.

6. Expert files store the GLOBAL expert id in the key; the load path rewrites global->local
   (engine.py: key.replace(f'{prefix}{global_expert_id}', f'{prefix}{local_expert_id}')).
   Expert files are the RAW state dict -- no 'module' wrapper. mp_rank_00 IS wrapped.

7. No bias anywhere (config attention_bias=false and no .bias tensors exist), so the run
   must pass --disable-bias-linear. RMSNorm (weight only) => --normalization rmsnorm, and
   rms_norm_eps=1e-6 => --layernorm-epsilon 1e-6 (Megatron defaults to 1e-5).
"""

import argparse
import json
import os
import re
import struct
import sys

MP_RANK_FILE = "mp_rank_00_model_states.pt"


# ----------------------------------------------------------------------------- helpers
def expert_ckpt_name(moe_ordinal, global_expert_id):
    """Mirrors DeepSpeedEngine._get_expert_ckpt_name for layer_id > -1, mp_rank 0."""
    return f"layer_{moe_ordinal}_expert_{global_expert_id}_mp_rank_00_model_states.pt"


def vocab_size_with_padding(orig_vocab_size, divisible_by, tp):
    """Mirror of megatron/tokenizer/tokenizer.py::_vocab_size_with_padding.

    Megatron sizes the embedding from the TOKENIZER (rounded up), never from
    config.vocab_size. A HF model whose config.vocab_size exceeds its tokenizer (DeepSeek:
    102400 vs 100015; Llama-3: 128256 vs 128000) therefore ships more embedding rows than
    Megatron will allocate, so the converter -- not the model -- must reconcile them.
    """
    multiple = divisible_by * tp
    after = orig_vocab_size
    while (after % multiple) != 0:
        after += 1
    return after


def detect_tokenizer_vocab_size(model_dir):
    """Token count from tokenizer.json, matching tokenizers' get_vocab_size(). None if absent."""
    path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            t = json.load(f)
        return len(t["model"]["vocab"]) + len(t.get("added_tokens", []))
    except (KeyError, TypeError, ValueError):
        return None


def read_safetensors_headers(model_dir):
    """{tensor_name: (shape, dtype, file)} using only the safetensors header. No torch."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]

    meta = {}
    for fname in files:
        path = os.path.join(model_dir, fname)
        with open(path, "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hdr_len))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            meta[k] = (tuple(v["shape"]), v["dtype"], fname)
    return meta


class Plan:
    """The full HF->DS mapping, derived from config.json.

    Built without touching tensor data so it can be validated in --dry-run.
    """

    def __init__(self, cfg, padded_vocab=None):
        self.h = cfg["hidden_size"]
        # Rows the emitted embedding/lm_head must have, = what Megatron will allocate.
        # None => keep config.vocab_size (no resize).
        self.padded_vocab = padded_vocab
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg.get("num_key_value_heads", self.heads)
        self.head_dim = self.h // self.heads
        self.layers = cfg["num_hidden_layers"]
        self.n_routed = cfg["n_routed_experts"]
        self.n_shared = cfg["n_shared_experts"]
        self.moe_inter = cfg["moe_intermediate_size"]
        self.dense_inter = cfg["intermediate_size"]
        self.first_k_dense = cfg["first_k_dense_replace"]
        self.moe_freq = cfg.get("moe_layer_freq", 1)
        self.vocab = cfg["vocab_size"]

        if self.heads != self.kv_heads:
            raise SystemExit(
                f"num_key_value_heads({self.kv_heads}) != num_attention_heads({self.heads}): "
                "this model uses GQA, which takes Megatron's separate query/key_value path, "
                "not the fused query_key_value this converter targets."
            )
        if self.moe_freq != 1:
            raise SystemExit(f"moe_layer_freq={self.moe_freq} unsupported (expected 1).")

    def is_moe_layer(self, layer):
        return layer >= self.first_k_dense

    def moe_ordinal(self, layer):
        """Transformer layer index -> MoE module ordinal (see note 5 in the docstring)."""
        return layer - self.first_k_dense

    # -- non-expert weights, written into mp_rank_00_model_states.pt under 'module'
    def non_expert_ops(self):
        """[(dst_key, op, [src_keys]), ...] -- op names are handled by apply_op()."""
        ops = [("language_model.embedding.word_embeddings.weight", "resize_vocab",
                ["model.embed_tokens.weight"])]

        for L in range(self.layers):
            p = f"language_model.encoder.layers.{L}"
            ops += [
                (f"{p}.input_layernorm.weight", "copy", [f"model.layers.{L}.input_layernorm.weight"]),
                (f"{p}.self_attention.query_key_value.weight", "fuse_qkv",
                 [f"model.layers.{L}.self_attn.q_proj.weight",
                  f"model.layers.{L}.self_attn.k_proj.weight",
                  f"model.layers.{L}.self_attn.v_proj.weight"]),
                (f"{p}.self_attention.dense.weight", "copy", [f"model.layers.{L}.self_attn.o_proj.weight"]),
                (f"{p}.post_attention_layernorm.weight", "copy",
                 [f"model.layers.{L}.post_attention_layernorm.weight"]),
            ]

            if not self.is_moe_layer(L):
                # Dense layer (first_k_dense_replace). Width = intermediate_size, which is
                # why the run needs --dense-ffn-hidden-size.
                ops += [
                    (f"{p}.mlp.dense_h_to_4h.weight", "fuse_gate_up",
                     [f"model.layers.{L}.mlp.gate_proj.weight", f"model.layers.{L}.mlp.up_proj.weight"]),
                    (f"{p}.mlp.dense_4h_to_h.weight", "copy", [f"model.layers.{L}.mlp.down_proj.weight"]),
                ]
                continue

            # Router. HF mlp.gate -> DeepSpeed TopKGate.wg (sharded_moe.py: self.wg = Linear).
            ops.append((f"{p}.mlp.deepspeed_moe.gate.wg.weight", "copy",
                        [f"model.layers.{L}.mlp.gate.weight"]))

            if self.n_shared > 0:
                sp = f"{p}.mlp.shared_experts.shared_mlp"
                ops += [
                    (f"{sp}.dense_h_to_4h.weight", "fuse_gate_up",
                     [f"model.layers.{L}.mlp.shared_experts.gate_proj.weight",
                      f"model.layers.{L}.mlp.shared_experts.up_proj.weight"]),
                    (f"{sp}.dense_4h_to_h.weight", "copy",
                     [f"model.layers.{L}.mlp.shared_experts.down_proj.weight"]),
                ]

        ops += [
            ("language_model.encoder.final_layernorm.weight", "copy", ["model.norm.weight"]),
            ("language_model.output_layer.weight", "resize_vocab", ["lm_head.weight"]),
        ]
        return ops

    # -- routed experts, one file each
    def expert_files(self):
        """{filename: [(dst_key, op, [src_keys]), ...]}"""
        out = {}
        for L in range(self.layers):
            if not self.is_moe_layer(L):
                continue
            ordinal = self.moe_ordinal(L)
            for E in range(self.n_routed):
                # Key uses the transformer layer index L and the GLOBAL expert id E;
                # the filename uses the MoE ordinal. These differ -- see note 5.
                k = (f"language_model.encoder.layers.{L}"
                     f".mlp.deepspeed_moe.experts.deepspeed_experts.{E}")
                src = f"model.layers.{L}.mlp.experts.{E}"
                out[expert_ckpt_name(ordinal, E)] = [
                    (f"{k}.dense_h_to_4h.weight", "fuse_gate_up",
                     [f"{src}.gate_proj.weight", f"{src}.up_proj.weight"]),
                    (f"{k}.dense_4h_to_h.weight", "copy", [f"{src}.down_proj.weight"]),
                ]
        return out

    def expected_shape(self, dst_key, op, srcs, meta):
        """Shape the emitted tensor must have, for dry-run checking."""
        if op == "copy":
            return meta[srcs[0]][0]
        if op == "resize_vocab":
            rows = self.padded_vocab or meta[srcs[0]][0][0]
            return (rows, meta[srcs[0]][0][1])
        if op == "fuse_qkv":
            return (3 * self.heads * self.head_dim, self.h)
        if op == "fuse_gate_up":
            g, u = meta[srcs[0]][0], meta[srcs[1]][0]
            return (g[0] + u[0], g[1])
        raise AssertionError(op)


# ---------------------------------------------------------------------------------------
# Universal-checkpoint emitter
#
# A universal checkpoint stores one directory per parameter -- <ckpt>/zero/<name>/fp32.pt --
# and each run slices those at load time for whatever parallelism it has. That is what makes
# it topology-free, and it is why an HF import wants this format rather than the per-rank
# DeepSpeed one: the same artifact then loads at any TP/EP/PP without being reconverted.
#
# Converting HF -> universal is much cheaper than converting a DeepSpeed checkpoint. The
# expensive half of ds_to_universal.py is *merging*: a parameter's bytes are scattered across
# ZeRO shards at arbitrary offsets, so it must extract fragments and reassemble them. HF
# safetensors already hold whole, unsharded tensors -- the merged form. So there is no
# extract phase and no merge phase; this is rename, reshape, stamp, write.
#
# Two properties fall out for free:
#   * Expert names are already global. There is no per-rank view of the model here, so
#     `deepspeed_experts.<E>` is the global expert id, which is exactly what the universal
#     format wants.
#   * The vocabulary is stored UNPADDED. Megatron pads it to a multiple of
#     make_vocab_size_divisible_by * TP, so the padded height is a property of the topology,
#     not of the model. The loader pads back to whatever the run needs, so --target-tp and
#     --make-vocab-size-divisible-by are irrelevant on this path.

def ds_version_string():
    """The loader asserts this is non-empty and, for ZeRO stage 1, that it parses >= 0.3.17."""
    try:
        from deepspeed.git_version_info import version
        return version
    except Exception:
        return "0.15.5"


FP32_FILE = "fp32.pt"

# Keys inside an atom, mirroring deepspeed/checkpoint/constants.py. Duplicated rather than
# imported so this script stays runnable without DeepSpeed on the path.
PARAM_KEY = "param"
CAT_DIM_KEY = "cat_dim"
VOCAB_TENSOR_KEY = "vocab_tensor"
PARAM_N_SUB_PARAMS_KEY = "param_n_sub_params"

# Row-parallel weights are split along their INPUT dimension, i.e. dim 1 of the stored
# [out, in] tensor; everything else that tensor parallelism splits is split along dim 0.
# The loader reads cat_dim to know which way to chunk. Only the row-parallel ones need
# naming -- dim 0 is the default when the key is absent, but it is written explicitly so an
# atom is self-describing.
_ROW_PARALLEL_SUFFIXES = (".self_attention.dense.weight", ".dense_4h_to_h.weight")

# Parameters whose first dimension is the padded vocabulary.
_VOCAB_SUFFIXES = ("embedding.word_embeddings.weight", "output_layer.weight")


def atom_metadata(dst_key, op):
    """The metadata a universal atom must carry, as {key: value}.

    `cat_dim` says which dimension tensor parallelism splits the tensor along, so the loader
    can chunk it. `vocab_tensor` says the first dimension is the vocabulary and is stored
    unpadded. `param_n_sub_params` says the tensor is really two tensors concatenated.

    That last one is the subtle case, and the emitter is the only place that can know it. A
    gated MLP stores gate and up as one [2*ffn, hidden] tensor -- which is what the
    `fuse_gate_up` op just built. Splitting it naively across N tensor-parallel ranks gives
    rank r a contiguous slice that straddles the gate/up boundary: the right number of
    elements and the wrong tensor, with nothing to raise. The loader instead splits each half
    separately and re-concatenates, but only when told the tensor has two sub-parameters.

    Keyed off the OP rather than the parameter name, because the op is the statement that the
    tensor is a concatenation -- a name suffix would only be a guess about it. Harmless at
    TP=1, where chunk(2) -> chunk(1) -> cat is the identity.
    """
    md = {CAT_DIM_KEY: 1 if dst_key.endswith(_ROW_PARALLEL_SUFFIXES) else 0}
    if dst_key.endswith(_VOCAB_SUFFIXES):
        md[VOCAB_TENSOR_KEY] = True
    if op == "fuse_gate_up":
        md[PARAM_N_SUB_PARAMS_KEY] = 2
    return md


def write_universal_model_states(out_dir, plan, num_moe_layers):
    """Write the small mp_rank_00_model_states.pt a universal load still requires.

    A universal load never reads the weights from this file -- engine._load_checkpoint skips
    load_module_state_dict entirely and the weights come from the fp32 atoms. But it does
    still LOAD the file, and dereferences `checkpoint['dp_world_size']` unconditionally. If
    the file is absent, _load_checkpoint returns (None, None); `load_path is None` then makes
    load_zero_checkpoint False, so the atoms are never read and the model trains on from its
    random initialisation, without an error anywhere.

    So it is written, but WITHOUT the 30 GB `module` payload that would never be read. That
    makes this folder universal-only: to also load it the ordinary --finetune way, emit the
    'ds' format alongside it with --emit both.
    """
    import torch
    state = {
        "module": {},                                  # deliberately empty -- see docstring
        "num_experts": [plan.n_routed] * num_moe_layers,
        "checkpoint_version": 2.0,
        "iteration": 0,
        "lr_scheduler": None,
        "data_sampler": None,
        "random_ltd": None,
        "sparse_tensor_module_names": None,
        "skipped_steps": 0,
        "global_steps": 0,
        "global_samples": 0,
        "dp_world_size": 1,
        "mp_world_size": 1,
    }
    torch.save(state, os.path.join(out_dir, MP_RANK_FILE))
    print(f"  {MP_RANK_FILE}  (no module payload; the atoms carry the weights)")


def write_universal(out_dir, all_ops, get, plan, ds_version, atom_dtype):
    """Write <out_dir>/zero/<name>/fp32.pt for every parameter, plus optimizer_state.pt."""
    import torch

    zero_dir = os.path.join(out_dir, "zero")
    os.makedirs(zero_dir, exist_ok=True)

    for i, (dst, op, srcs) in enumerate(all_ops, 1):
        tensor = apply_op(op, [get(s) for s in srcs], plan)
        ckpt = {PARAM_KEY: tensor.to(atom_dtype)}
        ckpt.update(atom_metadata(dst, op))
        param_dir = os.path.join(zero_dir, dst)
        os.makedirs(param_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(param_dir, FP32_FILE))
        if i % 500 == 0 or i == len(all_ops):
            print(f"  atoms: {i}/{len(all_ops)}")

    # The loader reads this file before any atom. It needs ds_version (asserted non-empty,
    # and required to be >= 0.3.17 for ZeRO stage 1); everything else it takes with a
    # default. param_groups is deliberately EMPTY: imported weights carry no optimizer
    # history, so every group hyperparameter -- lr, betas, weight decay, step -- must come
    # from the run's own configuration rather than from here.
    torch.save({"ds_version": ds_version, "param_groups": []},
               os.path.join(zero_dir, "optimizer_state.pt"))
    print(f"  zero/optimizer_state.pt  (param_groups=[], fresh optimizer)")


def apply_op(op, tensors, plan):
    """Execute a transform. torch is imported lazily so --dry-run stays dependency-free."""
    import torch  # noqa: F401  (only needed on the real path)

    if op == "copy":
        return tensors[0]

    if op == "resize_vocab":
        t = tensors[0]
        target = plan.padded_vocab
        if target is None or t.shape[0] == target:
            return t
        if t.shape[0] > target:
            # Trim. Safe only because the dropped rows are above the tokenizer's highest
            # id -- main() asserts that before we get here.
            return t[:target].contiguous()
        pad = torch.zeros(target - t.shape[0], t.shape[1], dtype=t.dtype)
        return torch.cat([t, pad], dim=0).contiguous()

    if op == "fuse_qkv":
        q, k, v = tensors
        nh, hd = plan.heads, plan.head_dim
        # [nh*hd, h] -> [nh, hd, h], concat along the head's row axis, flatten back.
        # Produces [h0_q, h0_k, h0_v, h1_q, ...] to match the [sq,b,np,3*hn] reshape.
        qh = q.reshape(nh, hd, -1)
        kh = k.reshape(nh, hd, -1)
        vh = v.reshape(nh, hd, -1)
        return torch.cat([qh, kh, vh], dim=1).reshape(3 * nh * hd, -1).contiguous()

    if op == "fuse_gate_up":
        gate, up = tensors
        # chunk(2)[0] must be gate: silu(x[0]) * x[1].
        return torch.cat([gate, up], dim=0).contiguous()

    raise AssertionError(op)


def to_pipe_name(key, num_layers, untied=True):
    """GPTModel parameter name -> GPTModelPipe parameter name.

    GPTModelPipe is a PipelineModule: every layer is add_module()'d under its GLOBAL spec index
    (deepspeed/runtime/pipe/module.py:319-335, `layer_idx = local_idx + self._local_start`).
    That index is what the parameter name starts with, and it does NOT depend on the pipeline
    degree -- which is exactly why a pipe-named checkpoint reshards across PP and a GPTModel-named
    one cannot be loaded by a PP>1 run at all.

    The spec order is fixed by GPTModelPipe.__init__ (megatron/model/gpt_model.py):

        0                 _to_float16                    (no parameters)
        1                 EmbeddingPipe                  -> 1.word_embeddings.weight
        2 .. 1+N          ParallelTransformerLayerPipe   -> {2+L}.<rest>
        2+N               final norm (LayerNorm/RMSNorm) -> {2+N}.weight
        3+N               LMHeadPipe                     -> {3+N}.lm_head.weight

    Offsets are computed from num_layers rather than hardcoded, so a different depth stays correct.
    `untied` reflects --untie-embeddings-and-output-weights; when tied, the embedding and the head
    are one TiedLayerSpec('embed') and are named tied_modules.embed.* instead.
    """
    EMB = "language_model.embedding.word_embeddings.weight"
    NORM = "language_model.encoder.final_layernorm.weight"
    HEAD = "language_model.output_layer.weight"
    if key == EMB:
        return "1.word_embeddings.weight" if untied else "tied_modules.embed.word_embeddings.weight"
    if key == NORM:
        return f"{2 + num_layers}.weight"
    if key == HEAD:
        return (f"{3 + num_layers}.lm_head.weight" if untied
                else "tied_modules.embed.word_embeddings.weight")
    m = re.match(r"^language_model\.encoder\.layers\.(\d+)\.(.+)$", key)
    if m:
        return f"{2 + int(m.group(1))}.{m.group(2)}"
    raise AssertionError(f"no GPTModelPipe name for {key!r} (update to_pipe_name)")


def nest_module(flat):
    """Flat 'language_model.encoder.layers.N...' keys -> the NESTED dict the loader wants.

    megatron GPTModel.load_state_dict / TransformerLanguageModel.load_state_dict route by
    top-level sub-dict, not by dotted prefix: they do `state_dict['encoder']`,
    `state_dict['embedding']`, `state_dict['output_layer']`. This mirrors
    state_dict_for_save_checkpoint (its inverse), NOT the flat module.state_dict() that
    DeepSpeedEngine.module_state_dict happens to write. Flat keys make 'encoder' absent as a
    key -> the encoder gets an empty dict -> every layer reported "Missing key(s)". So the
    non-expert weights must be nested:
        {'language_model': {'embedding': {'word_embeddings': {'weight': ...}},
                            'encoder':   {'layers.0.input_layernorm.weight': ..., ...,
                                          'final_layernorm.weight': ...},
                            'output_layer': {'weight': ...}}}
    Routed experts stay FLAT in their own files -- load_moe_state_dict re-injects them and
    they route via the moe_state_dict path, so they are not nested here.
    """
    lm = {"embedding": {"word_embeddings": {}}, "encoder": {}}
    out = {"language_model": lm}
    for k, v in flat.items():
        if k == "language_model.embedding.word_embeddings.weight":
            lm["embedding"]["word_embeddings"]["weight"] = v
        elif k.startswith("language_model.encoder."):
            # encoder sub-dict is FLAT relative to the encoder: 'layers.N...' /
            # 'final_layernorm.weight' (matches ParallelTransformer.state_dict()).
            lm["encoder"][k[len("language_model.encoder."):]] = v
        elif k == "language_model.output_layer.weight":
            lm.setdefault("output_layer", {})["weight"] = v
        else:
            raise AssertionError(f"unexpected non-expert key (update nest_module): {k}")
    return out


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Convert HF DeepSeek-MoE -> Megatron-DeepSpeed MoE checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hf-model-path", required=True, help="Folder with config.json + safetensors")
    ap.add_argument("--output-folder", required=True, help="DeepSpeed checkpoint root to write")
    ap.add_argument("--tag", default="global_step0", help="Checkpoint tag subfolder (default: global_step0)")
    ap.add_argument("--make-vocab-size-divisible-by", type=int, default=128,
                    help="Must match the training run's --make-vocab-size-divisible-by "
                         "(default 128). Used to reproduce Megatron's padded_vocab_size.")
    ap.add_argument("--target-tp", type=int, default=1,
                    help="Tensor-parallel size of the target run (default 1). Megatron's "
                         "padding multiple is divisible_by * tp, so this changes the "
                         "emitted embedding height.")
    ap.add_argument("--tokenizer-vocab-size", type=int, default=None,
                    help="Token count the run's tokenizer will report. Default: auto-detect "
                         "from tokenizer.json. Megatron pads THIS (not config.vocab_size) "
                         "to size the embedding.")
    ap.add_argument("--emit", choices=["ds", "universal", "both"], default="ds",
                    help="Output format. 'ds' (default) writes the per-rank DeepSpeed layout "
                         "this script has always written, loadable with --finetune at the "
                         "topology it was built for. 'universal' writes a universal "
                         "checkpoint, loadable at ANY TP/EP/PP with --universal-checkpoint "
                         "--finetune. 'both' writes each into its own tag folder.")
    ap.add_argument("--atom-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="dtype of the universal atoms (default fp32, matching what a real "
                         "conversion stores). bf16 halves the size and is lossless for a "
                         "bf16 source, but the file is still named fp32.pt.")
    ap.add_argument("--target-model", choices=["gptmodelpipe", "gptmodel"], default="gptmodelpipe",
                    help="Which Megatron model class the emitted checkpoint is for. "
                         "gptmodelpipe (DEFAULT) = runs WITHOUT --no-pipeline-parallel; names are "
                         "'1.word_embeddings.weight' style and reshard across any PP degree. "
                         "gptmodel = runs WITH --no-pipeline-parallel (PP=1 only); names are "
                         "'language_model.encoder.layers.N...' and CANNOT be loaded at PP>1.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate the full mapping + shapes from safetensors headers and print "
                         "the plan. Requires no torch and writes nothing.")
    args = ap.parse_args()

    # ---- target-model banner: printed on EVERY run, because picking the wrong one produces a
    # checkpoint that looks fine on disk and fails only when a job tries to load it. -----------
    if args.target_model == "gptmodelpipe":
        print("=" * 86)
        print(" TARGET MODEL: GPTModelPipe   (default)")
        print("=" * 86)
        print("  Emitted names : 1.word_embeddings.weight, 2.self_attention..., "
              f"{'{'}2+N{'}'}.weight, {'{'}3+N{'}'}.lm_head.weight")
        print("  Structure     : FLAT state dict")
        print("  Load with     : any --pipeline-model-parallel-size, and WITHOUT")
        print("                  --no-pipeline-parallel")
        print("  Reshards PP   : YES. PipelineModule names layers by GLOBAL spec index, so the")
        print("                  names do not change with the pipeline degree.")
        print()
        print("  !! Your slurm template forces --no-pipeline-parallel whenever PP_SIZE == 1.")
        print("     A PP=1 run therefore builds GPTModel and will REJECT this checkpoint.")
        print("     Use PP_SIZE >= 2, or change that else-branch to")
        print("     '--pipeline-model-parallel-size 1'.")
        print("=" * 86)
    else:
        print("=" * 86)
        print(" TARGET MODEL: GPTModel   (NOT the default -- PP=1 only)")
        print("=" * 86)
        print("  Emitted names : language_model.encoder.layers.N...")
        print("  Structure     : NESTED {embedding, encoder, output_layer}")
        print("  Load with     : --no-pipeline-parallel  (PP=1 only)")
        print("  Reshards PP   : NO. A PP>1 run builds GPTModelPipe, whose parameters are named")
        print("                  by global spec index. Loading this there fails with every key")
        print("                  Missing and every key Unexpected.")
        print("  UCP           : atoms land under language_model.* and can only be loaded back")
        print("                  by another GPTModel run. TP and EP still reshard; PP does not.")
        print("=" * 86)
    print()


    with open(os.path.join(args.hf_model_path, "config.json")) as f:
        cfg = json.load(f)

    tok_vocab = args.tokenizer_vocab_size or detect_tokenizer_vocab_size(args.hf_model_path)
    if tok_vocab is None:
        raise SystemExit(
            "Could not determine the tokenizer vocab size (no readable tokenizer.json). "
            "Pass --tokenizer-vocab-size explicitly: it must equal what the training run's "
            "tokenizer reports, because Megatron sizes the embedding from that."
        )
    padded = vocab_size_with_padding(tok_vocab, args.make_vocab_size_divisible_by, args.target_tp)
    if padded < tok_vocab:
        raise SystemExit(f"padded vocab {padded} < tokenizer vocab {tok_vocab}")

    plan = Plan(cfg, padded_vocab=padded)
    meta = read_safetensors_headers(args.hf_model_path)

    print(f"HF model      : {args.hf_model_path}")
    print(f"  arch        : {cfg.get('architectures')}  vocab={plan.vocab}  layers={plan.layers}")
    print(f"  hidden={plan.h} heads={plan.heads} head_dim={plan.head_dim}")
    print(f"  routed={plan.n_routed} shared={plan.n_shared} topk={cfg.get('num_experts_per_tok')}")
    print(f"  moe_inter={plan.moe_inter} dense_inter={plan.dense_inter} "
          f"first_k_dense={plan.first_k_dense}")
    print(f"  HF tensors  : {len(meta)}")
    print()

    print(f"Vocab reconciliation:")
    print(f"  config.vocab_size        : {plan.vocab}   (rows shipped by HF)")
    print(f"  tokenizer vocab          : {tok_vocab}   (what the run's tokenizer reports)")
    print(f"  -> Megatron padded_vocab : {padded}   "
          f"(round up to {args.make_vocab_size_divisible_by} x tp={args.target_tp})")
    if plan.vocab > padded:
        # Trimming is only safe if every dropped row sits above the highest real token id.
        if padded < tok_vocab:
            raise SystemExit(f"  REFUSING: trim to {padded} would drop real tokens (< {tok_vocab}).")
        print(f"  action                   : TRIM {plan.vocab} -> {padded} "
              f"(drops {plan.vocab - padded} unused rows above token id {tok_vocab - 1})")
    elif plan.vocab < padded:
        print(f"  action                   : ZERO-PAD {plan.vocab} -> {padded}")
    else:
        print(f"  action                   : none (already aligned)")
    print()

    non_expert = plan.non_expert_ops()
    expert_files = plan.expert_files()

    # ---- validate every source key exists and every emitted shape is what we expect
    missing, bad = [], []
    all_ops = [(MP_RANK_FILE, o) for o in non_expert]
    for fn, ops in expert_files.items():
        all_ops += [(fn, o) for o in ops]

    used = set()
    for fn, (dst, op, srcs) in all_ops:
        absent = [s for s in srcs if s not in meta]
        if absent:
            missing += absent
            continue
        used.update(srcs)
        got = plan.expected_shape(dst, op, srcs, meta)
        # sanity: fused rows must equal the sum of their parts
        if op == "fuse_gate_up":
            g, u = meta[srcs[0]][0], meta[srcs[1]][0]
            if g[1] != u[1]:
                bad.append(f"{dst}: gate/up input dims differ {g} vs {u}")
        if op == "fuse_qkv":
            for s in srcs:
                if meta[s][0] != (plan.heads * plan.head_dim, plan.h):
                    bad.append(f"{dst}: {s} has shape {meta[s][0]}, "
                               f"expected {(plan.heads * plan.head_dim, plan.h)}")
        _ = got

    unused = sorted(set(meta) - used)

    print(f"Planned outputs:")
    print(f"  {MP_RANK_FILE:<48} {len(non_expert):>5} tensors")
    print(f"  layer_*_expert_*_mp_rank_00_model_states.pt      {len(expert_files):>5} files"
          f"  ({sum(len(v) for v in expert_files.values())} tensors)")
    print(f"  total emitted tensors                            "
          f"{len(non_expert) + sum(len(v) for v in expert_files.values()):>5}")
    print()
    print(f"Coverage:")
    print(f"  HF tensors consumed : {len(used)} / {len(meta)}")
    print(f"  missing sources     : {len(set(missing))}")
    print(f"  shape problems      : {len(bad)}")
    if missing:
        print("  MISSING (first 10):")
        for m in sorted(set(missing))[:10]:
            print(f"    {m}")
    if bad:
        print("  SHAPE PROBLEMS (first 10):")
        for b in bad[:10]:
            print(f"    {b}")
    if unused:
        print(f"  UNCONSUMED HF tensors ({len(unused)}) -- these would be silently dropped:")
        for u in unused[:10]:
            print(f"    {u}  {meta[u][0]}")

    ok = not missing and not bad and not unused
    print()
    print("MAPPING COMPLETE - every HF tensor is accounted for." if ok
          else "MAPPING INCOMPLETE - fix the above before converting.")

    # ---- example of the index arithmetic, since it is the easiest thing to get wrong
    if plan.first_k_dense:
        L = plan.first_k_dense
        print(f"\nIndex check: transformer layer {L} (first MoE layer) -> "
              f"file {expert_ckpt_name(plan.moe_ordinal(L), 0)}")
        print(f"             but its key references layers.{L}  "
              f"(file index = MoE ordinal {plan.moe_ordinal(L)})")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0 if ok else 1

    if not ok:
        print("\nRefusing to convert with an incomplete mapping.")
        return 1

    # ------------------------------------------------------------------ real conversion
    import torch
    from safetensors import safe_open

    handles = {}
    for _, _, fname in meta.values():
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(args.hf_model_path, fname),
                                       framework="pt", device="cpu")

    def get(key):
        return handles[meta[key][2]].get_tensor(key)

    # ---- universal: one directory per parameter, loadable at any TP/EP/PP ----
    if args.emit in ("universal", "both"):
        uni_tag = f"{args.tag}_universal"
        uni_dir = os.path.join(args.output_folder, uni_tag)
        os.makedirs(uni_dir, exist_ok=True)
        # Every parameter, in one flat list: the per-file grouping the DeepSpeed layout needs
        # is exactly what the universal layout does away with.
        all_ops = list(non_expert) + [op for ops in expert_files.values() for op in ops]
        dtype = torch.float32 if args.atom_dtype == "fp32" else torch.bfloat16

        # Store the vocabulary at the TOKENIZER's size, not at any run's padded size.
        # Megatron pads to make_vocab_size_divisible_by * TP, so the padded height is a
        # property of the topology: 100096 at TP=1, 100352 at TP=4. The loader pads up from
        # what is stored and asserts target >= stored, so storing a padded height would make
        # the checkpoint unloadable at any SMALLER padding -- exactly the portability this
        # format exists to provide. HF ships 102400 rows for a 100015-token tokenizer; the
        # rows above the tokenizer's highest id are unused and are dropped here, which is the
        # same thing ds_to_universal does with ORIGINAL_VOCAB_SIZE.
        padded_for_ds, plan.padded_vocab = plan.padded_vocab, tok_vocab
        print(f"  vocabulary stored unpadded at {tok_vocab} rows "
              f"(a run pads it to its own multiple of {args.make_vocab_size_divisible_by} x TP)")
        print(f"\nWriting universal checkpoint to {uni_dir}  ({len(all_ops)} atoms, {args.atom_dtype})")
        write_universal(uni_dir, all_ops, get, plan, ds_version_string(), dtype)
        write_universal_model_states(
            uni_dir, plan, sum(1 for L in range(plan.layers) if plan.is_moe_layer(L)))
        plan.padded_vocab = padded_for_ds
        with open(os.path.join(args.output_folder, "latest_universal"), "w") as f:
            f.write(uni_tag)
        print(f"  latest_universal -> {uni_tag}")
        if args.emit == "universal":
            print(f"\nDone. Launch at ANY parallelism with:")
            print(f"  --load {args.output_folder} --universal-checkpoint --finetune "
                  f"--no-load-optim --no-load-rng")
            return 0

    out_dir = os.path.join(args.output_folder, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nWriting to {out_dir}")

    module_sd = {}
    for dst, op, srcs in non_expert:
        module_sd[dst] = apply_op(op, [get(s) for s in srcs], plan)

    # One entry per MoE layer (engine.py builds self.num_experts by appending each MoE
    # module's num_experts). DeepSpeed's load only checks isinstance(list) on this to pick
    # the NEW expert-file layout (layer_L_expert_E...) over the OLD (expert_E...) -- which is
    # what this converter writes -- and otherwise loads from the model's own self.num_experts.
    # So the value must be a LIST; the counts just mirror a real save.
    num_moe_layers = sum(1 for L in range(plan.layers) if plan.is_moe_layer(L))
    # GPTModelPipe takes a FLAT state dict keyed by global spec index; GPTModel takes the NESTED
    # {embedding, encoder, output_layer} form that nest_module() builds. Getting this wrong yields
    # a load_state_dict error listing every key as Missing AND every key as Unexpected.
    if args.target_model == "gptmodelpipe":
        module_sd = {to_pipe_name(k, plan.layers, untied=True): v for k, v in module_sd.items()}
        module_payload = module_sd                       # flat
    else:
        module_payload = nest_module(module_sd)          # nested

    state = {
        "module": module_payload,
        "num_experts": [plan.n_routed] * num_moe_layers,
        # 2.0, NOT 3.0. The --finetune (load_module_only) path routes through DeepSpeed's
        # Megatron SD loader (state_dict_factory.split_query_key_value), which only supports
        # ckpt_ver in {0, 1.0, 2.0} and asserts on 3.0. 2.0's QKV layout is [(np*3*hn), h] --
        # exactly the per-head-interleaved fusion this converter emits (note 2) -- and at TP=1
        # (num_to_split=1) the split is a passthrough, so nothing is re-permuted. (Megatron's
        # own fix_query_key_value_ordering only reorders for ver < 2.0, and is not in the
        # DeepSpeed load path anyway.) Megatron's normal save writes 3.0, but that path is
        # load_module_state_dict, which never calls split_query_key_value.
        "checkpoint_version": 2.0,
        "iteration": 0,
        "lr_scheduler": None,
        "data_sampler": None,
        "random_ltd": None,
        "sparse_tensor_module_names": None,
        "skipped_steps": 0,
        "global_steps": 0,
        "global_samples": 0,
        "dp_world_size": 1,
        "mp_world_size": 1,
    }
    torch.save(state, os.path.join(out_dir, MP_RANK_FILE))
    print(f"  {MP_RANK_FILE}  ({len(module_sd)} tensors)")

    for i, (fname, ops) in enumerate(sorted(expert_files.items()), 1):
        # Expert files are the RAW state dict -- no 'module' wrapper (engine.py reads
        # expert_state_dict.keys() directly).
        esd = {dst: apply_op(op, [get(s) for s in srcs], plan) for dst, op, srcs in ops}
        if args.target_model == "gptmodelpipe":
            esd = {to_pipe_name(k, plan.layers, untied=True): v for k, v in esd.items()}
        torch.save(esd, os.path.join(out_dir, fname))
        if i % 200 == 0 or i == len(expert_files):
            print(f"  expert files: {i}/{len(expert_files)}")

    with open(os.path.join(args.output_folder, "latest"), "w") as f:
        f.write(args.tag)
    with open(os.path.join(args.output_folder, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("0")

    print(f"\nDone. Launch with:")
    print(f"  --load {args.output_folder} --finetune --no-load-optim --no-load-rng")
    print(f"  (model_registry entry DeepSeek_16B supplies the matching ARCH_FLAGS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
