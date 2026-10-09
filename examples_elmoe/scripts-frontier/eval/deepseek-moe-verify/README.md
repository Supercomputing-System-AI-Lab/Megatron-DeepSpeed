# deepseek-moe-verify — execution-path verification

Purpose: prove the framework's forward for DeepSeek-MoE-16B matches HF's, component by
component, rather than trusting eval scores. Wherever HF and the framework first disagree
localizes the bug.

## Status of the 5 verification items

| # | Item | How verified | Status |
|---|------|--------------|--------|
| 1 | dense vs MoE layers mapped to right weights | `[consolidated_load] 1 missing, 0 unexpected` after `--expert-interval 1` (the 1 is the RoPE `inv_freq` buffer) | VERIFIED |
| 2 | each weight -> right expert/qkv/component | `verify_conversion.py` 3654/3654 exact vs HF + clean load report | VERIFIED (weights) |
| 3 | attention (MHA) execution | `compare.py` per-layer hidden-state diff (this suite) | run it |
| 4 | MoE execution (router/gate*expert/swiglu/shared) | `compare.py` + per-layer diff isolates the MoE sublayer | run it |
| 5 | output head + tokenizer | logit compare (this suite); eval tokenizer = HFTokenizer padded 100096 | tokenizer VERIFIED; logits: run it |

Items 1, 2, 5.3 are already settled. This suite targets 3, 4, 5.1/5.2 — the *live forward*.

## The idea

Feed the SAME token ids to HF DeepSeek and to the framework model, capture the hidden state
after every transformer layer plus the final logits, and diff them.

- If all per-layer hidden states match (cosine ~1, max|d| tiny) -> attention + MoE + swiglu +
  shared expert + output are all correct; any remaining eval gap is data (BOS) or tokenization.
- If they match up to layer L then diverge at L+1 -> the bug is in layer L+1. Since layer 0 is
  dense and 1..27 are MoE, the FIRST diverging layer tells you attention vs MoE, and whether
  it's the dense layer or an MoE layer.

## Run order (GPU node, same env as the eval)

1. HF reference activations (single GPU):
     srun -N1 -n1 --gpus-per-node=8 python hf_dump.py \
        --hf-model-path ../../../../models/deepseek-moe-16b-base \
        --out hf_acts.pt --with-bos

2. Framework activations (SAME 8-rank srun as the eval, so EP=8 experts exist). Copy the
   eval's srun line but run fw_dump.py with the eval's megatron args + --dump-out:
     srun ... python deepseek-moe-verify/fw_dump.py <same megatron_options as eval> \
        --dump-out fw_acts.pt
   (Easiest: temporarily point the eval SLURM template's python invocation at fw_dump.py, or
    copy the exact `python ... evaluate_v04.py ...` line from the eval slurm and swap the script
    + drop the eval-only flags, appending --dump-out fw_acts.pt.)

3. Compare:
     python compare.py --hf hf_acts.pt --fw fw_acts.pt

`compare.py` prints, per layer: cosine similarity and max abs diff of the hidden state, and at
the end the logit top-1 agreement. The first layer that drops below ~0.99 cosine is the culprit.

## Notes
- Use IDENTICAL ids in both (the scripts default to the same fixed prompt with BOS=100000).
- Keep the sequence SHORT (<= a few hundred tokens) so no MoE capacity overflow and a fast run.
- bf16 + different kernels mean tiny diffs are expected; a real bug shows as cosine << 1 or a
  whole-layer sign/scale break, not 1e-3 noise.
