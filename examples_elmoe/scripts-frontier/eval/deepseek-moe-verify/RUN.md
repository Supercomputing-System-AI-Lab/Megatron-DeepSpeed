# Forward verification: HF (transformers) vs our framework, per-layer

Goal: forward the SAME text / input_ids through HuggingFace `deepseek-moe-16b-base` and through our
converted-checkpoint framework forward, and see EXACTLY where the two diverge — before spending a
job on wikitext. The framework side runs THROUGH the eval (reusing its exact env + load path), so
there's nothing to reconstruct.

## Recommended: one-shot inline compare (`DUMP_COMPARE_REF`)

### Step 1 — HF reference (single GPU, once per probe text)
    cd deepseek-moe-verify
    srun -N1 -n1 --gpus-per-node=8 python hf_dump.py \
      --hf-model-path ../../../../../../models/deepseek-moe-16b-base \
      --out hf_acts.pt --with-bos
      # optional: --prompt "your sentence here"   (default is a short factual sentence)
This writes `hf_acts.pt` = {ids, hidden:[emb,L0,...], logits}. You do NOT need to copy the ids —
the framework reads them straight out of this file so both sides forward identical tokens.

### Step 2 — framework forward + inline compare (one eval job, exits before wikitext)
    cd ..
    export DUMP_COMPARE_REF=$PWD/deepseek-moe-verify/hf_acts.pt
    bash autorun_eval.sh
    unset DUMP_COMPARE_REF        # so later real evals run normally

Read `logs/job_<JOBID>_step0_X-MOE_DeepSeek_16B/rank_0.log`. It prints, per aligned layer, the
cosine + max|diff| of the hidden state, then the final-logit cosine + next-token top-1 match, then a
VERDICT naming the FIRST layer that diverges (`emb`/`L0` => dense-MLP or attention; `L1..L27` =>
attention or MoE router/experts/swiglu/shared). All-match => the forward is correct and any eval gap
is BOS/data. The job does NOT run wikitext when this flag is set.

To ALSO keep the framework tensors for offline poking, add:
    export DUMP_ACTIVATIONS=$PWD/deepseek-moe-verify/fw_acts.pt   # saved alongside the inline report

## Fallback: explicit 3-step (dump both sides to files, compare offline)

Use only if you want the raw `fw_acts.pt` and a standalone `compare.py` run.
    # step 1: hf_dump.py as above (note the printed "input ids (...)" list)
    export DUMP_ACTIVATIONS=$PWD/deepseek-moe-verify/fw_acts.pt
    export DUMP_IDS=100000,549,6077,280,7239,317,...    # the ids hf_dump printed, COMMAS, no spaces
    bash autorun_eval.sh
    unset DUMP_ACTIVATIONS DUMP_IDS
    cd deepseek-moe-verify && python compare.py --hf hf_acts.pt --fw fw_acts.pt

Both paths use the SAME per-layer cosine/first-divergence logic; the recommended path just prints it
inside the one job (and pulls ids from the ref so the two forwards can't drift apart).
