# Thin entrypoint for the Megatron-DeepSpeed lm-eval harness (lm-eval >= 0.4).
#
# The implementation lives in the megatron_ds_eval package next to this file:
#   megatron_ds_eval/loader.py           checkpoint load + Megatron/DeepSpeed setup
#   megatron_ds_eval/lm_eval_adapter.py  MegatronDeepSpeedLM -- the lm-eval <-> engine bridge
#   megatron_ds_eval/probes.py           deepseek-moe-verify dump/compare tooling
#   megatron_ds_eval/main.py             CLI + per-task eval loop + results/registry writing
#
# This file stays as the launch target so existing SLURM templates and
# MON_PROC_MATCH=evaluate_v04.py keep working:
#
#   python tasks/eval_harness/evaluate_v04.py \
#       --load <checkpoint parent dir> \
#       --task_list wikitext,lambada_openai,piqa \
#       --results_path results.json \
#       [standard Megatron flags]

import os
import sys

# Repo root on sys.path for `import megatron` / `import pretrain_gpt`. sys.path[0] is this
# file's directory (tasks/eval_harness), which is what makes `import megatron_ds_eval` and the
# sibling `import result_registry` resolve.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

# Re-exports for existing callers: deepseek-moe-verify/fw_dump.py imports the load path from
# here, and XMoELM is the adapter class's pre-refactor name.
from megatron_ds_eval.loader import load_ds_checkpoint_and_setup_megatron  # noqa: F401
from megatron_ds_eval.lm_eval_adapter import MegatronDeepSpeedLM, XMoELM   # noqa: F401
from megatron_ds_eval.main import main, tasks_args                         # noqa: F401

if __name__ == '__main__':
    main()
