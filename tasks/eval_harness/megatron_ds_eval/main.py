"""CLI + orchestration for the Megatron-DeepSpeed lm-eval harness.

Argument registration (tasks_args), the per-task simple_evaluate loop with partial
results.json saves, the response-cache path derivation, and registry row writing.
Launched through the tasks/eval_harness/evaluate_v04.py shim.
"""
import json
import codecs
import os
import sys
import time

import torch

import os as _os
# [2026-09-12] the datasets are prefetched into eval/cache by design (prefetch_datasets.py); a compute node must never ask the
# hub: 16 replicas resolving MMLU's 57 configs through the hub API were rate-limited (HTTP 429, six shards of job 5474995 lost).
# Offline unless the caller says otherwise.
_os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
_os.environ.setdefault('HF_HUB_OFFLINE', '1')
# One HF metrics cache per rank. Every rank runs lm-eval's process_results, and the HF evaluate library keeps one arrow
# cache file per experiment id under HF_HOME/metrics with a file lock: the 64 ranks of the 671B HumanEval run collided
# on it (job 5476520, rank 0: "another evaluation module instance is already using the local cache file", rank 11:
# FileNotFoundError on the file another rank had removed) after every completion was generated; smaller runs had won
# the race. evaluate reads HF_METRICS_CACHE at import, so it is set here, before anything imports it.
_os.environ.setdefault('HF_METRICS_CACHE', _os.path.join(
    _os.environ.get('HF_HOME', _os.path.expanduser('~/.cache/huggingface')), 'metrics',
    'rank' + (_os.environ.get('RANK') or _os.environ.get('SLURM_PROCID') or '0')))
from lm_eval import simple_evaluate

# --- host/GPU memory sampler (diagnosis of the 131k host oom-kill, 2026-08-30) ---
# One rank was OOM-KILLED by the kernel at seq 131072 with NO python traceback
# (slurm.err: "Detected 1 oom_kill event", job 5383151), so the spike must be
# caught in the act. Daemon thread; prints VmRSS + torch GPU allocated every 5 s.
# main.py is deliberately OUTSIDE FORWARD_SOURCES: this cannot rotate code_fp.
def _rss_sampler():
    import threading, time
    def read_rss():
        try:
            for line in open('/proc/self/status'):
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) // 1024   # MiB
        except Exception:
            return -1
        return -1
    def loop():
        import torch
        while True:
            try:
                g = torch.cuda.memory_allocated() // (1 << 20) if torch.cuda.is_available() else -1
                print(f'>>> [memwatch] rss={read_rss()} MiB gpu_alloc={g} MiB', flush=True)
            except Exception:
                pass
            time.sleep(5)
    t = threading.Thread(target=loop, daemon=True); t.start()
_rss_sampler()

from megatron import get_args, get_tokenizer, print_rank_0

from .cache import ResponseCache
from .dispatch import DispatchLM, DispatchClient, DEFAULT_BLOCK
from .loader import load_ds_checkpoint_and_setup_megatron
from .lm_eval_adapter import MegatronDeepSpeedLM
from .probes import _dump_activations_and_exit

# This file sits at tasks/eval_harness/megatron_ds_eval/main.py -- repo root is three dirnames up.
# result_registry.infer_repo_root() assumes a file directly in tasks/eval_harness/ (one level
# shallower), so anchoring on our own path here is the exact form of the same computation.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Provenance + durable results registry (pure stdlib, no Megatron/torch deps). Imported
# defensively: bookkeeping must never be able to break a multi-hour eval.
try:
    import result_registry as _result_registry
except ImportError:                       # package parent (tasks/eval_harness) not on sys.path
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import result_registry as _result_registry
    except Exception:
        _result_registry = None


class _DispatchPeer:
    """A non-leader rank's stand-in for the coordinator socket. Deliberately inert.

    It exists so that all 8 ranks of a worker take the SAME branch of DispatchLM._run and
    therefore call broadcast_object_list the same number of times. A rank that took the
    static branch while its siblings pulled blocks would call the adapter with a different
    sublist, which means a different miss set, which means a hung MoE all-to-all with no
    error message -- the failure cache.py:244-247 already calls a correctness step rather
    than an optimization.
    """
    lost = None

    def ready(self, *_a, **_k):                 # never reached: is_leader gates it
        return ('local', 'peer rank has no control plane')

    def done(self, *_a, **_k):
        return False

    def failed(self, *_a, **_k):
        return False

    def close(self):
        return None


def _dp_gather(payload):
    """Every rank's (indices, values) of its own group's blocks, gathered over the whole job (the data-parallel prompt split):
    after the call every rank holds every group's answers, so no position is left a placeholder and no collector is needed.
    Called by ALL ranks, unconditionally, once per method call."""
    if not torch.distributed.is_initialized():
        return [payload]
    box = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(box, payload)
    return box


def _dispatch_broadcast(obj):
    """The leader's assignment on every rank of this worker. Called by ALL of them.

    This is the existing pattern, not a new one: cache.py:244-249 already broadcasts the
    whole answer dict (4.5 MB) through the same call for the same reason. Here the payload
    is a two-element list holding a string and an integer.
    """
    if not torch.distributed.is_initialized():
        return obj
    box = [obj]
    torch.distributed.broadcast_object_list(box, src=0)
    return box[0]


def tasks_args(parser):
    group = parser.add_argument_group(title='lm-eval-harness 0.4+ options')
    group.add_argument('--task_list', type=str, default='wikitext',
                       help='Comma-separated lm-eval task names (0.4+ spelling: '
                            'wikitext, lambada_openai, piqa, hellaswag, ...).')
    group.add_argument('--results_path', type=str, default='./results_v04.json',
                       help='Output JSON path.')
    group.add_argument('--adaptive_seq_len', default=False, action='store_true',
                       help='Bucket sequence lengths for faster scoring.')
    group.add_argument('--num_fewshot', type=int, default=0)
    group.add_argument('--eval_fp32', default=False, action='store_true')
    group.add_argument('--no-optimizer', dest='no_optimizer', default=False, action='store_true',
                       help='Build the DeepSpeed engine without any optimizer (no fp32 master copies): '
                            'the weights alone on the GPU. For a model whose per-rank parameters '
                            'would not fit beside their masters (the released DeepSeek-V3 on 64 GCDs).')
    group.add_argument('--limit', type=int, default=None,
                       help='Cap examples per task (debug aid).')
    group.add_argument('--bootstrap_iters', type=int, default=100000,
                       help='Bootstrap iters for stderr (0 to skip).')
    group.add_argument('--consolidated_load', type=str, default=None,
                       help='Path to a consolidated flat state_dict .pt produced by '
                            'eval/xmoe_consolidate.py. When set, bypasses DeepSpeed '
                            'engine load_checkpoint() and loads directly into '
                            'model.module via load_state_dict(strict=False), with a '
                            'global->local expert index remap for this rank\'s EP slice.')
    group.add_argument('--correctness_tests', default=False, action='store_true',
                       help='Run runtime correctness probes (~30s) after model load '
                            'and before simple_evaluate. Fast fail-fast suite that '
                            'catches pipeline bugs (NaN, mask leaks, position drift, '
                            'tokenization mismatch) BEFORE the multi-hour eval.')
    group.add_argument('--no-correctness-tests', dest='no_correctness_tests', default=False, action='store_true',
                       help='Skip the runtime correctness probes even when --correctness_tests is on the line '
                            '(the launch template always passes it): for a model on which the probes cost a '
                            'large share of the wall and their thresholds do not apply (the released DeepSeek-V3 '
                            'on 64 GCDs: 29 minutes of a 2-hour job, five probes flagged at magnitudes inside the '
                            "model's own kernel-to-kernel floor). The registry rows carry probes=None (not run).")
    group.add_argument('--dump-activations', dest='dump_activations', type=str, default=None,
                       help='deepseek-moe-verify: forward a fixed token sequence, dump per-layer '
                            'hidden states + logits to this .pt (rank 0), then exit BEFORE lm-eval. '
                            'Pair with --dump-ids to match hf_dump.py exactly, then compare.py.')
    group.add_argument('--dump-ids', dest='dump_ids', type=str, default=None,
                       help='Comma-separated token ids for --dump-activations (must equal hf_dump ids).')
    group.add_argument('--dump-compare-ref', dest='dump_compare_ref', type=str, default=None,
                       help='deepseek-moe-verify: path to an HF reference .pt (from hf_dump.py) '
                            'holding {ids, hidden:[emb,L0,...], logits}. Forward the SAME ids '
                            'through this framework, print a per-layer HF-vs-framework cosine / '
                            'max|diff| divergence report + next-token logit agreement (rank 0), '
                            'then exit BEFORE lm-eval. If --dump-ids is omitted the ids are taken '
                            'from this ref, so HF and the framework forward identical inputs. '
                            'Combine with --dump-activations to also save the framework tensors.')
    group.add_argument('--dump-pad', dest='dump_pad', type=int, default=0,
                       help='deepseek-moe-verify: with --dump-compare-ref, forward the ref ids '
                            'RIGHT-PADDED to this length (e.g. 4096) using the eod pad token, then '
                            'compare only the original content positions against the (unpadded) HF '
                            'ref. Reproduces the real eval regime (padded-to-seq_length) so a bug '
                            'that only shows under padding/varlen — the causal_mask_sanity failure — '
                            'is isolated. 0 (default) = forward unpadded at the content length.')
    group.add_argument('--dump-pad-id', dest='dump_pad_id', type=int, default=0,
                       help='Pad token id used by --dump-pad. Default 0 = EXACTLY what the eval '
                            'right-pad path fills with (torch.zeros in _build_padded_batch). Change '
                            'only to probe a different pad token.')
    group.add_argument('--dump-sublayers', dest='dump_sublayers', type=str, default=None,
                       help='deepseek-moe-verify: comma layer indices (e.g. "0,1") to capture '
                            'OP-LEVEL tensors (ln1, q, k, v, q_rot, k_rot, ctx, attn_out, ln2, '
                            'router, mlp_out) during the dump forward, for op-by-op HF-vs-framework '
                            'localization. Saves to --dump-sublayers-out. Uses --dump-ids for input.')
    group.add_argument('--dump-sublayers-out', dest='dump_sublayers_out', type=str, default=None,
                       help='Output .pt path for --dump-sublayers (default: <dump_activations>.sub.pt '
                            'or ./fw_sublayers.pt).')
    group.add_argument('--padding-side', '--padding_side', dest='padding_side',
                       choices=['left', 'right'], default='right',
                       help="Padding direction for loglikelihood scoring. "
                            "'right' (default): zeros at the END of each row, "
                            "matches evaluate.py and is correct under standard "
                            "Megatron + Flash with no source patches. 'left': "
                            "zeros at the START of each row, requires "
                            "--use-left-pad-eval to activate the "
                            "position_id-aware rotary and FlashSelfAttention "
                            "varlen strip/restitch.")
    group.add_argument('--lm-eval-response-cache', '--lm_eval_response_cache',
                       dest='lm_eval_response_cache', default=False,
                       action='store_true',
                       help="Enable the adapter's resumable per-request response cache "
                            "(megatron_ds_eval/cache.py): append-only JSONL under "
                            "eval/cache/responses/, flushed after every computed batch, "
                            "keyed on checkpoint identity + code_fp + eval config. A job "
                            "killed at the walltime loses at most one batch; a re-run "
                            "answers cached requests from disk and computes only the "
                            "rest. Replaces lm-eval's use_cache (which only committed "
                            "after a full method call). Disabled by default.")
    group.add_argument('--task-metadata-json', '--task_metadata_json',
                       dest='task_metadata_json', type=str, default=None,
                       help='JSON dict passed to lm_eval simple_evaluate(metadata=...). '
                            'Flows into TaskManager -> ConfigurableTask custom datasets; '
                            'RULER reads {"tokenizer": <HF dir>, "max_seq_lengths": [N]} '
                            'from it (lm_eval evaluator.py:292 -> ruler/common_utils.py).')
    group.add_argument('--no-kv-decode', '--no_kv_decode', dest='no_kv_decode',
                       default=False, action='store_true',
                       help='Force generate_until onto the legacy full-reforward loop. '
                            'KV-cached decode (InferenceParams) is the default on '
                            'the no-pipeline path; this flag is the '
                            'equivalence oracle for validating it.')
    group.add_argument('--kv-per-row', '--kv_per_row', dest='kv_per_row',
                       default=False, action='store_true',
                       help='Use the v1 per-row (B=1) KV decode instead of the default '
                            'batched ragged decode (Phase 5b). Reference path for '
                            'validating batching; ~Bx slower.')
    group.add_argument('--task-metadata-json-file', '--task_metadata_json_file',
                       dest='task_metadata_json_file', type=str, default=None,
                       help='Like --task-metadata-json but read from a file -- inline JSON '
                            'does not survive the SLURM template\'s shell parse layers '
                            '(quote-stripping + brace expansion), a file path does.')
    group.add_argument('--allow-code-eval', '--allow_code_eval', dest='allow_code_eval',
                       default=False, action='store_true',
                       help='Enable pass@1 code-execution tasks (humaneval/mbpp): sets '
                            'HF_ALLOW_CODE_EVAL=1 and simple_evaluate('
                            'confirm_run_unsafe_code=True). Generated code runs as the '
                            'job user; Frontier compute nodes have no outbound network. '
                            'Per-row opt-in via the task table, never a global default.')
    group.add_argument('--max-gen-toks', '--max_gen_toks', dest='max_gen_toks',
                       type=int, default=None,
                       help='Generation budget from the task table, pinned over the '
                            'task YAML via simple_evaluate(gen_kwargs=...). The table '
                            'is what check_seqlens.py measures against, so the two must '
                            'agree or prompts get silently truncated (humaneval 2026-08-16).')
    group.add_argument('--extra-until', dest='extra_until', action='append', default=[],
                       help='A stop string appended to the `until` list of every generation request (repeatable; '
                            'python escapes such as \\n are decoded). The task YAML keeps its own stops. For a model '
                            'that closes a completion with a markdown fence the harness does not stop at: '
                            "--extra-until '```' (the bigcode harness stops at the fence; lm-eval's humaneval does not).")
    group.add_argument('--include-path', '--include_path', dest='include_path',
                       type=str, default=None,
                       help='Directory of custom lm-eval task YAMLs (TaskManager '
                            'include_path). Used for tasks we define ourselves, e.g. '
                            'alpaca_eval_gen (chat-generation transport whose judging '
                            'happens offline later).')
    group.add_argument('--task-samples-json-file', '--task_samples_json_file',
                       dest='task_samples_json_file', type=str, default=None,
                       help='JSON manifest of FIXED doc indices per (sub)task '
                            '(config/subsets/*.json from make_subsets.py) passed to '
                            'simple_evaluate(samples=...). The fast-cadence subsets: '
                            'seeded stratified random draws, committed for '
                            'auditability. Overrides --limit (a warning is printed); '
                            'the registry row records limit = total sampled docs.')
    group.add_argument('--experiment', '--answers-ns', '--answers_ns',
                       dest='experiment', type=str, default=None,
                       help='Mark this run as EXPERIMENT <tag>: a measurement made to '
                            'compare, never to publish. Two effects, and they are one '
                            'decision: (a) the answer file gets a private `_ns<tag>` '
                            'suffix (megatron_ds_eval/cache.py) so the run is forced '
                            'cold without touching a production answer file -- the '
                            'parallel shape is not in the answer-file identity, so a '
                            '4-node run otherwise reads the 1-node answers as hits and '
                            'a shape comparison on warm cache is vacuous; and (b) every '
                            'registry row this job writes carries experiment=<tag>, '
                            'which every consumer excludes from published values '
                            '(result_registry.experiment_of). Unset (the default) is a '
                            'PRODUCTION run: existing filenames stay byte for byte and '
                            'the rows define the published cell. --answers-ns is the '
                            'deprecated spelling of this flag.')
    group.add_argument('--shard-index', '--shard_index', dest='shard_index',
                       type=int, default=None,
                       help='REQUEST-LEVEL WORK SPLIT: this worker is shard '
                            '<shard-index> (0-based) of --shard-count. The worker '
                            'builds the WHOLE benchmark exactly as an unsliced run '
                            'does -- so every few-shot prompt is bit-identical -- and '
                            'then computes only its own slice of the materialized '
                            'request list, returning the full-length result list with '
                            'the rest answered from the shared answer file or left as '
                            'placeholders. Its registry rows are stamped '
                            "partial='shard k/N'; the collector scores. Set by "
                            'eval_lib at workers>1 split=request. See '
                            'rebuild/DISPATCH_ENGINE_DESIGN.md.')
    group.add_argument('--shard-count', '--shard_count', dest='shard_count',
                       type=int, default=None,
                       help='Number of request-level shards this benchmark is split '
                            'across. 1 or unset = no split, the code path that runs '
                            'today.')
    group.add_argument('--dp-split', '--dp_split', dest='dp_split', default=False, action='store_true',
                       help='Split the request list across the DATA-PARALLEL ATTENTION GROUPS of this model copy '
                            '(the tensor-parallel groups at pp=1): group g takes blocks g, g+G, ... of the list the '
                            'DispatchLM shard rule already cuts, the groups agree on the number of forwards of every '
                            'round (an idle group runs an empty forward), and the groups\' answers are all-gathered '
                            'after each method so the results are complete on every rank. Set by eval_lib from '
                            'dp_split=1. Ends the 8-fold replication of the 671B evaluation (2026-09-14).')
    group.add_argument('--dispatch-block', '--dispatch_block', dest='dispatch_block',
                       type=int, default=None,
                       help='Requests per dispatch block; shard k takes blocks k, '
                            'k+N, k+2N, ... Fixed by the DECLARATION (eval_lib passes '
                            '4 x micro_batch_size) and never derived from the shard '
                            'count, so block b holds the same requests whatever N is. '
                            'It is part of a result\'s identity and rides on the '
                            'registry row next to nodes/ep/tp/pp -- an A/B that '
                            'changes it must declare experiment=<tag>. Default '
                            '4 x micro_batch_size.')
    group.add_argument('--dispatch-addr', '--dispatch_addr', dest='dispatch_addr',
                       type=str, default=None,
                       help='STAGE 2: pull block indices from the coordinator at '
                            'host:port instead of computing the static set '
                            '--shard-index picks. Unset (the default) is the static '
                            'split, byte for byte the path that runs today. The '
                            'coordinator is a plain CPU python process in the batch '
                            'shell (no torch, no model); the payload is a block index. '
                            'Only RANK 0 of this worker holds the socket -- it fans each '
                            'assignment to its 7 peers with broadcast_object_list, so '
                            'all 8 ranks enter every adapter call together. If the '
                            'coordinator is unreachable or its answers stop, the worker '
                            'falls back to the static split and the job still finishes. '
                            'Set by eval_lib at workers>1 split=request dispatch=engine.')
    group.add_argument('--dispatch-id', '--dispatch_id', dest='dispatch_id',
                       type=int, default=None,
                       help='This worker\'s id on the dispatch wire. Defaults to '
                            '--shard-index. Identity only: it names the worker in the '
                            'coordinator log and journal, and nothing depends on it.')
    group.add_argument('--dispatch-heartbeat', '--dispatch_heartbeat',
                       dest='dispatch_heartbeat', type=float, default=None,
                       help='Seconds between the leader\'s heartbeat ticks (default '
                            '30). The tick is what lets the coordinator size its death '
                            'detector against SILENCE rather than against the longest '
                            'legitimate block -- the property no torch process group '
                            'can give. 0 disables it.')
    group.add_argument('--ckpt-step-dir', '--ckpt_step_dir', dest='ckpt_step_dir',
                       type=str, default=None,
                       help='The REAL global_step<N> directory being evaluated (not the '
                            '--load parent, and not the per-job .eval_ucp selector). Used '
                            'only for identity: the response-cache filename and registry '
                            'provenance. Without it, under UCP the cache falls back to '
                            'basename(--load), which carries no training-run identity.')
    return parser


def main():
    start = time.time()
    model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()
    # The loader merges the checkpoint's stored TRAINING args into the eval args
    # (loader.override_args), so a checkpoint trained with --intra-document-attention
    # arrives here with the flag ON. At eval that is wrong twice over: the scoring
    # path passes a dense bool mask where the varlen path expects cu_seqlens (bool-
    # subtraction crash on the first attention layer), and every eval prompt is a
    # single document, for which plain causal attention IS intra-doc semantics
    # (4,350-prompt EOD census, 2026-08-20: zero interior EODs). Neutralize both
    # the args flag and the per-module attribute the runtime gate actually reads.
    # main.py is deliberately outside FORWARD_SOURCES, so this does not fork the
    # response-cache identity.
    if getattr(args, 'intra_document_attention', False):
        args.intra_document_attention = False
        _n = 0
        for _m in getattr(model, 'module', model).modules():
            if getattr(_m, 'intra_document_attention', False):
                _m.intra_document_attention = False
                _n += 1
        print_rank_0(f'>>> [eval] intra_document_attention neutralized on {_n} attention '
                     f'modules (training-only flag inherited from checkpoint args)')
    if args.deepspeed and args.adaptive_seq_len:
        # Required so DeepSpeed's pipeline engine permits per-batch shape changes
        args.curriculum_learning_legacy = 1

    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    tokenizer = get_tokenizer()

    # Provenance BEFORE the adapter: the response cache is keyed on code_fp and the adapter
    # takes the cache at construction. Registry rows below reuse the same _prov.
    _prov = {'code_fp': 'nofp', 'git_sha': None, 'git_dirty': None}
    _repo_root = None
    if _result_registry is not None:
        try:
            _repo_root = REPO_ROOT
            _prov = _result_registry.provenance(_repo_root)
            _missing = _prov.get('code_fp_detail', {}).get('missing') or []
            print_rank_0(f">>> [provenance] code_fp={_prov['code_fp']} git={_prov['git_sha']}"
                         f"{' (dirty)' if _prov['git_dirty'] else ''}"
                         + (f" WARN missing sources: {_missing}" if _missing else ''))
        except Exception as _e:
            print_rank_0(f">>> [provenance] WARN: could not fingerprint code: {_e}")

    # Resumable per-request response cache. Replaces lm-eval's use_cache/CachingLM, which
    # only committed AFTER a full method call and therefore lost everything on a walltime
    # kill -- see cache.py for the full rationale and the rank protocol.
    response_cache = ResponseCache.build(
        args,
        code_fp=_prov.get('code_fp', 'nofp'),
        cache_dir=os.path.join(REPO_ROOT, 'examples_elmoe', 'scripts-frontier', 'eval',
                               'cache', 'responses'),
        print_fn=print_rank_0,
    )

    _extra_until = [codecs.decode(u, 'unicode_escape') for u in (getattr(args, 'extra_until', None) or [])]
    if _extra_until:
        print_rank_0(f'>>> [gen] extra stop strings appended to every generation request: {_extra_until!r}')
    lm = MegatronDeepSpeedLM(model, tokenizer, response_cache=response_cache, extra_until=_extra_until)

    # deepseek-moe-verify: dump per-layer activations and/or print an inline HF-vs-framework
    # divergence report for a fixed probe input, then exit BEFORE lm-eval. Triggered by either
    # --dump-activations (save framework tensors) or --dump-compare-ref (inline compare vs HF).
    if (getattr(args, 'dump_activations', None) or getattr(args, 'dump_compare_ref', None)
            or getattr(args, 'dump_sublayers', None)):
        _dump_activations_and_exit(lm, model, args)
        return

    # Optional pre-eval correctness probes. Fast fail-fast pipeline checks
    # (logits sanity, causal mask, position invariance, etc.) so we don't
    # burn a 3-hour eval on a silently-broken setup.
    # _probe_status lands in every registry row ('probes'): None = probes not
    # requested, 'passed', or 'failed:<name>,<name>'. Before this, a failing
    # probe was one WARNING line in a job log and the rows it produced were
    # indistinguishable from trustworthy ones -- five of the eight
    # baseline-capture jobs failed a probe (QA defect 1).
    _probe_status = None
    if getattr(args, 'no_correctness_tests', False) and getattr(args, 'correctness_tests', False):
        print('[correctness_tests] SKIPPED (--no-correctness-tests): the registry rows carry probes=None', flush=True)
        args.correctness_tests = False
    if getattr(args, 'correctness_tests', False):
        sys.path.insert(0, os.path.join(
            REPO_ROOT,
            'examples_elmoe', 'scripts-frontier', 'eval', 'tests'))
        try:
            import test_runtime as _runtime_tests
        except ImportError as _e:
            # HARD failure, not a warning: a job that was asked to probe and cannot
            # would otherwise run to completion writing rows whose probe status is
            # silently absent -- the silent-skip trap (this fires when tests/ is
            # moved; main.py imports it by absolute path).
            raise RuntimeError(
                f"--correctness-tests was requested but test_runtime could not be "
                f"imported: {_e}. The probe suite lives at "
                f"examples_elmoe/scripts-frontier/eval/tests/test_runtime.py and "
                f"must not move.") from _e
        # Probes run BEFORE the main eval loop's model.eval()+no_grad. If
        # we leave the model in train mode with autograd on, every probe
        # forward saves activations across all 24 layers + the MoE expert
        # MLPs for backward, which OOMs rank 6 at MBS=4 / B=4 / S=4096
        # (the chunking path pads input up to micro_batch_size, so even
        # 2 prompts → B=4). Switch to eval mode + no_grad here so each
        # probe forward is bounded by hidden-state memory only.
        model.eval()
        with torch.no_grad():
            ok, _probe_flags = _runtime_tests.run_all(lm)
        # run_all's verdict is identical on every rank: at PP>1 the probes are
        # judged on the last pipeline stage (the only one holding logits) and
        # broadcast from there, so stamping from world rank 0 -- pipeline
        # stage 0 -- stays truthful. _probe_flags carries failed probe names
        # plus 'skipped-under-pp:<name>' for any probe no rank could judge at
        # PP>1. A skip is not a failure (ok stays True) but must not read as
        # a pass either, so it rides after ';' on whatever the pass/fail
        # verdict is: 'passed;skipped-under-pp:x' / 'failed:a;skipped-...'.
        _probe_failed = [f for f in _probe_flags
                         if not f.startswith('skipped-under-pp:')]
        _probe_skipped = [f for f in _probe_flags
                          if f.startswith('skipped-under-pp:')]
        _probe_status = 'passed' if ok else 'failed:' + ','.join(_probe_failed)
        if _probe_skipped:
            _probe_status += ';' + ','.join(_probe_skipped)
        if not ok and (not torch.distributed.is_initialized()
                       or torch.distributed.get_rank() == 0):
            # Continue to eval anyway (don't sys.exit and waste the
            # warm process), but the failure is clearly logged above AND
            # stamped on every row this job records.
            print("[correctness_tests] WARNING: some tests FAILED "
                  f"({','.join(_probe_failed)}). Eval results below may not be "
                  "trustworthy; their registry rows carry "
                  f"probes={_probe_status!r}.", flush=True)

    # ---- request-level work split (Stage 1 of rebuild/DISPATCH_ENGINE_DESIGN.md) -------
    # A DOCUMENT split cannot carry a few-shot benchmark: lm-eval bakes each document's
    # few-shot prompt into its Instance at BUILD time from one rng stream consumed in
    # build order, so a worker handed a subset of the documents rebuilds every prompt
    # differently (gsm8k 8-shot: 1 of 20 requests in common with the unsliced run). A
    # REQUEST split has no such problem: every worker builds the identical full list --
    # simple_evaluate seeds deterministically and the adapter reports rank=0/world_size=1
    # on purpose -- and then computes only its own slice of it, which is bit-identical to
    # the unsliced run by construction (40 of 40, measured).
    #
    # The wrapper goes on HERE, after the probes: probes measure the model, not the
    # split, and _dump_activations_and_exit wants the adapter itself. And it is a
    # wrapper rather than an adapter edit because lm_eval_adapter.py is FORWARD_SOURCES[4]
    # -- main.py is deliberately outside that list (result_registry.py:47-48), so this
    # whole feature costs no fingerprint and orphans no cached answer.
    _dispatch = None
    _dispatch_client = None
    _dp_groups, _dp_rank = 1, 0
    if getattr(args, 'dp_split', False):
        # The data-parallel prompt split (2026-09-14): the request list is cut across the data-parallel attention groups of
        # this model copy (the tensor-parallel groups at pp=1; 8 nodes of TP 8 = 8 groups for the 671B, 8 ranks of TP 1 = 8
        # groups for a 16B on one node). The shard rule below is reused with (worker shard, group) as the shard, and the
        # groups' answers are all-gathered after each method (_dp_gather), so nothing is partial and no collector runs.
        from megatron.core import mpu as _mpu
        _dp_groups = int(_mpu.get_data_parallel_world_size())
        _dp_rank = int(_mpu.get_data_parallel_rank())
        lm.dp_split = _dp_groups > 1
        print_rank_0(f'>>> [dp-split] {_dp_groups} data-parallel attention group(s) in this model copy take interleaved '
                     f'request blocks; the forwards of every round are agreed over the job and an idle group runs an '
                     f'empty forward (rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0} '
                     f'is in group {_dp_rank})')
    if int(getattr(args, 'shard_count', None) or 1) > 1 or _dp_groups > 1:
        _block = int(getattr(args, 'dispatch_block', None) or 0) or \
            4 * int(getattr(args, 'micro_batch_size', None) or 8) or DEFAULT_BLOCK
        _shard_index = int(getattr(args, 'shard_index', None) or 0) * _dp_groups + _dp_rank
        # STAGE 2. Exactly ONE rank of this worker opens the socket -- RANK 0, which
        # cache.writer_rank(8, 1) also makes the answer-file writer, so the rank that
        # receives a block is the rank that persists its answers. The other 7 never touch
        # the control plane; they receive every assignment through the broadcast below,
        # which every rank calls unconditionally.
        _is_leader = (not torch.distributed.is_initialized()
                      or torch.distributed.get_rank() == 0)
        if getattr(args, 'dispatch_addr', None):
            if _is_leader:
                _dispatch_client = DispatchClient(
                    args.dispatch_addr,
                    worker_id=int(getattr(args, 'dispatch_id', None)
                                  if getattr(args, 'dispatch_id', None) is not None
                                  else _shard_index),
                    heartbeat=(30.0 if getattr(args, 'dispatch_heartbeat', None) is None
                               else float(args.dispatch_heartbeat)),
                    log=print_rank_0)
                _dispatch_client.connect()      # never raises: a failure means LOCAL mode
            else:
                # A peer still needs a non-None client so that every rank takes the
                # ENGINE branch of _run and therefore calls the broadcast the same number
                # of times. Its methods are never reached (is_leader gates them), but a
                # rank that silently took the static branch would desynchronise the EP
                # all-to-all, which is a hang with no error message.
                _dispatch_client = _DispatchPeer()
        _dispatch = DispatchLM(lm,
                               shard_index=_shard_index,
                               shard_count=int(getattr(args, 'shard_count', None) or 1) * _dp_groups,
                               block=_block,
                               print_fn=print_rank_0,
                               client=_dispatch_client,
                               is_leader=_is_leader,
                               broadcast_fn=(_dispatch_broadcast
                                             if getattr(args, 'dispatch_addr', None)
                                             else None),
                               gather_fn=(_dp_gather if _dp_groups > 1 else None))
        lm = _dispatch
        print_rank_0(f'>>> [dispatch] request-level split ON: shard '
                     f'{_dispatch.shard_index + 1}/{_dispatch.shard_count}, block='
                     f'{_dispatch.block}. This worker builds the WHOLE benchmark and '
                     f'computes its own slice of the request list; its rows are stamped '
                     f'partial and the collector scores from the shared answer file.')

    task_list = [t.strip() for t in args.task_list.split(',') if t.strip()]

    # Only world-rank 0 writes the JSON. With PP=1 + TP=1, every rank's
    # mpu.is_pipeline_last_stage() returns True and mpu.get_tensor_model_parallel_rank()
    # returns 0, so the old guard would let all N ranks race on the same file.
    # Gate explicitly on the real world rank.
    is_world_rank_zero = (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )

    # Per-task loop with partial-save after each task. If SLURM kills us
    # mid-task due to timeout, the results.json on disk still contains every
    # task we finished. The lm-eval `simple_evaluate` setup overhead per call
    # is small once the process is warm (no re-loading model, no re-loading
    # task registry beyond the first call).
    #
    # (Provenance and the response cache were both set up above, before the adapter was
    # constructed. The old sqlite use_cache path lived here; eval/cache/lm_eval_responses/
    # holds its dead .db files, safe to delete.)

    # ---- results registry plumbing -------------------------------------------------------
    # Rows are appended PER TASK as each one finishes, not only at the very end. An eval that
    # hits the SLURM walltime mid-suite must still keep the tasks it actually completed -- the
    # same reasoning that makes the response cache treat a timeout as resumable rather than
    # fatal. `_recorded` makes this idempotent, which also handles GROUP tasks (mmlu emits one
    # result key per subtask plus one for the group) and the final flush below re-checking.
    # Fixed-subset manifests (instruct-short): explicit doc indices replace --limit.
    _samples = None
    _slice_kn = None
    if getattr(args, 'task_samples_json_file', None):
        with open(args.task_samples_json_file) as _sf:
            _samples = json.load(_sf)
        # A SLICE manifest says so in its _meta ('slice': [k, n], written by
        # eval_lib._slice_jobs). Read it before the header is dropped: a slice job is
        # otherwise an ordinary eval job whose rows land at the SAME identity as the
        # full benchmark -- two slices of one declaration recorded conflicting values
        # at one identity (jobs 5354994 acc=0.98 vs 5355041 acc=0.96), a later
        # declaration read SKIP-done off a slice row, and report.py printed a slice's
        # partial score as the benchmark.
        _hdr = _samples.get('_meta')
        if isinstance(_hdr, dict):
            _slice_kn = _hdr.get('slice')
        _samples.pop('_meta', None)
        _n_sampled = sum(len(v) for v in _samples.values())
        if getattr(args, 'limit', None) is not None:
            print_rank_0(f'>>> [samples] WARN: --limit {args.limit} ignored; doc '
                         f'selection comes from {args.task_samples_json_file}')
            args.limit = None
        print_rank_0(f'>>> [samples] fixed subset: {_n_sampled} docs over '
                     f'{len(_samples)} (sub)task(s) from '
                     f'{os.path.basename(args.task_samples_json_file)}')

    _reg_path = None
    _run_meta = {}
    _recorded = set()
    if _result_registry is not None:
        _run_meta = {
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'model': os.environ.get('MODEL_SIZE'),
            'moe_type': os.environ.get('MOE_TYPE'),
            'step': os.environ.get('STEP'),
            'num_fewshot': getattr(args, 'num_fewshot', 0) or 0,
            'seq_len': getattr(args, 'seq_length', None),
            'padding_side': getattr(args, 'padding_side', None),
            # The parallel shape that produced this number. Without it a 4-node PP4/TP4
            # result and a 1-node result are the same row in results/registry/, so the
            # completion check credits one to the other and ACCEPTANCE Gate 2 ("measured
            # and explained") cannot be answered at all. Additive: every reader uses
            # .get(), and a row written before today carries none of these fields, which
            # eval_lib.row_shape() reads as the default 1-node EP8/TP1/PP1 shape -- what
            # all 3974 historical eval jobs actually ran at.
            'nodes': int(os.environ.get('SLURM_NNODES') or 0) or None,
            'ep': getattr(args, 'moe_expert_parallel_size', None),
            'tp': getattr(args, 'tensor_model_parallel_size', None),
            'pp': getattr(args, 'pipeline_model_parallel_size', None),
            # samples runs record their subset SIZE here: limit is the registry's
            # subset-identity field (DONE checks and table series key on it), and the
            # committed manifest pins WHICH docs that size means.
            'limit': (_n_sampled if _samples is not None
                      else getattr(args, 'limit', None)),
            'consolidated_load': getattr(args, 'consolidated_load', None),
            # Stable checkpoint identity (fills the provenance hole the UCP switch opened:
            # consolidated_load is null on the UCP path).
            'ckpt_step_dir': getattr(args, 'ckpt_step_dir', None),
            'ckpt_run': (os.path.basename(os.path.dirname(
                             os.path.normpath(args.ckpt_step_dir)))
                         if getattr(args, 'ckpt_step_dir', None) else None),
            'ckpt_mtime': (response_cache.meta.get('mtime')
                           if response_cache is not None else None),
            'weight_source': ('cons' if getattr(args, 'consolidated_load', None)
                              else 'ucp' if getattr(args, 'universal_checkpoint', False)
                              else 'ds'),
            'response_cache': (os.path.basename(response_cache.path)
                               if response_cache is not None else None),
            # EXPERIMENT IDENTITY. None -> this row may become the published value of
            # its cell; a tag -> it never can, at any consumer (the rule is
            # result_registry.experiment_of, and every reader goes through it). Before
            # this field existed an experiment run wrote a plain production row and its
            # only trace was the `_ns<tag>` suffix on `response_cache` above -- which is
            # how five published cells were silently overwritten on 2026-08-28.
            # `--answers-ns` is the deprecated spelling of `--experiment`; both land in
            # args.experiment, so the answer FILE and the registry ROW can never
            # disagree about which experiment this is.
            'experiment': getattr(args, 'experiment', None) or None,
            'results_path': args.results_path,
            'code_fp': _prov.get('code_fp'),
            'git_sha': _prov.get('git_sha'),
            'git_dirty': _prov.get('git_dirty'),
            # Correctness-probe verdict for THIS job: None (not requested), 'passed',
            # or 'failed:<probe>,<probe>'. Additive field; done-check semantics are
            # deliberately unchanged (whether probe-failed rows satisfy a done-check
            # is an owner decision pending WO-10's diagnosis).
            'probes': _probe_status,
        }
        if _slice_kn:
            # Slice rows are PARTIAL results at the full row's identity. `partial` is
            # the field rescore_partial.py already writes and eval_lib.rows_cover()
            # already skips, so a slice row can never satisfy a done-check or define
            # the benchmark's number -- the collecting run, which carries no slice
            # _meta, writes the full-set row. Additive: every reader uses .get().
            _run_meta['partial'] = 'slice {}/{}'.format(*_slice_kn)
            _run_meta['partial_source'] = 'slice'
        if _dispatch is not None and _dp_groups > 1:
            _run_meta['dp_split'] = _dp_groups
        if _dispatch is not None and int(getattr(args, 'shard_count', None) or 1) > 1:
            # A dispatched worker holds a SHARD of the request list, so its results.json
            # is built from its own answers plus whatever the shared answer file already
            # held plus placeholders -- a partial result at the full row's identity, the
            # same thing a slice row is. Stamping it partial is what stops it satisfying
            # a done-check (eval_lib.rows_cover skips partial rows) or being printed as
            # the benchmark (report.py excludes them). Only the COLLECTOR -- the final
            # unsliced step of the same job, carrying no shard flags -- writes the row
            # that counts. `dispatch_block` rides here next to nodes/ep/tp/pp for the
            # same reason the parallel shape does: it changes how the numbers were
            # produced without changing the forward math, so it is a row field, not part
            # of the code fingerprint and not part of the answer filename.
            _run_meta['partial'] = 'shard {}/{}'.format(_dispatch.shard_index + 1,
                                                        _dispatch.shard_count)
            _run_meta['partial_source'] = 'dispatch'
            _run_meta['dispatch_shard'] = '{}/{}'.format(_dispatch.shard_index + 1,
                                                         _dispatch.shard_count)
            _run_meta['dispatch_block'] = _dispatch.block
        # Sharded registry (Phase 3b): rows go to results/registry/<ckpt_run>.jsonl when
        # the training run is known; unknown runs fall back to the legacy single file.
        # Readers (report.py, submit_eval, render_results) union both via load_all_rows.
        _reg_path = _result_registry.shard_path(_repo_root or '.',
                                                _run_meta.get('ckpt_run'))

    def _flush_registry(results_by_task):
        """Append rows for any task not yet recorded. Never raises into the eval loop."""
        if _result_registry is None or _reg_path is None or not results_by_task:
            return
        fresh = {t: r for t, r in results_by_task.items() if t not in _recorded}
        if not fresh:
            return
        try:
            n = _result_registry.record(_reg_path, _run_meta, fresh)
            _recorded.update(fresh.keys())
            print_rank_0(f">>> [registry] +{n} rows for {sorted(fresh)} -> {_reg_path}")
        except Exception as _e:
            print_rank_0(f">>> [registry] WARN: append failed: {_e}")

    merged = None
    completed = []
    for task_idx, task_name in enumerate(task_list, start=1):
        print_rank_0(f">>> [{task_idx}/{len(task_list)}] evaluating task '{task_name}'")
        t0 = time.time()

        _kw = {}
        if getattr(args, 'task_metadata_json', None):
            _kw['metadata'] = json.loads(args.task_metadata_json)
        elif getattr(args, 'task_metadata_json_file', None):
            with open(args.task_metadata_json_file) as _mf:
                _kw['metadata'] = json.load(_mf)
        if getattr(args, 'allow_code_eval', False):
            os.environ['HF_ALLOW_CODE_EVAL'] = '1'
            _kw['confirm_run_unsafe_code'] = True
        if getattr(args, 'include_path', None):
            # An explicit task_manager suppresses simple_evaluate's own construction,
            # so metadata must ride along here too (it otherwise flows only into the
            # TaskManager that simple_evaluate builds internally).
            from lm_eval.tasks import TaskManager as _TM
            _kw['task_manager'] = _TM(include_path=args.include_path,
                                      metadata=_kw.get('metadata'))
        if _samples is not None:
            _kw['samples'] = _samples
            # A manifest may serve several tasks sharing one job; each task's registry
            # rows must record ITS OWN subset size (the DONE/table identity). Group
            # subtasks are keyed by LEAF name, which need not carry the group's prefix
            # (ruler's niah_*, longbench_e's longbench_*_e) -- prefix-matching would
            # under-count ruler (500 of 1300) and record longbench_e as None, i.e. a
            # FULL run. A single-task job owns the whole manifest by construction
            # (submit_eval groups jobs by extra_eval_flags, so rows with different
            # manifests never share one); the prefix rule only splits manifests
            # genuinely shared across a multi-task job (short_q512_fs0.json).
            # Always record THIS task's own share of the manifest. The old rule gave a
            # single-task job the manifest total on the assumption that such a job owns
            # the whole file -- true while submit_eval always grouped tasks sharing a
            # manifest, false once tasks are submitted individually: a solo hellaswag
            # run against short_q512_fs0.json (gsm8k 100 + hellaswag 100) recorded
            # limit=200 for a 100-document measurement, and every consumer that checks
            # provenance by limit then rejected a perfectly good score (2026-08-17).
            # The rule itself is result_registry.own_manifest_keys() -- ONE home for
            # every own-keys clone in the tree. Its RULER special case is why this no
            # longer records limit=500 for a 1300-document RULER: the leaves span two
            # name families (niah_*, ruler_*) and the bare prefix rule counted only the
            # ruler_* five, so a finished RULER never satisfied a done-check (2026-08-27).
            # Group tasks whose manifest keys are leaf names that do NOT carry the
            # group's prefix (longbench_e's longbench_*_e) still match nothing and keep
            # the manifest total via the fallback.
            _own = (sum(len(_samples[k]) for k in
                        _result_registry.own_manifest_keys(task_name, _samples))
                    if _result_registry is not None else 0)
            _run_meta['limit'] = _own or (_n_sampled if len(task_list) == 1 else None)
        # The task table's gen_toks is AUTHORITATIVE over the task YAML's own
        # max_gen_toks. Without this the two silently disagreed: the table sized
        # humaneval at gen_toks=512 (so check_seqlens declared 425+512 <= 1024 safe)
        # while the YAML asked for 1024, leaving zero room for the prompt -- every
        # HumanEval prompt was cut to its last token and scored 0.0 everywhere.
        # It also truncated minerva_math's OUTPUT at 256 when the table intended 512
        # (41% of generations hit that cap), depressing the math scores.
        if getattr(args, 'max_gen_toks', None):
            _kw['gen_kwargs'] = (_kw.get('gen_kwargs') + ',' if _kw.get('gen_kwargs') else '') \
                + f'max_gen_toks={int(args.max_gen_toks)}'
            print_rank_0(f'>>> [gen] max_gen_toks pinned to {args.max_gen_toks} '
                         f'from the task table (overrides the task YAML)')
        task_result = simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            bootstrap_iters=args.bootstrap_iters,
            **_kw,
        )
        dt = time.time() - t0
        print_rank_0(f">>> [{task_idx}/{len(task_list)}] task '{task_name}' done in "
                     f"{dt:.1f}s ({dt/60:.1f} min)")

        # Merge into the running results dict. Each call to simple_evaluate
        # returns a self-contained dict; we accumulate the per-task pieces.
        if merged is None:
            merged = task_result
        else:
            for top_key in ('results', 'configs', 'versions', 'n-shot',
                            'higher_is_better', 'n-samples', 'group_subtasks'):
                src = task_result.get(top_key)
                if isinstance(src, dict):
                    dst = merged.setdefault(top_key, {})
                    dst.update(src)
            # Keep the latest env/version metadata (it'll be the same across calls)
            for top_key in ('git_hash', 'date', 'transformers_version',
                            'lm_eval_version', 'upper_git_hash', 'pretty_env_info',
                            'config'):
                if top_key in task_result:
                    merged[top_key] = task_result[top_key]
        completed.append(task_name)

        # Save partial after each task. ALL ranks reach this point in lockstep
        # (DeepSpeed collective forwards keep them synchronized), but only
        # rank 0 writes — others would race.
        if is_world_rank_zero:
            saved = {k: v for k, v in merged.items() if k != 'samples'}
            saved['_completed_tasks'] = completed[:]
            saved['_in_progress'] = (task_list[task_idx:] if task_idx < len(task_list) else [])
            saved['_partial_save_time'] = time.time()
            try:
                with open(args.results_path, 'w') as f:
                    f.write(json.dumps(saved, indent=2, default=str))
                print_rank_0(f">>> [partial-save] wrote {args.results_path} "
                             f"(tasks done: {completed}; remaining: {saved['_in_progress']})")
            except Exception as _e:
                print_rank_0(f">>> [partial-save] WARN: failed to write JSON: {_e}")
            # Record this task NOW so a later walltime kill cannot lose it. Attempted even if
            # the JSON write above failed -- the two sinks are independent.
            _flush_registry(saved.get('results', {}))

    # End of loop — final dump (same content as the last partial-save).
    if is_world_rank_zero and merged is not None:
        saved = {k: v for k, v in merged.items() if k != 'samples'}
        saved['_completed_tasks'] = completed[:]
        saved['_in_progress'] = []
        saved['_partial_save_time'] = time.time()
        out = json.dumps(saved, indent=2, default=str)
        print(out)
        with open(args.results_path, 'w') as f:
            f.write(out)

        # Final safety net: records anything the per-task flushes missed (e.g. a group-level
        # result key that only materialises after the last task). Deduped by `_recorded`, so a
        # normal run appends nothing extra here.
        _flush_registry(saved.get('results', {}))

    end = time.time()
    if _dispatch is not None:
        print_rank_0(f'>>> [dispatch] {_dispatch.summary()}')
    if _dispatch_client is not None:
        # Close the socket explicitly so the coordinator sees a clean EOF from a worker
        # that FINISHED rather than waiting out its heartbeat timer on a process that is
        # merely exiting slowly.
        _dispatch_client.close()
    end = time.time()
    print_rank_0(
        f"Evaluation of {args.task_list} ended in {end - start:.1f}s "
        f"({(end - start) / 60:.1f} min); completed {len(completed)}/{len(task_list)} tasks"
    )


if __name__ == '__main__':
    main()
