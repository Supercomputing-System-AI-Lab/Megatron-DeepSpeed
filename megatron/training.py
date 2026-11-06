# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time
import json
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
import os
import socket

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron import print_rank_0, is_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import GPTModel
from megatron.core.enums import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
#from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.utils import report_memory, throughput_calculator, checkpoint_throughput_calculator, update_rotary_pos_emb
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd
from megatron.model.transformer import ParallelTransformerLayer

from deepspeed import comm as dist

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None
from torch.profiler import profile, record_function, ProfilerActivity, schedule

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

'''
Since v0.9.0, deepspeed.initialize() has forbidden simultaneous setting of args.deepspeed_config (Path) and ds_config dict.
So, we use ds_config dict which is the more flexible option. 
'''
def _create_ds_config_dict():
    args = get_args()
    with open(args.deepspeed_config, 'r', encoding='utf-8') as config_file:
        ds_config_dict = json.load(config_file)
    
    if args.universal_checkpoint:
        ds_config_dict["checkpoint"] = {"load_universal": True}
    sequence_parallel_pipe_config(args, ds_config_dict)
    ulysses_pipe_batch_config(args, ds_config_dict)

    # Clear config path
    args.deepspeed_config = None 

    return ds_config_dict


def ulysses_pipe_batch_config(args, ds_config_dict):
    """[X-MoE 2026-09-17, the Ulysses blueprint C3] under a pipeline the DeepSpeed engine's data-parallel size is the pipe grid's data
    axis, which is dp x P (gpt_model.py builds the topology so, for ZeRO and the gradient reduction). DeepSpeed derives the micro-batches
    per rank as train_batch_size / (micro_batch x that size), so with the global batch as given it would run P times too few: every P
    rank of one loader must process all of that loader's micro-batches (they hold shards of the same sequences). The engine's
    train_batch_size is therefore the global batch times P; Megatron's own counters (global_batch_size, consumed samples) are unchanged.
    Without a pipeline the engine's data-parallel size excludes P and nothing changes."""
    P = getattr(args, 'ds_sequence_parallel_size', 1) or 1
    if P > 1 and args.pipeline_model_parallel_size > 1:
        if 'train_batch_size' in ds_config_dict:
            ds_config_dict['train_batch_size'] = int(ds_config_dict['train_batch_size']) * P
        if 'gradient_accumulation_steps' in ds_config_dict:
            ds_config_dict['gradient_accumulation_steps'] = int(ds_config_dict['gradient_accumulation_steps']) * P
        print_rank_0(f'[ulysses] DeepSpeed train_batch_size x {P} under the pipeline (the grid data axis is dp x P): '
                     f'{ds_config_dict.get("train_batch_size")}, gradient_accumulation_steps {ds_config_dict.get("gradient_accumulation_steps")}')


def sequence_parallel_pipe_config(args, ds_config_dict):
    """Under Megatron's companion sequence parallelism (--sequence-parallel) with tensor and pipeline parallelism, switch off
    DeepSpeed's partitioning of the activations and gradients that cross a pipeline boundary (ds_config pipeline
    pipe_partitioned / grad_partitioned, both True by default). That partitioning cuts the tensor into TP slices on the sender
    and reassembles it by an all-gather over the receiver's tensor-parallel group, which is right only when every TP rank
    holds the same activation; under sequence parallelism each rank holds its own sequence shard, so the reassembled tensor is
    half of one shard and half of the other (2026-09-11, runs A2b/A3b: the stage-0 layers bit-identical with and without SP,
    the first stage-1 layer off by 1.4 relative; the loss 8.72 instead of 7.53 from the same weights). Returns True when it
    changed the config; a no-op for every other layout."""
    if not getattr(args, 'sequence_parallel', False):
        return False
    if getattr(args, 'tensor_model_parallel_size', 1) <= 1 or getattr(args, 'pipeline_model_parallel_size', 1) <= 1:
        return False
    if getattr(args, 'no_pipeline_parallel', False):
        return False
    pipe = ds_config_dict.setdefault('pipeline', {})
    pipe['pipe_partitioned'] = False
    pipe['grad_partitioned'] = False
    print_rank_0('[sequence-parallel] pipeline activations and gradients are sent whole, not partitioned across the '
                 'tensor-parallel group (pipe_partitioned = grad_partitioned = False): each TP rank holds its own sequence shard')
    return True
    

def _debug_dump_loaded_weights(model):
    """Print a fingerprint of selected loaded weights, to compare against the universal atoms.

    Under --universal-checkpoint the bf16 weights are rebuilt from the fp32 atoms, so this is
    the direct way to check that a TP-sharded parameter received the slice it should.

    NOTE: for a 2-D weight of shape [R, C] split over TP, chunk(dim=0) and chunk(dim=1) produce
    the SAME number of elements, so load_hp_checkpoint_state's numel assertion cannot tell them
    apart. weight[1,0] does: with the correct row-parallel slice it is W[1,0]; with the wrong
    axis it is another element of W (edit 24.1 of the pathway page traces it).  Enable with DEBUG_WEIGHT_DUMP=1.
    """
    import os
    if os.environ.get('DEBUG_WEIGHT_DUMP', '0') != '1':
        return
    want = os.environ.get('DEBUG_WEIGHT_DUMP_PARAMS',
                          'layers.0.self_attention.dense.weight,'
                          'layers.0.self_attention.query_key_value.weight,'
                          'layers.0.mlp.dense_h_to_4h.weight,'
                          'layers.0.mlp.dense_4h_to_h.weight,'
                          'layers.1.self_attention.dense.weight,'
                          'layers.1.self_attention.query_key_value.weight,'
                          'layers.1.mlp.dense_h_to_4h.weight,'
                          'layers.1.mlp.dense_4h_to_h.weight').split(',')
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    tp = mpu.get_tensor_model_parallel_world_size()
    tpr = mpu.get_tensor_model_parallel_rank()
    for m in model:
        mod = m.module if hasattr(m, 'module') else m
        for n, p in mod.named_parameters():
            if not any(w.strip() and w.strip() in n for w in want):
                continue
            d = p.detach().float().cpu()
            flat = d.flatten()
            extra = ""
            if d.dim() == 2:
                extra = f" w[1,0]={d[1,0].item():+.8f} w[2,0]={d[2,0].item():+.8f}"
            print(f"[WDUMP] rank={rank} tp={tpr}/{tp} {n} shape={tuple(d.shape)} "
                  f"flat[:4]={[round(v,8) for v in flat[:4].tolist()]}{extra}", flush=True)


# [X-MoE resilience 2026-10-09] XMOE_PG_TIMEOUT_ALL_MIN: the timeout of EVERY process group created without one (Megatron's
# tensor/data/pipeline groups and DeepSpeed's), so a hang inside a TP or DP collective is detected as fast as one in the expert
# group (XMOE_PG_TIMEOUT_MIN covers DeepSpeed's groups only). Unset: torch's default (10 min), as before.
if os.environ.get('XMOE_PG_TIMEOUT_ALL_MIN') and not getattr(torch.distributed.new_group, '_xmoe', False):
    from datetime import timedelta as _xmoe_td
    _xmoe_orig_new_group = torch.distributed.new_group
    def _xmoe_new_group(*a, **k):
        if k.get('timeout') is None and len(a) < 2:
            k['timeout'] = _xmoe_td(minutes=float(os.environ['XMOE_PG_TIMEOUT_ALL_MIN']))
        return _xmoe_orig_new_group(*a, **k)
    _xmoe_new_group._xmoe = True
    torch.distributed.new_group = _xmoe_new_group


# [X-MoE resilience 2026-10-09] the heartbeat, the detectors and the fault injection live in
# examples_elmoe/scripts-frontier/resilience/bin/hb.py; nothing is imported unless XMOE_HB=1, XMOE_DETECT=1 or an injection is set.
_XMOE_RES = {'mod': None, 'tried': False}


def _xmoe_res():
    if not _XMOE_RES['tried']:
        _XMOE_RES['tried'] = True
        if os.environ.get('XMOE_HB') == '1' or os.environ.get('XMOE_DETECT') == '1' or \
                os.environ.get('XMOE_INJECT') or os.environ.get('XMOE_INJECT_SCHEDULE'):
            _d = os.environ.get('RES_BIN') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                           'examples_elmoe', 'scripts-frontier', 'resilience', 'bin')
            if _d not in sys.path:
                sys.path.insert(0, _d)
            import hb as _hbmod
            sys.modules['xmoe_hb'] = _hbmod            # the call sites in deepspeed look it up by this name
            _XMOE_RES['mod'] = _hbmod
    return _XMOE_RES['mod']


# [ft-sys-v2 2026-11-05] the in-process part of the heartbeat system (examples_elmoe/scripts-frontier/ft-sys-v2/heartbeat-sys/
# hb_monitor.py): a count of every torch.distributed communication call, and a monitor thread that writes this rank's line to host
# memory every 5 s for the node's agent. Nothing is imported unless XMOE_FT_V2=1, so training is unchanged by default. The monitor
# makes no GPU call (design map 1.5); the trainer only hands it Python values (the step, the phase).
_XMOE_FTV2 = {'mod': None, 'tried': False}


def _xmoe_ftv2():
    if not _XMOE_FTV2['tried']:
        _XMOE_FTV2['tried'] = True
        if os.environ.get('XMOE_FT_V2') == '1':
            _d = os.environ.get('FTV2_HB_DIR') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                               'examples_elmoe', 'scripts-frontier', 'ft-sys-v2', 'heartbeat-sys')
            if _d not in sys.path:
                sys.path.insert(0, _d)
            import hb_monitor as _m
            _m.install_counter()
            _XMOE_FTV2['mod'] = _m
    return _XMOE_FTV2['mod']


# [ft-sys-v2 2026-11-06] the in-process recovery (examples_elmoe/scripts-frontier/ft-sys-v2/recovery/): on only with XMOE_FT_V2=1
# AND FTV2_RECOVERY=1. Off, nothing below is imported and pretrain() runs exactly as before. The hooks in pretrain(): the standby's
# start (no world until it joins), Keep installed after the first build, and train() run inside the recovery's loop, which enters
# it again at the resume step after a recovery (train() starts at args.iteration). No line of train() changes.
_XMOE_FTV2_RC = {'obj': None, 'tried': False}


def _xmoe_ftv2_rc():
    if not _XMOE_FTV2_RC['tried']:
        _XMOE_FTV2_RC['tried'] = True
        if _xmoe_ftv2() is not None and os.environ.get('FTV2_RECOVERY') == '1':
            _d = os.environ.get('FTV2_RC_DIR') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                               'examples_elmoe', 'scripts-frontier', 'ft-sys-v2', 'recovery')
            if _d not in sys.path:
                sys.path.insert(0, _d)
            import rc_controller as _rcm
            _XMOE_FTV2_RC['obj'] = _rcm.Recovery(_XMOE_FTV2['mod'], None)
    return _XMOE_FTV2_RC['obj']


# [X-MoE resilience 2026-10-10] THE WALL-AWARE EXIT (XMOE_WALL_AWARE_EXIT=1; unset or 0 = today's behaviour, unchanged).
# WHY: in the 260-node job 5645677 (wall 2 h, steps of 43 min) the exit was --exit-duration-in-mins = wall - 20, checked by
# Megatron only at the END of an iteration against the trainer's own start time; the steps ended at minute 51 and 94, the third
# step started at 94 and the wall killed it at 120, so 22 minutes of 256 nodes were lost, and a hard stop at the wall can land in
# the middle of a save. The user (2026-10-09 23:50): the exit must follow the job's real Slurm end time, and a save must never be
# cut by the wall.
# HOW: the arithmetic lives in examples_elmoe/scripts-frontier/resilience/bin/wall_exit.py (no torch, so it is unit-tested on a
# login node). At the start of train() the end time is read from SLURM_JOB_END_TIME (Slurm exports the job's projected end as an
# epoch), else from `scontrol show job` once, else from --exit-duration-in-mins (the trainer's start + the minutes); the source is
# logged. After every iteration rank 0 computes remaining = end - now and the cost of one more step = the slowest of the last 3
# iterations x (1 + XMOE_WALL_MARGIN_FRAC) + the measured save (its synchronous part + the asynchronous engine's commit-to-'latest'
# time; XMOE_SAVE_EST_S before the first save) + XMOE_WALL_MARGIN_S, and shares the decision with every rank through one all-reduce.
# When remaining < cost the trainer saves (unless this step's save is already in flight), waits until that save is complete on
# every rank and 'latest' names it, writes one sentence with the arithmetic (rank_0.log, training.log, data/wall_exit.tsv) and
# exits with code 0. While the flag is on, --exit-duration-in-mins is NOT checked (the wall rule takes precedence; it is only the
# fallback source of the end time).
# THE EXIT IS HARD (2026-10-10 02:30, from job 5646541): the first version ended with sys.exit(0), and the ranks did not end. The
# evidence: the exit sentence was printed at 01:31:41; within 0-4 s every rank printed torch's "destroy_process_group() was not
# called before program exit" (the ProcessGroupNCCL destructors had started, so Python's teardown had already passed the thread
# joins and the atexit handlers); the destructors then finalized the six RCCL communicators of every rank one by one, the 8-rank
# (node) step finished on all 8 ranks of a node within 1 ms at 01:31:48/50, and the 16-rank communicators never finished until the
# Slurm pre-wall handler killed the step 46 s later ("Abort COMPLETE" x3 per rank at 01:32:27). The "new 8-rank communicator after
# the exit" seen in rank_0.log was only RCCL's C stdio buffer being flushed at exit (that communicator's Init START is at line 1186
# of the same log, during the first step). So the trainer must never enter torch's teardown: after the save is published on every
# rank and ONE confirming all-reduce, every rank flushes its logs, stops the asynchronous engine's writer thread (a bounded join;
# the commit threads are already joined by wait_published) and calls os._exit(0), with no further collective and no teardown.
# This is the same exit path hb.die() and the profiled branch of pretrain_gpt_deepspeed.py already use.
_XMOE_WALL = {'on': os.environ.get('XMOE_WALL_AWARE_EXIT') == '1', 'mod': None, 'pred': None, 'end': None, 'src': None, 'tried': False}


# [X-MoE 2026-10-15, the Adam warm-start probe (assistant_deep_dive/2026-10-15-adam_warmstart)] two probe knobs, both off by default.
# XMOE_SAVE_STEPS="2,3,5,10,20,40": the training loop saves at the listed iterations IN ADDITION to --save-interval (the checkpointing
# block below and the 'saved' flag of training.log). Unset or empty = today's behaviour, unchanged.
# XMOE_LOG_DW=1: one line per iteration on rank 0 with the global update of the fp32 master weights across that iteration's optimizer
# step: |dw|_2 and max|dw| summed/maxed over every rank's ZeRO partition (every partition is held exactly once in the world: the dense
# groups are sharded over their data-parallel group, each expert group over its own), the number of changed elements, the optimizer's
# step counter (apex FusedAdam keeps it in param_groups[0]['step']), the lr, and rank 0's local |exp_avg_sq|_2. The snapshot before
# the step is a host copy of this rank's partitions (the 16B at 1 node: about 8 GB per rank, 1-2 s), so it is a probe instrument.
def _xmoe_parse_steps(text):
    out = set()
    for tok in (text or '').replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return out


_XMOE_SAVE_STEPS = _xmoe_parse_steps(os.environ.get('XMOE_SAVE_STEPS'))
_XMOE_LOG_DW = os.environ.get('XMOE_LOG_DW') == '1'


def _xmoe_dw_groups(engine):
    """This rank's fp32 master partitions: ZeRO-1/2 (single_partition_of_fp32_groups) or the bf16 optimizer (fp32_groups_flat_partition)."""
    opt = getattr(engine, 'optimizer', None)
    for name in ('single_partition_of_fp32_groups', 'fp32_groups_flat_partition'):
        g = getattr(opt, name, None)
        if g is not None:
            return list(g)
    return []


def _xmoe_dw_snapshot(engine):
    return [p.detach().to('cpu', copy=True) for p in _xmoe_dw_groups(engine)]


def _xmoe_adam_step(engine):
    """apex FusedAdam's step counter (param_groups[0]['step']; None before the first step of a fresh optimizer)."""
    try:
        return getattr(engine.optimizer, 'optimizer', engine.optimizer).param_groups[0].get('step', None)
    except Exception:
        return None


def _xmoe_sumsq_chunked(t, chunk=1 << 26):
    acc = 0.0
    for c in t.flatten().split(chunk):
        cd = c.double()
        acc += float((cd * cd).sum().item())
    return acc


def _xmoe_dw_report(engine, optimizer, snap, iteration):
    """Every rank computes its partitions' delta against the snapshot; one all-reduce each for the sum of squares, the max and the
    changed count; rank 0 prints the line."""
    dev = get_accelerator().current_device_name()
    sq, mx, nz, n = 0.0, 0.0, 0.0, 0.0
    groups = _xmoe_dw_groups(engine)
    for p, s0 in zip(groups, snap):
        d = p.detach().float() - s0.to(p.device).float()
        sq += _xmoe_sumsq_chunked(d)
        mx = max(mx, float(d.abs().max().item()) if d.numel() else 0.0)
        nz += float((d != 0).sum().item())
        n += float(d.numel())
        del d
    v2 = 0.0
    base = getattr(engine.optimizer, 'optimizer', None)
    if base is not None:
        for p in groups:
            st = base.state.get(p, {})
            if 'exp_avg_sq' in st:
                v2 += _xmoe_sumsq_chunked(st['exp_avg_sq'])
    t = torch.tensor([sq, nz, n], dtype=torch.float64, device=dev)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    m = torch.tensor([mx], dtype=torch.float64, device=dev)
    torch.distributed.all_reduce(m, op=torch.distributed.ReduceOp.MAX)
    try:
        lr = float(optimizer.param_groups[0]['lr'])
    except Exception:
        lr = float('nan')
    print_rank_0(f'[dw] iteration {iteration} | lr {lr:.6e} | adam step {_xmoe_adam_step(engine)} | |dw|_2 {t[0].item() ** 0.5:.6e} | '
                 f'max|dw| {m.item():.6e} | changed {int(t[1].item())} of {int(t[2].item())} fp32 masters | '
                 f'rank0 |exp_avg_sq|_2 {v2 ** 0.5:.6e}')


def _xmoe_res_bin():
    """The folder of the resilience helpers (RES_BIN, else the tree's examples_elmoe/scripts-frontier/resilience/bin)."""
    return os.environ.get('RES_BIN') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    'examples_elmoe', 'scripts-frontier', 'resilience', 'bin')


def _xmoe_wall_mod():
    """Import wall_exit.py once; None when the flag is off or the import fails (then the old behaviour stays)."""
    if not _XMOE_WALL['tried']:
        _XMOE_WALL['tried'] = True
        if _XMOE_WALL['on']:
            try:
                _d = _xmoe_res_bin()
                if _d not in sys.path:
                    sys.path.insert(0, _d)
                import wall_exit as _we
                _XMOE_WALL['mod'] = _we
                _XMOE_WALL['pred'] = _we.Predictor()
            except Exception as e:
                print(f'[wall-aware exit] XMOE_WALL_AWARE_EXIT=1 but wall_exit.py could not be imported ({e}): the old exit rule stays', flush=True)
                _XMOE_WALL['on'] = False
    return _XMOE_WALL['mod']


def _xmoe_wall_init(args):
    """Called once before the training loop on every rank: resolve the end time (rank 0 asks Slurm; the value is shared through
    an all-reduce so that every rank holds the same number), seed the save cost from the job folder, log the rule once."""
    we = _xmoe_wall_mod()
    if we is None:
        return
    rank = torch.distributed.get_rank()
    end, src, note = (None, None, '')
    if rank == 0:
        end, src, note = we.resolve_end_time(env=os.environ, job_id=os.environ.get('SLURM_JOB_ID'), train_start=_TRAIN_START_TIME,
                                             exit_duration_min=args.exit_duration_in_mins)
    # share rank 0's end time and the source index: [end epoch, source code]; the other ranks contribute zeros (MAX keeps rank 0's)
    srcs = ['SLURM_JOB_END_TIME', 'scontrol show job EndTime', '--exit-duration-in-mins']
    t = torch.tensor([float(end or 0.0), float(srcs.index(src) + 1 if src in srcs else 0)], dtype=torch.float64,
                     device=get_accelerator().current_device_name())
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    end_v, src_i = t[0].item(), int(t[1].item())
    if end_v <= 0:
        _XMOE_WALL['on'] = False
        print_rank_0(f'[wall-aware exit] {note}; the old exit rule (--exit-duration-in-mins {args.exit_duration_in_mins}) stays in force')
        return
    _XMOE_WALL['end'], _XMOE_WALL['src'] = end_v, srcs[src_i - 1]
    pred = _XMOE_WALL['pred']
    pred.seed_from_files(os.environ.get('RES_DATA_DIR') or (os.path.join(os.environ['RES_JOB_DIR'], 'data') if os.environ.get('RES_JOB_DIR') else None))
    save_s, save_src = pred.save_cost()
    print_rank_0(f'[wall-aware exit] on: {note}; {we.fmt(end_v - time.time())} s remain now; after every iteration the trainer exits when the '
                 f'remaining time is shorter than the slowest of the last 3 iterations x {1 + pred.margin_frac:.1f} + the save cost '
                 f'({we.fmt(save_s)} s now, from {save_src}) + {we.fmt(pred.margin_s)} s; '
                 + (f'--exit-duration-in-mins {args.exit_duration_in_mins} (the rider\'s EXIT_DURATION) is ignored while this rule is on'
                    if args.exit_duration_in_mins and _XMOE_WALL['src'] != '--exit-duration-in-mins' else
                    'no --exit-duration-in-mins is set' if not args.exit_duration_in_mins else 'the end time itself came from --exit-duration-in-mins'))


def _xmoe_wall_latest(args, model):
    """(expected tag, what 'latest' names now): the DeepSpeed tag of the save of this step and the content of <save>/latest."""
    tag = f'global_step{model[0].global_steps}' if args.deepspeed else None
    try:
        cur = open(os.path.join(args.save, 'latest')).read().strip()
    except Exception:
        cur = None
    return tag, cur


def _xmoe_hard_exit(eng, code=0):
    """End this rank NOW, symmetrically on every rank, without torch's teardown (see the block above for why): flush Python's
    stdout/stderr and every C stdio stream (the rank log is fd 1; RCCL writes its INFO lines through C stdio), give the
    asynchronous checkpoint engine's writer thread a bounded chance to return (its queue is empty once 'latest' names the save,
    so the join is a formality), wait 3 s so that every rank's last all-reduce kernels have drained before any rank's buffers
    vanish, then os._exit(code). The heartbeat thread is a daemon and dies with the process; the node writer child sees its
    parent gone and copies once more by itself."""
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL(None).fflush(None)            # every C stdio stream, so that the RCCL lines in the rank log are not lost
    except Exception:
        pass
    try:
        if eng is not None and hasattr(eng, 'stop_for_exit'):
            eng.stop_for_exit(timeout=30.0)
    except Exception as e:
        print(f'[wall-aware exit] the engine\'s writer thread was not joined cleanly ({e}); exiting anyway', flush=True)
    time.sleep(3.0)
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    os._exit(int(code))


def _xmoe_wall_decide(iteration, model, optimizer, opt_param_scheduler, saved_checkpoint, step_seconds):
    """After an iteration (the point where Megatron checks --exit-duration-in-mins): rank 0 decides whether one more step and
    its save fit before the end time; every rank receives the same decision; on 'exit' the save is made (or waited for), its
    completion is verified on every rank ('latest' names the step), one sentence is logged, and the process exits with code 0."""
    we = _XMOE_WALL['mod']
    if we is None or not _XMOE_WALL['on']:
        return
    args = get_args()
    pred = _XMOE_WALL['pred']
    pred.record_step(step_seconds)
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    eng = getattr(model[0], 'checkpoint_engine', None) if args.deepspeed else None
    last = getattr(eng, 'last_complete', None)                       # rank 0 of the async engine: the last save's commit-to-'latest' time
    if rank == 0 and last:
        pred.record_save(async_s=last.get('commit_to_latest_s'), src=f'the last save ({last.get("tag")}): the synchronous part + {we.fmt(last.get("commit_to_latest_s"))} s from the commit to \'latest\'')
    d = pred.decide(time.time(), _XMOE_WALL['end']) if rank == 0 else dict(done=False, remaining=0, step_s=0, factor=0, step_pred=0, save_s=0, margin_s=0, cost=0)
    # one all-reduce (MAX) of rank 0's numbers: the decision and the figures of the sentence reach every rank; the others send zeros
    t = torch.tensor([1.0 if d['done'] else 0.0, float(d.get('remaining') or 0), float(d.get('step_s') or 0), float(d.get('factor') or 0),
                      float(d.get('save_s') or 0), float(d.get('margin_s') or 0), float(d.get('cost') or 0)], dtype=torch.float64,
                     device=get_accelerator().current_device_name())
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    done = bool(t[0].item() > 0.5)
    d = dict(done=done, remaining=t[1].item(), step_s=t[2].item(), factor=t[3].item(), save_s=t[4].item(), margin_s=t[5].item(), cost=t[6].item())
    if not done:
        # a line per step only near the end (the remaining time under 3 x the cost) or on request (XMOE_WALL_VERBOSE=1)
        if os.environ.get('XMOE_WALL_VERBOSE') == '1' or d['remaining'] < 3 * max(d['cost'], 1.0):
            print_rank_0('[wall-aware exit] ' + we.continue_sentence(iteration, d, _XMOE_WALL['end'], _XMOE_WALL['src']))
        return
    _res = _XMOE_RES['mod']
    t_exit0 = time.time()
    started_save = False
    if args.save and not saved_checkpoint:
        print_rank_0(f'[wall-aware exit] step {iteration}: {we.fmt(d["remaining"])} s remain and one more step would need {we.arithmetic(d)}: saving now, then exiting')
        save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler)
        started_save = True
    elif args.save:
        print_rank_0(f'[wall-aware exit] step {iteration}: {we.fmt(d["remaining"])} s remain and one more step would need {we.arithmetic(d)}: the save of this step is already in flight, no second save is started; waiting for it')
    else:
        print_rank_0(f'[wall-aware exit] step {iteration}: {we.fmt(d["remaining"])} s remain and one more step would need {we.arithmetic(d)}: no --save, nothing to wait for')
    # wait until the save is complete on EVERY rank: the asynchronous engine's commit thread of this rank, then 'latest' on disk
    # (rank 0's commit thread writes it only after every rank's marker exists), then one barrier so that no rank exits early
    if _res is not None:
        _res.save_mark(f'wall-aware exit: waiting for the save of step {iteration} to be complete on all ranks')
    tag, cur = _xmoe_wall_latest(args, model) if args.save else (None, None)
    wait_s, save_ok = 0.0, (not args.save)
    if args.save:
        budget = max(30.0, d['remaining'] - 30.0)
        if eng is not None and hasattr(eng, 'wait_published'):
            save_ok, wait_s, cur = eng.wait_published(tag, timeout=budget)
        else:
            # the synchronous engine wrote 'latest' inside save_checkpoint itself; verify
            save_ok, wait_s = (cur == tag), time.time() - t_exit0
    t_ok = torch.tensor([0.0 if save_ok else 1.0], dtype=torch.float64, device=get_accelerator().current_device_name())
    torch.distributed.all_reduce(t_ok, op=torch.distributed.ReduceOp.MAX)   # the ONE confirming collective: every rank must see the save complete
    save_ok = bool(t_ok.item() < 0.5)
    # no barrier after this point and no other collective: the all-reduce above is the last time the ranks talk to each other
    sentence = we.exit_sentence(iteration, d, _XMOE_WALL['end'], _XMOE_WALL['src'], world, cur, save_ok, time.time() - t_exit0, started_save)
    print_rank_0('[wall-aware exit] ' + sentence)
    jd = os.environ.get('RES_JOB_DIR')
    writer = 0 if mpu.get_pipeline_model_parallel_world_size() == 1 else world - 1    # training.log's writer rank (hb.training_log_line)
    if jd and rank == writer:
        try:
            with open(os.path.join(jd, 'training.log'), 'a') as fh:
                fh.write(f'# {time.strftime("%Y-%m-%d %H:%M:%S")} the planned exit (wall-aware) at step {iteration}: {sentence}\n')
        except Exception as e:
            print(f'[wall-aware exit] training.log not written: {e}', flush=True)
    if rank == 0:
        dd = os.environ.get('RES_DATA_DIR') or (os.path.join(jd, 'data') if jd else None)
        if dd:
            try:
                p = os.path.join(dd, 'wall_exit.tsv')
                new = not os.path.exists(p)
                with open(p, 'a') as fh:
                    if new:
                        fh.write(we.RECORD_HEADER)
                    fh.write(we.record_line(os.environ.get('RES_ATTEMPT', '0'), iteration, d, _XMOE_WALL['end'], _XMOE_WALL['src'], cur, save_ok,
                                            time.time() - t_exit0, sentence))
            except Exception as e:
                print(f'[wall-aware exit] wall_exit.tsv not written: {e}', flush=True)
    print_datetime(f'exiting program after the wall-aware exit at iteration {iteration}')
    _xmoe_hard_exit(eng, 0)


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """
    rank = int (os.getenv ("RANK"))
    # torch.cuda.set_device(local_rank)

    _xmoe_ftv2()             # [ft-sys-v2] the call counter goes in before the first communication call
    _rc = _xmoe_ftv2_rc()    # [ft-sys-v2 recovery] None unless FTV2_RECOVERY=1
    _rc_standby = _rc is not None and _rc.role == 'standby'
    if _rc is not None:
        from adapters.megatron_train import MegatronTrainer
        _rc.tr = MegatronTrainer(model_provider, model_type, train_valid_test_dataset_provider, data_post_process)
        _rc.split_mpi_world()    # [ft-sys-v2 recovery] the standby shares the Slurm step: MPI's world is split by role first
    # Initalize and get arguments, timers, and Tensorboard writer.
    if _rc_standby:
        _rc.standby_init(extra_args_provider, args_defaults)   # [ft-sys-v2 recovery] arguments, device, kernels; no world yet
    else:
        initialize_megatron(extra_args_provider=extra_args_provider,
                            args_defaults=args_defaults)
    print_rank_0 (f'after initialize_megatron')
    if _XMOE_FTV2['mod'] is not None and not _rc_standby:
        _XMOE_FTV2['mod'].start(torch.distributed.get_rank(), int(os.getenv('LOCAL_RANK', os.getenv('SLURM_LOCALID', '0'))),
                                on_failure=_rc.on_failure if _rc is not None else None)
        if _rc is not None:
            _rc.on_start(torch.distributed.get_rank(), torch.distributed.get_world_size())
    _res = _xmoe_res()
    if _res is not None:
        _res._ctx['rank'] = torch.distributed.get_rank()
        _res._ctx['local_rank'] = int(os.getenv('LOCAL_RANK', '0'))
        _res._ctx['job_dir'] = os.environ.get('RES_JOB_DIR')
        _res.start(torch.distributed.get_rank(), int(os.getenv('LOCAL_RANK', '0')))
        if os.environ.get('XMOE_DETECT') == '1':
            sys.excepthook = _res.excepthook        # D4 GPU OOM -> 11, D9 HIP -> 16, any other exception -> 17
    
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    #if get_accelerator().device_name() == 'cuda':
    #    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    print (f'[megatron/training.py] before start_time_tensor = get_accelerator().DoubleTensor([_TRAIN_START_TIME]) {rank=} {get_accelerator().current_device()=}', flush=True)
    global _TRAIN_START_TIME
    # device_id = get_accelerator().current_device()
    device_id = int (os.getenv ("LOCAL_RANK"))
    # print ()
    start_time_tensor = torch.tensor([_TRAIN_START_TIME], dtype=torch.float64, device=f'cuda:{device_id}')
    # start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
    print (f'[megatron/training.py] before init all-reduce {rank=} {start_time_tensor=}', flush=True)
    if not _rc_standby:      # [ft-sys-v2 recovery] the standby has no world yet; it keeps its own start time
        torch.distributed.all_reduce(start_time_tensor,
                                     op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    if not _rc_standby:      # [ft-sys-v2 recovery] print_datetime is a barrier; the standby has no world yet
        print_datetime('after megatron is initialized')
    print (f'[megatron/training.py] after init all-reduce', flush=True)

    args = get_args()
    timers = get_timers()

    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()
        if "curriculum_learning" in args.deepspeed_config_dict and \
            "enabled" in args.deepspeed_config_dict["curriculum_learning"]:
            args.curriculum_learning_legacy = args.deepspeed_config_dict[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_config_dict["curriculum_learning"])
        if "compression_training" in args.deepspeed_config_dict:
            args.compression_training = True

    if _rc_standby:          # [ft-sys-v2 recovery] wait for a failure list naming this node, join, build with everyone, train
        _box = _rc.standby_join(_rc.standby_wait())
        _rc.train(train, _box, forward_step_func, process_non_loss_data_func)
        _rc.finish()         # ends the process (it trained after a recovery)
        return

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    if _res is not None: _res.set_phase('load')
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, teacher=False, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0]
                               for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1]
                               for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2]
                              for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    if args.data_efficiency_curriculum_learning:
        if args.deepspeed_dataloader is not None:
            # We use args to pass the deepspeed_dataloader because adding
            # output to setup_model_and_optimizer will break the API for other
            # cases. We clear args.deepspeed_dataloader after updating
            # train_data_iterator because args will be saved in checkpoint and
            # attempting to save the whole deepspeed_dataloader will lead to
            # "AttributeError: Can't pickle local object...".
            train_data_iterator = iter(args.deepspeed_dataloader)
            args.deepspeed_dataloader = None
        else:
            train_data_iterator = None
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')
    if _rc is not None:      # [ft-sys-v2 recovery] Keep on the inner optimizer; every partner's shadow seeded (collective)
        _rc.after_build(model, optimizer, opt_param_scheduler, args.iteration)
    if _res is not None: _res.set_phase('init')

    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model()
    args.teacher_model = None
    if args.mos or args.kd: # Set up teacher model
        args.teacher_model = setup_teacher_model(args, model_provider)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_add_retriever:
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0 and _rc is not None:
            # [ft-sys-v2 recovery] the old objects must be free to go during a rebuild: the box holds the only references
            _box = [model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator]
            del model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator
            iteration = _rc.train(train, _box, forward_step_func, process_non_loss_data_func)
            model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator = _box
            _rc.finish()     # after a recovery: ends the process here (MPI's finalize would wait for the dead ranks)
        elif args.do_train and args.train_iters > 0:
            iteration = train(forward_step_func,
                            model, optimizer, opt_param_scheduler,
                            train_data_iterator, valid_data_iterator,
                            process_non_loss_data_func)

        print_datetime('after training is done')
        # 2026-09-18: the micro-batch profiler window keeps its per-rank traces unmerged inside the iteration; merge one rank per
        # pipeline stage into a Perfetto-ready .json.gz now that every rank has finished (profiler_manager.merge_after_training).
        from megatron.profiler_manager import merge_after_training
        merge_after_training()
        # Clean the model
        if args.compression_training:
            print (f"[training.py] {rank=} before cleaning the model", flush=True )
            model = [redundancy_clean(model[0], args.deepspeed_config_dict, mpu)]
            print (f"[training.py] {rank=} after cleaning the model", flush=True )
        

        if args.save and iteration != 0:
            print (f"[training.py] {rank=} before saving checkpoint", flush=True )
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
            print (f"[training.py] {rank=} after saving checkpoint", flush=True )
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    print (f"[training.py] {rank=} before core_transformer_config_from_args", flush=True )
    config = core_transformer_config_from_args(args)
    print (f"[training.py] {rank=} returning pretrain", flush=True )
    '''
    if args.do_valid:
        prefix = f'iteration {iteration} on {args.eval_iters * args.global_batch_size}-sample draw from validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:
        prefix = f'iteration {iteration} on {args.eval_iters * args.global_batch_size}-sample draw from test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train, test=True)
    return model
    '''


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def setup_teacher_model(args, model_provider):        
    
    print_rank_0('***>>>>> Student model checkpoint iteration:{}'.format(args.iteration))
    iteration_stuent = args.iteration
    num_layers_student = args.num_layers
    num_experts_student = args.num_experts
    hidden_size_student = args.hidden_size
    num_attention_heads_student = args.num_attention_heads
    load_student = args.load

    print_rank_0('***>>>>> Setting up the teacher model')

    args.num_layers = args.num_layers_teacher
    args.num_experts = args.num_experts_teacher
    args.hidden_size = args.hidden_size_teacher
    args.num_attention_heads = args.num_attention_heads_teacher
    args.load = args.load_teacher
    teacher_model, _, _ = load_model_weights_only(model_provider)
    print_rank_0('***>>>>> Teacher model:{}'.format(teacher_model))

    args.num_layers = num_layers_student
    args.num_experts = num_experts_student
    args.hidden_size = hidden_size_student
    args.num_attention_heads = num_attention_heads_student
    args.load = load_student
    args.iteration = iteration_stuent

    return teacher_model

def mark_norm_params_for_sequence_parallel(model_modules):
    """Tag replicated norm weights with the `sequence_parallel` attribute.

    Under companion sequence parallelism the norm activations are sharded
    seq-wise across the TP group, so every TP replica of a norm weight
    accumulates gradient over a disjoint sequence shard and the gradients
    must be summed across the TP group. Megatron's fused LayerNorm sets the
    attribute on its own weights (fused_layer_norm.py), but the apex
    MixedFusedRMSNorm modules built under `--normalization rmsnorm`
    (transformer.py) are constructed bare and never mark theirs -- so both
    Megatron's allreduce_layernorm_grads sweep and the DeepSpeed engine's
    _allreduce_sequence_parallel_grads hook (the only reachable path under
    DeepSpeed) would sweep nothing. Tensor-parallel-partitioned params are
    excluded: their gradients are per-partition, not sequence-shard
    replicas.
    """
    marked = 0
    for model_module in model_modules:
        for module in model_module.modules():
            if 'norm' not in type(module).__name__.lower():
                continue
            for param in module.parameters(recurse=False):
                if getattr(param, 'tensor_model_parallel', False):
                    continue
                setattr(param, 'sequence_parallel', True)
                marked += 1
    return marked


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type


    if not isinstance(model, list):
        model = [model]

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Companion sequence parallelism: rmsnorm weights must carry the
    # `sequence_parallel` attribute so their gradients get the TP-group sum
    # (see mark_norm_params_for_sequence_parallel). Must run before the
    # deepspeed early return below.
    if args.sequence_parallel:
        mark_norm_params_for_sequence_parallel(model)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p,'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.to(get_accelerator().current_device_name())
 

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = get_accelerator().current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler

def load_model_weights_only(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    print_rank_0('***>>>>> Args:{}'.format(args))

    model = get_model(model_provider_func)

    optimizer = None
    lr_scheduler = None

    if args.deepspeed:
        # When loading just the model weights, ZeRO can be disabled.
        if 'zero_optimization' in args.deepspeed_config_dict:
            del args.deepspeed_config_dict['zero_optimization']

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            config=args.deepspeed_config_dict
        )

        assert not isinstance(model, deepspeed.PipelineEngine), \
            'Weight loading only mode is not supported in pipeline parallelism yet.'

        model = [model]

    print_datetime('before load checkpoint')
    if args.load is not None:
        iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=True, load_only_weights=True)

    print_datetime('after load checkpoint weights')

    return model, optimizer, lr_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              teacher=False,
                              data_post_process=None,
                              build_train_valid_test_datasets_provider=None):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)

    # initialize the compression here
    student_global_steps = 0
    if args.kd or args.mos:
        model, _, _, _ = deepspeed.initialize(
                model=model[0],
                args=args,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
        model = [model]
        if args.load is not None:
            args.iteration = load_checkpoint(model, None, None, strict=False)
        else:
            args.iteration = 0
        student_global_steps = model[0].global_steps
        print_rank_0('***>>>>> Student model, global step:{}'.format(student_global_steps))

    if args.compression_training:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None,
            config=args.deepspeed_config_dict,
        )
        model = [model]
        model = [init_compression(model[0].module, args.deepspeed_config_dict, mpu)]

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    if args.inference:
        optimizer = None
        opt_param_scheduler = None
    else:
        if teacher:
            optimizer = None
        else:
            if getattr(args, 'dsa_freeze_non_indexer', False):
                # the V3.2 dense warm-up (blueprint CS7): only the indexer learns; frozen by learning rate (lr_mult 0 outside
                # indexer.*) so every optimizer group and the ZeRO layout stay as they are (indexer.py warmup_lr_condition)
                from megatron.model.deepseek_v3.indexer import warmup_lr_condition
                scale_lr_cond, lr_mult = warmup_lr_condition, 0.0
            optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        # opt_param_scheduler is the old lr_scheduler plus weight decay scheduling
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        pp = mpu.get_pipeline_model_parallel_world_size()
        if args.data_efficiency_curriculum_learning and build_train_valid_test_datasets_provider is not None:
            train_ds = None
            # Only need to build dataset on tp rank 0 since Megatron has the
            # broadcast_data() function that broadcast data from tp rank 0.
            if mpu.get_tensor_model_parallel_rank() == 0:
                # Number of train/valid/test samples.
                if args.train_samples:
                    train_samples = args.train_samples
                    update_train_iters(args)
                else:
                    train_samples = args.train_iters * args.global_batch_size
                # eval_iters and test_iters here are not actually used, only for
                # satisfying the input of build_train_valid_test_datasets_provider.
                # We only need to build the training data here. And we follow
                # baseline's logic to build eval/test dataset later in
                # build_train_valid_test_data_iterators.
                eval_iters = (args.train_iters // args.eval_interval + 1) * \
                            args.eval_iters
                test_iters = args.eval_iters
                train_val_test_num_samples = [train_samples,
                                            eval_iters * args.global_batch_size,
                                            test_iters * args.global_batch_size]
                # Build the datasets.
                train_ds, _, _ = build_train_valid_test_datasets_provider(
                    train_val_test_num_samples)
            model, optimizer, args.deepspeed_dataloader, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                training_data=train_ds,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
            model.set_data_post_process_func(data_post_process)
        else:
            model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_sequence_data_parallel_rank()   # [X-MoE 2026-09-17, C3] the grid's data axis is dp x P
        model = [model]

    # Compression has its own checkpoint loading path (e.g, loading both teacher and student models). So if compression is enabled, we skip the following checkpoint loading.
    no_post_init_checkpoint_loading = args.kd or args.mos
    if not no_post_init_checkpoint_loading:
        if args.load is not None:
            timers = get_timers()
            timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
            _debug_dump_loaded_weights(model)
            timers('load-checkpoint').stop(barrier=True)
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0
    else:
        model[0].global_steps = student_global_steps

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # random-LTD requires converting transformer layers
    if args.random_ltd:
        model[0] = convert_to_random_ltd(model[0], ParallelTransformerLayer)

    return model, optimizer, opt_param_scheduler



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if args.deepspeed and args.ds_pipeline_enabled:
        skipped_iter = 0
        num_zeros_in_grad = 0
        # print (f'[megatron/training.py] {model[0]=}')
        assert isinstance(model[0], deepspeed.PipelineEngine)
        loss = model[0].train_batch(data_iter=data_iterator)
        grad_norm = model[0].get_global_grad_norm()
        loss_dict = {'lm loss' : loss}
        # the prediction head's parts (DeepSeek-V3 CS5): 'lm main loss' and 'mtp loss' from the last stage; {} without the head
        from megatron.model.deepseek_v3.mtp import pop_pipe_loss_parts
        loss_dict.update(pop_pipe_loss_parts())
        return loss_dict, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        optimizer.zero_grad()

    # Forward pass.
    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    print (f'[megatron/training] after forward-backward timer')
    forward_backward_func = get_forward_backward_func()
    if args.mos or args.kd:
        # args.teacher_forward is used as global variable to enable kd loss
        # calculation in forward pass. Users do not need to set it in the
        # command line to use kd.
        args.teacher_forward = True

    # set timers to None if none of the timers in fwd_bwd are active, just to save the checks
    if args.timing_log_level < 2:
        config.timers = None

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    # reset timers if necessary
    if config.timers is None:
        config.timers = timers
    timers('forward-backward').stop()
    if args.mos or args.kd:
        args.teacher_forward = False

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    if not args.deepspeed:
        optimizer.reduce_model_grads(args, timers)

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    if _XMOE_RES['mod'] is not None: _XMOE_RES['mod'].set_phase('step')
    if args.deepspeed:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        model[0].step(lr_kwargs={'increment': increment})
        update_successful = model[0].was_step_applied()
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()

    # Gather params.
    if not args.deepspeed and update_successful:
        optimizer.gather_model_params(args, timers)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        # [2026-10-13] the single-stage DeepSpeed path never read the norm (only the pipeline path did, so every PP=1 run printed no
        # 'grad norm'); the engine caches it at step() (the bf16/ZeRO optimizer's all-groups norm, the one used for clipping), so
        # reading it here costs nothing and changes no training arithmetic
        try:
            grad_norm = model[0].get_global_grad_norm()
        except Exception:
            grad_norm = None
        num_zeros_in_grad = None
        
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    else:
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None, optimizer=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        writer.add_scalar('steps-vs-samples/y=steps,x=samples', iteration, args.consumed_train_samples)
        writer.add_scalar('steps-vs-samples/y=samples,x=steps', args.consumed_train_samples, iteration)
        writer.add_scalar('steps-vs-tokens/y=steps,x=tokens', iteration, args.consumed_train_tokens)
        writer.add_scalar('steps-vs-tokens/y=tokens,x=steps', args.consumed_train_tokens, iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate/learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate/learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            writer.add_scalar('learning-rate/learning-rate vs tokens', learning_rate,
                              args.consumed_train_tokens)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size/batch-size', batch_size, iteration)
            writer.add_scalar('batch-size/batch-size vs samples', batch_size,
                              args.consumed_train_samples)
            writer.add_scalar('batch-size/batch-size vs tokens', batch_size,
                              args.consumed_train_tokens)
        for key in loss_dict:
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs tokens', loss_dict[key],
                              args.consumed_train_tokens)
        if args.fp16 and args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale/loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale/loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            writer.add_scalar('loss-scale/loss-scale vs tokens', loss_scale,
                              args.consumed_train_tokens)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size/world-size', args.world_size, iteration)
            writer.add_scalar('world-size/world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            writer.add_scalar('world-size/world-size vs tokens', args.world_size,
                              args.consumed_train_tokens)
        if grad_norm is not None:
            writer.add_scalar('grad-norm/grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm/grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            writer.add_scalar('grad-norm/grad-norm vs tokens', grad_norm,
                              args.consumed_train_tokens)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros/num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros/num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            writer.add_scalar('num-zeros/num-zeros vs tokens', num_zeros_in_grad,
                              args.consumed_train_tokens)
        if params_norm is not None:
            writer.add_scalar('params-norm/params-norm', params_norm, iteration)
            writer.add_scalar('params-norm/params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            writer.add_scalar('params-norm/params-norm vs tokens', params_norm,
                              args.consumed_train_tokens)
        if hasattr(args, 'actual_seq_length'):
            writer.add_scalar('seqlen/actual_seq_length', args.actual_seq_length,
                              iteration)
            writer.add_scalar('seqlen/actual_seq_length vs samples', args.actual_seq_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/actual_seq_length vs tokens', args.actual_seq_length,
                              args.consumed_train_tokens)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            writer.add_scalar('seqlen/curriculum_seqlen', args.curriculum_seqlen,
                              iteration)
            writer.add_scalar('seqlen/curriculum_seqlen vs samples', args.curriculum_seqlen,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/curriculum_seqlen vs tokens', args.curriculum_seqlen,
                              args.consumed_train_tokens)
        if args.random_ltd:
            writer.add_scalar('seqlen/random_ltd_reserved_length', args.random_ltd_reserved_length,
                              iteration)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs samples', args.random_ltd_reserved_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs tokens', args.random_ltd_reserved_length,
                              args.consumed_train_tokens)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.tensorboard_log_interval == 0:
        # This logging write various optimizer states to tensorboard. This
        # feature may consume extra GPU memory thus is set at false by default.
        if args.log_optimizer_states_to_tensorboard and optimizer is not None:
            opt_stats = [0.0] * 8
            opt_stats_2 = [0.0] * 4
            for _, group in enumerate(optimizer.param_groups):
                for _, param in enumerate(group['params']):
                    opt_stats[0] += (torch.norm(optimizer.state[param]['exp_avg_sq']).item())**2
                    opt_stats[1] += (torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt()).item())**2
                    opt_stats[2] += (torch.norm(optimizer.state[param]['exp_avg']).item())**2
                    opt_stats[3] += (torch.norm(param).item())**2
                    opt_stats[4] += torch.norm(optimizer.state[param]['exp_avg_sq'],p=1).item()
                    opt_stats[5] += torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt(),p=1).item()
                    opt_stats[6] += torch.norm(optimizer.state[param]['exp_avg'],p=1).item()
                    opt_stats[7] += torch.norm(param,p=1).item()
                    opt_stats_2[0] = max(opt_stats_2[0], abs(optimizer.state[param]['exp_avg_sq'].max().item()), abs(optimizer.state[param]['exp_avg_sq'].min().item()))
                    opt_stats_2[1] = max(opt_stats_2[1], optimizer.state[param]['exp_avg_sq'].sqrt().abs_().max().item())
                    opt_stats_2[2] = max(opt_stats_2[2], abs(optimizer.state[param]['exp_avg'].max().item()), abs(optimizer.state[param]['exp_avg'].min().item()))
                    opt_stats_2[3] = max(opt_stats_2[3], abs(param.max().item()), abs(param.min().item()))
            # print('step {} rank {} before sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if args.zero_stage > 0:
                # ZeRO partiions optimizer states
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_sequence_data_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_sequence_data_parallel_group())

            if args.tensor_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_tensor_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_tensor_model_parallel_group())

            if args.pipeline_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_pipeline_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_pipeline_model_parallel_group())

            # print('step {} rank {} after sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if writer and is_last_rank():
                writer.add_scalar('optimizer/variance_l2 vs tokens', opt_stats[0]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l2 vs tokens', opt_stats[1]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l2 vs tokens', opt_stats[2]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l2 vs tokens', opt_stats[3]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_l1 vs tokens', opt_stats[4], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l1 vs tokens', opt_stats[5], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l1 vs tokens', opt_stats[6], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l1 vs tokens', opt_stats[7], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_abs_max vs tokens', opt_stats_2[0], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_abs_max vs tokens', opt_stats_2[1], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_abs_max vs tokens', opt_stats_2[2], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_abs_max vs tokens', opt_stats_2[3], args.consumed_train_tokens)

                writer.add_scalar('optimizer/variance_l2', opt_stats[0]**0.5, iteration)
                writer.add_scalar('optimizer/variance_sqrt_l2', opt_stats[1]**0.5, iteration)
                writer.add_scalar('optimizer/momentum_l2', opt_stats[2]**0.5, iteration)
                writer.add_scalar('optimizer/weight_l2', opt_stats[3]**0.5, iteration)
                writer.add_scalar('optimizer/variance_l1', opt_stats[4], iteration)
                writer.add_scalar('optimizer/variance_sqrt_l1', opt_stats[5], iteration)
                writer.add_scalar('optimizer/momentum_l1', opt_stats[6], iteration)
                writer.add_scalar('optimizer/weight_l1', opt_stats[7], iteration)
                writer.add_scalar('optimizer/variance_abs_max', opt_stats_2[0], iteration)
                writer.add_scalar('optimizer/variance_sqrt_abs_max', opt_stats_2[1], iteration)
                writer.add_scalar('optimizer/momentum_abs_max', opt_stats_2[2], iteration)
                writer.add_scalar('optimizer/weight_abs_max', opt_stats_2[3], iteration)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        seq_len = args.seq_length
        if hasattr(args, 'actual_seq_length'):
            seq_len = args.actual_seq_length
        samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(
            model,
            args,
            elapsed_time,
            total_iterations
        )
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            tput = {
                'throughput/iteration-time': elapsed_time_per_iteration,  # 1000 ms / s
                'throughput/samples_per_sec': samples_per_sec,
                'throughput/samples_per_sec_per_replica': samples_per_sec_per_replica,
                'throughput/tokens_per_sec': tokens_per_sec,
                'throughput/tokens_per_sec_per_replica': tokens_per_sec_per_replica,
                'throughput/tflops': tflops,
                'throughput/approx_params_in_billions': approx_parameters_in_billions,
                'throughput/elapsed_ms_per_iteration': elapsed_time_per_iteration,
            }
            wandb.run.log(tput)
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time/iteration-time',
                                  elapsed_time_per_iteration, iteration)
                writer.add_scalar('iteration-time/iteration-time vs samples',
                                  elapsed_time_per_iteration, args.consumed_train_samples)
                writer.add_scalar('iteration-time/iteration-time vs tokens',
                                  elapsed_time_per_iteration, args.consumed_train_tokens)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' consumed tokens: {:12d} |'.format(
            args.consumed_train_tokens)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        loss_for_wandb = {}
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                    loss_for_wandb[key] = avg
                total_loss_dict[key] = get_accelerator().FloatTensor([0.0])
        if loss_scale is not None:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            log_string += ' curriculum seqlen: {:5d} |'.format(args.curriculum_seqlen)
        if args.random_ltd:
            log_string += ' random ltd reserved length: {:5d} |'.format(args.random_ltd_reserved_length)
        log_string += ' actual seqlen: {:5d} |'.format(seq_len)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
        log_string += ' TFLOPs: {:.2f} |'.format(tflops)
        skipped_count = total_loss_dict[skipped_iters_key]
        nan_count = total_loss_dict[nan_iters_key]
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        if _XMOE_RES['mod'] is not None and os.environ.get('RES_JOB_DIR') and os.environ.get('XMOE_HB') == '1':
            # [X-MoE resilience 2026-10-09] training.log (design 3.11): one line per step across attempts. Every rank is here
            # (the interval timer above already used a barrier); one MAX all-reduce of a single float gives the HBM maximum.
            try:
                _xm = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device=get_accelerator().current_device_name())
                torch.distributed.all_reduce(_xm, op=torch.distributed.ReduceOp.MAX)
                _xm = float(_xm.item())
            except Exception:
                _xm = None
            _w = 0 if mpu.get_pipeline_model_parallel_world_size() == 1 else torch.distributed.get_world_size() - 1
            if torch.distributed.get_rank() == _w:
                _XMOE_RES['mod'].training_log_line(iteration, loss_for_wandb.get('lm loss'), tflops, elapsed_time_per_iteration, _xm,
                                                   bool(args.save and ((args.save_interval and iteration % args.save_interval == 0)
                                                                       or iteration in _XMOE_SAVE_STEPS)))   # [XMOE_SAVE_STEPS]
        print_rank_last(log_string)

        # Wandb: log the rest of the iteration-line fields + memory.
        # Throughput keys are already covered by the wandb.run.log(tput) block above.
        # Gated on wandb.run is not None -- only the last rank ever has it set
        # (init lives in pretrain_gpt_deepspeed.py:_maybe_init_wandb).
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb_log = {
                'train/iteration': iteration,
                'train/consumed_samples': args.consumed_train_samples,
                'train/consumed_tokens': args.consumed_train_tokens,
                'train/learning_rate': learning_rate,
                'train/global_batch_size': batch_size,
                'train/actual_seqlen': seq_len,
                'train/skipped_iterations': skipped_count,
                'train/nan_iterations': nan_count,
                'memory/allocated_gb': torch.cuda.memory_allocated() / (1024.0 ** 3),
                'memory/max_allocated_gb': torch.cuda.max_memory_allocated() / (1024.0 ** 3),
                'memory/reserved_gb': torch.cuda.memory_reserved() / (1024.0 ** 3),
                'memory/max_reserved_gb': torch.cuda.max_memory_reserved() / (1024.0 ** 3),
            }
            if loss_scale is not None:
                wandb_log['train/loss_scale'] = float(loss_scale)
            if grad_norm is not None:
                wandb_log['train/grad_norm'] = float(grad_norm)
            if num_zeros_in_grad is not None:
                wandb_log['train/num_zeros_in_grad'] = float(num_zeros_in_grad)
            if params_norm is not None:
                wandb_log['train/params_norm'] = float(params_norm)
            for k, v in loss_for_wandb.items():
                wandb_log[f'loss/{k}'] = v
            try:
                wandb.run.log(wandb_log, step=iteration)
            except Exception as e:
                print_rank_last(f'[wandb] log failed at iter {iteration}: {e}')

        # if report_memory_flag and learning_rate > 0.:
        if (iteration % 2 == 0) and (learning_rate > 0.):
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    _res = _XMOE_RES['mod']
    if _XMOE_FTV2['mod'] is not None:
        _XMOE_FTV2['mod'].set_phase('save')           # [ft-sys-v2] a save: the coordinator uses its longer still-limit
    if _res is None:
        timers('save-checkpoint', log_level=0).start(barrier=True)
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        timers('save-checkpoint').stop(barrier=True)
    else:
        with _res.SaveDeadline(iteration):           # D3: exit 12 when the save is later than its deadline
            timers('save-checkpoint', log_level=0).start(barrier=True)
            _res.set_phase('save')
            _res.inject('save', iteration)           # fshang: this rank's save sleeps 10 min
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
            timers('save-checkpoint').stop(barrier=True)
    checkpoint_throughput_calculator(model, timers('save-checkpoint').elapsed(reset=False))
    if _XMOE_WALL['pred'] is not None:
        # [wall-aware exit] the synchronous part of this save (the host copy and the barriers; with the asynchronous engine the
        # background writes and the commit are not in it: the engine's commit-to-'latest' time is added by _xmoe_wall_decide)
        _XMOE_WALL['pred'].record_save(sync_s=timers('save-checkpoint').elapsed(reset=False))
    timers.log(['save-checkpoint'])


# 1. Get the directory containing training.py (.../Megatron-DeepSpeed-X-MoE/megatron)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root (.../Megatron-DeepSpeed-X-MoE)
project_root = os.path.dirname(current_dir)

# 3. Build the absolute path to the examples_xmoe directory
examples_dir = os.path.join(project_root, "examples_xmoe")

# 4. Add it to sys.path so Python can find profiler_manager.py
if examples_dir not in sys.path:
    sys.path.append(examples_dir)

# 5. Now you can safely import!
from megatron.profiler_manager import _get_profiler, finalize_profiler, rank_selected

def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    print (f'[megatron/training.py] train()', flush=True)
    
    # Zixian: 10/01/2025: Try to profile megatron pp 
    # Zixian: 04/15/2025: Enabled a central file to control profiler args in X-MoE/Megatron-DeepSpeed-X-MoE/profiler_manager.py
    
    to_profile=os.getenv ("to_profile")
    # the routing record (2026-10-06): deepspeed/moe/route_log.py, on under XMOE_ROUTE_LOG=1, flushed once per step after the optimizer step
    try:
        from deepspeed.moe import route_log as _route_log
        _route_on = _route_log.enabled()
    except Exception:
        _route_log, _route_on = None, False
    if to_profile == 'True' and not rank_selected():
        to_profile = 'False'   # PROFILE_RANKS (2026-10-06): this rank is not in the traced set
    if mpu.get_data_parallel_rank() == 0:
        print (f'[training.py] {to_profile=}')
    if to_profile=='True': 
        # the schedule in iterations; PROFILE_WAIT / PROFILE_WARMUP / PROFILE_ACTIVE override it (2026-09-13: one active iteration
        # at GBS 2,048, where an iteration holds hundreds of micro-batches); the defaults are the previous constants
        wait=int(os.getenv('PROFILE_WAIT', '1'))
        warmup=int(os.getenv('PROFILE_WARMUP', '2'))
        active=int(os.getenv('PROFILE_ACTIVE', '2'))
        repeat=1
        max_prof_steps = (wait + warmup + active) * repeat 
        current_prof_step = 0
        # prof = _get_profiler(wait=wait, warmup=warmup, active=active, repeat=repeat)
        
        # from ../examples_xmoe import 
        # profiler_manager import _get_profiler, finalize_profiler
        prof = _get_profiler(wait=wait, warmup=warmup, active=active, repeat=repeat)
        if os.getenv('PROFILE_MICROBATCHES') == '1' and (hasattr(model[0], '_exec_schedule') or (args.deepspeed and not args.ds_pipeline_enabled)):
            # 2026-09-13: the schedule counts micro-batches instead of iterations (an iteration of GBS 2,048 at micro-batch 1
            # holds 256 micro-batches per rank, about 12 GB of trace per rank): the pipeline engine calls this after every
            # backward pass (deepspeed/runtime/pipe/engine.py _exec_schedule), and without a pipeline (--no-pipeline-parallel,
            # 2026-09-14: run Y1 traced nothing because only the pipe engine was hooked) backward_step in
            # megatron/core/pipeline_parallel/schedules.py calls it after model.backward, so PROFILE_WAIT / WARMUP / ACTIVE count
            # micro-batches; the per-iteration step below is switched off, and the per-rank traces are not merged (rank 0
            # would read the other ranks' files while they are still being written)
            _engine, _micro = model[0], {'n': 0}
            def _xmoe_micro_step():
                prof.step()
                _micro['n'] += 1
                if _micro['n'] >= max_prof_steps:
                    _engine._xmoe_micro_step = None
                    finalize_profiler(merge=False)
            _engine._xmoe_micro_step = _xmoe_micro_step
            to_profile = 'False'

    # Standalone CUDA-allocator memory snapshot (.pickle for https://pytorch.org/memory_viz).
    # Independent of the torch profiler (`to_profile`): runs whenever profile_memory=True, so
    # memory tracing works even with kineto OFF and isn't taken down if kineto crashes.
    profile_memory = os.getenv('profile_memory') == 'True'
    mem_snapshot_after_steps = int(os.getenv('mem_snapshot_after_steps', '3'))
    _mem_snapshot_done = False
    if profile_memory:
        from megatron.profiler_manager import enable_memory_history
        if enable_memory_history():
            if mpu.get_data_parallel_rank() == 0:
                print(f"[training.py] Memory history ON; dumping .pickle after "
                      f"{mem_snapshot_after_steps} steps", flush=True)
        else:
            profile_memory = False
        

    # Write args to tensorboard
    write_args_to_tensorboard()

    if args.random_ltd:
        # random-ltd requires different randomness on each rank
        import random
        random.seed(args.seed + torch.distributed.get_rank())

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Translate args to core configuration
    config = core_transformer_config_from_args(args)
    if not args.deepspeed:
        config.grad_scale_func = optimizer.scale_loss
    config.timers = timers

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('[megatron/training.py] before the start of training step')
    report_memory_flag = True
    if args.random_ltd:
        assert model[0].random_ltd_enabled()
        args.random_ltd_layer_num = model[0].random_ltd_scheduler.get_random_ltd_layer_num()
        
    _res = _XMOE_RES['mod']
    if _res is not None:
        _res.S['step'] = iteration                    # the heartbeat's step starts at the resumed iteration, not 0
        _res._ctx['resumed_from'] = iteration         # training.log names the save the attempt resumed from
        _res.inject('loop_start', iteration)          # inithang: sleep 20 min before the first step of this attempt
        _fsf = forward_step_func
        def forward_step_func(*a, **k):               # one line per micro-batch: the phase, and the 'slow' injection
            _res.set_phase('forward')
            _res.inject('micro', args.iteration)
            out = _fsf(*a, **k)
            _res.set_phase('backward')
            return out
    if _XMOE_WALL['on']:
        _xmoe_wall_init(args)                         # [wall-aware exit] the end time from Slurm, logged once; off when none is known
    if _XMOE_SAVE_STEPS:
        print_rank_0(f'[XMOE_SAVE_STEPS] saves at iterations {sorted(_XMOE_SAVE_STEPS)} in addition to --save-interval {args.save_interval}')
    if _XMOE_LOG_DW:
        print_rank_0(f'[dw] XMOE_LOG_DW=1: one line per iteration with the global fp32 master update (|dw|_2, max|dw|, changed count), '
                     f'the Adam step counter, the lr and rank 0\'s |exp_avg_sq|_2; adam step before the first iteration: {_xmoe_adam_step(model[0])}')
    _dw_snap = None
    while iteration < args.train_iters and (args.train_tokens is None or \
        args.consumed_train_tokens < args.train_tokens):
        if _XMOE_LOG_DW:
            _dw_snap = _xmoe_dw_snapshot(model[0])    # [dw] this rank's fp32 masters before the step
        if _XMOE_FTV2['mod'] is not None:
            _XMOE_FTV2['mod'].set_step(iteration)     # [ft-sys-v2] the step on this rank's heartbeat line (a Python value)
            _XMOE_FTV2['mod'].set_phase('train')
        _xmoe_wall_t0 = time.time()                   # [wall-aware exit] the iteration's wall time, measured from here to the save block
        update_num_microbatches(args.consumed_train_samples)
        if _res is not None:
            _res.begin_step()
            _res.set_phase('forward')
            _res.inject('step_start', iteration)      # hang / exit / kill / oom / slow / nan (each once per job)
            _xmoe_t0 = time.time()
        # [X-MoE 2026-10-08] fault injection for the hang-detection test: XMOE_HANG_INJECT="<rank>:<iteration>" makes that one rank
        # sleep forever at the start of that iteration (a fail-silent stop of a single rank); every other rank then blocks in its next
        # collective. It measures what NCCL_TIMEOUT, the process-group timeout and the watchers actually do: in V3X0 (job 5638129) a
        # wedged node stopped 8 ranks and 2,040 waited 86 min without one error line. Off unless the variable is set.
        _hang = os.environ.get('XMOE_HANG_INJECT')
        if _hang:
            _hr, _hit = (int(x) for x in _hang.split(':'))
            if torch.distributed.get_rank() == _hr and iteration == _hit:
                print(f'[hang-inject] rank {_hr} sleeps forever at iteration {_hit} ({time.strftime("%Y-%m-%d %H:%M:%S")})', flush=True)
                while True:
                    time.sleep(60)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = mpu.get_data_parallel_world_size() * \
                                args.micro_batch_size * \
                                get_num_microbatches()
            # [X-MoE 2026-09-17, the Ulysses blueprint C3] under a pipeline the engine's data-parallel size is the grid's dp x P: its
            # train batch is the global batch times P so that every P rank runs all the micro-batches of its loader (see
            # ulysses_pipe_batch_config); this call re-applies Megatron's global batch every iteration and must carry the same factor
            _P = args.ds_sequence_parallel_size if (args.ds_sequence_parallel_size > 1 and args.pipeline_model_parallel_size > 1) else 1
            model[0].set_train_batch_size(global_batch_size * _P)

        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                    args.iteration + 1)
            if iteration == 0 or curriculum_seqlen != args.curriculum_seqlen:
                if args.use_rotary_position_embeddings:
                    update_rotary_pos_emb(curriculum_seqlen)
            args.curriculum_seqlen = curriculum_seqlen
        args.curr_iteration = iteration
        
        # Zixian: 10/01/2025: profiling 
        # if to_profile=='True': 
        #     with record_function("pretrain"):
        #         loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
        #         train_step(forward_step_func,
        #                 train_data_iterator,
        #                 model,
        #                 optimizer,
        #                 opt_param_scheduler,
        #                 config)
        #     prof.step()
            
        #     print (f'[training.py] processing {iteration=}')
            
        if to_profile == 'True': 
            print(f'[training.py] processing {iteration=} (Profiler step {current_prof_step}/{max_prof_steps})')
            with record_function("pretrain"):
                loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config)
            print(f"[training.py] Profiling schedule step!")
            prof.step()
            current_prof_step += 1
            
            # ---> THE FIX <---
            if current_prof_step >= max_prof_steps:
                if os.getenv('PROFILE_EXPORT_AT_END') == '1':
                    # 2026-10-06: the export (a file write of seconds) and the close happen after the training loop
                    # (profiler_manager.finalize_if_pending, called from merge_after_training), so no iteration after the traced one pays for it
                    print(f"[training.py] Profiling schedule complete! Export deferred to the end of training (PROFILE_EXPORT_AT_END=1).")
                else:
                    print(f"[training.py] Profiling schedule complete! Closing handler early to save memory.")
                    finalize_profiler()   # 1. Close the background trace threads safely
                to_profile = 'False'  # 2. Dynamically turn off profiling for remaining steps!
            
                
            
            
        else: 
            print(f'[training.py] processing training_step {iteration=}')
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config)
        iteration += 1
        args.iteration = iteration
        if _XMOE_LOG_DW:
            _xmoe_dw_report(model[0], optimizer, _dw_snap, iteration)   # [dw] the update of this iteration, one line on rank 0
            _dw_snap = None
        if _res is not None:
            _xmoe_loss = None
            try:
                if loss_dict and 'lm loss' in loss_dict:
                    _lv = loss_dict['lm loss']
                    _xmoe_loss = float(_lv.item() if hasattr(_lv, 'item') else _lv)
                    _mul = _res.inject('loss', iteration - 1)   # nan: this rank's loss reads NaN for 3 steps
                    if _mul is not None and _mul != 1.0:
                        _xmoe_loss = _xmoe_loss * _mul
            except Exception:
                _xmoe_loss = None
            if _xmoe_loss is not None and os.environ.get('XMOE_DETECT') == '1':
                if _res.loss_rule(iteration, _xmoe_loss):     # D7: NaN or spike -> counted as skipped; 3 in succession -> exit 13
                    skipped_iter = 1
            _res.end_step(iteration, time.time() - _xmoe_t0, _xmoe_loss)
            _res.set_phase('step')
        if _route_on:
            _route_log.flush(iteration)   # the routing record: one host copy + one append per step

        # One-shot allocator snapshot after a few warm steps (standalone; no kineto needed).
        if profile_memory and not _mem_snapshot_done and iteration >= mem_snapshot_after_steps:
            from megatron.profiler_manager import dump_memory_snapshot
            dump_memory_snapshot()
            _mem_snapshot_done = True

        new_samples = mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()
        args.consumed_train_samples += new_samples
        # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
        args.actual_seq_length = args.seq_length
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            args.actual_seq_length = args.curriculum_seqlen
        if args.random_ltd:
            args.random_ltd_reserved_length = model[0].random_ltd_scheduler.get_current_seq()
            if args.random_ltd_reserved_length < args.actual_seq_length:
                args.actual_seq_length = (args.actual_seq_length * (args.num_layers - args.random_ltd_layer_num) + args.random_ltd_reserved_length * args.random_ltd_layer_num) // args.num_layers
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                act_mbsz = args.data_efficiency_curriculum_learning_numel / args.curriculum_seqlen
                act_token = act_mbsz * args.actual_seq_length
                args.consumed_train_tokens += mpu.get_data_parallel_world_size() * \
                        get_num_microbatches() * act_token
            else:
                args.consumed_train_tokens += new_samples * args.actual_seq_length
        else:
            args.consumed_train_tokens += new_samples * args.actual_seq_length
        
        # Logging.
        if args.deepspeed:
            if hasattr(model[0].optimizer, 'cur_scale'):
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = None
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad,
                                          model, optimizer)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, False)

        # Checkpointing
        _xmoe_wall_step_s = time.time() - _xmoe_wall_t0   # [wall-aware exit] this iteration's time without its save
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
                print_datetime('exiting program after receiving SIGTERM.')
                sys.exit()

        if args.save and ((args.save_interval and iteration % args.save_interval == 0)
                          or iteration in _XMOE_SAVE_STEPS):   # [XMOE_SAVE_STEPS, 2026-10-15] the listed iterations save too
            if iteration in _XMOE_SAVE_STEPS:
                print_rank_0(f'[XMOE_SAVE_STEPS] iteration {iteration} is in the listed save steps {sorted(_XMOE_SAVE_STEPS)}: saving')
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler)
            saved_checkpoint = True

        # [wall-aware exit, 2026-10-10] with XMOE_WALL_AWARE_EXIT=1 the decision below replaces the duration check: it exits
        # when one more step and its save would not fit before the job's Slurm end time, and never before the save is complete
        if _XMOE_WALL['on']:
            _xmoe_wall_decide(iteration, model, optimizer, opt_param_scheduler, saved_checkpoint, _xmoe_wall_step_s)

        # Exiting based on duration (not checked while the wall-aware exit is on: the wall rule takes precedence)
        if args.exit_duration_in_mins and not _XMOE_WALL['on']:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = get_accelerator().IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                # [2026-09-20] the same guard the exit-interval branch below carries: with --exit-duration-in-mins
                # reached and no --save (a probe that keeps no checkpoint), get_checkpoint_name(None) raised
                # TypeError on all ranks at exit (P3p 5517094)
                if args.save and not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()


    if _route_on:
        _route_log.close()   # the routing record: flush + fsync + close the per-rank file
    return iteration


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            if args.deepspeed and args.ds_pipeline_enabled:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{'lm loss' : loss}] * get_num_microbatches()
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if 'moe' not in key:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
            args.iteration + 1)
        if args.curriculum_seqlen < args.seq_length:
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    return total_loss_dict, collected_non_loss_data

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True, test=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            data_type = 'test' if test else 'validation'
            writer.add_scalar(f'lm-loss-validation/{key} {data_type}',
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar(f'lm-loss-validation/{key} {data_type} vs samples',
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            writer.add_scalar(f'lm-loss-validation/{key} {data_type} vs tokens',
                              total_loss_dict[key].item(),
                              args.consumed_train_tokens)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl', ppl,
                                  iteration)
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl vs samples',
                                  ppl, args.consumed_train_samples)
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl vs tokens',
                                  ppl, args.consumed_train_tokens)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    return build_train_valid_test_datasets_provider(train_val_test_num_samples)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    ds_sequence_parallel = mpu.get_sequence_parallel_world_size() > 1 or args.force_ds_sequence_parallel
    # [X-MoE 2026-09-17, C4] with a sequence degree the loader lives on the first rank of the TP x P block that shares a sample
    # (TP rank 0 and sequence rank 0): one builder per block, as the dataset's index-cache sync expects (gpt_dataset.py counts)
    rank_in_parallel_group = mpu.get_data_broadcast_rank() if ds_sequence_parallel else mpu.get_tensor_model_parallel_rank()
    if rank_in_parallel_group == 0:
        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = get_accelerator().LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = get_accelerator().LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if ds_sequence_parallel:
        torch.distributed.broadcast(flags,
                                    mpu.get_data_broadcast_src_rank(),
                                    group=mpu.get_data_broadcast_group())   # [X-MoE 2026-09-17, C4]
    else:
        torch.distributed.broadcast(flags,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
