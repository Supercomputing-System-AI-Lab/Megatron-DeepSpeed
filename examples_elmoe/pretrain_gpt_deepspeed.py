# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
import math
from functools import partial
import sys
sys.path.append('/lustre/orion/gen150/scratch/pinaster/smore/Megatron-DeepSpeed-tutel')
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group, update_rotary_pos_emb
from megatron.arguments import core_transformer_config_from_args
from megatron.model.transformer import ParallelMLP
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.distributed as dist

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import os
import subprocess

from torch import nn
import torch.nn.functional as F

import os
import datetime
from datetime import timedelta


from deepspeed.moe.layer import MoE


try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


def _maybe_init_wandb():
    """Opt-in wandb on the last rank only.

    Activates only when WANDB_PROJECT is set in the env (so a commented-out
    wandb block in the SLURM template means no wandb activity). Uses RANK /
    WORLD_SIZE env vars set by the SLURM template before pretrain runs to
    gate to the last rank without needing torch.distributed up.

    Once initialized here, the wandb.run.log() blocks inside megatron/training.py
    light up automatically (they're already gated on `wandb.run is not None`):
      - throughput keys (TFLOPs, samples_per_sec, ...) at line ~1060
      - iteration-line metrics + memory (lm_loss, learning_rate, grad_norm,
        nan iters, memory_*_gb, ...) right after print_rank_last

    Auth comes from WANDB_API_KEY env var or ~/.netrc (`wandb login`).
    """
    if wandb is None:
        return
    if not os.environ.get('WANDB_PROJECT'):
        return
    if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
        return
    # Read SLURM_PROCID/SLURM_NTASKS first: under srun these are the only env
    # vars guaranteed to be per-task. RANK/WORLD_SIZE in this template are
    # exported once at script level (where SLURM_PROCID=0) and then propagated
    # by --export=ALL, so every child task would see RANK=0 and the gate would
    # let every rank init -> N duplicate wandb runs. Fall back to RANK only
    # for non-SLURM launchers (torchrun etc.).
    rank = int(os.environ.get('SLURM_PROCID', os.environ.get('RANK', '0')))
    world = int(os.environ.get('SLURM_NTASKS', os.environ.get('WORLD_SIZE', '1')))
    if rank != world - 1:
        return
    try:
        wandb.init(
            project=os.environ['WANDB_PROJECT'],
            name=os.environ.get('WANDB_NAME'),
            entity=os.environ.get('WANDB_ENTITY'),
            group=os.environ.get('WANDB_GROUP'),
            dir=os.environ.get('WANDB_DIR'),
            mode=os.environ.get('WANDB_MODE', 'online'),
        )
        # Use Megatron's iteration counter as the x-axis for every chart.
        # Continuation runs land iter 1801..N (logged by the comprehensive
        # block in training.py) and overlay cleanly with the original run's
        # iter 0..1800 in the wandb UI — no per-run x-axis offset.
        wandb.define_metric('train/iteration')
        wandb.define_metric('*', step_metric='train/iteration')
        run = getattr(wandb, 'run', None)
        url = getattr(run, 'url', None) or getattr(run, 'dir', '?')
        print(f'[wandb] initialized "{run.name}" -> {url}', flush=True)
    except Exception as e:
        print(f'[wandb] init failed: {e}; continuing without wandb', flush=True)


master_port = "29500"
default_pg_timeout = timedelta(minutes=1)
# def setup_distributed_env(init_method=None, rank = 0, world_size=16):
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     world_size = comm.Get_size()
#     world_rank = rank = comm.Get_rank()
#     backend = None
#     os.environ['MASTER_ADDR'] = master_addr
#     os.environ['MASTER_PORT'] = master_port
#     os.environ['WORLD_SIZE'] = str(world_size)
#     os.environ['RANK'] = str(world_rank)
#     os.environ['LOCAL_RANK'] = "0"#str(world_rank % 8)
#     print("initialization parameters:", init_method, backend, rank, world_size)
#     torch.distributed.init_process_group(backend,
#                                         timeout=default_pg_timeout,
#                                         init_method=init_method,
#                                         rank=rank,
#                                         world_size=world_size)
#     using_mpi = torch.distributed.get_backend() == 'mpi'
#     print("using_mpi=", using_mpi)

def _set_env_variables(args):
    from mpi4py import MPI
    
    print (f'[pretrain_gpt_deepspeed.py] inside _set_env_variables')
    
    # Call the init process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    master_addr = args.master_addr
    
    

    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])
    
    # print (f'{rank=}, {comm=}, {world_size=}, {master_addr=}, {proc_name=}, {all_procs=}, {local_rank=}')
    
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # os.environ['LOCAL_RANK'] = "0"#str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    print("world_size, rank, master_addr, local_rank:", world_size, rank, master_addr, local_rank)
    using_mpi = torch.distributed.get_backend() == 'mpi'
    print("using_mpi=", using_mpi)
    
    # print (f'[pretrain_gpt_deepspeed.py] after _set_env_variables')

def get_env_variables(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print("world_size, rank, master_addr, master_port, local_rank:", world_size, rank, master_addr, master_port, local_rank)
    using_mpi = torch.distributed.get_backend() == 'mpi'
    print("using_mpi=", using_mpi)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # print("XXX", world_size)

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    if args.using_mpi:
        _set_env_variables(args)
        # print (f'[pretrain_gpt_deepspeed.py] commenting out _set_env_variables(args)')
    else:
        get_env_variables(args)

    config = core_transformer_config_from_args(args)
    
    
    # if os.getenv ('profile_memory') == 'True': 
    #     print (f'[pretrain_gpt_deepspeed.py] init_memory_logger() ')
    #     from profiling_utils.memory_profiler import init_memory_logger
    #     init_memory_logger() 
    # else: 
    #     print (f'[pretrain_gpt_deepspeed.py] DISABLED init_memory_logger() ')
    
    
    
    
    #CHANGED
    #with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
    print("#####")
    print(mpu.get_sequence_data_parallel_group())
    with deepspeed.zero.Init(sequence_data_parallel_group=mpu.get_sequence_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        
        
        if args.deepspeed and not args.no_pipeline_parallel:
            
            model = GPTModelPipe(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                # pre_process=pre_process,
                # post_process=post_process
            )
            model._megatron_batch_fn = get_batch_pipe
            
            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=get_accelerator().current_device_name())).view(
                    1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

            # For prertaining, since sequence length is fixed, cache rotary embedding in args, to avoid communicating around
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.seq_length)
            
        else: 
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
            


    '''
    model = GPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    '''
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'loss_mask'] if args.answer_loss_only else ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    # SFT: shift the answer mask [:, 1:] so it aligns with labels, not tokens.
    answer_mask = data_b['loss_mask'].float()[:, 1:].contiguous() \
        if args.answer_loss_only else None


    #ADDED
    # Get the masks and postition ids.
    skip_mask = args.use_flash_attn or args.use_flash_attn_triton
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        skip_mask)
    if answer_mask is not None:
        # Multiply, don't replace: composes with whatever the line above built.
        loss_mask = loss_mask * answer_mask
        if int(os.getenv('SFT_DEBUG_MASK', '0')):
            # gate 4: must match supervised_density_of_real x (1 - padding_waste)
            # from the corpus manifest
            print_rank_0('[sft] loss_mask density {:.4f}'.format(
                loss_mask.sum().item() / loss_mask.numel()))


    # For DS's sequence parallel
    seq_parallel_world_size = mpu.get_sequence_parallel_world_size()
    seq_parallel_world_rank = mpu.get_sequence_parallel_rank()

    # For Megatron's sequence parallel
    if args.sequence_parallel:
        seq_parallel_world_size = mpu.get_tensor_model_parallel_world_size()
        seq_parallel_world_rank = mpu.get_tensor_model_parallel_rank()
    seq_length = tokens.size(1)

    assert seq_length % seq_parallel_world_size == 0
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

    tokens = tokens[:, sub_seq_start:sub_seq_end]
    position_ids = position_ids[:, sub_seq_start:sub_seq_end]
    # For DS's sequence parallel
    if mpu.get_sequence_parallel_world_size() > 1:
        labels = labels[:, sub_seq_start:sub_seq_end]

    return tokens, labels, loss_mask, attention_mask, position_ids

def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            if current_seqlen < args.seq_length:
                data['text'] = data['text'][:, :(current_seqlen+1)].contiguous()
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.seq_length:
                orig_num_token = torch.numel(data['text'])
                reshape_len = (data['text'].size()[1] // (current_seqlen+1)) * (current_seqlen+1)
                data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen+1),
                    data['text'][:, -(current_seqlen+1):]), 0).contiguous()
                num_row = math.ceil(orig_num_token / (current_seqlen+1))
                num_row = min(num_row, data['text'].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data['text'] = data['text'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'loss_mask'] if args.answer_loss_only else ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    # SFT: shift the answer mask [:, 1:] so it aligns with labels, not tokens.
    answer_mask = data_b['loss_mask'].float()[:, 1:].contiguous() \
        if args.answer_loss_only else None

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if answer_mask is not None:
        # Multiply, don't replace: composes with whatever the line above built,
        # and must precede the curriculum truncation below.
        loss_mask = loss_mask * answer_mask
    if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    # clamp: under --answer-loss-only a microbatch can contain zero supervised
    # tokens (all-prompt); an unguarded denominator is a silent nan. Identity in
    # pretraining, where the mask is all ones.
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum().clamp(min=1)
    
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    if args.mos or args.kd:
        # assert max(args.num_experts) >= 1
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'mos loss': mos_loss}
        elif args.kd:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'kd loss': mos_loss}
        print_rank_0('>>> total loss: {}, lm loss {}, kd loss {}'.format(loss, averaged_loss[0], mos_loss))
    else:
        if max(args.num_experts) <= 1:
            return loss, {'lm loss': averaged_loss[0]}
        else:
            loss = loss + moe_loss
            return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}

def calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp
    
    if teacher_model:
        with torch.no_grad():
            if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
                assert args.curriculum_seqlen is not None
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
                # No need to truncate labels as we do not need it for the teacher logits
            tea_output, tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
            assert stu_output.size() == tea_output.size(), 'teacher and student output should match in size. Student: {}, Teacher: {}, CL seq length {}'.format(stu_output.size(), tea_output.size(), args.curriculum_seqlen)

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        tea_logits = F.softmax(tea_output / kd_temp, dim=2) # The target logits is expected to be probabilities. If we use log_softmax, then we need to set target_log to true when initializing the KLDivLoss.

        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)

        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss

# timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# ---------------------------
# Global profiler object
# ---------------------------
# ================================================ Original pretrain_gpt_deepspeed.py ==================================================
# _PROF = None

# def _get_rank_world():
#     # Works under DeepSpeed/Megatron once dist is initialized
#     if dist.is_available() and dist.is_initialized():
#         return dist.get_rank(), dist.get_world_size()
#     return 0, 1

# def _trace_handler(prof):
#     # out_dir = os.path.abspath("../scripts/torch_profile/xmoe_prof_yes_torch_tensor_v3")
#     out_dir = os.path.abspath("../scripts/torch_profile/ds_prof_no_torch_tensor_v6")
#     os.makedirs(out_dir, exist_ok=True)
#     r, w = _get_rank_world()
#     fname = os.path.join(out_dir, f"trace_rank{r}_of_{w}.json")

#     # optional: print summary to stdout
#     try:
#         print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
#     except Exception:
#         pass

#     # export *one* json per rank containing multiple steps
#     prof.export_chrome_trace(fname)

# def _get_profiler():
#     """Create the global profiler if not yet created."""
#     global _PROF
#     if _PROF is None:
#         sched = schedule(wait=1, warmup=1, active=5, repeat=1)
#         _PROF = profile(
#             activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#             schedule=sched,
#             record_shapes=True,
#             with_stack=True,
#             with_flops=True,
#             with_modules=True,
#             profile_memory=True,
#             on_trace_ready=_trace_handler,
#         )
#         _PROF.__enter__()   # manually enter context once
#     return _PROF

# ================================================ Copied from training.py ==================================================
_PROF = None


def _get_rank_world():
    # Works under DeepSpeed/Megatron once dist is initialized
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def _trace_handler(prof):
    # Fallback to '.' if 'profile_dir' environment variable is not set
    out_dir = os.path.abspath(os.getenv('profile_dir', '.'))
    os.makedirs(out_dir, exist_ok=True)
    r, w = _get_rank_world()
    
    # # 1. Safely export chrome trace
    # fname_json = os.path.join(out_dir, f"trace_rank{r}_of_{w}.json")
    # try:
    #     print(f"--- Profiler Exporting chrome trace for rank {r} ---")
    #     prof.export_chrome_trace(fname_json)
    # except Exception as e:
    #     print(f"[Rank {r}] Warning: Failed to export chrome trace: {e}")

    # # 2. Optional: print summary to stdout
    # try:
    #     print(f"--- Profiler Summary for Rank {r} ---")
    #     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # except Exception:
    #     pass

    # # 3. Safely export memory timeline
    # fname_html = os.path.join(out_dir, f"memory_trace_rank{r}_of_{w}.html")
    # try:
    #     # On ROCm, sometimes specifying the exact device (e.g., "cuda:0") helps,
    #     # but the primary fix is preventing the empty sequence error from crashing training.
    #     device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cuda"
    #     prof.export_memory_timeline(fname_html, device=device_str)
    # except ValueError:
    #     print(f"[Rank {r}] Warning: No memory events found to export timeline (ignoring empty sequence).")
    # except Exception as e:
    #     print(f"[Rank {r}] Warning: Failed to export memory timeline: {e}")
        
    # Memory pickle
    try:
        pickle_fname = os.path.join(out_dir, f"snapshot_rank{r}_of_{w}.pickle")
        torch.cuda.memory._dump_snapshot(pickle_fname)
        if r == 0:
            print(f"[training.py] Dumped memory snapshot to {pickle_fname}")
    except Exception as e:
        print(f"[Rank {r}] Warning: Failed to dump memory snapshot: {e}")

    # Finally, turn off the memory tracker AFTER dumping
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except AttributeError:
        pass


def _get_profiler(wait=0, warmup=0, active=2, repeat=1):
    """Create the global profiler if not yet created."""
    global _PROF
    if _PROF is None:
        print (f'[pretrain_gpt_deepspeed.py]: global _PROF is None, initiating a new one')
        sched = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        _PROF = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            profile_memory=True,
            on_trace_ready=_trace_handler,
        )
        _PROF.__enter__()   # manually enter context once
    else:
        print (f'[pretrain_gpt_deepspeed.py] loading pre-allocated global _PROF')
    return _PROF

# ================================================ Original pretrain_gpt_deepspeed.py finalize prof ==================================================
def finalize_profiler():
    """Call this once at the very end of training (after pretrain)."""
    rank, _ = _get_rank_world()
    
    print (f'[pretrain_gpt_deepspeed.py] {rank=} closing profiler')
    global _PROF
    if _PROF is not None:
        _PROF.__exit__(None, None, None)
        _PROF = None
        
    # # Ensure all ranks have finished writing their trace files
    # if dist.is_available() and dist.is_initialized():
    #     dist.barrier()

    # Have only rank 0 perform the merge
    
    if rank == 0:
        print (f'[pretrain_gpt_deepspeed.py] {rank=} merging traces', flush=True)
        merge_traces()
        
import json
import glob
import torch.distributed as dist

def merge_traces():
    """
    Merges multiple Chrome trace files from different ranks into a single file.
    This function should be called by only one rank after all ranks have
    finished profiling and saved their individual traces.
    """
    # Directory where individual rank traces are saved
    trace_dir = os.path.abspath(os.getenv ("profile_dir"))
    
    # Path for the final merged trace file
    merged_file_path = os.path.join (trace_dir, "merged.json")

    # Use glob to find all trace files. This is more robust than hardcoding names.
    trace_files = sorted(glob.glob(os.path.join(trace_dir, "trace_rank*.json")))

    if not trace_files:
        print("No trace files found to merge.")
        return

    print(f"Found {len(trace_files)} trace files to merge.")

    all_events = []
    
    # We will remap the original PIDs to new, unique PIDs
    # Let's just use the rank number as the new PID for simplicity and clarity.
    for rank, trace_file in enumerate(trace_files):
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # Original PIDs in this file. We need to track them to remap TIDs correctly.
        original_pids = set()

        # Add a metadata event to name this process in the Perfetto UI
        all_events.append({
            "name": "process_name", 
            "ph": "M", 
            "pid": rank,  # The new, unique PID
            "args": {"name": f"Rank {rank}"}
        })
        
        for event in data['traceEvents']:
            original_pid = event.get('pid')
            if original_pid is not None:
                original_pids.add(original_pid)

                # Overwrite the PID with our new unique rank-based PID
                event['pid'] = rank
                all_events.append(event)

    # Write the final merged file
    with open(merged_file_path, 'w') as f:
        json.dump({'traceEvents': all_events}, f)

    print(f"Successfully merged {len(trace_files)} traces into {merged_file_path}")


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # """
    # def trace_handler(prof):
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
    #     world_size = comm.Get_size()
    #     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    #     # prof.export_chrome_trace(f"/lustre/orion/gen150/scratch/pinaster/smore/layer_test/system-benchmark/tmp/test_trace_{rank}_of_{world_size}_" + str(prof.step_num) + ".json")
        
    #     base_dir = f"../scripts/torch_profile/prof_one"
    #     os.makedirs(base_dir, exist_ok=True)
    #     prof.export_chrome_trace (os.path.join(base_dir, f"trace_rank{rank}_of_{world_size}_step{prof.step_num}.json"))
    to_profile = os.getenv ("to_profile")
    # print (f'inside forward_step')
    # print (f'{to_profile=}')
    if to_profile=="True":
        # """
        # Profiler is owned solely by megatron/training.py (single owner). Here we only
        # add record_function labels -- they attach to that active profiler. Do NOT
        # create/step a profiler here (that was a second, separate profiler object).
        with record_function("get_batch"):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

        # Forward pass
        with record_function("forward_pass"):
            if args.mos or args.kd:
                stu_output, other_losses = model(tokens, position_ids, attention_mask)
                output_tensor = tensor_parallel.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
            else:
                output_tensor, other_losses = model(tokens, position_ids, attention_mask, labels=labels)

        # Loss calculation
        with record_function("loss_calculation"):
            moe_losses = [moe_loss for moe_loss in other_losses if moe_loss is not None]
            moe_loss = sum(moe_losses) * args.moe_loss_coeff

            mos_loss = 0
            if args.mos or args.kd:
                if args.teacher_forward and args.teacher_model is not None:
                    mos_loss = calculate_mos_loss(args, stu_output, args.teacher_model[0], tokens, position_ids, attention_mask)
            loss = partial(loss_func, loss_mask, moe_loss, mos_loss)
        return output_tensor, loss
    
    else: 
        # """
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

        if args.mos or args.kd:
            stu_output, other_losses = model(tokens, position_ids, attention_mask)
            output_tensor = tensor_parallel.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
        else:
            output_tensor, other_losses = model(tokens, position_ids, attention_mask, labels=labels)


        moe_losses = [moe_loss for moe_loss in other_losses if moe_loss is not None]
        moe_loss = sum(moe_losses) * args.moe_loss_coeff

        mos_loss = 0
        if args.mos or args.kd:
            if args.teacher_forward and args.teacher_model is not None:
                mos_loss = calculate_mos_loss(args, stu_output, args.teacher_model[0], tokens, position_ids, attention_mask)

        return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)
        # """


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')

def get_data(train_val_test_num_samples):
    import datasets
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")#("GPT2-XL/", local_files_only=True, truncation=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = datasets.load_from_disk("/lustre/orion/world-shared/stf218/sajal/owt-new/")
    dataset_dict = dataset.train_test_split(test_size=0.3)
    train_dataset, val_dataset = dataset_dict["train"], dataset_dict["test"]
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text']), batched=True)
    val_dataset = val_dataset.map(lambda examples: tokenizer(examples['text']), batched=True)

    return train_dataset, val_dataset, val_dataset

if __name__ == "__main__":
    git_ds_info()
    _maybe_init_wandb()    # opt-in via WANDB_PROJECT env var; runs once before pretrain

    import sys

    # print(f"[pretrained_gpt_deepspeed.py] Python executable: {sys.executable} \n Python version: {sys.version}", flush=True)
    # print("[pretrained_gpt_deepspeed.py] \nModule search paths:")
    # for path in sys.path:
    #     print(f"  {path}", flush=True)
    

    to_profile=os.getenv ("to_profile")
    print (f'{to_profile=}')
    
    if to_profile=='True':
        _exit_code = 0
        try:
            # The profiler is created/owned by megatron/training.py (sole owner), so it
            # starts at the training loop -- NOT during model build. Here we only wrap
            # pretrain and guarantee the trace is flushed on exit (safety net for an
            # early crash before training.py's own finalize fires).
            from megatron.profiler_manager import finalize_profiler

            with record_function("pretrain"):
                print (f'Trying to profile')
                pretrain(train_valid_test_datasets_provider,
                        model_provider,
                        ModelType.encoder_or_decoder,
                        forward_step,
                        args_defaults={},
                        data_post_process=data_post_process)
            print (f'pretrain_gpt_deepspee.py: after pretrain')
        except BaseException:
            # IMPORTANT: os._exit() below bypasses normal exception printing, so without
            # this the real error (e.g. a profiler crash in the first train_step) is
            # silently swallowed and the job exits 0. Print the traceback and fail loud.
            import traceback
            _exit_code = 1
            print("[pretrain_gpt_deepspeed.py] EXCEPTION during profiled pretrain "
                  "(see traceback below):", flush=True)
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
            print (f'[pretrain_gpt_deepspee.py] {rank=}, before finalize_profiler')
            finalize_profiler()

            # # 1. Ensure all GPUs finish writing their traces before anyone exits
            # if dist.is_initialized():
            #     print(f"[Rank {dist.get_rank()}] Waiting for all ranks to finish...", flush=True)
            #     dist.barrier()
            #     dist.destroy_process_group()

            print(f"[pretrained_gpt_deepspeed.py] Exiting (code={_exit_code}).", flush=True)

            # 2. Force the OS to terminate the process immediately
            os._exit(_exit_code)
    else: 
        pretrain(train_valid_test_datasets_provider,
                    model_provider,
                    ModelType.encoder_or_decoder,
                    forward_step,
                    args_defaults={},
                    data_post_process=data_post_process)

    
    # pretrain(train_valid_test_datasets_provider,
    #          model_provider,
    #          ModelType.encoder_or_decoder,
    #          forward_step,
    #          args_defaults={},
    #          data_post_process=data_post_process)
