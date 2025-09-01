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
    # Call the init process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    master_addr = args.master_addr

    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = "0"#str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    print("world_size, rank, master_addr, local_rank:", world_size, rank, master_addr, local_rank)
    using_mpi = torch.distributed.get_backend() == 'mpi'
    print("using_mpi=", using_mpi)

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
    else:
        get_env_variables(args)

    config = core_transformer_config_from_args(args)
    
    #CHANGED
    #with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
    print("#####")
    print(mpu.get_sequence_data_parallel_group())
    with deepspeed.zero.Init(sequence_data_parallel_group=mpu.get_sequence_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
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
    keys = ['text']
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
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
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
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    
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
    def _get_rank_world():
        # Works under DeepSpeed/Megatron once dist is initialized
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        # Fallback (single process)
        return 0, 1

    def trace_handler(prof):
        # _PROFILE_DIR = os.path.abspath("../scripts/torch_profile/ds_prof_yes_torch_tensor_v2")
        _PROFILE_DIR = os.path.abspath("../scripts/torch_profile/xmoe_prof_yes_torch_tensor_v3")
        rank, world = _get_rank_world()
        # Defensive: re-ensure dir exists inside handler (writer thread)
        try:
            os.makedirs(_PROFILE_DIR, exist_ok=True)
        except Exception:
            pass
        # Unique, stable filename per rank/step
        fname = os.path.join(
            _PROFILE_DIR,
            f"trace_rank{rank}_of_{world}_step{prof.step_num}.json"
        )
        # Print table to the rank’s stdout (optional)
        try:
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        except Exception:
            pass
        # Export trace
        prof.export_chrome_trace(fname)

    sched = schedule(wait=1, warmup=1, active=5, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        schedule=sched,
        record_shapes=True, 
        with_stack=True,
        with_flops=True,
        with_modules=True,
        profile_memory=True,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir')  # Save trace for TensorBoard
        on_trace_ready=trace_handler
    ) as prof:

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

        # Backward pass (in pretrain function, for example)
        prof.step()  # Record each iteration or step
    return output_tensor, loss
    """
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
    """


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
    
    # def _get_rank_world():
    #     # Works under DeepSpeed/Megatron once dist is initialized
    #     if dist.is_available() and dist.is_initialized():
    #         return dist.get_rank(), dist.get_world_size()
    #     # Fallback (single process)
    #     return 0, 1

    # def trace_handler(prof):
    #     _PROFILE_DIR = os.path.abspath("../scripts/torch_profile/xmoe_2_steps_prof_yes_torch_tensor_v2")
    #     rank, world = _get_rank_world()
    #     # Defensive: re-ensure dir exists inside handler (writer thread)
    #     try:
    #         os.makedirs(_PROFILE_DIR, exist_ok=True)
    #     except Exception:
    #         pass
    #     # Unique, stable filename per rank/step
    #     fname = os.path.join(
    #         _PROFILE_DIR,
    #         f"trace_rank{rank}_of_{world}_step{prof.step_num}.json"
    #     )
    #     # Print table to the rank’s stdout (optional)
    #     try:
    #         print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    #     except Exception:
    #         pass
    #     # Export trace
    #     prof.export_chrome_trace(fname)

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #     record_shapes=True, 
    #     with_stack=True,
    #     with_flops=False,
    #     with_modules=True,
    #     profile_memory=True,
    #     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir')  # Save trace for TensorBoard
    #     on_trace_ready=trace_handler
    # ) as prof:
    #     with record_function("begin training"):
    #         pretrain(train_valid_test_datasets_provider,
    #             model_provider,
    #             ModelType.encoder_or_decoder,
    #             forward_step,
    #             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    #             data_post_process=data_post_process)

        
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             data_post_process=data_post_process)
