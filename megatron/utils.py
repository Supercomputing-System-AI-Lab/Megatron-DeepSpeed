# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import sys
import os

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as torchDDP

from deepspeed.accelerator import get_accelerator
if get_accelerator().device_name() == 'cuda':
    from apex.multi_tensor_apply import multi_tensor_applier
    import amp_C

from megatron import (
    get_args,
    get_adlr_autoresume,
    get_num_microbatches
)
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.model.module import param_is_not_shared
from megatron.model.rotary_pos_embedding import RotaryEmbedding


def update_rotary_pos_emb(seq_length):
    args = get_args()
    rotary_dim = args.hidden_size // args.num_attention_heads \
        if args.kv_channels is None else args.kv_channels

    if args.rotary_percent < 1.0:
        rotary_dim = int(rotary_dim * args.rotary_percent)

    # partial rotary embeddings, which is better than full rotary
    # Wang and Komatsuzaki et al
    # https://github.com/kingoflolz/mesh-transformer-jax/
    rotary_pos_emb = RotaryEmbedding(rotary_dim, base=args.rotary_base)(
        seq_length).to(get_accelerator().current_device_name())
    args.rotary_pos_emb = rotary_pos_emb


def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = get_accelerator().IntTensor([0])
    
    if get_accelerator().device_name() == 'cuda':

        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm,
            dummy_overflow_buf,
            [params_data],
            False # no per-parameter norm
        )
    else :
        norm = torch.norm(params_data,p=2.0)
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])

    # print("Current rank:", torch.distributed.get_rank(),
    #       "Group world size:", torch.distributed.get_world_size(group=mpu.get_data_parallel_group()), 
    #       "My rank in group:", torch.distributed.get_rank(group=mpu.get_data_parallel_group()))
    
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    giga_bytes = 1024.0 * 1024.0 * 1024.0
    string = name + ' memory (GB)'
    string += ' | allocated: {}'.format(
        # get_accelerator().memory_allocated() / giga_bytes)
        torch.cuda.memory_allocated() / giga_bytes)
    string += ' | max allocated: {}'.format(
        # get_accelerator().max_memory_allocated() / giga_bytes)
        torch.cuda.max_memory_allocated() / giga_bytes)
    string += ' | reserved: {}'.format(
        # get_accelerator().memory_reserved() / giga_bytes)
        torch.cuda.memory_reserved() / giga_bytes)
    string += ' | max reserved: {}'.format(
        # get_accelerator().max_memory_reserved() / giga_bytes)
        torch.cuda.max_memory_reserved() / giga_bytes)
    # if mpu.get_data_parallel_rank() % 8 == 0:
    print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
            flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    skip_mask=False):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None
    if not skip_mask:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length))).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask and not skip_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    if not skip_mask:
        attention_mask = (attention_mask < 0.5)
        attention_mask = attention_mask.to(data.device)

    return attention_mask, loss_mask, position_ids


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ

def is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
            is_aml() and torch.distributed.get_rank() % get_accelerator().device_count() == 0
            ):
            return True
        else:
            return False
    else:
        return True

def get_parameters_in_billions(model):
    gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
                                        for model_module in model])

    return approx_parameters_in_billions*gpus_per_model/(1e9)

def throughput_calculator(model, args, iteration_time, total_iterations):
    batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
    approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
    elapsed_time_per_iter = iteration_time/total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    #flops calculator
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    head_dim = hidden_size // num_attention_heads
    ffn_hidden_size = args.ffn_hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size
    gqa = args.num_attention_heads // args.num_key_value_heads
    num_experts_routed_to = args.topk
    ffn_multiplier = 3 if args.swiglu else 2
    macs_per_flops = 2

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # correction has been made to TFLOPs formula due to incorrect behavior
    # observed with selective recompute when GQA not used and for all with GQA
    seq_len = args.seq_length
    if hasattr(args, 'actual_seq_length'):
        seq_len = args.actual_seq_length

    # 05/06/2026: Zixian: changed from (2 // gqa) --> (2 / gqa) to correctly account for GQA (q_heads / kv_heads)>2. 
    pre_and_post_mha_gemm_macs = batch_size * num_layers * (1 + (2 / gqa) + 1) * (hidden_size**2) * seq_len
    mha_bgemm_macs = batch_size * num_layers * 2 * head_dim * num_attention_heads * (seq_len**2)
    ffn_gemm_macs = batch_size * num_layers * ffn_multiplier * ffn_hidden_size * hidden_size * seq_len * num_experts_routed_to
    logit_lmhead_gemm_macs = batch_size * vocab_size * hidden_size * seq_len
    
    
    # 06/30/2026: Zixian: IMPLEMENTING SHARED EXPERT: 
    # num_shared_experts (the arg is a list; take a scalar).
    num_shared_experts = getattr(args, 'num_shared_experts', 0)
    if isinstance(num_shared_experts, (list, tuple)):
        num_shared_experts = num_shared_experts[0] if len(num_shared_experts) > 0 else 0

    # IMPLEMENTING SHARED EXPERT: a single wider SwiGLU MLP of width
    # (ffn_hidden_size * num_shared_experts) runs on EVERY token (no top-k), so its MACs
    # equal `num_shared_experts` experts of width ffn_hidden_size executed once per token.
    shared_ffn_gemm_macs = batch_size * num_layers * ffn_multiplier * ffn_hidden_size * hidden_size * seq_len * num_shared_experts

    if getattr(args, 'attention_variant', None) == 'deepseek_v3':
        # DeepSeek-V3 (2026-09-11): the latent attention's projections are not 4 h^2, its core uses two head widths, the
        # first layers are dense with their own width, the prediction head adds a layer and a second head, the V3.2
        # indexer and sparse core replace the core's key count; the standard terms above would overcount the projections
        # and undercount the core (deepseek_v3_macs_per_token holds the derivation; the test in the V3 scope checks it
        # against the toy model's own weights).
        c = deepseek_v3_macs_per_token(args, seq_len)
        tokens = batch_size * seq_len
        pre_and_post_mha_gemm_macs = c['attention_projections'] * tokens
        mha_bgemm_macs = c['attention_core'] * tokens
        ffn_gemm_macs = c['routed_and_dense_ffn'] * tokens
        shared_ffn_gemm_macs = c['shared_ffn'] * tokens
        logit_lmhead_gemm_macs = c['head'] * tokens
        extra_fwd_only_macs = c['forward_only'] * tokens
    else:
        extra_fwd_only_macs = 0
    fwd_macs = pre_and_post_mha_gemm_macs + mha_bgemm_macs + ffn_gemm_macs + logit_lmhead_gemm_macs + shared_ffn_gemm_macs
    bwd_macs = 2 * fwd_macs
    fwd_macs += extra_fwd_only_macs    # the V3.2 warm-up teacher: recomputed in the forward, detached, no backward
    fwd_bwd_macs = fwd_macs + bwd_macs

    # if (hasattr(args, 'checkpoint_activations') and args.checkpoint_activations) or (hasattr(args, 'recompute_granularity') and args.recompute_granularity == 'full'):
    #     fwd_bwd_macs += fwd_macs
    # if hasattr(args, 'recompute_granularity') and args.recompute_granularity == 'selective':
    #     fwd_bwd_macs += mha_bgemm_macs

    flops_per_iteration = fwd_bwd_macs * macs_per_flops
    tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    # print (f'[megatron/utils] {flops_per_iteration=}, {elapsed_time_per_iter=}, {args.world_size=}')
    return samples_per_second, tflops, approx_parameters_in_billions


def deepseek_v3_macs_per_token(args, seq_len):
    """The multiply-accumulates one token costs in the forward of a DeepSeek-V3 model built here, by part (2026-09-11).
    The attention core follows the convention of throughput_calculator (every query against every key of the sequence,
    the causal half not deducted); the sparse core of V3.2 counts its selected keys instead. Returns a dict whose values
    are MACs per token; the sum times the tokens of an iteration times 2 is the forward FLOPs."""
    h = args.hidden_size
    H = args.num_attention_heads
    L = args.num_layers
    q_lora = getattr(args, 'q_lora_rank', None)
    kv_lora = args.kv_lora_rank
    d_n, d_r, d_v = args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim
    d_qk = d_n + d_r
    # the latent attention's projections (megatron/model/deepseek_v3/attention.py)
    q_proj = (h * q_lora + q_lora * H * d_qk) if q_lora else h * H * d_qk
    attn_proj = q_proj + h * (kv_lora + d_r) + kv_lora * H * (d_n + d_v) + H * d_v * h
    core_dense = seq_len * H * (d_qk + d_v)
    # the V3.2 indexer and sparse core (indexer.py, sparse_core.py), when --index-topk > 0
    topk_keys = int(getattr(args, 'index_topk', 0) or 0)
    H_i, d_i = int(getattr(args, 'index_n_heads', 0) or 0), int(getattr(args, 'index_head_dim', 0) or 0)
    indexer = (q_lora * H_i * d_i + h * d_i + h * H_i + seq_len * H_i * d_i) if topk_keys > 0 else 0
    warmup = bool(getattr(args, 'dsa_dense_warmup', False))
    if topk_keys > 0 and not warmup:
        k_eff = min(topk_keys, seq_len)
        core = k_eff * H * (kv_lora + d_r) + k_eff * H * kv_lora + H * kv_lora * (d_n + d_v)   # logits, values, absorb + un-absorb
        forward_only = 0
    else:
        core = core_dense
        forward_only = (seq_len * H * d_qk) if topk_keys > 0 else 0     # the warm-up teacher: the dense scores once more
    # the feed-forward: the first k layers dense with their own width, then the routed top-k and the shared experts
    ffn_mult = 3 if args.swiglu else 2
    n_dense = int(getattr(args, 'first_k_dense_replace', 0) or 0)
    dense_ffn = int(getattr(args, 'dense_ffn_hidden_size', None) or args.ffn_hidden_size)
    n_experts = args.num_experts[0] if isinstance(args.num_experts, (list, tuple)) else args.num_experts
    n_shared = getattr(args, 'num_shared_experts', 0)
    n_shared = (n_shared[0] if n_shared else 0) if isinstance(n_shared, (list, tuple)) else (n_shared or 0)
    moe_layer_ffn = args.topk * ffn_mult * h * args.ffn_hidden_size + h * n_experts        # the experts and the gate
    dense_layer_ffn = ffn_mult * h * dense_ffn
    shared_layer_ffn = n_shared * ffn_mult * h * args.ffn_hidden_size
    n_moe = L - n_dense
    mtp = int(getattr(args, 'num_nextn_predict_layers', 0) or 0)
    head = h * args.padded_vocab_size
    parts = {
        'attention_projections': (L + mtp) * attn_proj,
        'attention_core': (L + mtp) * core + mtp * 0,
        'routed_and_dense_ffn': n_moe * moe_layer_ffn + n_dense * dense_layer_ffn + mtp * moe_layer_ffn,
        'shared_ffn': (n_moe + mtp) * shared_layer_ffn,
        'head': head * (1 + mtp) + mtp * 2 * h * h,                                            # the second head and eh_proj
        'indexer': (L + mtp) * indexer,
        'forward_only': (L + mtp) * forward_only,
    }
    parts['attention_projections'] += parts.pop('indexer')
    return parts


def checkpoint_throughput_calculator(model, latency_second):
    approx_parameters_in_billions = get_parameters_in_billions(model)
    checkpoint_multiplier = 14  # fp16 weights (2), fp32 weights (4), fp32 momentum (4), fp32 variance (4)
    checkpoint_GB = approx_parameters_in_billions * checkpoint_multiplier
    GB_per_second = checkpoint_GB / latency_second
    print_rank_0(f"Checkpoint Save GB: {round(checkpoint_GB, 3)}, GB/Sec: {round(GB_per_second,2)}, Latency(second): {round(latency_second, 3)}")


def get_fingerprint_header():
    return f"{'min':^13} {'max':^13} {'mean':^13} {'l2 norm':^12} metadata"

def get_fingerprint(p):
    return f"{p.min():13.6e} {p.max():13.6e} {p.mean():13.6e} {p.norm():12.6e}"


def dump_position_embed_weights(preamble, iteration, model):
    # return 
    from deepspeed.utils import safe_get_full_fp32_param
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    get_fingerprint_header()
    for n, p in model[0].named_parameters():
        if 'position_embeddings' in n:
            tag = "pos_embed"
        elif "word_embeddings" in n:
            tag = "word_embed"
        else:
            continue 
        print(f"iter {iteration} {preamble} {tag} lp {tp_rank}/{pp_rank}/{dp_rank}: {get_fingerprint(p)} {p.shape}\n")
        fp32_value = safe_get_full_fp32_param(p)
        if fp32_value is not None: 
            print(f"iter {iteration} {preamble} {tag} hp {tp_rank}/{pp_rank}/{dp_rank}: {get_fingerprint(fp32_value)} {p.shape}\n")

def dump_weights(preamble, iteration, model, optimizer, tensor=None):
    # return
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    fn = f"debug-bf16-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"

    # only care for first and last pp stages and dp0 tp0
    #if not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()):
    #    return

    #if not (tp_rank == 0 and dp_rank == 0):
    #    return

    if tensor is not None:
        orig_tensor = tensor
        if hasattr(tensor, "_hp_param"):
            numel = tensor._hp_param.numel() # // dp_size
            tensor = tensor.flatten().narrow(0, 0, numel)

    #print(fn)
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")

        if tensor is not None:
            fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
        else:
            for n, p in model[0].named_parameters():
                fh.write(f"{get_fingerprint(p)} {n} {p.shape}\n")


    return


    # until we figure out how to dump the actual fp32 values don't do this
    fn = f"debug-fp32-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")
        if tensor is not None:
            tensor = orig_tensor
            if hasattr(tensor, "_hp_param"):
                fh.write(f"{get_fingerprint(tensor._hp_param)} tensor {tensor._hp_param.shape}\n")
                #fh.write(f"{get_fingerprint(tensor._hp_grad)} tensor grad\n")
            else:
                fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
                #fh.write(f"{get_fingerprint(tensor.grad)} tensor grad\n")

        else:
            if hasattr(model[0].module.tied_modules, "embed"):
                p = model[0].module.tied_modules.embed.word_embeddings.weight._hp_param
                fh.write(f"{get_fingerprint(p)} module.tied_modules.embed.word_embeddings.weight._hp_param {p.shape}\n")

