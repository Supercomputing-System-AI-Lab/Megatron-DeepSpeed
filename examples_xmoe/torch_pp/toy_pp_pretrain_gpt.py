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
from torch.profiler import profile, record_function, ProfilerActivity

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import os
import subprocess

from torch import nn
import torch.nn.functional as F

import os
from datetime import timedelta


from deepspeed.moe.layer import MoE

master_port = "29500"
default_pg_timeout = timedelta(minutes=1)


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
    
    print (f'args generated in model_provider: \n{args}')
    print (f'config generated in model_provider: \n{config}')
    print (f'model generated in model_provider: \n{model}')
    
    # assert False 


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


def print_model_parameters(model, print_values=False, max_elements=10):
    """
    Print detailed information about model parameters
    
    Args:
        model: The PyTorch model
        print_values: Whether to print actual parameter values (default: False)
        max_elements: Maximum number of elements to print for each parameter
    """
    rank = int (os.getenv ('RANK'))
    print("="*80)
    print(f"{rank=} MODEL PARAMETER ANALYSIS")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            
        print(f"\n{rank=} Parameter: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Size: {param_count:,} elements")
        print(f"  Dtype: {param.dtype}")
        print(f"  Device: {param.device}")
        print(f"  Requires grad: {param.requires_grad}")
        
        if param.is_cuda:
            print(f" {rank=}  Memory usage: {param.element_size() * param_count / 1024**2:.2f} MB")
        
        # Print parameter statistics
        if param.numel() > 0:
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
        
        # Print actual values if requested
        if print_values and param.numel() > 0:
            if param.numel() <= max_elements:
                print(f"  Values: {param.flatten()}")
            else:
                print(f"  First {max_elements} values: {param.flatten()[:max_elements]}")
                print(f"  Last {max_elements} values: {param.flatten()[-max_elements:]}")
        
        print("-" * 40)
    
    print(f"\n{rank=} SUMMARY:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Total model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB")




def save_gradients_for_comparison(model, rank, step=0):
    """Save gradients from this rank for later comparison"""
    gradients = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Create a meaningful name that maps back to original model
            if rank == 1:
                # Rank 1: layers 0,1 in stage are actually layers 2,3 in full model
                if 'encoder.layers.0' in name:
                    mapped_name = name.replace('encoder.layers.0', 'encoder.layers.2')
                elif 'encoder.layers.1' in name:
                    mapped_name = name.replace('encoder.layers.1', 'encoder.layers.3')
                else:
                    # output_layer, final_layernorm, etc. keep as-is
                    mapped_name = name
            else:
                # Rank 0: keep names as-is (embedding, layers 0,1)
                mapped_name = name
            
            gradients[mapped_name] = param.grad.detach().cpu().clone()
    
    # Save to file
    torch.save(gradients, f'gradients_rank_{rank}_step_{step}.pt')
    print(f"[Rank {rank}] Saved {len(gradients)} gradients to file")
    
    return gradients


# Separate script for reference model (save as reference_model.py):
def create_reference_model():
    """Create full model on single GPU for comparison"""
    from megatron.initialize import initialize_megatron
    from megatron import get_args
    
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    args = get_args()
    args.model_type = "ModelType.encoder_or_decoder"
    
    # Create full model (your existing model_provider)
    model = model_provider()
    model.return_moe_loss = False
    
    # Keep all layers - don't delete anything
    # Set to handle full pipeline
    # model.language_model.pre_process = True
    # model.language_model.encoder.pre_process = True 
    # model.post_process = True
    # model.language_model.post_process = True
    # model.language_model.encoder.post_process = True
    
    # # Convert layer norms to FP16 (same as your pipeline model)
    # def convert_layer_norms_to_fp16(module):
    #     for name, child in module.named_children():
    #         if isinstance(child, (torch.nn.LayerNorm, torch.nn.GroupNorm)) or \
    #            'layernorm' in name.lower() or 'layer_norm' in name.lower():
    #             if hasattr(child, 'weight') and child.weight is not None:
    #                 child.weight.data = child.weight.data.to(torch.float16)
    #             if hasattr(child, 'bias') and child.bias is not None:
    #                 child.bias.data = child.bias.data.to(torch.float16)
    #         else:
    #             convert_layer_norms_to_fp16(child)
    
    convert_layer_norms_to_fp16(model)
    
    return model
from collections import OrderedDict

def save_model_for_pipeline_parallel(model, save_dir="./model_weights", model_name="gpt_model"):
    """
    Save model weights in a format that can be loaded for pipeline parallel execution.
    
    Args:
        model: The PyTorch model to save
        save_dir: Directory to save the weights
        model_name: Base name for the saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all model parameters
    state_dict = model.state_dict()
    
    # Save the complete model state dict
    complete_path = os.path.join(save_dir, f"{model_name}_complete.pt")
    torch.save(state_dict, complete_path)
    print(f"✅ Saved complete model to: {complete_path}")
    
    # Organize parameters by layer for easier pipeline parallel loading
    layer_weights = {}
    embedding_weights = {}
    other_weights = {}
    
    for param_name, param_value in state_dict.items():
        if 'embedding' in param_name:
            embedding_weights[param_name] = param_value
        elif 'encoder.layers.' in param_name:
            # Extract layer number from patterns like "language_model.encoder.layers.0.xxx"
            parts = param_name.split('.')
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            
            if layer_idx is not None:
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = {}
                layer_weights[layer_idx][param_name] = param_value
            else:
                other_weights[param_name] = param_value
        else:
            other_weights[param_name] = param_value
    
    # Save embedding weights (typically for rank 0)
    if embedding_weights:
        embedding_path = os.path.join(save_dir, f"{model_name}_embedding.pt")
        torch.save(embedding_weights, embedding_path)
        print(f"✅ Saved embedding weights to: {embedding_path}")
    
    # Save each layer separately
    for layer_idx, layer_params in layer_weights.items():
        layer_path = os.path.join(save_dir, f"{model_name}_layer_{layer_idx}.pt")
        torch.save(layer_params, layer_path)
        print(f"✅ Saved layer {layer_idx} weights to: {layer_path}")
    
    # Save other weights (final layer norm, output layers, etc.)
    if other_weights:
        other_path = os.path.join(save_dir, f"{model_name}_other.pt")
        torch.save(other_weights, other_path)
        print(f"✅ Saved other weights to: {other_path}")
    
    # Save metadata about the model structure
    metadata = {
        'num_layers': len(layer_weights),
        'layer_indices': list(layer_weights.keys()),
        'embedding_params': list(embedding_weights.keys()),
        'other_params': list(other_weights.keys()),
        'total_params': len(state_dict)
    }
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.pt")
    torch.save(metadata, metadata_path)
    print(f"✅ Saved metadata to: {metadata_path}")
    
    return {
        'complete_path': complete_path,
        'embedding_path': embedding_path if embedding_weights else None,
        'layer_paths': {idx: os.path.join(save_dir, f"{model_name}_layer_{idx}.pt") 
                       for idx in layer_weights.keys()},
        'other_path': other_path if other_weights else None,
        'metadata_path': metadata_path
    }


def load_model_parameters_directly(model, save_dir="./saved_weights", model_name="gpt_model", 
                                 rank=None, total_ranks=None, layer_mapping=None):
    """
    Alternative approach: Load parameters directly into the model without using load_state_dict
    """
    
    # Load metadata
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location='cpu')
        print(f"📋 Loaded metadata: {metadata}")
    else:
        print("⚠️  No metadata found")
        return {'loaded': False, 'error': 'No metadata found'}
    
    print(f"🔧 Loading for pipeline parallel: rank {rank}/{total_ranks}")
    
    # Determine layer mapping if not provided
    if layer_mapping is None and metadata:
        num_layers = metadata['num_layers']
        layers_per_rank = num_layers // total_ranks
        remainder = num_layers % total_ranks
        
        start_layer = rank * layers_per_rank + min(rank, remainder)
        end_layer = start_layer + layers_per_rank + (1 if rank < remainder else 0)
        
        layer_mapping = {}
        for i, orig_layer in enumerate(range(start_layer, end_layer)):
            layer_mapping[orig_layer] = i
        
        print(f"📊 Auto-generated layer mapping for rank {rank}: {layer_mapping}")
    
    # Get model parameters
    model_state_dict = model.state_dict()
    loaded_count = 0
    skipped_count = 0
    loaded_components = []
    
    # Load embedding weights for rank 0
    if rank == 0:
        embedding_path = os.path.join(save_dir, f"{model_name}_embedding.pt")
        if os.path.exists(embedding_path):
            embedding_weights = torch.load(embedding_path, map_location='cpu')
            for param_name, param_value in embedding_weights.items():
                if param_name in model_state_dict:
                    # Copy parameter data directly
                    with torch.no_grad():
                        model_state_dict[param_name].copy_(param_value)
                    loaded_count += 1
                    print(f"✅ Loaded: {param_name}")
                else:
                    skipped_count += 1
                    print(f"⚠️  Skipped (not in model): {param_name}")
            loaded_components.append("embedding")
    
    # Load layer weights based on mapping
    if layer_mapping:
        for orig_layer_idx, new_layer_idx in layer_mapping.items():
            layer_path = os.path.join(save_dir, f"{model_name}_layer_{orig_layer_idx}.pt")
            if os.path.exists(layer_path):
                layer_weights = torch.load(layer_path, map_location='cpu')
                
                for param_name, param_value in layer_weights.items():
                    # Transform parameter name
                    new_param_name = param_name.replace(f'encoder.layers.{orig_layer_idx}.', f'encoder.layers.{new_layer_idx}.')
                    
                    if new_param_name in model_state_dict:
                        # Copy parameter data directly
                        with torch.no_grad():
                            model_state_dict[new_param_name].copy_(param_value)
                        loaded_count += 1
                        print(f"✅ Loaded: {param_name} -> {new_param_name}")
                    else:
                        skipped_count += 1
                        print(f"⚠️  Skipped (not in model): {new_param_name}")
                
                loaded_components.append(f"layer_{orig_layer_idx}->layer_{new_layer_idx}")
    
    # Load other weights for last rank
    if rank == total_ranks - 1:
        other_path = os.path.join(save_dir, f"{model_name}_other.pt")
        if os.path.exists(other_path):
            other_weights = torch.load(other_path, map_location='cpu')
            for param_name, param_value in other_weights.items():
                if param_name in model_state_dict:
                    # Copy parameter data directly
                    with torch.no_grad():
                        model_state_dict[param_name].copy_(param_value)
                    loaded_count += 1
                    print(f"✅ Loaded: {param_name}")
                else:
                    skipped_count += 1
                    print(f"⚠️  Skipped (not in model): {param_name}")
            loaded_components.append("other")
    
    print(f"📦 Summary: Loaded {loaded_count} parameters, skipped {skipped_count}")
    
    return {
        'loaded': True,
        'rank': rank,
        'loaded_components': loaded_components,
        'loaded_params': loaded_count,
        'skipped_params': skipped_count,
        'layer_mapping': layer_mapping
    }


# Use this instead of load_model_for_pipeline_parallel
def safe_load_model_for_pipeline_parallel(model, save_dir="./saved_weights", model_name="gpt_model", 
                                        rank=None, total_ranks=None, layer_mapping=None):
    """
    Safe wrapper that tries the standard approach first, then falls back to direct loading
    """
    try:
        # Try the standard approach first
        result = load_model_for_pipeline_parallel(
            model, save_dir, model_name, rank, total_ranks, layer_mapping, 
            load_complete=False, strict=False
        )
        if result.get('loaded', False):
            return result
        else:
            print("🔄 Standard loading failed, trying direct parameter loading...")
    except Exception as e:
        print(f"🔄 Standard loading error: {e}, trying direct parameter loading...")
    
    # Fall back to direct parameter loading
    return load_model_parameters_directly(
        model, save_dir, model_name, rank, total_ranks, layer_mapping
    )    
    
def debug_parameter_matching(model, save_dir="./saved_weights", model_name="my_gpt", rank=0):
    """
    Debug helper to see what parameters are expected vs what's available
    """
    print(f"=== DEBUG PARAMETER MATCHING FOR RANK {rank} ===")
    
    # Get model parameters
    model_params = set(model.state_dict().keys())
    print(f"Model has {len(model_params)} parameters:")
    for i, param in enumerate(sorted(model_params)):
        print(f"  {i:2d}: {param}")
    
    print("\n" + "="*60)
    
    # Load saved parameters for this rank
    if rank == 0:
        # Embedding
        embedding_path = f"{save_dir}/{model_name}_embedding.pt"
        if os.path.exists(embedding_path):
            embedding_params = torch.load(embedding_path, map_location='cpu')
            print(f"Embedding parameters ({len(embedding_params)}):")
            for param in sorted(embedding_params.keys()):
                print(f"  📦 {param}")
        
        # Layers 0, 1
        for layer_idx in [0, 1]:
            layer_path = f"{save_dir}/{model_name}_layer_{layer_idx}.pt"
            if os.path.exists(layer_path):
                layer_params = torch.load(layer_path, map_location='cpu')
                print(f"\nLayer {layer_idx} parameters ({len(layer_params)}):")
                for param in sorted(layer_params.keys()):
                    # Show what it would be renamed to
                    new_name = param.replace(f'encoder.layers.{layer_idx}.', f'encoder.layers.{layer_idx}.')
                    match = "✅" if new_name in model_params else "❌"
                    print(f"  {match} {param} -> {new_name}")
    
    elif rank == 1:
        # Layers 2->0, 3->1
        for orig_layer, new_layer in [(2, 0), (3, 1)]:
            layer_path = f"{save_dir}/{model_name}_layer_{orig_layer}.pt"
            if os.path.exists(layer_path):
                layer_params = torch.load(layer_path, map_location='cpu')
                print(f"\nLayer {orig_layer}->{new_layer} parameters ({len(layer_params)}):")
                for param in sorted(layer_params.keys()):
                    # Show what it would be renamed to
                    new_name = param.replace(f'encoder.layers.{orig_layer}.', f'encoder.layers.{new_layer}.')
                    match = "✅" if new_name in model_params else "❌"
                    print(f"  {match} {param} -> {new_name}")
        
        # Other params
        other_path = f"{save_dir}/{model_name}_other.pt"
        if os.path.exists(other_path):
            other_params = torch.load(other_path, map_location='cpu')
            print(f"\nOther parameters ({len(other_params)}):")
            for param in sorted(other_params.keys()):
                match = "✅" if param in model_params else "❌"
                print(f"  {match} {param}")


def run_reference_model(x=None, y=None, position_ids=None, attention_mask=None):
    """Run reference model and save gradients"""
    
    stage_index = int (os.getenv ("RANK") )
    rank = stage_index
    num_stages = int (os.getenv ("WORLD_SIZE"))
    device = torch.device(f"cuda:{stage_index}")
    
    
    model = create_reference_model()
    
    # Saving model weights for PP2 to use 
    save_paths = save_model_for_pipeline_parallel(model, save_dir="./saved_weights", model_name="my_gpt")
    print("Save paths:", save_paths)
    
    # Print PP1 model weights 
    print_model_parameters (model, print_values=True)
    
    model.to(device)
    
    debug_layer_types (model, f"model-{rank}")
    debug_parallel_state()
    
    
    if (rank == 0): 
        # Same dummy data as your pipeline
        
        x = x.to(device)
        y = y.to(device) 
        position_ids = position_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        model.zero_grad()
        output = model(x, position_ids=position_ids, attention_mask=attention_mask, labels=y)
        
        print(f"[toy_pp_pretrain_gpt.py] {output.shape=}, {output=}")
        
        # Compute loss (same as your tokenwise_loss_fn)
        loss = output.mean()
        print(f"Reference model loss: {loss.item()}")
        
        # Backward pass
        loss.backward()
        
        # Save gradients
        reference_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                reference_grads[name] = param.grad.detach().cpu().clone()
        
        torch.save(reference_grads, 'gradients_reference.pt')
        print(f"Saved {len(reference_grads)} reference gradients")



# Comparison script (save as compare_gradients.py):
def compare_pipeline_vs_reference():
    """Load and compare gradients from pipeline vs reference"""
    
    # Load gradients
    ref_grads = torch.load('gradients_reference.pt')
    rank0_grads = torch.load('gradients_rank_0_step_0.pt') 
    rank1_grads = torch.load('gradients_rank_1_step_0.pt')
    
    # Combine pipeline gradients
    pipeline_grads = {**rank0_grads, **rank1_grads}
    
    print(f"Reference gradients: {len(ref_grads)} parameters")
    print(f"Pipeline gradients: {len(pipeline_grads)} parameters")
    
    # Compare each gradient
    matches = 0
    total = 0
    rtol, atol = 1e-5, 1e-8
    
    for name, ref_grad in ref_grads.items():
        if name in pipeline_grads:
            pipe_grad = pipeline_grads[name]
            
            if torch.allclose(ref_grad, pipe_grad, rtol=rtol, atol=atol):
                print(f"✓ {name}: MATCH")
                matches += 1
            else:
                max_diff = torch.max(torch.abs(ref_grad - pipe_grad)).item()
                print(f"✗ {name}: DIFFER (max diff: {max_diff:.2e})")
            total += 1
        else:
            print(f"✗ {name}: MISSING in pipeline")
            total += 1
    
    # Check for extra parameters in pipeline
    for name in pipeline_grads:
        if name not in ref_grads:
            print(f"✗ {name}: EXTRA in pipeline")
    
    print(f"\nResult: {matches}/{total} gradients match")
    return matches == total


# Convert only layer norm parameters to FP16 to match activations
def convert_layer_norms_to_fp16(module):
    print (f'[toy_pp_pretrain_gpt.py] converting layer_norms_to_fp16')
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.LayerNorm, torch.nn.GroupNorm)) or \
        'layernorm' in name.lower() or 'layer_norm' in name.lower():
            if hasattr(child, 'weight') and child.weight is not None:
                child.weight.data = child.weight.data.to(torch.float16)
            if hasattr(child, 'bias') and child.bias is not None:
                child.bias.data = child.bias.data.to(torch.float16)
        else:
            convert_layer_norms_to_fp16(child)
            

def debug_layer_types(model, name):
    """Debug what types of linear layers we have"""
    print (f'debug_layer_types debug_layer_types debug_layer_types')
    print(f"\n=== {name} LAYER TYPES ===")
    for layer_name, module in model.named_modules():
        if any(x in layer_name for x in ['dense', 'query_key_value', 'output_layer', 'word_embeddings']):
            print(f"  {layer_name}: {type(module).__name__}")
            if hasattr(module, 'world_size'):
                print(f"    world_size: {module.world_size}")
            if hasattr(module, 'tensor_model_parallel_size'):
                print(f"    tensor_model_parallel_size: {module.tensor_model_parallel_size}")
    print(f"=== END {name} ===\n")
    
def debug_parallel_state():
    """Debug parallel state settings"""
    print ('debug_parallel_state debug_parallel_state debug_parallel_state')
    try:
        import megatron.core.parallel_state as mpu
        print(f"Tensor model parallel size: {mpu.get_tensor_model_parallel_world_size()}")
        print(f"Pipeline model parallel size: {mpu.get_pipeline_model_parallel_world_size()}")
        print(f"Data parallel size: {mpu.get_data_parallel_world_size()}")
    except:
        print("Could not get parallel state info")




if __name__ == "__main__":
    # git_ds_info()
    # pretrain(train_valid_test_datasets_provider,
    #          model_provider,
    #          ModelType.encoder_or_decoder,
    #          forward_step,
    #          args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    #          data_post_process=data_post_process)
    
    torch.manual_seed(42) 
    
    
    x = torch.ones(12, 2048, dtype=torch.long) * 12
    # y = torch.randint(0, 1280, (12, 2048), dtype=torch.long)
    y = torch.ones (12, 2048, dtype=torch.long) * 13
    position_ids = torch.arange(0, 2048, dtype=torch.long).unsqueeze(0).expand(12, -1)
    attention_mask = torch.ones(12, 2048, dtype=torch.bool)
    
    
    if (os.getenv ('RUN_REF_DEBUG') == 'True'): 
        stage_index = int (os.getenv ("RANK") )
        rank = stage_index

        run_reference_model(x=x, y=y, position_ids=position_ids, attention_mask=attention_mask)
        
        if (rank == 0): 
            compare_pipeline_vs_reference()
        
        sys.exit(0)
    
    
    from megatron.initialize import initialize_megatron
    from megatron import get_args
    
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    
    args = get_args()
    args.model_type = "ModelType.encoder_or_decoder"
    
    print (f'[toy_pp_pretrain_gpt.py] model is:')
    model = model_provider()
    debug_parameter_matching(model, save_dir="./saved_weights", model_name="my_gpt", rank=torch.distributed.get_rank())
    # For 2 GPU, PP=2 case:
    # Load PP1 model weights 
    # On rank 0 (loads layers 0, 1 + embedding)
    if torch.distributed.get_rank() == 0:
        result = safe_load_model_for_pipeline_parallel(
            model, 
            save_dir="./saved_weights", 
            model_name="my_gpt",
            rank=0, 
            total_ranks=2
        )
        print("Rank 0 load result:", result)
    
    # On rank 1 (loads layers 2, 3 as layers 0, 1 + final norm)
    elif torch.distributed.get_rank() == 1:
        # If you want rank 1 to load original layers 2,3 as its layers 0,1:
        custom_mapping = {2: 0, 3: 1} 
        result = safe_load_model_for_pipeline_parallel(
            model, 
            save_dir="./saved_weights", 
            model_name="my_gpt",
            rank=1, 
            total_ranks=2,
            layer_mapping=custom_mapping
        )
        print("Rank 1 load result:", result)
    
    # from megatron.core.transformer.transformer_config import TransformerConfig
    # config = TransformerConfig(
    #     num_layers=4,
    #     hidden_size=2048,
    #     num_attention_heads=16,
    #     # optional: override the FFN size; if you omit, it will default to 4*hidden_size
    #     ffn_hidden_size=8192,
    #     # model parallel options
    #     tensor_model_parallel_size=1,
    #     pipeline_model_parallel_size=1,
    #     # precision options
    #     fp16=True,
    #     bf16=False,
    #     params_dtype=torch.float16,
    #     activation_func=F.gelu,
    #     init_method_std=0.014
    # )
    
    
    # model = GPTModel(
    #             config=config,
    #             num_tokentypes=0,
    #             parallel_output=True,
    #             pre_process=True,
    #             post_process=True
    #         )
    
    
    
    stage_index = int (os.getenv ("RANK") )
    rank = stage_index
    num_stages = int (os.getenv ("WORLD_SIZE"))
    device = torch.device(f"cuda:{stage_index}")
    
    print (f'[toy_pp_pretrain_gpt.py] rank of model: {os.getenv ("RANK")}')
    print (f'[toy_pp_pretrain_gpt.py] {model}')
    
    # print (f'[toy_pp_pretrain_gpt.py] rank of model: {os.getenv ("RANK")} model.language_model.encoder.layers[2]: \n{model.language_model.encoder.layers[2]}')
    
    
    # print (f'[toy_pp_pretrain_gpt.py] rank of model: {os.getenv ("RANK")} model.language_model.encoder.layers[3]: \n{model.language_model.encoder.layers[3]}')
    
    
    # Dummy data
    # x = torch.ones(12, 2048, dtype=torch.long) * 12
    # # y = torch.randint(0, 1280, (12, 2048), dtype=torch.long)
    # y = torch.ones (12, 2048, dtype=torch.long) * 13
    # # Create position_ids and attention_mask
    # position_ids = torch.arange(0, 2048, dtype=torch.long).unsqueeze(0).expand(12, -1)
    # attention_mask = torch.ones(12, 2048, dtype=torch.bool)  # All tokens are attended to
    

    num_microbatches = 2
    
    from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe

    
    
    print (f'[toy_pp_pretrain_gpt.py] model.language_model.embedding.word_embeddings: {model.language_model.embedding.word_embeddings}')
    print (f'[toy_pp_pretrain_gpt.py] {model.language_model.pre_process=}')
    print (f'[toy_pp_pretrain_gpt.py] {model.language_model.encoder.pre_process=}')
    print (f'[toy_pp_pretrain_gpt.py] {model.language_model.post_process=}')
    print (f'[toy_pp_pretrain_gpt.py] {model.language_model.encoder.post_process=}')
    
    model.return_moe_loss = False 
    
    def manual_model_split(model) -> PipelineStage:
        if stage_index == 0:
            # prepare the first stage model
            # for i in reversed (range(2, 4)):
            #     del model.language_model.encoder.layers[i]
            # model.language_model.encoder.final_layernorm = None
            del model.language_model.encoder.final_layernorm
            model.language_model.pre_process = True  
            model.language_model.encoder.pre_process = True 
            model.post_process = False 
            model.language_model.post_process = False 
            model.language_model.encoder.post_process = False 
            # model.output = None

        elif stage_index == 1:
            # prepare the second stage model
            # for i in reversed (range(2)):
            #     del model.language_model.encoder.layers[i]
            # model.language_model.embedding = None
            model.language_model.pre_process = False 
            model.language_model.encoder.pre_process = False 
            model.post_process = True 
            model.language_model.post_process = True 
            model.language_model.encoder.post_process = True 
            
        # model.language_model.encoder.num_layers = 2
        

        
        convert_layer_norms_to_fp16(model)

        print (f'[toy_pp_pretrain_gpt.py] {stage_index=}, {num_stages=}, {device=}')
        stage = PipelineStage(
            model,
            stage_index,
            num_stages,
            device,
        )
        return stage
    
    stage = manual_model_split (model)
    
    print_model_parameters (model, print_values=True)
    
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    debug_layer_types (stage.submod, f'rank-{rank}')
    debug_parallel_state ()
    
    print (f'[toy_pp_pretrain_gpt.py] rank of model: {os.getenv ("RANK")}, stage.submod: \n{stage.submod}')
    
    
    def tokenwise_loss_fn(outputs, targets):
        print (f'[toy_pp_pretrain_gpt.py] {os.getenv ("RANK")=}, {outputs[0].requires_grad=}, {outputs[0].shape=} {type (outputs)=} {outputs=}')
        
        loss_scalar = outputs[0].mean()
        # loss_scalar = outputs[0]
        print (f'[toy_pp_pretrain_gpt.py] {os.getenv ("RANK")=}, {loss_scalar=} ')
        return loss_scalar

    
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
    # schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches)
    
    
    if rank == 0:
        print (f'[toy_pp_pretrain_gpt.py] {rank=}, begin to process {x=} \n{position_ids=} \n{attention_mask=}')
        schedule.step(x, position_ids, attention_mask)
        
        # Save for grad comparison 
        save_gradients_for_comparison(stage.submod, rank)
    elif rank == 1:
        print (f'[toy_pp_pretrain_gpt.py] {rank=}, begin to process')
        losses = []
        # output = schedule.step(attention_mask=attention_mask, target=y, losses=losses)
        output = schedule.step(position_ids=position_ids, attention_mask=attention_mask, labels=y, target=y, losses=losses)
        print(f"[toy_pp_pretrain_gpt.py] {output.shape=}, {output=}")
        print(f"[toy_pp_pretrain_gpt.py] {rank=}, losses: {losses}")
        
        # Save for grad comparison 
        save_gradients_for_comparison(stage.submod, rank)

    
    
    
    # from torch.profiler import profile, ProfilerActivity
    # from contextlib import nullcontext

    # # Replace your final if/elif block with this exact code:
    # to_profile = os.getenv("to_profile", "False") == "True"
    # prof_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #                     on_trace_ready=lambda p: p.export_chrome_trace(f"rank_{rank}.json")) if to_profile else nullcontext()

    # with prof_context as prof:
        
        
    #     if rank == 0:
    #         print (f'[toy_pp_pretrain_gpt.py] {rank=}, begin to process {x=} \n{position_ids=} \n{attention_mask=}')
    #         schedule.step(x, position_ids, attention_mask)
    #     elif rank == 1:
    #         print (f'[toy_pp_pretrain_gpt.py] {rank=}, begin to process')
    #         losses = []
    #         # output = schedule.step(target=y, losses=losses)
    #         output = schedule.step(attention_mask=attention_mask, target=y, losses=losses)
    #         print(f"[toy_pp_pretrain_gpt.py] losses: {losses}")
            
    #     if to_profile and prof:
    #         prof.step()
    #         print(f"Profile saved to rank_{rank}.json")
    

        