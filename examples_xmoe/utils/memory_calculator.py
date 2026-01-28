import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import matplotlib.patches as mpatches
import math

class MemoryPredictor:
    """
    A class to predict and visualize GPU memory usage for training large-scale 
    Mixture-of-Experts (MoE) transformer models.
    """

    FP8_BYTES = 1
    FP16_BYTES = 2
    BF16_BYTES = 2
    FP32_BYTES = 4
    GIGA = 1024**3

    def __init__(self, config: Dict):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        required_keys = [
            'd_model', 'seqlen', 'pp_stages', 'num_layers', 'dp', 'ep', 
            'num_experts', 'expert_dim', 'mbs', 'gbs', 'topk', 'vocab_size',
            'attention_heads', 'tied_embedding'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: '{key}'")
        
        gbs = self.config['gbs']
        dp = self.config['dp']
        b = self.config['mbs']
        if gbs % (dp * b) != 0:
            raise ValueError(f"Global batch size (gbs={gbs}) must be divisible by dp*mbs ({dp*b}).")

        if 'pp_partitioning' in self.config:
            parts = self.config['pp_partitioning']
            if not isinstance(parts, list):
                 raise ValueError("pp_partitioning must be a list of integers.")
            if len(parts) != self.config['pp_stages']:
                raise ValueError(f"pp_partitioning length ({len(parts)}) must match pp_stages ({self.config['pp_stages']})")
            if sum(parts) != self.config['num_layers']:
                raise ValueError(f"Sum of pp_partitioning ({sum(parts)}) must match num_layers ({self.config['num_layers']})")
        
        # [NEW] Validate Dynamic Checkpointing
        if 'dynamic_checkpointing' in self.config:
            dyn_ckpt = self.config['dynamic_checkpointing']
            if not isinstance(dyn_ckpt, list):
                raise ValueError("dynamic_checkpointing must be a list of integers.")
            if len(dyn_ckpt) != self.config['pp_stages']:
                 raise ValueError(f"dynamic_checkpointing length ({len(dyn_ckpt)}) must match pp_stages ({self.config['pp_stages']})")
            # We cannot easily validate max layers here without duplicating partitioning logic, 
            # so w

    # --- Calculation Methods ---

    def _model_param_attention_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        return num_layers_in_stage * 4 * h**2 * self.FP16_BYTES

    def _model_grad_attention_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        return self._model_param_attention_per_gpu_stage(num_layers_in_stage)

    def _model_optim_attention_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        dp = self.config['dp']
        return 3 * num_layers_in_stage * 4 * h**2 * self.FP32_BYTES / dp

    def _attention_activation_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        s = self.config['seqlen']
        b = self.config['mbs']
        num_heads = self.config['attention_heads']
        
        input_act = self.FP16_BYTES * b * s * h
        qkv = self.FP16_BYTES * 3 * b * s * h
        softmax = self.FP16_BYTES * b * num_heads * s**2
        dropout = self.FP8_BYTES * b * s * h
        layernorm = self.FP16_BYTES * b * s * h
        out = self.FP16_BYTES * b * s * h
        return (input_act + qkv + out + layernorm) * num_layers_in_stage

    def _model_param_moe_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        num_experts = self.config['num_experts']
        expert_dim = self.config['expert_dim']
        ep = self.config['ep']
        
        router = self.FP16_BYTES * h * num_experts * num_layers_in_stage
        experts = self.FP16_BYTES * (num_experts / ep) * h * expert_dim * 2 * num_layers_in_stage
        return router + experts

    def _model_grad_moe_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        return self._model_param_moe_per_gpu_stage(num_layers_in_stage)

    def _model_optim_moe_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        num_experts = self.config['num_experts']
        expert_dim = self.config['expert_dim']
        ep = self.config['ep']
        
        router = 3 * self.FP32_BYTES * h * num_experts * num_layers_in_stage
        experts = 3 * self.FP32_BYTES * (num_experts / ep) * h * expert_dim * 2 * num_layers_in_stage
        return router + experts

    def _moe_activation_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        s = self.config['seqlen']
        b = self.config['mbs']
        dp = self.config['dp']
        ep = self.config['ep']
        num_experts = self.config['num_experts']
        expert_dim = self.config['expert_dim']
        topk = self.config['topk']
        
        # X-MoE's memory consumption (https://arxiv.org/pdf/2508.13337)
        input_act = self.FP16_BYTES * b * s * h
        softmax = self.FP32_BYTES * b * s * num_experts
        routing_map = self.FP16_BYTES * b * s * topk
        act_router = input_act + softmax + routing_map
        
        dispatch_buf = self.FP16_BYTES * b * s * h * topk 
        a2a_buf = 2 * dispatch_buf
        
        up_proj = self.FP16_BYTES * topk * b * s * expert_dim
        gelu = self.FP16_BYTES * topk * b * s * expert_dim
        down_proj = self.FP16_BYTES * topk * b * s * h
        expert_act = up_proj + gelu 
        
        layernorm = self.FP16_BYTES * b * s * h
        
        return (act_router + a2a_buf + expert_act + layernorm) * (dp / ep) * num_layers_in_stage

    def _embedding_param(self) -> float:
        return self.FP16_BYTES * self.config['vocab_size'] * self.config['d_model']

    def _embedding_grad(self) -> float:
        return self._embedding_param()

    def _embedding_optim(self) -> float:
        return self.FP32_BYTES * 2 * self.config['vocab_size'] * self.config['d_model']
        
    def _embedding_act(self) -> float:
        return self.FP16_BYTES * self.config['mbs'] * self.config['seqlen'] * self.config['d_model']

    def _output_layer_param(self) -> float:
        return 0 if self.config.get('tied_embedding') else self._embedding_param()
        
    def _output_layer_grad(self) -> float:
        return 0 if self.config.get('tied_embedding') else self._embedding_grad()
        
    def _output_layer_optim(self) -> float:
        return 0 if self.config.get('tied_embedding') else self._embedding_optim()

    def _output_layer_act(self) -> float:
        b, s, h, V = self.config['mbs'], self.config['seqlen'], self.config['d_model'], self.config['vocab_size']
        
        input_act = self.FP16_BYTES * b * s * h
        
        # [UPDATE] Loss Computation Memory Physics
        # 1. Logits (Forward): [B, S, V] in FP32 (usually cast to FP32 for stability)
        logits = self.FP32_BYTES * b * s * V
        
        # 2. Probs/Softmax (Forward/Backward): [B, S, V] in FP32
        probs = self.FP32_BYTES * b * s * V
        
        # 3. Gradients of Logits (Backward): [B, S, V]. 
        # Even if mixed precision, the gradient w.r.t logits is often kept in FP32 or BF16.
        # Let's assume BF16/FP16 minimum, effectively typically FP32 workspace.
        # Taking a conservative average: FP32.
        grad_logits = self.FP32_BYTES * b * s * V
        
        return input_act + logits + probs + grad_logits

    def _calculate_activation_memory(self, stage_id: int, num_layers_in_stage: float) -> Dict:
        pp_stages = self.config['pp_stages']
        is_interleaved = self.config.get('interleaved_pipeline', False)
        special_activations = (self._embedding_act() if stage_id == 0 else 0) + \
                              (self._output_layer_act() if stage_id == pp_stages - 1 else 0)

        # 1. Base Metrics (Total for the stage if everything was standard)
        full_attn_total = self._attention_activation_per_gpu_stage(num_layers_in_stage)
        full_moe_total = self._moe_activation_per_gpu_stage(num_layers_in_stage)
        
        # 2. Per-Layer Metrics (To mix and match)
        # Avoid division by zero if stage has 0 layers (rare but possible in some schedulers)
        denom = max(1, num_layers_in_stage)
        full_attn_per_layer = full_attn_total / denom
        full_moe_per_layer = full_moe_total / denom
        
        # Input Buffer (Size of saved input per layer)
        input_buffer_per_layer = self.FP16_BYTES * self.config['mbs'] * self.config['seqlen'] * self.config['d_model']

        # 3. Determine how many layers are Checkpointed vs Standard
        num_ckpt_layers = 0.0
        num_std_layers = num_layers_in_stage

        # [NEW] Dynamic Checkpointing Logic takes precedence
        if 'dynamic_checkpointing' in self.config:
            num_ckpt_layers = self.config['dynamic_checkpointing'][stage_id]
            # Safety check
            if num_ckpt_layers > num_layers_in_stage:
                print(f"Warning: Stage {stage_id} asks for {num_ckpt_layers} ckpt layers but only has {num_layers_in_stage}. Clamping.")
                num_ckpt_layers = num_layers_in_stage
            
            num_std_layers = num_layers_in_stage - num_ckpt_layers
        else:
            # Fallback to Global String Setting
            ckpt_setting = self.config.get('activation_checkpointing', 'full')
            if ckpt_setting in ['attention', 'layer', 'moe']:
                num_ckpt_layers = num_layers_in_stage
                num_std_layers = 0.0
        
        # 4. Calculate Effective Stored Memory (The "Stack")
        # Standard Layers pay Full Price. Checkpointed Layers pay Input Buffer Price.
        
        # Attention Cost
        eff_attn = (num_std_layers * full_attn_per_layer) + (num_ckpt_layers * input_buffer_per_layer)
        
        # MoE Cost 
        # Note: If ckpt_setting was 'attention' only, MoE acts might not be ckpt'd. 
        # But for 'dynamic_checkpointing' (list), we assume it applies 'layer' level (both attn+moe).
        # If using global string 'moe', only moe is ckptd. 
        # To keep this logic clean, if 'dynamic_checkpointing' is used, we treat it as 'layer' strategy.
        eff_moe = (num_std_layers * full_moe_per_layer) + (num_ckpt_layers * input_buffer_per_layer)

        # 5. Workbench Calculation
        # We only need a workbench if AT LEAST ONE layer is checkpointed.
        recompute_workbench = 0.0
        if num_ckpt_layers > 0:
            # Size of one single layer recomputation
            single_layer_max = max(full_attn_per_layer, full_moe_per_layer)
            # Physics: Act + Grad(Act)
            recompute_workbench = single_layer_max * 2.0
        
        effective_core_act_base = eff_attn + eff_moe
        
        if not is_interleaved:
            pipeline_multiplier = pp_stages - stage_id
            
            # Stored stack scales with pipeline depth
            stored_activations = effective_core_act_base * pipeline_multiplier
            
            # Workbench is transient
            total_activations = stored_activations + recompute_workbench
            
            metadata = {
                "pipeline_multiplier": pipeline_multiplier,
                "attention_act_base_gb": full_attn_total / self.GIGA,
                "moe_act_base_gb": full_moe_total / self.GIGA,
                "effective_attention_act_total_gb": (eff_attn * pipeline_multiplier) / self.GIGA,
                "effective_moe_act_total_gb": (eff_moe * pipeline_multiplier) / self.GIGA,
                "full_attention_act_total_gb": (full_attn_total * pipeline_multiplier) / self.GIGA,
                "full_moe_act_total_gb": (full_moe_total * pipeline_multiplier) / self.GIGA,
                "recompute_workbench_gb": recompute_workbench / self.GIGA,
                "ckpt_layers_count": num_ckpt_layers # Useful for debug
            }
            return {"pipelined_core": total_activations, "special": special_activations,
                    "metadata": metadata}
        else:
            stored_activations = effective_core_act_base * (pp_stages - stage_id)
            total_activations = stored_activations + recompute_workbench
            metadata = {"recompute_workbench_gb": recompute_workbench / self.GIGA} 
            return     
        
    def calculate_model_size_in_billions(self) -> float:
        config = self.config
        attn_params_per_layer = 4 * config['d_model']**2
        moe_params_per_layer = (
            (config['num_experts'] * 2 * config['d_model'] * config['expert_dim']) +
            (config['d_model'] * config['num_experts'])
        )
        core_params = config['num_layers'] * (attn_params_per_layer + moe_params_per_layer)
        embedding_params = config['vocab_size'] * config['d_model']
        output_params = 0 if config.get('tied_embedding', False) else embedding_params
        total_params = core_params + embedding_params + output_params
        return total_params / 1024**3
    
    def _generate_plot_title(self, base_title: str) -> str:
        model_size_b = self.calculate_model_size_in_billions()
        model_size_str = f"{model_size_b:.1f}B"

        if self.config.get('interleaved_pipeline', False):
            interleaving_type = self.config.get('interleaving_type', 'v_shape')
            schedule_str = "1F1B Interleaved"
        else:
            schedule_str = "1F1B"

        # [UPDATE] Title logic for dynamic vs global
        if 'dynamic_checkpointing' in self.config:
            ckpt_str = f"Dynamic {self.config['dynamic_checkpointing']}"
        else:
            ckpt_str = self.config.get('activation_checkpointing', 'full')
            if ckpt_str != 'full':
                ckpt_str = f"'{ckpt_str}'"
        
        schedule_str += f" w/ {ckpt_str} CKPT"
        
        if 'pp_partitioning' in self.config:
            part_str = str(self.config['pp_partitioning'])
        else:
            part_str = "Even"

        title = (
            f"{model_size_str} Model - {base_title}\n"
            f"DP={self.config['dp']}, EP={self.config['ep']}, PP={self.config['pp_stages']} ({part_str}) | Schedule: {schedule_str}"
        )
        return title

    def calculate_stage_memory(self, stage_id: int) -> Dict:
        pp_stages = self.config['pp_stages']
        
        if 'pp_partitioning' in self.config:
            layers_count = self.config['pp_partitioning'][stage_id]
        else:
            layers_count = self.config['num_layers'] / pp_stages

        param = self._model_param_attention_per_gpu_stage(layers_count) + self._model_param_moe_per_gpu_stage(layers_count)
        grad = self._model_grad_attention_per_gpu_stage(layers_count) + self._model_grad_moe_per_gpu_stage(layers_count)
        optim = self._model_optim_attention_per_gpu_stage(layers_count) + self._model_optim_moe_per_gpu_stage(layers_count)
        
        if stage_id == 0:
            param += self._embedding_param()
            grad += self._embedding_grad()
            optim += self._embedding_optim()
        if stage_id == pp_stages - 1:
            param += self._output_layer_param()
            grad += self._output_layer_grad()
            optim += self._output_layer_optim()
            
            optim += self._embedding_optim()
            
        activation_results = self._calculate_activation_memory(stage_id, layers_count)
        
        final_metadata = {"layers_per_stage": layers_count}
        final_metadata.update(activation_results["metadata"])

        eff_attn_act = final_metadata.get("effective_attention_act_total_gb", 0)
        eff_moe_act = final_metadata.get("effective_moe_act_total_gb", 0)
        recompute_wb = final_metadata.get("recompute_workbench_gb", 0)
        
        full_attn_act = final_metadata.get("full_attention_act_total_gb", eff_attn_act)
        full_moe_act = final_metadata.get("full_moe_act_total_gb", eff_moe_act)
        
        saved_attn = max(0, full_attn_act - eff_attn_act)
        saved_moe = max(0, full_moe_act - eff_moe_act)

        # Recomputation generally causes a small amount of fragmentation/overhead
        # because the allocator has to handle large transient spikes.
        # [UPDATE] Fragmentation estimation
        # If we have a large workbench (big swing), we assume ~30% overhead on that swing
        dynamic_frag = recompute_wb * 0.30 

        return {
            "Model Parameters": param / self.GIGA,
            "Gradients": grad / self.GIGA,
            "Optimizer States": optim / self.GIGA,
            "Attention Activations": eff_attn_act,
            "MoE Activations": eff_moe_act,
            "Recomputation Workbench": recompute_wb, 
            "Saved by Checkpointing (Attention)": saved_attn,
            "Saved by Checkpointing (MoE)": saved_moe,
            "Special Activations (Emb/Out)": activation_results["special"] / self.GIGA,
            "Communication Buffers": 0,
            "System Overhead": 7.0 + dynamic_frag,
            "_metadata": final_metadata
        }

    def generate_report(self):
        print("="*80)
        print("GPU Memory Prediction Report")
        
        # [UPDATE] Report Dynamic Config
        if 'dynamic_checkpointing' in self.config:
            print(f"Activation Checkpointing: Dynamic {self.config['dynamic_checkpointing']}")
        else:
            ckpt_str = self.config.get('activation_checkpointing', 'full')
            print(f"Activation Checkpointing: '{ckpt_str}'")
        
        if 'pp_partitioning' in self.config:
            print(f"Partitioning Strategy:    {self.config['pp_partitioning']}")
        else:
            print("Partitioning Strategy:    Even Split")
            
        print("="*80)

        for stage_id in range(self.config['pp_stages']):
            memory_breakdown = self.calculate_stage_memory(stage_id)
            metadata = memory_breakdown.pop("_metadata")
            total_memory = sum(v for k, v in memory_breakdown.items() if "Saved by" not in k)
            layers = metadata.get("layers_per_stage", "N/A")
            ckpt_count = metadata.get("ckpt_layers_count", "N/A")

            print(f"\n--- PP Stage {stage_id} (Layers: {layers} | Ckpt: {ckpt_count}) ---")
            print(f"  {'Model Parameters':<30}: {memory_breakdown['Model Parameters']:7.2f} GB")
            print(f"  {'Gradients':<30}: {memory_breakdown['Gradients']:7.2f} GB")
            print(f"  {'Optimizer States':<30}: {memory_breakdown['Optimizer States']:7.2f} GB")
            print(f"  Activations (Effective):")

            print(f"    {'Attention Blocks':<25}: {memory_breakdown['Attention Activations']:7.2f} GB")
            if memory_breakdown['Saved by Checkpointing (Attention)'] > 0:
                print(f"      (Saved: {memory_breakdown['Saved by Checkpointing (Attention)']:7.2f} GB)")
                
            print(f"    {'MoE Blocks':<25}: {memory_breakdown['MoE Activations']:7.2f} GB")
            if memory_breakdown['Saved by Checkpointing (MoE)'] > 0:
                print(f"      (Saved: {memory_breakdown['Saved by Checkpointing (MoE)']:7.2f} GB)")
            
            if memory_breakdown.get('Recomputation Workbench', 0) > 0:
                print(f"    {'Recomputation Workbench':<25}: {memory_breakdown['Recomputation Workbench']:7.2f} GB")

            print(f"    {'Embedding/Output':<25}: {memory_breakdown['Special Activations (Emb/Out)']:7.2f} GB")
            
            total_acts = (memory_breakdown['Attention Activations'] + memory_breakdown['MoE Activations'] + 
                          memory_breakdown['Special Activations (Emb/Out)'] + memory_breakdown.get('Recomputation Workbench', 0))
            print(f"    {'-- Total Activation --':<25}: {total_acts:7.2f} GB")
            
            print(f"  {'Communication Buffers':<30}: {memory_breakdown['Communication Buffers']:7.2f} GB")
            print(f"  {'System Overhead':<30}: {memory_breakdown['System Overhead']:7.2f} GB")
            print("-" * 75)
            print(f"  {'Total Estimated Memory':<30}: {total_memory:7.2f} GB")
        print("\n" + "="*80)

    
    def plot_memory_usage(self, memory_limit_gb=64):
        stage_ids = list(range(self.config['pp_stages']))
        first_stage_data = self.calculate_stage_memory(0)
        first_stage_data.pop("_metadata", None)
        all_components = list(first_stage_data.keys())
        memory_data = {key: [] for key in all_components}

        for stage_id in stage_ids:
            memory_breakdown = self.calculate_stage_memory(stage_id)
            for component, memory in memory_breakdown.items():
                if component in memory_data:
                    memory_data[component].append(memory)

        # [UPDATE] Filter out components with ~0 memory to clean up legend
        active_components = []
        for comp in all_components:
            if sum(memory_data[comp]) > 0.01: 
                active_components.append(comp)

        fig, ax = plt.subplots(figsize=(14, 9))
        colors = plt.cm.get_cmap('viridis', len(active_components))
        component_colors = {comp: colors(i) for i, comp in enumerate(active_components)}
        
        # [UPDATE] Assign specific color for workbench
        if "Recomputation Workbench" in component_colors:
            component_colors["Recomputation Workbench"] = "#FF6B6B"

        total_stack_height = np.zeros(len(stage_ids))
        
        for component in active_components:
            memories = memory_data[component]
            
            if "Saved by" in component:
                p = ax.bar(stage_ids, memories, label=component, bottom=total_stack_height, width=0.6,
                           color=component_colors[component], hatch='///', alpha=0.5, edgecolor='grey')
            elif component == "Recomputation Workbench":
                # [UPDATE] Distinct style for workbench
                p = ax.bar(stage_ids, memories, label=component, bottom=total_stack_height, width=0.6,
                           color=component_colors[component], hatch='..', edgecolor='black', linewidth=0.5)
                total_stack_height += np.array(memories)
                labels = [f'{mem:.1f}' if mem > 0.1 else '' for mem in memories]
                ax.bar_label(p, labels=labels, label_type='center', color='black', fontweight='bold', fontsize=8)
            else:
                p = ax.bar(stage_ids, memories, label=component, bottom=total_stack_height, width=0.6,
                           color=component_colors[component])
                total_stack_height += np.array(memories)
                labels = [f'{mem:.1f}' if mem > 1.0 else '' for mem in memories]
                ax.bar_label(p, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=9)

        ax.set_xlabel("Pipeline Stage", fontweight='bold')
        ax.set_ylabel("Memory (GB)", fontweight='bold')
        title = self._generate_plot_title("Predicted GPU Memory Usage per Pipeline Stage")
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        ax.set_xticks(stage_ids)
        if 'pp_partitioning' in self.config:
            parts = self.config['pp_partitioning']
            ax.set_xticklabels([f"Stage {i}\n({parts[i]}L)" for i in stage_ids])
        else:
            ax.set_xticklabels([f"Stage {i}" for i in stage_ids])
        
        if memory_limit_gb > 0:
            ax.axhline(y=memory_limit_gb, color='r', linestyle='--', linewidth=2, 
                       label=f'Hardware Limit ({memory_limit_gb}GB)')

        ax.legend(title="Memory Component", loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        effective_totals = np.zeros(len(stage_ids))
        for comp in active_components:
            if "Saved by" not in comp:
                effective_totals += np.array(memory_data[comp])

        for i, total in enumerate(effective_totals):
            ax.text(i, total + 0.5, f"Total: {total:.1f}", ha='center', fontweight='bold')

        ax.set_ylim(bottom=0, top=max(max(effective_totals)*1.2, memory_limit_gb * 1.1))
        
        plt.tight_layout()
        plt.show()

    def plot_memory_vs_mbs(self, mbs_values, memory_limit_gb=64):
        # NOTE: This plot ignores partitioning changes during the loop for simplicity
        # (It uses the currently set config)
        original_mbs = self.config['mbs']
        original_gbs = self.config['gbs']
        acc_steps = original_gbs // (self.config['dp'] * original_mbs)

        sample_calc = self.calculate_stage_memory(0)
        sample_calc.pop("_metadata", None)
        components = list(sample_calc.keys())
        memory_results = {}

        try:
            for mbs in mbs_values:
                self.config['mbs'] = mbs
                self.config['gbs'] = self.config['dp'] * mbs * acc_steps
                self._validate_config()
                
                memory_results[mbs] = {key: [] for key in components}
                for stage_id in range(self.config['pp_stages']):
                    mem_breakdown = self.calculate_stage_memory(stage_id)
                    for comp in components:
                        if "Saved by" not in comp:
                            memory_results[mbs][comp].append(mem_breakdown[comp])
        finally:
            self.config['mbs'] = original_mbs
            self.config['gbs'] = original_gbs

        num_mbs = len(mbs_values)
        num_stages = self.config['pp_stages']
        x = np.arange(num_mbs)
        width = 0.8 / num_stages

        fig, ax = plt.subplots(figsize=(18, 10))
        
        plot_components = [c for c in components if "Saved by" not in c]
        colors = plt.cm.get_cmap('viridis', len(plot_components))
        component_colors = {comp: colors(i) for i, comp in enumerate(plot_components)}
        
        total_bar_heights = np.zeros((num_mbs, num_stages))

        for stage_idx in range(num_stages):
            offset = width * (stage_idx - (num_stages - 1) / 2)
            bottoms_for_current_stage = np.zeros(num_mbs)

            for component in plot_components:
                mem_values = np.array([memory_results[mbs][component][stage_idx] for mbs in mbs_values])
                
                rects = ax.bar(x + offset, mem_values, width,
                               label=component if stage_idx == 0 else "",
                               bottom=bottoms_for_current_stage,
                               color=component_colors[component],
                               edgecolor='black',
                               linewidth=0.7)
                
                labels = [f'{val:.1f}' if val > 1.5 else '' for val in mem_values]
                ax.bar_label(rects, labels=labels, label_type='center',
                            color='white', fontsize=8, fontweight='bold')

                bottoms_for_current_stage += mem_values
            
            total_bar_heights[:, stage_idx] = bottoms_for_current_stage

        title = self._generate_plot_title("GPU Memory vs. Micro-Batch Size")
        ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel("Micro-Batch Size (mbs)", fontweight='bold', fontsize=12)
        ax.set_ylabel("Total Memory (GB)", fontweight='bold', fontsize=12)
        ax.set_xticks(x, mbs_values)

        for i in range(num_mbs):
            for stage_idx in range(num_stages):
                offset = width * (stage_idx - (num_stages - 1) / 2)
                ax.text(x[i] + offset, -0.05, f"S{stage_idx}",
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=10, color='dimgray')
        
        if memory_limit_gb > 0:
            ax.axhline(y=memory_limit_gb, color='r', linestyle='--', linewidth=2, 
                       label=f'Hardware Limit ({memory_limit_gb}GB)')

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0, top=max(np.max(total_bar_heights) * 1.1, memory_limit_gb * 1.1))
        
        ax.legend(title="Memory Component", loc='upper left', bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        plt.show()

    def find_best_parallelism_config(self, ep_size, max_gpus=128, mbs=1, memory_limit_gb=64):
        # NOTE: This function does not currently sweep pp_partitioning options.
        # It assumes default even splitting for the scan.
        valid_configs = []
        for pp in range(1, (max_gpus // ep_size) + 1):
            total_gpus = ep_size * pp
            if self.config['num_layers'] % pp == 0:
                valid_configs.append({'ep': ep_size, 'pp': pp, 'total_gpus': total_gpus})

        if not valid_configs:
            print(f"No valid EP/PP configurations found for EP={ep_size} and max_gpus={max_gpus} "
                f"with {self.config['num_layers']} layers.")
            return

        original_config = self.config.copy()
        all_results = {}

        sample_calc = self.calculate_stage_memory(0)
        sample_calc.pop("_metadata", None)
        components = [c for c in sample_calc.keys() if "Saved by" not in c]

        try:
            for config in valid_configs:
                ep, pp = config['ep'], config['pp']

                self.config['ep'] = ep
                self.config['pp_stages'] = pp
                self.config['dp'] = ep
                self.config['mbs'] = mbs
                # Ensure we reset partitioning if changing pp counts to avoid mismatch error
                if 'pp_partitioning' in self.config:
                    del self.config['pp_partitioning']

                acc_steps = original_config['gbs'] // (self.config['dp'] * self.config['mbs'])
                self.config['gbs'] = self.config['dp'] * self.config['mbs'] * acc_steps
                self._validate_config()

                config_key = f"EP={ep}, PP={pp}"
                all_results[config_key] = {comp: [] for comp in components}

                for stage_id in range(pp):
                    mem_breakdown = self.calculate_stage_memory(stage_id)
                    for comp in components:
                        all_results[config_key][comp].append(mem_breakdown[comp])

        finally:
            self.config = original_config

        num_configs = len(valid_configs)
        ncols = min(num_configs, 3)
        nrows = math.ceil(num_configs / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
        axes = axes.flatten()

        colors = plt.cm.get_cmap('viridis', len(components))
        component_colors = {comp: colors(i) for i, comp in enumerate(components)}

        for i, (config_key, memory_data) in enumerate(all_results.items()):
            ax = axes[i]
            num_stages = len(memory_data[components[0]])
            stage_ids = list(range(num_stages))

            bottom = np.zeros(num_stages)
            for component, memories in memory_data.items():
                p = ax.bar(stage_ids, memories, label=component, bottom=bottom, width=0.7, color=component_colors[component])
                bottom += np.array(memories)

            peak_memory = np.max(bottom)

            ax.set_title(f"{config_key} | Peak Memory: {peak_memory:.1f} GB", fontweight='bold')
            ax.set_xlabel("Pipeline Stage")
            ax.set_ylabel("Memory (GB)")
            ax.set_xticks(stage_ids)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            if memory_limit_gb > 0:
                ax.axhline(y=memory_limit_gb, color='r', linestyle='--', linewidth=2)

            ax.set_ylim(bottom=0, top=max(peak_memory * 1.15, memory_limit_gb * 1.15))

        for i in range(num_configs, len(axes)):
            axes[i].set_visible(False)

        handles = [mpatches.Patch(color=color, label=comp) for comp, color in component_colors.items()]
        fig.legend(handles=handles, title="Memory Component", loc='upper right', bbox_to_anchor=(0.98, 0.95))

        suptitle = self._generate_plot_title(f"Parallelism Options for EP={ep_size} (MBS={mbs})")
        fig.suptitle(suptitle, fontsize=20, fontweight='bold', y=1.0)

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        plt.show()
        
        
    # =========================================================================
    #  PORT FUNCTION FOR PLANNER
    # =========================================================================

    def query_stage_memory(self, stage_id: int, num_layers: int, num_ckpt_layers: int, micro_batch_size: int) -> float:
        """
        Calculates the estimated memory usage (GB) for a specific stage configuration.
        Designed to be called iteratively by an external Planner.
        """
        # 1. Snapshot State
        original_mbs = self.config.get('mbs')
        original_gbs = self.config.get('gbs')
        original_parts = self.config.get('pp_partitioning')
        original_dyn_ckpt = self.config.get('dynamic_checkpointing')

        try:
            # --- Sanity Check ---
            # If planner asks for more layers than the model has, it's impossible.
            if num_layers > self.config['num_layers']:
                return float('inf')

            # 2. Apply Simulation Overrides
            self.config['mbs'] = micro_batch_size
            # Dummy GBS to pass validation (must be divisible by mbs * dp)
            self.config['gbs'] = micro_batch_size * self.config['dp'] * 256 

            # --- Construct VALID Partitioning ---
            # We need a list that sums exactly to self.config['num_layers']
            # to satisfy strict validation if it were called.
            
            # Start with 0s
            temp_parts = [0] * self.config['pp_stages'] 
            
            # Set the requested stage
            temp_parts[stage_id] = num_layers
            
            # Dump the remaining layers into a different stage (e.g., the next one)
            # This ensures sum(temp_parts) == total_layers
            remainder = self.config['num_layers'] - num_layers
            if remainder > 0:
                # Find a neighbor index to dump the rest
                neighbor_idx = (stage_id + 1) % self.config['pp_stages']
                # If single stage pipeline, neighbor is self, which is fine (handled by sanity check above)
                temp_parts[neighbor_idx] += remainder

            self.config['pp_partitioning'] = temp_parts

            # --- Construct Dynamic Checkpointing ---
            # Just set the specific stage we care about. 
            temp_ckpt = [0] * self.config['pp_stages']
            temp_ckpt[stage_id] = num_ckpt_layers
            self.config['dynamic_checkpointing'] = temp_ckpt

            # 3. Run Calculation
            result = self.calculate_stage_memory(stage_id)

            # 4. Sum Total
            # Sum everything except the "Saved by" visualization keys and metadata
            total_gb = sum(v for k, v in result.items() if "Saved by" not in k and k != "_metadata")
            
            return total_gb

        except Exception as e:
            # print(f"Error in query: {e}") # Debugging
            return float('inf')

        finally:
            # 5. Restore State
            self.config['mbs'] = original_mbs
            self.config['gbs'] = original_gbs
            if original_parts is None: self.config.pop('pp_partitioning', None)
            else: self.config['pp_partitioning'] = original_parts
            
            if original_dyn_ckpt is None: self.config.pop('dynamic_checkpointing', None)
            else: self.config['dynamic_checkpointing'] = original_dyn_ckpt
        
# example config:
# config_60b_pp1_mbs1 = {'d_model': 5120, 
#     'seqlen': 4096, 
#     'pp_stages': 4, 
#     'num_layers': 32, 
#     'dp':8,
#     'ep':8,
#     'num_experts': 128, 
#     'expert_dim': 1536, 
#     'mbs': 1, 
#     'gbs': 32, 
#     'topk': 6, 
#     'vocab_size': 50257,
#     'attention_heads': 16,  
#     'tied_embedding': False, 
#     'moe-train-capacity-factor': 1.25,
#     'activation_checkpointing': 'none', 
#     'pp_partitioning': [7,8,8,9],
#     'dynamic_checkpointing': [4, 3, 0, 0]}
# predictor_v = MemoryPredictor(config_60b_pp1_mbs1)
# predictor_v.generate_report()
# predictor_v.plot_memory_usage()