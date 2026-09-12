import argparse
import sys

# --- CENTRAL CONFIGURATION ---
# This matches exactly what was in your SLURM template
#
# Required fields: h, ffn, heads, ep, k, s, layers.
# Optional architecture fields (omit to keep the historical launch behavior):
#   shared            -> --num-shared-experts N
#   first_k_dense     -> --first-k-dense-replace N   (dense MLP for the first N layers)
#   dense_ffn         -> --dense-ffn-hidden-size N   (width of those dense layers)
#   norm: "rmsnorm"   -> --normalization rmsnorm
#   norm_eps          -> --layernorm-epsilon E   (Megatron defaults to 1e-5)
#   bias: False       -> --disable-bias-linear
#   swiglu: True      -> --swiglu
#   untie_embeddings  -> --untie-embeddings-and-output-weights
#   tokenizer_type    -> --tokenizer-type T
#   eod_token         -> --eod-token S       (for tokenizers whose EOD isn't a builtin probe)
#   softmax_before_topk -> --softmax-before-topk  (routed weight = prob, not raw logit)
#
# NOTE: paths (vocab file, data, checkpoints) deliberately do NOT live here -- they are
# environment, not model identity, and keep this registry portable across machines.
# Entries without these fields emit an empty ARCH_FLAGS, so existing runs and their
# checkpoints are unaffected.
MODEL_SPECS = {
    "10B":    {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 2048, "layers": 24},
    "10B_k7":    {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 7, "s": 2048, "layers": 24},
    "10B_k8":    {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 8, "s": 2048, "layers": 24},
    "50B":    {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 24},
    "56B":    {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 27},
    "63B":    {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 32},
    "173B":   {"h": 7168, "ffn": 2048, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 24},
    "537B":   {"h": 7168, "ffn": 2560, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 60},
    "1T":     {"h": 7168, "ffn": 2560, "heads": 56, "ep": 512, "k": 8, "s": 4096, "layers": 60},
    
    # Mirrors HF deepseek-ai/deepseek-moe-16b-base so its weights can be converted
    # in and loaded. ffn == moe_intermediate_size (1408); layer 0 is a dense MLP of
    # width intermediate_size (10944) via first_k_dense/dense_ffn. vocab 102400.
    "DeepSeek_16B": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 4096, "layers": 28,
                     "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                     "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                     "untie_embeddings": True, "vocab": 102400,
                     "tokenizer_type": "HFTokenizer",
                     "eod_token": "<｜end▁of▁sentence｜>",
                     "softmax_before_topk": True},

    # deepseek-ai/DeepSeek-V3 (2026-09-12, the milestone's second stage): the released 671B model, every value from its config.json
    # (hidden 7168, 61 layers, 128 heads, q_lora 1536, kv_lora 512, 256 routed + 1 shared
    # experts, top 8, 8 groups of which 4, scaling 2.5, dense first 3 layers of 18432, rope theta
    # 10000 with YaRN factor 40 over 4096, eps 1e-06, vocabulary 129280); the prediction head's
    # layer (num_nextn_predict_layers 1) is left out of the forward comparison (0 here; transformers' class skips it too).
    # The weights are fp8 block-quantized in the release: the converter dequantizes them to bf16 atoms. The run tokenizes nothing
    # (XMOE_NULL_TOKENIZER=1, --vocab-size 129279); a forward-only pipeline (PP 8, EP 8, 8 nodes) holds the 1.37 TB of bf16 weights.
    "DeepSeek_V3_671B": {"h": 7168, "ffn": 2048, "heads": 128, "ep": 256, "k": 8, "s": 8192, "layers": 61,
                "shared": 1, "first_k_dense": 3, "dense_ffn": 18432, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-06, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 129280,
                "tokenizer_type": "NullTokenizer", "null_vocab": 129279, "rotary_base": 10000,
                "v3": {"q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0,
                       "num_nextn_predict_layers": 0,
                       "rope_scaling": {"factor": 40, "original_max_position_embeddings": 4096, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}}},
    # moonshotai/Moonlight-16B-A3B (2026-09-12, the HF comparison milestone): the released DeepSeek-V3 architecture at 16B, every
    # value from its config.json (hidden 2048, 27 layers, 16 heads, kv_lora 512 and no q_lora, 64 routed + 2 shared experts, top 6,
    # sigmoid gate with the bias, one group, scaling 2.446, dense first layer 11264, rope theta
    # 50000, eps 1e-05, vocabulary 163840). The run tokenizes nothing (NullTokenizer sized to the
    # embedding's 163840 rows: --vocab-size 163839, EOD = 163839); the prompts arrive as token ids
    # through the XMOE_FIXED_TOKENS rider. Weights: tools/convert_checkpoint/hf_to_ds_moe.py --emit universal.
    "Moonlight_16B_A3B": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 8192, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 11264, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-05, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 163840,
                "tokenizer_type": "NullTokenizer", "null_vocab": 163839, "rotary_base": 50000,
                "v3": {"q_lora_rank": None, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.446, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0,
                       "num_nextn_predict_layers": 0, "rope_scaling": None}},
    # DeepSeek-V3 in the 16B shape (blueprint decision V1): DeepSeek-V2-Lite's outer shape with V3's attention and router fields.
    # The "v3" sub-dict uses the released config.json keys; get_v3_flags renders it. rope_scaling None until an extension stage
    # (CS4) sets {"factor": 40, "original_max_position_embeddings": 4096, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}.
    # num_nextn_predict_layers stays 0 until the prediction head (CS5) is built.
    "V3S_16B": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0,   # the bias alone (smokes A-H); V3S_16B_SEQ adds the loss
                       "num_nextn_predict_layers": 0, "rope_scaling": None}},
    # V3S_16B plus the report's complementary sequence-wise balance loss (decision V3, alpha 0.0001; the batch-wise loss and
    # --moe-loss-coeff are replaced by it). Under pipeline parallelism the loss reaches the total only with PIPE_MOE_AUX=1 in the
    # run's environment (the template's --pipe-moe-aux-loss); the argument validation refuses the run otherwise.
    "V3S_16B_SEQ": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0001,
                       "num_nextn_predict_layers": 0, "rope_scaling": None}},
    # V3S_16B plus the prediction head (blueprint CS5, decision V4): depth 1, lambda 0.3 (the report's first phase); the head lives on
    # the pipeline path (PP > 1 or --no-pipeline-parallel off) with the untied head. V3S_16B_MTP0 is the probe with lambda 0: the head
    # is built and run but adds nothing to the loss, so 'lm main loss' must follow V3S_16B's curve.
    "V3S_16B_MTP": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0,
                       "num_nextn_predict_layers": 1, "mtp_loss_scaling_factor": 0.3, "rope_scaling": None}},
    "V3S_16B_MTP0": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0,
                       "num_nextn_predict_layers": 1, "mtp_loss_scaling_factor": 0.0, "rope_scaling": None}},
    # V3.2 (blueprint CS7): V3S_16B plus the lightning indexer (16 heads of 128, top-2,048 keys; decision V7) in the report's two
    # stages. _DSA_WARMUP: the attention stays dense, only the indexer trains by its loss (the freeze); _DSA: the sparse core and the
    # sparse-stage loss with everything trainable. Both carry the V3 recipe's sequence-wise balance loss (alpha 0.0001; run L2 showed the
    # batch-wise loss on sigmoid affinities at 32 per layer) and need PIPE_MOE_AUX=1 (the indexer loss rides the pipe beside it).
    "V3S_16B_DSA_WARMUP": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0001,
                       "num_nextn_predict_layers": 0, "index_n_heads": 16, "index_head_dim": 128, "index_topk": 2048,
                       "indexer_loss_coeff": 1.0, "dsa_dense_warmup": True, "dsa_freeze_non_indexer": True, "rope_scaling": None}},
    "V3S_16B_DSA": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0001,
                       "num_nextn_predict_layers": 0, "index_n_heads": 16, "index_head_dim": 128, "index_topk": 2048,
                       "indexer_loss_coeff": 1.0, "rope_scaling": None}},
    # The first smoke of the latent attention alone: V3S_16B's attention fields with today's gate (softmax before top-k, as DeepSeek_16B).
    "V3S_16B_MLA": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "softmax_before_topk": True,
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "num_nextn_predict_layers": 0, "rope_scaling": None}},
    # The YaRN extension smoke (CS4): V3S_16B at 8,192 tokens with the rotary table scaled by 2 from an original 4,096 (V3's mscale 1.0).
    "V3S_16B_YARN8K": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 8192, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
                       "routed_scaling_factor": 2.5, "moe_router_bias_update_speed": 0.001, "moe_seq_balance_coeff": 0.0,
                       "num_nextn_predict_layers": 0,
                       "rope_scaling": {"factor": 2, "original_max_position_embeddings": 4096, "beta_fast": 32, "beta_slow": 1,
                                        "mscale": 1.0, "mscale_all_dim": 1.0}}},
    # DeepSeek-V2-Lite as released (config_deepseek_v2_lite.json): the import probe of the latent attention (no query latent, softmax router).
    "DeepSeek_V2_Lite": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64, "k": 6, "s": 4096, "layers": 27,
                "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                "untie_embeddings": True, "vocab": 102400,
                "tokenizer_type": "HFTokenizer",
                "eod_token": "<｜end▁of▁sentence｜>",
                "v3": {"q_lora_rank": None, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                       "scoring_func": "softmax", "topk_method": "greedy", "n_group": 1, "topk_group": 1, "norm_topk_prob": False,
                       "routed_scaling_factor": 1.0, "moe_router_bias_update_speed": 0.0, "num_nextn_predict_layers": 0,
                       "rope_scaling": {"factor": 40, "original_max_position_embeddings": 4096, "beta_fast": 32, "beta_slow": 1,
                                        "mscale": 0.707, "mscale_all_dim": 0.707}}},
    
    "DeepSeek_16B_8k": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 8192, "layers": 28,
                         "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                         "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                         "untie_embeddings": True, "vocab": 102400,
                         "tokenizer_type": "HFTokenizer",
                         "eod_token": "<｜end▁of▁sentence｜>",
                         "softmax_before_topk": True},
    "DeepSeek_16B_12k": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 12288, "layers": 28,
                         "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                         "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                         "untie_embeddings": True, "vocab": 102400,
                         "tokenizer_type": "HFTokenizer",
                         "eod_token": "<｜end▁of▁sentence｜>",
                         "softmax_before_topk": True},
    "DeepSeek_16B_16k": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 16384, "layers": 28,
                         "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                         "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                         "untie_embeddings": True, "vocab": 102400,
                         "tokenizer_type": "HFTokenizer",
                         "eod_token": "<｜end▁of▁sentence｜>",
                         "softmax_before_topk": True},
    "DeepSeek_16B_32k": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 32768, "layers": 28,
                         "shared": 2, "first_k_dense": 1, "dense_ffn": 10944, "expert_interval": 1,
                         "norm": "rmsnorm", "norm_eps": 1e-6, "bias": False, "swiglu": True,
                         "untie_embeddings": True, "vocab": 102400,
                         "tokenizer_type": "HFTokenizer",
                         "eod_token": "<｜end▁of▁sentence｜>",
                         "softmax_before_topk": True},
    

    "FORGE_10B":    {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 4096, "layers": 24},
    "Llama31_8B":    {"h": 4096, "ffn": 14336, "heads": 32, "kv_heads": 8,"ep": 1, "k": 1, "s": 4096, "layers": 32},
    "65B_Dense":    {"h": 8192, "ffn": 32768, "heads": 64, "ep": 1, "k": 1, "s": 2048, "layers": 80},
    "63B_Dense":    {"h": 4096, "ffn": 16384, "heads": 32, "ep": 1, "k": 1, "s": 4096, "layers": 32},
    "63B_Coarse":    {"h": 4096, "ffn": 8192, "heads": 32, "ep": 8, "k": 2, "s": 4096, "layers": 32},
    "63B_Sparse":    {"h": 4096, "ffn": 1024, "heads": 32, "ep": 64, "k": 16, "s": 4096, "layers": 32},
    "63B_Sparse_16L":    {"h": 4096, "ffn": 1024, "heads": 32, "ep": 64, "k": 16, "s": 4096, "layers": 16},
    "63B_Sparse_24L":    {"h": 4096, "ffn": 1024, "heads": 32, "ep": 64, "k": 16, "s": 4096, "layers": 24},
    "63B_Sparse_40L":    {"h": 4096, "ffn": 1024, "heads": 32, "ep": 64, "k": 16, "s": 4096, "layers": 40},
    
    # Planner/Profiling specific (1 Layer variants)
    "10B_1L": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 2048, "layers": 1},
    "50B_1L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 1},
    "63B_1L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 1},
    "173B_1L":{"h": 7168, "ffn": 2048, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 1},
    "537B_1L":{"h": 7168, "ffn": 2560, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 1},
    "1T_1L":  {"h": 7168, "ffn": 2560, "heads": 56, "ep": 512, "k": 8, "s": 4096, "layers": 1},
    
    "1.5T_1L":  {"h": 7168, "ffn": 256, "heads": 56, "ep": 1024, "k": 8, "s": 4096, "layers": 1},
    
    
    "10B_2L": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 2048, "layers": 2},
    "50B_2L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 2},
    "63B_2L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 2},
    "173B_2L":{"h": 7168, "ffn": 2048, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 2},
    "537B_2L":{"h": 7168, "ffn": 2560, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 2},
    "1T_2L":  {"h": 7168, "ffn": 2560, "heads": 56, "ep": 512, "k": 8, "s": 4096, "layers": 2},
    
    # Planner/Profiling specific (1 Layer variants)
    "10B_4L": {"h": 2048, "ffn": 1408, "heads": 16, "ep": 64,  "k": 6, "s": 2048, "layers": 4},
    "50B_4L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 4},
    "63B_4L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 4},
    "173B_4L":{"h": 7168, "ffn": 2048, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 4},
    "537B_4L":{"h": 7168, "ffn": 2560, "heads": 56, "ep": 256, "k": 8, "s": 4096, "layers": 4},
    "1T_4L":  {"h": 7168, "ffn": 2560, "heads": 56, "ep": 512, "k": 8, "s": 4096, "layers": 4},
    
    "63B_8L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 8},
    "63B_16L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 16},
    "63B_30L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 30},
    "63B_24L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 24},
    "63B_20L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 20},
    "63B_36L": {"h": 5120, "ffn": 1536, "heads": 40, "ep": 128, "k": 6, "s": 4096, "layers": 36},
    
}

def get_spec(model_size):
    if model_size not in MODEL_SPECS:
        raise ValueError(f"Unknown model size: {model_size}")
    return MODEL_SPECS[model_size]

def get_v3_flags(v3):
    """The DeepSeek-V3 flags of a spec's "v3" sub-dict: one flag per key (config.json key with dashes), None skipped, True as a
    bare flag, the rope_scaling sub-dict flattened to the --yarn-* flags. The variant switch is always emitted."""
    flags = ["--attention-variant deepseek_v3"]
    for key, value in v3.items():
        if key == "rope_scaling":
            if value:
                for k2, v2 in value.items():
                    flags.append(f"--yarn-{k2.replace('_', '-')} {v2}")
            continue
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        flags.append(flag if value is True else f"{flag} {value}")
    return flags


def get_arch_flags(spec):
    """Megatron flags implied by a spec's optional architecture fields.

    Returns '' for specs that declare none, which keeps the emitted launch line
    identical to what it was before these fields existed.
    """
    flags = []
    if spec.get("norm") == "rmsnorm":
        flags.append("--normalization rmsnorm")
    if spec.get("norm_eps") is not None:
        flags.append(f"--layernorm-epsilon {spec['norm_eps']}")
    if spec.get("bias") is False:
        flags.append("--disable-bias-linear")
    if spec.get("swiglu"):
        flags.append("--swiglu")
    if spec.get("untie_embeddings"):
        flags.append("--untie-embeddings-and-output-weights")
    if spec.get("shared", 0) > 0:
        flags.append(f"--num-shared-experts {spec['shared']}")
    if spec.get("first_k_dense", 0) > 0:
        flags.append(f"--first-k-dense-replace {spec['first_k_dense']}")
    if spec.get("expert_interval") is not None:
        # 1 => EVERY layer is MoE (DeepSeek). Default (2) makes only alternating
        # layers MoE, leaving the rest as randomly-initialized dense MLPs.
        flags.append(f"--expert-interval {spec['expert_interval']}")
    if spec.get("dense_ffn") is not None:
        flags.append(f"--dense-ffn-hidden-size {spec['dense_ffn']}")
    if spec.get("tokenizer_type"):
        flags.append(f"--tokenizer-type {spec['tokenizer_type']}")
    if spec.get("null_vocab") is not None:
        # NullTokenizer's size (2026-09-12): the run tokenizes nothing; Megatron pads this + 1 to the embedding's rows
        flags.append(f"--vocab-size {spec['null_vocab']}")
    if spec.get("rotary_base") is not None:
        flags.append(f"--rotary-base {spec['rotary_base']}")
    if spec.get("eod_token"):
        # Single-quoted: the token contains < and >. The launcher expands
        # ${megatron_options} INTO a `bash -c "..."` string, which bash then
        # RE-TOKENIZES -- at which point bare < and > become stdin/stdout redirections
        # ("bash: ...: No such file or directory"). The single quotes survive the
        # get_vars print, the eval, and the concatenation as literal characters, and are
        # consumed by that final bash -c parse, protecting the redirection chars.
        # (Assumes the token has no single quote itself; DeepSeek's does not.)
        flags.append(f"--eod-token '{spec['eod_token']}'")
    if spec.get("softmax_before_topk"):
        flags.append("--softmax-before-topk")
    if spec.get("v3"):
        flags += get_v3_flags(spec["v3"])
    return " ".join(flags)

def get_filename(model_size, nodes, total_gpus, mbs):
    """
    Generates the exact filename required by the planner/profiler.
    Format: N{nodes}_n{gpus}_d{h}_e{ep}_f{ffn}_k{k}_s{s}_b{mbs}_profile.json
    """
    spec = get_spec(model_size)
    
    filename = (
        f"N{nodes}_n{total_gpus}_"
        f"d{spec['h']}_"
        f"e{spec['ep']}_"
        f"f{spec['ffn']}_"
        f"k{spec['k']}_"
        f"s{spec['s']}_"
        f"b{mbs}_"
        f"profile.json"
    )
    # print (f'{filename=}')
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Command to get filename (used by autorun.sh)
    cmd_name = subparsers.add_parser("get_filename")
    cmd_name.add_argument("--model_size", required=True)
    cmd_name.add_argument("--nodes", type=int, required=True)
    cmd_name.add_argument("--total_gpus", type=int, required=True)
    cmd_name.add_argument("--mbs", type=int, required=True)  # <--- Added MBS argument

    # Command to get specific params (used by SLURM template)
    cmd_vars = subparsers.add_parser("get_vars")
    cmd_vars.add_argument("--model_size", required=True)

    args = parser.parse_args()

    if args.command == "get_filename":
        try:
            # Added args.mbs to the call
            print(get_filename(args.model_size, args.nodes, args.total_gpus, args.mbs))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "get_vars":
        # Output strictly in Bash export format
        try:
            s = get_spec(args.model_size)
            print(f"export HIDDEN_SIZE={s['h']}")
            print(f"export FFN_HIDDEN_SIZE={s['ffn']}")
            print(f"export NUM_ATTN_HEADS={s['heads']}")
            print(f"export EP_SIZE={s['ep']}")
            print(f"export topk={s['k']}")
            print(f"export SEQ_LEN={s['s']}")
            print(f"export NUM_LAYERS={s['layers']}")
            print(f"export KV_HEADS={s.get('kv_heads', s['heads'])}")
            print(f"export VOCAB_SIZE={s.get('vocab', '')}")
            print(f'export ARCH_FLAGS="{get_arch_flags(s)}"')
        except Exception as e:
            sys.exit(1)