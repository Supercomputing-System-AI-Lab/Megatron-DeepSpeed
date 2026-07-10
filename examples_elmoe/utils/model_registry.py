import argparse
import sys

# --- CENTRAL CONFIGURATION ---
# This matches exactly what was in your SLURM template
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
        except Exception as e:
            sys.exit(1)