# Open problem (2026-11-12): commits on a side stream at 63B

State of the branch at this commit, for whoever looks into it next. Facts are from Frontier jobs of 2026-11-12; the layout in
all of them is X-MoE 63B (ELMOE-3D), 8 training nodes + 1 standby, PP 4, TP 2, EP 8, DP_moe 2, seq 4096, micro-batch 1,
GBS 4096 (512 micro-batches per stage per step), ZeRO-1, bf16, no activation checkpointing, XMOE_TIMING=0.

## What a commit costs at 63B (seq 4096)

- One micro-batch computes for about 0.32 s per stage (forward 0.12-0.13 s, backward 0.19-0.20 s); a step is 184 s without commits
  (68.1 TFLOP/s per GCD, job 5912585).
- Per GPU, in bf16: the raw gradient is 5.1 GB (stages 0, 3) / 4.9 GB (stages 1, 2); the slice a rank owns after ZeRO's reduction
  is 2.1 GB. A node's 8 GPUs share 4 NICs of 25 GB/s.
- Reduce modes (FTV2_COMMIT_MODE=p2p or keep), on the side stream (FTV2_COMMIT_OVERLAP=1): 0.62-0.76 s per commit
  (reduction 0.40-0.52 s + copy 0.20-0.24 s). Synchronous k=1: 25.6-30.2 TFLOP/s per GCD (jobs 5909546, 5908958).
- Raw mode (FTV2_COMMIT_MODE=raw, recovery/rc_commit_raw.py): snapshot 13 ms + exchange 0.54-0.57 s per commit; k=1 steps of
  335 s and 303 s (job 5914433, first two steps only). No commit mode hides at k=1 with a 0.32 s micro-batch.

## Problem 1: the reduce modes on the side stream turn NaN or crash at 63B, from step 2 or 3 on

- Runs: keep_k2_ov 5912169, p2p_k4_ovnt 5912586, p2p_k1_ovnt 5912587, keep_k2_ovnt 5912999, keep_k4_ovnt 5912588, P9 5914005,
  P10 5914433 (arm p2p_k4_ovnt_al). Synchronous commits at 63B and every side-stream run of the 10B model stay finite
  (10B side-stream runs are bit-identical to synchronous ones in determinism mode; a 10B run with ELMOE-3D, TP 2, no
  checkpointing and a 400 ms delay before each reduction stayed finite too, job 5913925).
- The gradient norm of the step before the NaN equals the synchronous run's at the same k to every printed digit.
- P9 (FTV2_NAN_CHECK=2): step 1 is finite everywhere (every micro-batch, the sums, the weights after the update). In step 2
  the first non-finite value is in stage 0's own forward output, on all 8 GPUs of one node in the same micro-batch (13 on one
  node, 31 on the other); stage 1 receives it at those micro-batches. Stage 0 receives no activations.
- P10: step 1 finite, allocator counters quiet; 16 s into step 2 stage-0 ranks log "expandable_segments: memory mapping failed
  with OOM", and 3 s later a stage-0 GPU faults in the MoE dispatch kernel (_padded_copy). Stage 0 is the fullest stage
  (52-53 GB reserved of 64 GB with the overlap).
- Hypothesis, NOT proven: ZeRO's reduction, run on the side stream, allocates and frees temporaries there (stage_1_and_2.py
  buckets, get_flat_partition in the epilogue); they are freed on the host while the side stream still writes them. Under memory
  pressure the caching allocator hands that memory to the compute stream. Not excluded: RCCL's in-node all-to-all delivering
  wrong data while an all-reduce runs beside it (seen without commits on 2026-11-05; NCCL_P2P_DISABLE=1 removed it then).
  p2p_k1_ovnt turned NaN without any allocator warning, which speaks against "memory only".
- Raw mode allocates nothing on the side stream and calls no DeepSpeed code there; it ran two 63B steps finite (5914433).

## Problem 2: a survivor that cannot report ends the recovery for every rank (job 5912586)

- A rank's main thread sat in a plain GPU copy at the end of the step (deepspeed/runtime/pipe/engine.py:650, not a
  torch.distributed call, so StopForRecovery cannot be raised there); its aborts did not return within 30 s and dump_stuck
  (rc_stop.py) never came back. At level B only the main thread writes the report, so _join (rc_controller.py) gave up after
  120 s, every other rank exited 77, and the stuck rank held the Slurm step to the wall (--kill-on-bad-exit=0).
- The heartbeat's hang rule names the rank with the LOWEST call count; that rank had the highest of its stage. The stuck rule's
  GPU probes time out on GPUs that are busy with the stalled job, so every node "failed" and a wrong node was blamed.

## Problem 3 (not understood): at 63B the step-1 gradient norm depends on k even with synchronous commits

none 37.7, k=1 34.6, k=4 40.6, k=2 42.2 (the same data and seed); at 10B every arm reads the same value. Raw mode does not
change the sums (DeepSpeed's own reduction sees every micro-batch).

## How to reproduce

Settings (environment, on top of runs/ftv2_run_common.sh's recovery mode): FTV2_PP=4, MP_SIZE=2,
PP layout "8:64" / "4:8", run command " 1:512:4:ELMOE-3D:63B:no-ckpt:0:even:no-planner", XMOE_TIMING=0, FTV2_SHADOW=cpu,
FTV2_CPU_ADAM=torch, FTV2_OOM_RECONFIG=0, and per arm:
- reduce + p2p on the side stream: FTV2_DS_SET=zero_optimization.use_multi_rank_bucket_allreduce=false FTV2_COMMIT=1
  FTV2_COMMIT_MODE=p2p FTV2_COMMIT_K=4 FTV2_COMMIT_OVERLAP=1 (NaN or a memory fault in step 2);
- all-reduce + keep: FTV2_COMMIT=1 FTV2_COMMIT_MODE=keep FTV2_COMMIT_K=2 FTV2_COMMIT_OVERLAP=1;
- raw: FTV2_COMMIT=1 FTV2_COMMIT_MODE=raw FTV2_COMMIT_K=1.
Diagnosis knobs: FTV2_NAN_CHECK=1 (per step: non-finite sums and weights, allocator counters), =2 (also per micro-batch: forward
output, received activation, input gradient, received gradient); FTV2_EVENT_TRACE=<from>:<to> (GPU time per pipeline instruction
and per commit); FTV2_SIDE_DELAY_MS.

## What is tested

- recovery/tests/cpu_commit_test.py, cpu_commit_raw_test.py: pass (login node). gpu_commit_overlap_test.py: pass for the reduce
  modes (one host-wait check at keep k=1 fails by the test's own allocation).
- Raw mode: the healthy path ran on GPUs (two 63B steps); its restore path (decide_microbatch_raw, .grad set from the snapshot
  before the first backward past the resume point) is tested on CPU only.
