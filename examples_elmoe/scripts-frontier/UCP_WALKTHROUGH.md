# Universal Checkpointing (UCP) — Step-by-Step Walkthrough

A top-down trace of both directions — **distributed checkpoint → universal**, and **universal →
running job** — with exact files, line numbers, file layouts, and the reasoning behind each design
choice.

**All line numbers are from the pristine tree**
`/lustre/orion/gen150/scratch/zixianw4/ELMOE/X-MoE/deepspeed/` at commit `c765b3e8`
(vendored DeepSpeed **0.15.5**), i.e. stock UCP before any X-MoE modification. Where X-MoE's
expert-parallel work changes something, it is flagged **[X-MoE]** and is *not* part of stock UCP.

**Reference checkpoint used for all concrete numbers** — `2026-08-02_5142659/global_step60`,
DeepSeek_16B:

| Symbol | Meaning | Value |
|---|---|---|
| `TP` | tensor parallel | 1 |
| `PP` | pipeline parallel | 1 |
| `DP` | data parallel | 8 |
| `EP` | expert parallel | 8 |
| ZeRO | optimizer sharding stage | 1 |
| — | compute dtype | bf16, fp32 master weights |
| — | non-expert parameters | **198** |
| — | routed expert tensors | **3456** (27 MoE layers × 64 experts × 2 matrices) |
| — | source checkpoint | 1745 files, 215 GiB |
| — | converted checkpoint | 3655 atom dirs, 198 GiB |

---

## 0. The one idea: the *atom*

Everything in UCP follows from a single definition. From the paper (§5.2):

> An *atomic checkpoint* contains a consolidated view of the model states corresponding to a tensor
> operation … it is no longer coupled with any parallelism strategy or hardware configuration, e.g.
> **it does not include any rank id, partitioning information, or additional data from specific
> sharding strategies such as padding for alignment.**

Concretely, one atom is **one directory named after one parameter**, holding three files:

```
<ckpt>_universal/zero/language_model.embedding.word_embeddings.weight/
    fp32.pt          819,987,988 B     the fp32 master weight
    exp_avg.pt       819,988,073 B     Adam m
    exp_avg_sq.pt    819,988,094 B     Adam v
```

No rank. No `tp`/`dp`/`pp` in the path. No padding. **The parameter's name is its address.**

That is the whole contract, and it is also the whole limitation: UCP assumes *the name is
topology-invariant*. §16 is about where that assumption breaks.

> **bf16 weights are not atoms.** The source checkpoint's `layer_*_expert_*.pt` and the `module`
> dict inside `mp_rank_*` are bf16, bundle several tensors, and carry `mp_rank` in the path. They are
> **Extract input**, exactly like `bf16_zero_pp_rank_*`. Under `--universal-checkpoint` the bf16
> weights are never read at all — they are regenerated from the fp32 atoms (§14).

---

## 1. Which code is actually live

Two forks decide which of four code paths you are reading. Establish both before reading anything.

**Fork 1 — ZeRO stage.** `ds_to_universal.py:469` `main()`:

```python
optim_files = _get_optim_files(args.input_folder)     # :474
zero_stage  = _get_zero_stage(optim_files)            # :475
if zero_stage <= 2:
    ...                                               # <-- X-MoE (ZeRO-1)
else:
    ...                                               # ZeRO-3: extract_zero_shards_stage3, merge_zero3_slices
```

Stage 3 has an entirely separate extract/merge pair (`:152`, `:338`). **Ignore them.**
`engine.py:1611` asserts `"MoE not supported with Stage 3"`, so for X-MoE only the `zero_stage <= 2`
branch is reachable.

**Fork 2 — optimizer class.** The load side is implemented on the ZeRO optimizer, and there are
three entry points:

| Optimizer | Entry point | Group name |
|---|---|---|
| `DeepSpeedZeroOptimizer` (ZeRO-1/2) | `stage_1_and_2.py:2298` | `"bit16_groups"` |
| `BF16_Optimizer` | `bf16_optimizer.py` | `"bf16_groups"` |
| `DeepSpeedZeroOptimizer_Stage3` | `stage3.py` | — |

X-MoE uses **`DeepSpeedZeroOptimizer`** (`stage_1_and_2.py`). The `bf16_` prefix on the shard
filenames only reflects `bfloat16_enabled()`, not the optimizer class — an easy misread.

**The live path:**

| Stage | Function | File |
|---|---|---|
| Convert · read source | `DeepSpeedCheckpoint` | `checkpoint/deepspeed_checkpoint.py:35` |
| Convert · extract | `extract_zero_shards` | `checkpoint/ds_to_universal.py:112` |
| Convert · merge | `merge_tp_slices` | `checkpoint/ds_to_universal.py:232` |
| Convert · finalise | `_save_optimizer_state` | `checkpoint/ds_to_universal.py:410` |
| Load · engine | `load_checkpoint` | `runtime/engine.py:2799` |
| Load · optimizer | `load_hp_checkpoint_state_from_checkpoint_dir` | `runtime/base_optimizer.py:20` |
| Load · per-parameter | `load_hp_checkpoint_state` | `checkpoint/universal_checkpoint.py:22` |
| Load · bf16 refill | `update_lp_params` | `runtime/zero/stage_1_and_2.py:1938` |

---

## 2. The two directions

§0 defined the atom and §1 said which code produces it. This is the round trip both halves of the
document trace — Part I builds atoms, Part II consumes them, and they are exact inverses:

```
  TRAIN (any topology)                                    TRAIN (any other topology)
        │                                                            ▲
        │ save                                                       │ --universal-checkpoint
        ▼                                                            │
  global_step60/                                          global_step60_universal/
    mp_rank_00_model_states.pt        ── PART I ──►         zero/<param>/{fp32,exp_avg,exp_avg_sq}.pt
    bf16_zero_pp_rank_{0..7}_*.pt      converter            zero/optimizer_state.pt
    layer_{L}_expert_{E}_*.pt          (offline, CPU)       mp_rank_00_model_states.pt
    expp_rank_{0..7}_*.pt                                              │
                                                            ── PART II ──►  optimizer + model
```

**Part I is offline and has no process group.** That single fact drives most of the design: the
converter cannot ask `groups._get_expert_parallel_rank()` or any other distributed question, so
everything it needs must already be *in* the files.

---

# PART I — CONVERT

## 3. `main()` — five phases

To turn §2's left column into its right column, the converter has to answer four questions and then
do one thing. That is where the phases come from — they are not arbitrary stages:

| It must know | before it can | phase |
|---|---|---|
| which ZeRO stage, and what `(pp, tp, dp)` produced this | iterate the right set of shards | **0** |
| which parameters exist and what shape each should end up | know what to cut and what to rebuild | **0** |
| — | cut every rank's flat buffers into per-parameter fragments | **1** |
| — | reassemble fragments into one atom per parameter | **2** |
| — | preserve the state that is *not* per-parameter (lr, step, loss scale) | **3** |

`ds_to_universal.py:469-545`, in order:

```python
optim_files   = _get_optim_files(args.input_folder)                    # :474  phase 0
zero_stage    = _get_zero_stage(optim_files)                           # :475
ds_checkpoint = DeepSpeedCheckpoint(args.input_folder)                 # :478  phase 0
_check_for_required_state(ds_checkpoint)                               # :482
iteration     = ds_checkpoint.get_iteration()                          # :485
slice_shapes  = union of mp_sd[PARAM_SHAPES] over mp_rank files        # :489-492
temp_dir      = os.path.join(args.output_folder, 'tmp')                # :494

_extract_zero_shard_files(args, ds_checkpoint, temp_dir)               # :499  phase 1
_merge_tp_slice_files(args, ds_checkpoint, slice_shapes, temp_dir)     # :502  phase 2
_save_optimizer_state(args, ds_checkpoint)                             # :505  phase 3

shutil.rmtree(temp_dir)                                                # :508
for f in glob.glob(os.path.join(args.input_folder, 'mp*')):            # :511  phase 4
    shutil.copy2(f, args.output_folder)
```

Note what `main()` does **not** do: it never looks at `layer_*_expert_*` files. The MoE expert
weights are invisible to the converter — only the ZeRO optimizer shards matter.

## 4. Phase 0 — reading the source

Phase 0 answers the two "must know" rows of §3, and it does so in three steps, in this order, because
each depends on the last: **find the shards → learn the topology → learn the parameter list.** Nothing
is written in this phase.

### 4.1 Find the shards — `_get_optim_files:431`

```python
def _get_optim_files(checkpoint_dir):
    return _get_checkpoint_files(checkpoint_dir, "*_optim_states.pt")
```

A bare glob. `[M]` On the reference checkpoint it returns **16** files for an 8-rank run: 8 real
shards (22.9 GiB each) plus 8 `expp_rank_*` placeholders (1,493 B, empty under ZeRO). `optim_files[0]`
is the right one only because `'b' < 'e'` sorts `bf16_zero_*` first.

### 4.2 Learn the stage — `_get_zero_stage:448`

Which of §1's two forks to take. One integer, at a startling price:

```python
state_dict = torch.load(optim_files[0], map_location=torch.device('cpu'))   # :449
return state_dict[OPTIMIZER_STATE_DICT][ZERO_STAGE]
```

**No `mmap`.** This reads a whole 22.9 GiB shard into RAM to learn one integer.

### 4.3 Learn the topology — `DeepSpeedCheckpoint`, `deepspeed_checkpoint.py:35`

Phase 1 must visit **every** `(pp, tp, dp)` triple (§5.2 says why), so it needs the source topology.
Nothing in the checkpoint states it directly — it is inferred from filenames. `__init__`:

```python
pipeline_parallel = len(get_files_with_prefix(get_files(dir), LAYER_FILE_PREFIX)) > 0   # :46
self._validate_folder(dir, pipeline_parallel)                                           # :48
self.zero_checkpoint = ZeROCheckpoint(dir)                                              # :50
self.file_list     = get_files(dir)                                                     # :52
self.layer_files   = get_files_with_prefix(self.file_list, LAYER_FILE_PREFIX)           # :53
self.mp_rank_files = get_files_with_prefix(self.file_list, MODEL_FILE_PREFIX)           # :54
self.layer_keys    = self._get_layer_keys()                                             # :56
```

`ZeROCheckpoint` (`zero_checkpoint.py:50`) then derives `(pp, tp, dp)` from the shard filenames via
`get_model_3d_descriptor`, giving the `_3d_range_list` that phase 1 iterates.

> **[X-MoE] This is where a MoE checkpoint dies.** `LAYER_FILE_PREFIX` is the bare string `'layer_'`,
> and DeepSpeed MoE writes `layer_<L>_expert_<E>_mp_rank_<NN>_model_states.pt`, which matches. 1728
> expert files ⇒ `pipeline_parallel=True` on a `PP=1` run ⇒ `_validate_folder` demands `layer_01*`
> ⇒ `AssertionError`. See `0804_2026_UCP_BLOCKERS_ENCOUNTERED.md` Blocker 2.

### 4.4 Learn the parameter list — `PARAM_SHAPES`, `:489-492`

Topology tells the converter *which files* to read. This tells it *what is inside them*:

```python
slice_shapes = []
for mp_rank_file in ds_checkpoint.mp_rank_files:
    mp_sd = torch.load(mp_rank_file, map_location=torch.device('cpu'))
    slice_shapes += mp_sd[PARAM_SHAPES]
slice_shapes = {k: v for d in slice_shapes for k, v in d.items()}
```

`param_shapes` is a **list of per-group `OrderedDict[name → shape]`**, written at save time by
`engine._get_zero_param_shapes()`. It is the converter's only source of truth for *what parameters
exist and what shape each one should end up*. Everything phase 2 does is driven by this dict.

`[M]` On the reference checkpoint: 13 groups, **630 entries**.

> **[X-MoE] 630, not 3654.** `param_shapes` is written by whichever rank has `dp_rank == 0`, so its
> expert entries are that rank's **8 local** experts — not all 64. Stock UCP has no notion of this;
> X-MoE adds `expand_moe_param_shapes()` to materialise 630 → 3654.

## 5. Phase 1 — Extract

### 5.0 The intuition: ZeRO destroys parameter identity, and Extract puts it back

Before reading a line of `extract_zero_shards`, understand what a ZeRO optimizer actually stores,
because the whole function is a consequence of it.

**ZeRO-1 does not keep per-parameter optimizer state.** It takes every parameter in a *param group*,
concatenates them into **one flat 1-D fp32 buffer**, then gives each DP rank a **contiguous slice** of
that buffer. (A *param group* is the optimizer's own unit — §5.1 explains what the groups are here and
why there are 13 of them. For now: one flat buffer per group.) Three reasons:

1. Adam becomes one elementwise kernel over a big tensor instead of *N* small launches.
2. Redistributing updated weights is **one** collective, not one per parameter.
3. No per-tensor alignment padding — the group pads once, at the end (`group_paddings`).

`[M]` What that looks like on disk — rank 0 of the reference checkpoint:

```
single_partition_of_fp32_groups[0] : shape (177209344,)   1-D fp32
base_optimizer_state['state'][0]   : exp_avg (177209344,)  exp_avg_sq (177209344,)
```

Two things follow immediately:

- **The parameter boundaries are gone.** `fp32_groups[0]` is a flat run of 177 M floats. Nothing in it
  says where the embedding ends.
- **Adam's state mirrors the master weights exactly.** `exp_avg.numel() == fp32.numel()` `[M]` — Adam
  state is per **element** of the master weights, not per parameter. That is why the three atom files
  (`fp32`, `exp_avg`, `exp_avg_sq`) always have identical shapes, and why extract loops over them
  uniformly.

**A checkpoint of just that would be unusable** — a pile of anonymous floats, meaningful only to the
exact rank layout that produced it. So ZeRO also records a **receipt**:

```
param_slice_mappings[group_id] :  parameter name  ->  fragment_address(start, numel)
```

built at `stage_1_and_2.py:553` / `:575` and stored at `:2173`. `fragment_address` is a two-field
dataclass (`utils/tensor_fragment.py:13`):

```python
@dataclass
class fragment_address:
    numel: int
    start: int
```

`[M]` It sums to exactly the partition, in every group:

```
group  0:   1 named param  on this rank;  sum(numel) = 177,209,344 = 100.0% of fp32[0]
group  1:   8 named params on this rank;  sum(numel) =      14,592 = 100.0% of fp32[1]
group  2:  41 named params on this rank;  sum(numel) = 178,782,208 = 100.0% of fp32[2]
```

**So: values + index = named tensors.** That is the entire content of Extract. Every object the
function fetches is one of those two things:

| Object | What it is | Which half |
|---|---|---|
| `sd` | the shard **file** — also holds `ds_config`, `ds_version` | container |
| `optim_sd = sd[OPTIMIZER_STATE_DICT]` | the optimizer's portion of it | container |
| `fp32_groups = optim_sd[SINGLE_PARTITION_OF_FP32_GROUPS]` | this rank's fp32 master slice, per group | **values** |
| `state_groups = optim_sd[BASE_OPTIMIZER_STATE]["state"]` | Adam `exp_avg` / `exp_avg_sq`, per group | **values** |
| `param_slice_mappings` | name → `(start, numel)`, per group | **index** |

Nothing else is needed, and nothing listed is redundant. Drop the index and you have anonymous
floats; drop either value list and the atom is incomplete.

### 5.1 Why "groups" at all, and what `partition_count` means

§5.0 said "one flat buffer per group" and left *group* undefined. This section defines it, because the
one field that exposes the whole structure is `optim_sd[PARTITION_COUNT]`. `[M]` On the reference
checkpoint it reads:

```
partition_count = [8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

**Thirteen numbers, because there are thirteen param groups — one entry per group.** And
**`partition_count[g]` answers exactly one question: over how many ranks is group `g`'s flat buffer
split?** It is not a pattern to be read left-to-right; it is thirteen independent answers:

```
partition_count = [ 8 ,  8 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ]
   group index →    0    1    2    3    4    5    6    7    8    9   10   11   12
                  └───┬───┘  └──────────────────────┬──────────────────────────┘
                  non-expert                    routed experts
                  split over 8 ranks            not split at all (1 = "all mine")
```

So the array raises two questions, and the rest of this section answers them in order: **why are the
first two 8?** and **why are the remaining eleven 1?**

To answer either, you need to know what each group actually holds. `[M]` Every group, read off the
same shard:

```
 g  partition_count   fp32 numel/rank  #params  kind      group name
 0                8       177,209,344        1  non-exp   wd_no_scale_lr
 1                8            14,592        8  non-exp   no_wd_no_scale_lr
 2                1       178,782,208       41  EXPERT    ep_size_8
 3                1       175,898,624       41  EXPERT    ep_size_8
 4 … 11           1   178.8M / 175.9M        41  EXPERT    ep_size_8   (alternating)
12                1        95,158,272       22  EXPERT    ep_size_8
```

Column 2 of this table **is** the array above, one row per element. The `kind` column is measured, not
assumed: a group is "EXPERT" iff its `param_slice_mappings` entries contain `deepspeed_experts.`.

#### Why two 8s

Groups 0 and 1 are the **non-expert** parameters, and they exist for the ordinary reason any PyTorch
optimizer has param groups — different hyperparameters:

| Group | Name | Contents |
|---|---|---|
| 0 | `wd_no_scale_lr` | everything that gets **weight decay**: embeddings, attention, MLP weights |
| 1 | `no_wd_no_scale_lr` | everything that must **not**: layernorm weights, biases (`[M]` 14,592/rank) |

Both are replicated across the data-parallel group, so ZeRO shards each of them over

```
DP_non_moe = world / (TP × PP) = 8 / (1 × 1) = 8
```

Hence `8, 8`.

#### Why eleven 1s

Groups 2–12 are the **routed experts**. Two separate facts produce eleven ones.

**(a) Why the value is 1 — there is nothing to shard.** ZeRO's whole premise is *"this state is
replicated on every DP rank, so let each rank keep only 1/N of it."* At `EP=8` on 8 GPUs, each rank
owns its 8 local experts **alone** — no other rank holds a copy. The expert-data-parallel degree is

```
DP_moe = world / (EP × PP) = 8 / (8 × 1) = 1
```

**`partition_count = 1` means "not partitioned — this rank owns the whole thing."** ZeRO cannot
deduplicate what was never duplicated.

> This is the single most important line in the array. **`partition_count[g]` is literally
> `DP_non_moe` for non-expert groups and `DP_moe` for expert groups** — two different data-parallel
> degrees in one checkpoint. A converter that assumes one global `dp_degree` is wrong for half the
> groups the moment `DP_moe ≠ 1`.

**(b) Why eleven groups and not one.** `moe/utils.py`'s
`split_params_into_different_moe_groups_for_optimizer` caps a group at
`max_group_size = 178,956,971` elements, so one enormous flat buffer (and its all-reduce) can't grow
unboundedly. `[M]` The arithmetic is exact:

```
expert params per rank = 27 MoE layers × 8 local experts × (2816·2048 + 2048·1408)
                       = 27 × 8 × 8,650,752
                       = 1,868,562,432

1,868,562,432 / 178,956,971 = 10.44   ->   ceil = 11 groups        (measured: 11)
```

#### Why the sizes alternate 178.8M / 175.9M

`[M]` A nice tell that the packing is greedy and parameter-granular. One expert is two matrices:

```
dense_h_to_4h  2816 × 2048 = 5,767,168
dense_4h_to_h  2048 × 1408 = 2,883,584          pair = 8,650,752
```

A group boundary lands **mid-expert**:

```
group 2 : 20 pairs + one orphaned h_to_4h  = 20·8,650,752 + 5,767,168 = 178,782,208   (41 params)
group 3 : the orphan's 4h_to_h + 20 pairs  = 2,883,584 + 20·8,650,752 = 175,898,624   (41 params)
```

so every group inherits one half-expert from the previous one, and the two sizes alternate.
`[M]` Totals check out: `10 × 41 + 22 = 432 = 27 layers × 8 experts × 2 matrices`.

#### Why extract iterates over groups

```python
for param_group_id in range(len(state_groups)):
```

**The group is the unit ZeRO flattened over**, so it is the unit whose `(start, numel)` offsets are
meaningful. `fragment_address.start` is an offset into *that group's* buffer on *this* rank —
meaningless against any other group. Extract must therefore cut along exactly the same seams ZeRO
used, which is why the loop structure mirrors the storage structure.

#### The array, fully resolved

```
partition_count = [8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                   │  │  └──────────── 11 expert groups ──────────┘
                   │  │     value 1  = DP_moe     = world/(EP×PP) = 1  -> nothing to shard
                   │  │     count 11 = ceil(1,868,562,432 / 178,956,971)
                   │  └─ no-weight-decay params, value 8 = DP_non_moe = world/(TP×PP)
                   └──── weight-decay params,    value 8 = DP_non_moe
```

Two independent things are encoded here, and keeping them apart is the point: the **count** of entries
comes from how parameters are grouped (hyperparameters, then MoE size-capping); the **value** of each
entry comes from which data-parallel degree that group is replicated over.

### 5.2 Why the mapping is *partial* on every rank

§5.1 established *which* buffer each `(start, numel)` indexes into. This section is about the other
surprise in `param_slice_mappings`: it does not list every parameter.

`[M]` Rank 0's group 0 contains exactly **one** name:

```
language_model.embedding.word_embeddings.weight    start=0   numel=177,209,344
```

But the embedding is `100096 × 2048 = 204,996,608` elements. Rank 0's fragment is **86.4%** of it.

That is the point: **`fragment_address` describes the intersection of one parameter with one rank's
partition**, not a whole parameter. ZeRO slices the flat buffer at an arbitrary offset, so a
parameter can straddle two ranks, and a rank can hold a piece of one parameter and nothing else.

`stage_1_and_2.py:573` records a parameter only `if lp._hp_mapping is not None` — i.e. only when it
intersects this rank's partition. **The union over all DP ranks is the complete list.**

Two consequences that matter:

1. The extract loop *must* cover every `(pp, tp, dp)` triple. A missing rank does not raise — it
   silently yields a truncated atom.
2. Stage 2a (§6.1) is a `torch.cat` of exactly these fragments, in `dp` order, which is why the
   fragment filename carries the `dp` index and why they must be concatenated sorted.

### 5.3 The code

Everything above determines the shape of this function, so read it as three questions already
answered:

| The code does | Because (§) |
|---|---|
| loops over **groups** | the group is what ZeRO flattened, so it is what must be cut along — §5.1 |
| loops over **`param_slice_mappings`**, not over parameters | only this rank's fragments are here, and only the index knows their names — §5.0, §5.2 |
| loops over **three states with one offset** | Adam state is per-element of the master weights, so all three share the same cut — §5.0 |
| is fanned over every **`(pp, tp, dp)`** | each rank's mapping is partial; only the union is complete — §5.2 |

`_extract_zero_shard_files:363` fans `extract_zero_shards` over every `(pp, tp, dp)` triple with a
`ProcessPoolExecutor(num_extract_workers)`. Each task loads **one whole shard** — which is why memory
scales with worker count, not with model size (§17).

`extract_zero_shards:112`:

```python
sd = ds_checkpoint.get_zero_checkpoint_state(pp_index, tp_index, dp_index)   # :114  one shard
optim_sd = sd[OPTIMIZER_STATE_DICT]
param_slice_mappings = optim_sd[PARAM_SLICE_MAPPINGS]                        # :117  the INDEX
state_groups = optim_sd[BASE_OPTIMIZER_STATE]["state"]                       #       the VALUES
fp32_groups  = optim_sd[SINGLE_PARTITION_OF_FP32_GROUPS]                     #       the VALUES

for param_group_id in range(len(state_groups)):                              # group = flattening unit
    flat_state = dict(exp_avg    = state_groups[param_group_id]["exp_avg"],
                      exp_avg_sq = state_groups[param_group_id]["exp_avg_sq"],
                      fp32       = fp32_groups[param_group_id])
    for name, fragment_mapping in param_slice_mappings[param_group_id].items():
        for state_key in flat_state.keys():                                  # same cut, 3 buffers
            dump_param_fragment(dir, tp_index, dp_index, state_key, flat_state[state_key],
                                name, fragment_mapping.start, fragment_mapping.numel)   # :148
```

Read it as three nested loops with one job each:

```
for each group          ->  pick the three flat buffers ZeRO flattened together
  for each named param  ->  look up where it lives:  (start, numel)
    for each of 3 states->  cut the same window out of each buffer, write it under the param's name
```

`dump_param_fragment:179` performs the cut and decides the layout:

```python
param_base_path = os.path.join(dir, param_name, str(tp_index))               # :183
path = os.path.join(param_base_path, f"{state_name}.{dp_index_to_str(dp_index)}")
if state_name != "step" and torch.is_tensor(state_flat_tensor):
    state_flat_tensor = state_flat_tensor.narrow(0, offset, numel).clone()   # <- the cut
_save_checkpoint(path, state_flat_tensor)
```

`.narrow(0, start, numel)` is exactly "apply the receipt", and `.clone()` detaches the view so the
23 GiB parent buffer can be freed.

The resulting tree:

```
tmp/<param_name>/<tp_index>/<state>.<dp_index:02d>
```

Concretely:

```
tmp/language_model.encoder.layers.1.input_layernorm.weight/0/
    fp32.00  exp_avg.00  exp_avg_sq.00          <- small param, lives entirely on rank 0
```

**Note what the path encodes and what it does not.** `tp_index` is a directory level because TP
slices must stay distinguishable until stage 2b decides how to merge them. `dp_index` is only a
filename suffix because DP fragments are merged unconditionally by concatenation — no decision
needed. **`<dp_index>` means "shard `dp` of this one tensor".**

> **[X-MoE] That last sentence is the one EP breaks.** For an expert parameter, every EP rank emits
> the *same* name, so one directory receives 8 *different experts* labelled as 8 fragments of one.
> `[M]` Measured after a real extract: 8 files of 23,070,249 B in one directory — each a complete
> `[2816, 2048]` expert — and only 8 distinct expert ids present out of 64.
> **Why that happens, and why no pattern can fix it, is derived in §16.** Read the rest of the
> walkthrough first; §16 assumes §§0-15.

## 6. Phase 2 — Merge

Extract left one directory per parameter, holding fragments indexed by `(tp, dp)`. Merge turns each
directory into one atom.

**Why two nested concatenations, and not one:** two *independent* shardings were applied to the same
tensor, and they are not the same kind of operation.

| Sharding | Applied by | To undo it you need |
|---|---|---|
| **ZeRO-DP** — a flat 1-D cut at an arbitrary offset | the optimizer | nothing but `cat` + `reshape`; the cut has no model meaning |
| **TP** — a cut along a specific tensor axis, chosen per layer type | the model | to know **which axis**, i.e. what kind of parameter this is |

So stage 2a undoes DP and cannot be wrong; stage 2b undoes TP and needs the model to declare itself
(§7). Keeping them apart is the key to reading this file.

`_merge_tp_slice_files:378` fans `merge_tp_slices` over `slice_shapes.items()`.

### 6.1 Stage 2a — undoing the ZeRO-DP split

#### What ZeRO-DP actually did, at training time

Go back to §5.0. For each param group, ZeRO built **one flat 1-D fp32 buffer** holding every parameter
in that group, end to end, each in row-major order. Then it cut that buffer into
`partition_count[g]` **contiguous** pieces and gave piece *r* to rank *r*.

```
group 0's flat buffer  (8 x 177,209,344 = 1,417,674,752 elements)
├────────────┬────────────┬────────────┬─── … ───┬────────────┤
│  rank 0    │  rank 1    │  rank 2    │         │  rank 7    │
└────────────┴────────────┴────────────┴─── … ───┴────────────┘
 ◄──── word_embeddings.weight (204,996,608) ────►
```

Two properties of that cut are the whole reason stage 2a is easy:

- it is **one-dimensional** — the buffer has already been flattened, so there is no axis to choose;
- it is **contiguous and rank-ordered** — rank *r* holds a later slice than rank *r−1*, always.

#### What a "fragment" therefore is

A parameter's elements land wherever they land. `[M]` The embedding, traced across all 8 shards:

```
rank   in group-0 mapping?        start          numel       rank's partition
  0            yes                    0    177,209,344         177,209,344
  1            yes                    0     27,787,264         177,209,344
  2-7          no                     -              -         177,209,344

sum of fragments = 204,996,608  =  100096 x 2048  = the whole parameter   [M] exact
```

Read that table carefully — it contains the three facts stage 2a depends on:

1. **A parameter appears only on the ranks it overlaps.** The embedding is 205 M elements, so it
   spills out of rank 0's 177 M partition into rank 1 and stops. Ranks 2–7 have no entry for it at all.
2. **`start` is an offset into the *rank's partition*, not into the parameter.** Both rank 0 and rank 1
   report `start = 0` — yet rank 0 holds the parameter's elements `[0, 177209344)` and rank 1 holds
   `[177209344, 204996608)`. The `start` field says *where in my slice this lives*; it says nothing
   about *where in the parameter*.
3. **So the parameter's element order is carried by the rank index alone.** Rank order **is** element
   order. That is the invariant.

#### Why concatenating in rank order inverts it

Given (1)–(3), the fragments **tile** the parameter: they are disjoint, contiguous, in rank order, and
`[M]` sum to exactly its numel. The inverse of *"cut a 1-D array into ordered contiguous pieces"* is
*"concatenate the pieces in that order"* — and there is **exactly one** such operation.

`_merge_zero_shards:198`:

```python
for tp_index in range(tp_degree):
    prefix_path = os.path.join(param_base_path, str(tp_index), f"{state}")
    paths = glob.glob(f"{prefix_path}.*")                       # which ranks contributed?
    if len(paths) == 0: continue                                # this tp rank has none
    dp_indices = {int(pattern.match(p).group(1)) for p in paths}
    paths  = [f"{prefix_path}.{dp_index_to_str(i)}" for i in sorted(dp_indices)]   # <- RANK ORDER
    shards = [torch.load(p) for p in paths]
    slice  = torch.cat(shards, dim=0).reshape(slice_shape)      # :226
    slices.append(slice)
return slices                                                   # one entry per TP rank
```

Every line maps to a fact above:

| Code | Why |
|---|---|
| `glob(f"{prefix_path}.*")` then parse the suffixes | **fact 1** — it cannot assume all `dp_degree` ranks contributed; it must discover which did (`[M]` the embedding has 2 of 8) |
| `sorted(dp_indices)` | **fact 3** — rank order is element order, so the concat order is not free |
| `torch.cat(..., dim=0)` | the buffer was already flat; dim 0 is the only dimension a fragment has |
| `.reshape(slice_shape)` | restore the N-D shape from `param_shapes` (§4.4); flattening was ZeRO's doing, not the model's |

#### What "always correct" means — and what it does not

**It means the operation has no free parameter.** Inverting a flat ordered split admits exactly one
answer, and it is derivable from the fragments themselves. Stage 2a never consults a pattern, never
asks what kind of parameter this is, and cannot be *mis-configured*. Whether a tensor is a layernorm,
an attention matrix or an expert makes no difference here.

Contrast stage 2b (§6.2): TP cut the tensor **along an axis of the N-D tensor**, and which axis is a
property of the layer type. Given only the pieces you cannot tell — the shapes are identical either
way (§7). There, five different answers are possible and picking wrong is silent.

**It does not mean "cannot fail".** Stage 2a is correct *given its inputs are the right fragments*.
Two ways that premise is violated:

- **A missing rank.** If one `(pp, tp, dp)` task failed, or a stale `tmp/` from an aborted run is
  present, the glob simply returns a different set. `.reshape(slice_shape)` is the only thing standing
  between that and a silently wrong atom — it raises when the element count is off.
- **[X-MoE] Fragments that are not fragments.** Stage 2a trusts the `.dp` suffix to mean "piece `dp`
  of this tensor". If two *different* tensors are written under one name, it concatenates them as
  though they were pieces — and the reshape fires with an exact multiple of the expected size. That is
  §16, and it is precisely the reshape guard doing its job.

`group_paddings` `[M]` is `[0, 0, …]` here, so alignment padding never enters the picture; when it is
non-zero it lives at the tail of the last rank's *partition* and is excluded from every
`fragment_address`, so it never reaches an atom.

### 6.2 Stage 2b — undoing the tensor-parallel split

#### First, the four words the code overloads

`ds_to_universal.py` uses *shard*, *fragment* and *slice* for three different objects, and reuses
`shard` for a fourth. Pin them down before reading anything:

| Word | What it is | Shape | Where it lives |
|---|---|---|---|
| **ZeRO shard** (file) | one rank's whole optimizer file | — | `bf16_zero_pp_rank_<dp>_mp_rank_<tp>_optim_states.pt` |
| **fragment** | one rank's piece of **one parameter**, cut out of the flat buffer (§5) | 1-D, `fragment_address.numel` | `tmp/<name>/<tp>/<state>.<dp>` |
| **slice** | one **TP rank's** complete copy of that parameter, rebuilt from its DP fragments by stage 2a | `slice_shape` (N-D) | in memory, `slices[tp]` |
| **atom** | the merged, topology-free parameter (§0) | full shape | `zero/<name>/<state>.pt` |

So the pipeline is: **fragments → (2a) → slices → (2b) → atom.** In code:

```python
slices = _merge_zero_shards(slice_base_path, state, tp_degree, shape)   # :275  -> len == tp_degree
```

`slices[i]` is what TP rank *i* held. `shards` inside `_merge_zero_shards` is the list of *fragments*
for one TP rank — the same word, one level down.

#### Second, what `shape` is — and why it is not the answer

`shape` is `slice_shapes[name]`, from `param_shapes` (§4.4). **It is the per-TP-rank shape, not the
parameter's true shape.** `[M]` The same parameters in a TP=1 and a TP=4 checkpoint of this model:

```
parameter                          param_shapes @TP=1   param_shapes @TP=4   axis that shrank
word_embeddings (vocab)                (100096, 2048)        (25088, 2048)   dim 0  /4
query_key_value (column-parallel)        (6144, 2048)         (1536, 2048)   dim 0  /4
self_attention.dense (ROW-parallel)      (2048, 2048)          (2048, 512)   dim 1  /4
shared_expert h_to_4h (swiglu)           (5632, 2048)         (1408, 2048)   dim 0  /4
input_layernorm (replicated)                  (2048,)              (2048,)   none — identical
```

That table *is* the merge problem, stated in data:

- **The axis that shrank is the axis to concatenate on.** `self_attention.dense` shrank on dim 1, so
  its four slices must be joined on dim 1. `query_key_value` shrank on dim 0, so dim 0.
- **A replicated parameter shrank on no axis**, so joining is wrong for it entirely — the four copies
  are identical and one must be picked.
- **And the converter cannot see this table.** It has *one* checkpoint, so it sees only the right-hand
  column. Nothing records the unsharded shape. Two parameters that both read `(2048, 512)` may have
  come from `(2048, 2048)` cut on dim 1 or `(8192, 512)` cut on dim 0, and the fragments are identical
  in both cases.

**That is the exact gap §7 fills.** `shape` tells the converter what each slice *is*; only the model's
declaration tells it how the slices *relate*.

At `TP=1` all of this is vacuous — `len(slices) == 1`, and `cat([x], dim=anything) == x`. Every branch
below is the identity. **This is why a TP=1 conversion succeeds with a completely wrong pattern set**
(§17).

#### The code

`merge_tp_slices:232`. `slices` is the list above; `name` is the parameter; `ckpt_dict` is the atom
being built.

```python
for state in ("fp32", "exp_avg", "exp_avg_sq"):                 # three atoms, one decision
    slices = _merge_zero_shards(slice_base_path, state, tp_degree, shape)   # 2a -> tp_degree slices
    ckpt_dict = {}

    if get_matched_pattern(replicated_parameters, name):        # :281  layernorms, router
        assert all slices equal                                 #       ...so verify, then take one
        param = slices[0]                                       #       records NOTHING (§7.3)

    elif get_matched_pattern(parameters_to_average, name):      # :286  (unused by this model)
        param = sum(slices) / len(slices)

    elif get_matched_pattern(parameters_with_2_sub_params_cat_dim_0, name):  # :289  swiglu
        chunked = [torch.chunk(s, 2, dim=0) for s in slices]    #       each slice = [gate_r ; up_r]
        param   = torch.cat([cat(first halves), cat(second halves)], dim=0)  # -> [gate_all ; up_all]
        ckpt_dict[CAT_DIM] = 0 ; ckpt_dict[PARAM_N_SUB_PARAMS] = 2

    elif matched_sub_params_shape:                              # :297  GQA, unequal q/k/v blocks
        ...narrow per sub-dimension, cat each...
        ckpt_dict[SUB_PARAM_SHAPE] = matched_sub_params_shape

    else:                                                       # :318  THE DEFAULT
        cat_dim = 1 if get_matched_pattern(parameters_with_row_parallelism, name) else 0
        param   = torch.cat(slices, dim=cat_dim)                #       the axis that shrank
        ckpt_dict[CAT_DIM] = cat_dim

    if get_matched_pattern(vocabulary_parameters, name):        # :323  applied ON TOP of the above
        param = param[:original_vocab_size, :]                  #       100096 -> 100015  [M]
        ckpt_dict[VOCAB_TENSOR] = True

    ckpt_dict[PARAM] = param
    _save_checkpoint(final_path, ckpt_dict)                     # -> zero/<name>/<state>.pt
```

Four structural points, none of them obvious from the code alone:

1. **The loop is over the three states, not over parameters.** `fp32`, `exp_avg` and `exp_avg_sq` are
   the same tensor shape (§5.0), so the same branch and the same axis apply to all three. One
   decision, three atoms.
2. **The `if/elif` chain is ordered, and the first match wins** — but `:252` asserts at most one
   pattern matches anyway (§7.4), so the order is defensive, not semantic.
3. **Vocabulary is a separate `if`, not an `elif`.** Padding-stripping composes *on top of* whatever
   merge already happened, because a vocab parameter is also column-parallel. `[M]` The atom ends up
   `(100015, 2048)` with **both** `cat_dim: 0` and `vocab_tensor: True`.
4. **`ckpt_dict` is the atom's metadata, and the loader replays it verbatim** (§13).

#### Why a wrong pattern here is invisible until TP changes

This is the single most important property of stage 2b:

> `CAT_DIM`, `PARAM_N_SUB_PARAMS`, `SUB_PARAM_SHAPE` and `VOCAB_TENSOR` are written **into the atom**.
> §13 inverts **whatever was recorded**; it never re-derives.

So a wrong choice is *self-consistent*. Merge with the wrong axis, split with the same wrong axis, and
at unchanged TP you get back exactly what you started with:

```
S_q^t ∘ M_q^t = id        for any numel-preserving wrong pattern q, at unchanged degree t
```

The error only materialises when the degree changes — because then merge used `t_src` and split uses
`t_tgt`, and the two mistakes no longer cancel. Worse, the two cases that matter most for this model
are numel-preserving, so §13's only guard (`assert full_param_numel == tp_world_size * tp_slice_numel`)
passes:

- **row-parallel merged on dim 0** — `[2048,512] × 4` gives `[8192,512]` instead of `[2048,2048]`;
  numel identical, layout a near-total permutation.
- **swiglu merged plainly** — `gate_0;up_0;gate_1;up_1` instead of `gate_all;up_all`; **same shape**,
  half the gate weights swapped with up weights.

Both are silent. See §17.

## 7. The pattern contract

### 7.1 Background: what "row" and "column" parallel actually mean

The pattern lists are named after Megatron's TP classes, and those names describe a matrix that is
**not** the one stored on disk. Getting this straight first makes §7.2 mechanical.

#### The convention: input on the left, weight stored transposed

`[C]` Megatron's own docstring, `core/tensor_parallel/layers.py`, `ColumnParallelLinear`:

> The linear layer is defined as **Y = XA + b**. A is parallelized along its **second** dimension as
> `A = [A_1, ..., A_p]`.

and `RowParallelLinear`:

> A is parallelized along its **first** dimension and X along its second.

So the maths is `Y = X A`, **input on the left**: `X` is `[tokens, in]`, `A` is `[in, out]`.

But the tensor in the checkpoint is `Aᵀ`. `[C]` `layers.py:273`:

```python
output = torch.matmul(total_input, weight.t())      # weight is [out, in]
```

That is PyTorch's `nn.Linear` layout. **The transpose is the whole source of the confusion** — the
class names describe `A`; the file stores `Aᵀ`:

| Class | splits `A` `[in, out]` along | allocates `[C]` | stored shape | split axis **as stored** |
|---|---|---|---|---|
| `ColumnParallelLinear` | dim 1 — its **columns** | `torch.empty(output_size_per_partition, input_size)` | `[out/t, in]` | **0** |
| `RowParallelLinear` | dim 0 — its **rows** | `torch.empty(output_size, input_size_per_partition)` | `[out, in/t]` | **1** |

`[M]` Confirmed by the TP=4 measurement in §6.2: `self_attention.dense` — a `RowParallelLinear` —
shrank on dim **1**, `(2048, 2048) → (2048, 512)`, while `query_key_value` — column-parallel — shrank
on dim **0**.

> **The trap, stated once:** `parameter_with_row_parallelism_patterns` means *"this parameter belongs
> to a `RowParallelLinear`"*, and the axis UCP must concatenate on is **1**. The list says "row"; the
> axis is 1.

#### Why the MLP is column-then-row

`[C]` `ParallelMLP`, `megatron/model/transformer.py`:

```python
self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(..., gather_output=False)
self.dense_4h_to_h = tensor_parallel.RowParallelLinear(...,  input_is_parallel=True)
```

Those two keyword arguments are the design. The pairing makes an entire MLP block cost **one**
collective:

```
X  [s, h]   replicated on every rank
   │
h_to_4h  COLUMN-parallel      Aᵢ = [h, 4h/t]      Yᵢ = X·Aᵢ → [s, 4h/t]
   │                          gather_output=False   each rank owns a column slice — NO COMM
   │
activation (GELU / SwiGLU)    ELEMENTWISE           works per column slice — NO COMM
   │
4h_to_h  ROW-parallel         Aᵢ = [4h/t, h]      input_is_parallel=True → do not re-split
   │                          Zᵢ = Yᵢ·Aᵢ → [s, h]   a PARTIAL SUM
   ▼
all-reduce  Σᵢ Zᵢ → [s, h]                          ← the ONE collective
```

It works because **the output split of the first GEMM is exactly the input split the second one
needs**: `h_to_4h` leaves rank *i* holding output columns `[4h·i/t, 4h·(i+1)/t)`, and row-parallel
`4h_to_h` wants precisely those rows of its own matrix.

Reverse the order and you pay twice — row-parallel first needs an all-reduce to complete its output,
the elementwise activation then needs the full tensor, and column-parallel needs it re-split: **two
collectives plus a scatter.** The same pairing runs in attention: `query_key_value` column-parallel
(each rank gets whole heads), `self_attention.dense` row-parallel.

This is why the row-parallel list has exactly three entries in this model `[M]` — `self_attention.dense`,
`mlp.dense_4h_to_h`, and the shared expert's `dense_4h_to_h`. **Every one is the second half of such a
pair.**

### 7.1b Why the converter must be *told*, and cannot deduce

Stage 2b (§6.2) has `slices`, one per TP rank, and must produce one tensor. Take a weight whose stored
form is `[4096, 2048]` at `TP=2`, so each rank holds `[2048, 2048]`:

```
if it came from a ColumnParallelLinear        if it came from a RowParallelLinear
  stored A is [4096, 2048], split on dim 0      stored A is [2048, 4096], split on dim 1
  rank0 = stored[0:2048, :]                     rank0 = stored[:, 0:2048]
  rank1 = stored[2048:4096, :]                  rank1 = stored[:, 2048:4096]
  merge = cat(dim=0) -> [4096, 2048]            merge = cat(dim=1) -> [2048, 4096]
```

Both cases hand stage 2b **two `[2048, 2048]` tensors**. Same count, same shapes, same dtype, same
numel. The full shapes differ — but the converter never sees a full shape: `param_shapes` records only
the *per-rank* shape (§6.2), and nothing in the checkpoint records what it was before sharding.

**So the fragments are genuinely ambiguous.** The distinguishing fact is which `nn.Module` produced the
parameter, and the converter is an offline script with no model (§2).

**UCP therefore inverts the dependency: the model declares, the converter obeys.** That declaration is
`universal_checkpoint_info`, built by `megatron/checkpointing.py:775` at save time from
`model[0].universal_checkpoint_info()`, and read back by the converter.

#### What the PRISTINE tree declares — and why it is all dead

`[C]` `ELMOE/X-MoE/…/megatron/model/gpt_model.py:220`, verbatim:

```python
info[VOCABULARY_PARAMETER_PATTERNS] = [
    r"tied_modules.embed.word_embeddings.weight"]                     # 1 entry
info[TP_REPLICATED_PARAMETER_PATTERNS] = [
    r"tied_modules.embed.position_embeddings.weight",
    r"\d+.input_layernorm.weight",   r"\d+.input_layernorm.bias",
    r"\d+.post_attention_layernorm.weight", r"\d+.post_attention_layernorm.bias",
    r"\d+.self_attention.dense.bias", r"\d+.mlp.dense_4h_to_h.bias",
    r"\d+.weight",                    r"\d+.bias"]                    # 9 entries
info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = [
    r"\d+.mlp.dense_4h_to_h.weight",
    r"\d+.self_attention.dense.weight"]                               # 2 entries
```

Twelve regexes, and **there is no `PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0` list at all** — pristine has
no swiglu handling whatsoever.

More importantly, **all twelve match nothing in this model.** They are written in the `GPTModelPipe`
namespace: `tied_modules.embed.*` is the pipeline-tied embedding module, and a leading `\d+.` is a
pipeline *stage index*. This run uses `--no-pipeline-parallel`, so `pretrain_gpt_deepspeed.py` builds
**`GPTModel`**, whose parameters are named `language_model.encoder.layers.1.…`. `get_matched_pattern`
uses `re.match`, which anchors at position 0 — so `\d+.input_layernorm.weight` cannot fire against a
name beginning `language_model`.

> **Consequence.** On the pristine tree every parameter falls through to the default branch
> (`cat_dim = 0`). At `TP=1` that is harmless — every branch is the identity (§6.2) — which is why the
> conversion attempt got as far as the merge phase at all. At any `TP > 1` it is silently wrong in the
> two numel-preserving ways of §6.2, and with `--strict` (the default) all twelve unmatched patterns
> trip the assert at `:388`.

#### What the WORKING tree declares

`[M]` The lists below are read out of the **checkpoint** `2026-08-02_5142659/global_step60`, which was
written by the **`X-MoE/` working tree**, not by `ELMOE/X-MoE`. They are shown because they are what a
*correct* declaration for this model looks like — the pristine tree cannot express most of them (it has
no shared experts, and no swiglu list):

```
original_vocab_size = 100015          padded_vocab_size = 100096

vocabulary_parameter_patterns (2)
    language_model\.embedding\.word_embeddings\.weight
    language_model\.output_layer\.weight                              <- untied output layer
tp_replicated_parameter_patterns (4)
    …layers\.\d+\.input_layernorm\.weight
    …layers\.\d+\.post_attention_layernorm\.weight
    …encoder\.final_layernorm\.weight
    …layers\.\d+\.mlp\.deepspeed_moe\.gate\.wg\.weight                <- the router
parameter_with_row_parallelism_patterns (3)
    …layers\.\d+\.self_attention\.dense\.weight
    …layers\.\d+\.mlp\.dense_4h_to_h\.weight
    …layers\.\d+\.mlp\.shared_experts\.shared_mlp\.dense_4h_to_h\.weight   <- X-MoE only
parameter_with_2_sub_params_cat_dim_0 (2)
    …layers\.\d+\.mlp\.dense_h_to_4h\.weight
    …layers\.\d+\.mlp\.shared_experts\.shared_mlp\.dense_h_to_4h\.weight   <- X-MoE only
```

Every entry is the `language_model.…` namespace, and each row-parallel entry is the second half of a
column-then-row pair (§7.1). Note what is **absent**: `dense_h_to_4h` appears only in the swiglu list,
and `query_key_value` appears nowhere. Both are column-parallel, and column-parallel is the *default*
branch — **only deviations from the default are declared.**

### 7.2 Why *these* classes, and not others

The lists are not a taxonomy of layer types — they are a taxonomy of **inverse operations**. A class
exists precisely when undoing it needs an operation the others cannot express:

| Class | What TP did | Merge | Split (the inverse) | Why it needs its own class |
|---|---|---|---|---|
| *(default)* | cut dim 0 | `cat(dim=0)` | `chunk(dim=0)` | the baseline |
| `..._ROW_PARALLELISM` | cut dim 1 | `cat(dim=1)` | `chunk(dim=1)` | a different **axis** |
| `TP_REPLICATED_...` | nothing — every rank identical | take `slices[0]` | broadcast | there is no cut to invert; concatenating would inflate numel `t`× |
| `..._TO_AVERAGE` | ranks drifted apart | `mean(slices)` | copy | merging must be **lossy**; no `cat` can express it |
| `..._2_SUB_PARAMS_CAT_DIM_0` | cut dim 0, but the tensor is **two** logical matrices | de-interleave, then `cat` per half | `chunk` twice, re-interleave | one tensor, two independent cuts |
| `..._WITH_SUB_PARAMS` | cut dim 0, sub-blocks of **unequal** size (GQA q/k/v) | per-sub-dim `narrow` + `cat` | inverse | the split points are irregular, so shape info must travel with the atom |
| `VOCABULARY_...` | cut dim 0, **plus** padding to a TP multiple | `cat(dim=0)` then strip pad | re-pad for the *target* TP, then `chunk` | numel is **not** conserved: padding depends on the target degree |

Two entries earn their place for reasons worth stating explicitly:

**Swiglu (`2_SUB_PARAMS`).** `dense_h_to_4h` is one tensor holding gate and up stacked on dim 0,
consumed as `chunk(x, 2, dim=-1)`. Because that chunk runs on each rank's *local* output, the sharded
convention is per-rank interleaved — rank *r* stores `[gate_r ; up_r]` — while the canonical form is
`[gate_all ; up_all]`. A plain `cat(dim=0)` produces `gate_0 ; up_0 ; gate_1 ; up_1`: **same shape,
same numel, wrong permutation.** Nothing downstream can detect it (§17).

**Vocabulary.** `padded_vocab_size` is `make_vocab_size_divisible_by × TP`, so it is a function of the
topology. An atom must be topology-free (§0), so the padding is stripped on merge and recomputed on
load. `[M]` Visible in the artifact: `padded_vocab_size = 100096`, but the atom is

```
word_embeddings.weight   PARAM shape (100015, 2048)   {'cat_dim': 0, 'vocab_tensor': True}
```

**100015 = `original_vocab_size`.** The 81 padding rows are gone from the atom, exactly as §0 requires.

### 7.3 What gets recorded, and why replication records *nothing*

Merge writes its decision into the atom (§6.2), and the loader replays it (§13). `[M]` Every
distinct metadata shape actually present, read off the converted checkpoint:

```
parameter                              PARAM shape       metadata besides PARAM
word_embeddings (vocab)              (100015, 2048)      {'cat_dim': 0, 'vocab_tensor': True}
output_layer (vocab)                 (100015, 2048)      {'cat_dim': 0, 'vocab_tensor': True}
input_layernorm (replicated)               (2048,)      {}
router gate.wg (replicated)              (64, 2048)      {}
self_attention.dense (row-parallel)    (2048, 2048)      {'cat_dim': 1}
query_key_value (default col-par)      (6144, 2048)      {'cat_dim': 0}
layer-0 mlp.dense_h_to_4h (swiglu)    (21888, 2048)      {'cat_dim': 0, 'param_n_sub_params': 2}
shared_expert h_to_4h (swiglu)         (5632, 2048)      {'cat_dim': 0, 'param_n_sub_params': 2}
routed expert h_to_4h (Unique)         (2816, 2048)      {'cat_dim': 0}
```

`[M]` Over a random sample of 120 of the 3654 atoms: **117 carry `{cat_dim}` alone, 3 carry `{}`.**

The `{}` rows are the point. **Replication is encoded by absence** — the replicated branch of §6.2
sets no key at all. That is not an oversight: on load, a replicated atom already has the parameter's
full shape, so `universal_checkpoint.py:57`'s `full_hp_param.shape == self.shape` fires, forces
`tp_rank, tp_world_size = 0, 1`, and copies it whole. The metadata is unnecessary because the shape
re-derives it. Every other class *must* record, because its inverse is not recoverable from shapes.

Note the last row: routed experts get the plain `cat_dim: 0` default. That is **correct here** —
`enable_expert_tensor_parallelism=False`, so experts are never TP-cut, and with one slice
`cat([x], dim=0)` is the identity. They are the paper's `Unique` pattern, and they need no entry.

### 7.4 Three rules the contract imposes

1. **Omission is not neutral.** An unregistered parameter is not skipped — it silently takes the
   default column-parallel isomorphism (`:318`).
2. **At most one pattern may match.** `:252` asserts `len(matched_) <= 1`. So regexes must be mutually
   exclusive and fully anchored — note above how `\.mlp\.dense_h_to_4h` stays disjoint from
   `\.mlp\.shared_experts\.shared_mlp\.dense_h_to_4h` only because of the extra path segments.
3. **Terminology trap.** `parameter_with_row_parallelism_patterns` means *"belongs to a
   `RowParallelLinear`"*. The **columns** of the `[out, in]` weight are split and the merge axis is
   **1**. The list says "row"; the axis is 1.

**`--strict` is the only automatic check that the declaration matches the model** (`:388`): each merge
worker starts with the union of all pattern strings, `discard`s those it matched, and the intersection
across workers — matched by nobody — trips the assert.

## 8. Phase 3 — `optimizer_state.pt`

Phases 1–2 handled everything that is *per parameter*. But an optimizer also carries state belonging to
no parameter at all — learning rate, betas, `step`, the loss scaler, the clip threshold. It is
identical on every rank, so it needs neither sharding nor merging, only to be preserved once.

`_save_optimizer_state:410`:

```python
sharded_states = [BASE_OPTIMIZER_STATE, PARAM_SLICE_MAPPINGS, SINGLE_PARTITION_OF_FP32_GROUPS]
sd        = ds_checkpoint.get_zero_checkpoint_state(pp_index=0, tp_index=0, dp_index=0)  # ANY rank
optim_sd  = sd[OPTIMIZER_STATE_DICT]
output_sd = {k: v for k, v in optim_sd.items() if k not in sharded_states}   # drop the per-rank data
output_sd[PARAM_GROUPS] = optim_sd[BASE_OPTIMIZER_STATE][PARAM_GROUPS]       # lift out the scalars
_save_checkpoint(os.path.join(args.output_folder, "zero", "optimizer_state.pt"), output_sd)
```

### 8.1 Every key, and why it is kept or dropped

`[M]` The input `optim_sd` has 11 keys. Three are dropped, eight survive, one is added:

| Key | | What it is | Why |
|---|---|---|---|
| `base_optimizer_state` | **drop** | Adam `exp_avg` / `exp_avg_sq`, per group | per-rank; **it is now the atoms** (§5–6) |
| `single_partition_of_fp32_groups` | **drop** | this rank's fp32 master slice | per-rank; **it is now the atoms** |
| `param_slice_mappings` | **drop** | name → `fragment_address(start, numel)` | the index into a flat buffer that no longer exists; meaningless once atoms are per-parameter |
| `loss_scaler` | keep | `LossScaler` object | scalar training state |
| `dynamic_loss_scale`, `overflow` | keep | `False`, `False` | scalar |
| `clip_grad` | keep | `1.0` | scalar |
| `zero_stage` | keep | `ZeroStageEnum(1)` | provenance |
| `ds_version` | keep | `'0.15.5+unknown'` | provenance; `_load_global_state` **asserts** it is non-empty |
| `group_paddings` | keep | `[0] × 13` | see 8.2 |
| `partition_count` | keep | `[8] × 13` | see 8.2 — **and it was rewritten in transit** |
| `param_groups` | **add** | lifted from `base_optimizer_state[PARAM_GROUPS]` | **this is the point of phase 3** |

`param_groups` is why the phase exists — the optimizer hyperparameters, with no tensors in them.
`[M]` Group 0:

```
name wd_no_scale_lr   lr 8.1199104e-06   betas (0.9, 0.95)   eps 1e-08
weight_decay 0.1      wd_mult 1.0        lr_mult 1.0         step 60      params [0]
```

`step = 60` is the one that matters most: it is the Adam bias-correction counter, and losing it would
silently restart bias correction at iteration 0.

**`[M]` The entire file is 3,357 bytes** — 198 GiB of atoms, and 3 KB of everything else.

Reading from `(0, 0, 0)` is safe precisely *because* nothing left in `output_sd` is rank-dependent —
that is the same claim the `sharded_states` filter makes, just stated positively.

### 8.2 The one thing that is **not** rank-independent — a stock-UCP defect

`[M]` The source shard and the produced file disagree:

```
SOURCE  bf16_zero_pp_rank_0 …  partition_count = [8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SOURCE  bf16_zero_pp_rank_1 …  partition_count = [8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
OUTPUT  zero/optimizer_state.pt partition_count = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
```

`ds_to_universal.py` never mentions the field `[C]`. The rewrite happens inside the *read*, at
`zero_checkpoint.py:143`, called from `get_state_for_rank:74`:

```python
def _update_partition_count(self, sd):
    partition_counts = self._get_optimizer_state(sd, PARTITION_COUNT)
    if partition_counts:
        num_groups = len(partition_counts)
        sd[OPTIMIZER_STATE_DICT][PARTITION_COUNT] = [self.target_3d.dp_degree] * num_groups
```

**It overwrites every group with one global `dp_degree`.** For a dense model that is correct and even
necessary — resharding DP changes the partition count, and the target value is what a resumed run
should see. For MoE it is wrong: §5.1 established that `partition_count[g]` is `DP_non_moe` for
non-expert groups and **`DP_moe` for expert groups**, and the save side knows this
(`stage_1_and_2.py:629` sets expert groups from `expert_dp_process_group`). This line flattens two
distinct data-parallel degrees into one.

It is **latent today**: `[C]` `_load_global_state` reads only `loss_scaler`, `dynamic_loss_scale`,
`overflow`, `clip_grad` and `ds_version` — never `partition_count` — so nothing currently consumes the
wrong value. But it is the same assumption that produces blocker 2 and §16: **one global degree for
all groups**, applied to a checkpoint that has two.

`group_paddings` survives for a related reason — `[M]` it is `[0] × 13` here, and
`_clear_group_paddings` zeroes it when `strip_tensor_paddings` is on, since padding is a property of
the source partitioning and must not follow the atoms (§0).

## 9. The output

Phases 1–3 in one tree — per-parameter atoms from 1–2, the rank-independent remainder from 3, and the
client state copied verbatim:

```
global_step60_universal/
├── zero/
│   ├── optimizer_state.pt                          scalar/global state
│   └── <param_name>/                                3655 atom dirs  [M]
│       ├── fp32.pt        {PARAM: tensor, CAT_DIM: 0, ...}
│       ├── exp_avg.pt
│       └── exp_avg_sq.pt
└── mp_rank_00_model_states.pt                       copied verbatim (client state, args, rng)
```

plus, one level up, a `latest_universal` file containing `global_step60_universal`.

`[M]` 3655 = 199 non-expert + 3456 expert. 198 GiB.

Note the atom is a **dict**, not a bare tensor: `{PARAM: <tensor>, CAT_DIM: 0}` or
`{PARAM: <tensor>, VOCAB_TENSOR: True}`. That metadata is the merge decision, recorded for the loader.

---

# PART II — LOAD AND RUN

## 10. The flag

`--universal-checkpoint` is declared at `megatron/arguments.py:1078` and does exactly one thing —
`megatron/training.py:78-79`:

```python
if args.universal_checkpoint:
    ds_config_dict["checkpoint"] = {"load_universal": True}
```

**It affects load only.** Saving is byte-identical with or without it.

`--load` takes the **parent** directory, not the `_universal` one; the engine appends the tag it
reads from `latest_universal`.

## 11. `engine.load_checkpoint` — `engine.py:2799`

§10's flag reaches the engine as a single boolean, and the engine consults it in **four** places.
Reading them together is the fastest way to see what a universal load actually changes:

```python
latest_tag = "latest_universal" if self.load_universal_checkpoint() else "latest"     # :2832
...
self._load_checkpoint(...)                                                           # :2851
    DeepSpeedEngine.load_moe_state_dict(...)                                         # :2925
    if not self.load_universal_checkpoint():                                         # :2933
        self.load_module_state_dict(checkpoint=checkpoint, ...)                      # :2934   SKIPPED
...
if (load_optimizer_states and not load_module_only) or self.load_universal_checkpoint():   # :2861
    self._load_zero_checkpoint(load_dir, tag, ...)                                   # :2862
        checkpoint_folder = f'{os.path.join(load_dir, tag)}' if load_universal else None   # :3038
        self.optimizer.load_state_dict(..., checkpoint_folder=checkpoint_folder)
if self.load_universal_checkpoint():                                                 # :2884
    self.optimizer.update_lp_params()                                                # :2885
```

**Four behavioural switches, all on the same flag.** The one to internalise is `:2933`: under UCP,
`load_module_state_dict` is **skipped entirely**. The bf16 weights in `mp_rank_*` are never applied.
They are regenerated from the fp32 atoms at `:2885` (§14).

**Consequence: there is no "UCP for the dense part only."** Once you resume universally, *every*
parameter's value comes from an atom.

## 12. `base_optimizer.py:20` — resolving atoms

§11 handed the optimizer a folder and stopped. Now the central question of the load path: given a live
model, **which atom belongs to which parameter, and which slice of it does this rank need?**

Reached via `stage_1_and_2.py:2298`, which passes the group name `"bit16_groups"`:

```python
def load_hp_checkpoint_state_from_checkpoint_dir(self, lp_groups_name, checkpoint_dir):
    checkpoint_dir = os.path.join(checkpoint_dir, "zero")                       # :21
    optim_sd = torch.load(os.path.join(checkpoint_dir, "optimizer_state.pt"))   # :23
    self._load_global_state(optim_sd)                                           # :25

    tp_rank       = bwc_tensor_model_parallel_rank(mpu=self.mpu)                # :27
    tp_world_size = self.mpu.get_tensor_model_parallel_world_size()             # :32

    for i, (param_group, loaded_param_group) in enumerate(
            zip(self.optimizer.param_groups, optim_sd['param_groups'])):        # :38
        for lp in getattr(self, lp_groups_name)[i]:
            if lp._hp_mapping is not None:
                step = lp.load_hp_checkpoint_state(
                    os.path.join(checkpoint_dir, self.param_names[lp]),         # :47  <- the address
                    tp_rank, tp_world_size)
        map_to_flat_opt_states(param_group['params'][0], lp_groups[i], self.optimizer.state, opt_keys)
        for key, value in loaded_param_group.items():
            if key != 'params': param_group[key] = value
```

Three things happen here:

1. **`self.param_names[lp]` is the atom address.** Straight `named_parameters()` name, joined onto
   `zero/`. This is the whole addressing scheme.
2. **`lp._hp_mapping is not None`** filters to parameters this rank owns a slice of, established by
   `_link_all_hp_params` (`stage_1_and_2.py:580`). Each rank loads only its own share — the read is
   naturally distributed.
3. **`zip(param_groups, optim_sd['param_groups'])` is positional** (`:38`). Fine when the group
   *count* is topology-independent.

> **[X-MoE] Both `:47` and `:38` break under EP.** `param_names[lp]` is rank-local for experts, so
> every EP rank resolves to the same `…deepspeed_experts.0/` folder — **and raises nothing**. And
> the MoE param-group count is a function of `num_experts/EP`, so the positional `zip` mis-associates
> the moment EP changes.

## 13. `universal_checkpoint.py:22` — the per-parameter read

§12 resolved *which folder*. This is what happens inside it — and it is exactly §6.2 run backwards,
driven by the metadata that stage recorded.

Installed onto each parameter by `enable_universal_checkpoint` (`:144`), which sets
`param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)`.

```python
for key in hp_keys:                       # {fp32, exp_avg, exp_avg_sq, step} discovered by listdir
    ckpt_dict     = torch.load(os.path.join(folder, f"{key}.pt"))
    full_hp_param = ckpt_dict[PARAM]

    if full_hp_param.shape == self.shape:                    # :57  replicated / TP=1 short-circuit
        tp_rank, tp_world_size = 0, 1

    if ckpt_dict.get(VOCAB_TENSOR, False):                   # :65  re-pad for the TARGET tp
        padded_target_vocab_size = self.shape[0] * tp_world_size
        full_hp_param = F.pad(full_hp_param, (0, 0, 0, padding_size))

    assert full_param_numel == tp_world_size * tp_slice_numel     # :82  THE ONLY correctness guard

    chunk_dim    = ckpt_dict.get(CAT_DIM, 0)                 # :92  <- read back, never re-derived
    n_sub_params = ckpt_dict.get(PARAM_N_SUB_PARAMS, 1)      # :93
    if sub_param_shape:        ...                           # :88  GQA inverse
    elif n_sub_params > 1:                                   # :113 swiglu inverse
        sub_params  = full_hp_param.chunk(n_sub_params, dim=chunk_dim)
        tp_hp_slice = torch.cat([p.chunk(tp_world_size, chunk_dim)[tp_rank] for p in sub_params], chunk_dim)
    else:
        tp_hp_slice = full_hp_param.chunk(tp_world_size, chunk_dim)[tp_rank]   # :119
    tp_hp_slice = tp_hp_slice.flatten()                      # :121
```

**This is the exact inverse of §6.2, driven by the recorded metadata.** The symmetry is the design:
the converter decides, the loader obeys.

Two properties worth stating plainly:

- **The `:57` early-out is why TP=1 works with completely wrong patterns.** If the atom already has
  the parameter's shape, the loader short-circuits to a straight copy and no pattern is applied.
- **`:82` is the only correctness guard, and it is a numel check.** A wrong merge that preserves
  numel — row-parallel merged on dim 0, or swiglu halves interleaved — is **undetectable by
  construction**. Those are exactly the two cases that matter for a swiglu MoE model.

The slice is finally written into the fp32 master partition and Adam state via `hp_mapping`, and
`map_to_flat_opt_states` (back in `base_optimizer.py`) reassembles the optimizer's flat buffers.

## 14. `update_lp_params` — bf16 from fp32

§13 restored the **fp32** master weights and Adam state. But the model computes in **bf16**, and §11
showed `load_module_state_dict` is skipped — so nothing has written the bf16 working weights yet.
This is where they come from.

`stage_1_and_2.py:1938`, called at `engine.py:2885`:

```python
for i, (bit16_partitions, fp32_partition) in enumerate(
        zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
    partition_id = dist.get_rank(group=self.real_dp_process_group[i])
    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
```

The bf16 working weights are a **downcast of the fp32 masters**. Because the same copy runs after
every optimizer step during normal training, `bf16 == fp32.to(bfloat16)` bit-exactly at save time —
which is why the bf16 files are redundant under UCP and why skipping `load_module_state_dict` loses
nothing.

Note `real_dp_process_group[i]` — per-group, so this is already EP-correct once the atoms are.

## 15. Megatron touchpoints — the whole surface

Everything in §§3–14 was DeepSpeed. Megatron's contribution is small but load-bearing: it supplies the
one thing DeepSpeed cannot know — **what kind of parameter each tensor is** (§7) — plus the flag and
one RNG fix-up.

| File:line | Lines | Role |
|---|---|---|
| `megatron/arguments.py:1078` | 2 | `--universal-checkpoint` |
| `megatron/training.py:78-79` | 2 | injects `{"checkpoint": {"load_universal": True}}` |
| `megatron/checkpointing.py:259` | 1 | emits `UNIVERSAL_CHECKPOINT_INFO` at save |
| `megatron/checkpointing.py:775` | 10 | `_universal_checkpoint_info()` builds it |
| `megatron/checkpointing.py:696-716` | 20 | TP-seed reconfiguration (`:716`) when the mp degree changed |
| `megatron/model/gpt_model.py:220` / `:393` | 28 / 33 | the pattern lists |
| `megatron/model/module.py:118` | 2 | default `{}` |

That is the entire Megatron-side integration: **one flag, one save line, one info dict, one seed
reconfiguration.**

The seed reconfiguration at `:716` is easy to overlook and has real numerical consequences:

```python
if args.universal_checkpoint:
    ...
    tensor_parallel.model_parallel_reconfigure_tp_seed(args.seed + iteration)
```

A TP-changing universal load re-seeds the model-parallel RNG, so dropout masks differ from an
uninterrupted run. `[M]` This alone accounted for a 0.055 loss gap in a TP1-vs-TP4 comparison until
`--attention-dropout 0 --hidden-dropout 0` were set.

---

## 16. Where it breaks for expert parallelism — **[X-MoE]**

Read after §§0–15. Self-contained: every file, line and number used here is given below.

### 16.1 The setup, in plain numbers

The model has **64 experts**. The job runs on **8 GPUs** with `EP=8`, so each GPU owns 8 of them:

```
GPU 0  owns experts   0  1  2  3  4  5  6  7
GPU 1  owns experts   8  9 10 11 12 13 14 15
GPU 2  owns experts  16 17 18 19 20 21 22 23
 ...
GPU 7  owns experts  56 57 58 59 60 61 62 63
```

No expert is split. Each one lives whole, on exactly one GPU.

### 16.2 The problem: every GPU calls its own experts "0 … 7"

`[C]` `deepspeed/moe/experts.py:40` builds each GPU's experts as an ordinary local list:

```python
self.deepspeed_experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
```

A list of 8 is indexed 0–7. So `named_parameters()` on **GPU 1** returns:

```
…deepspeed_experts.0.dense_h_to_4h.weight     ← this is actually expert 8
…deepspeed_experts.1.dense_h_to_4h.weight     ← actually expert 9
 …
…deepspeed_experts.7.dense_h_to_4h.weight     ← actually expert 15
```

`[M]` All 8 GPUs report the **same 432 expert-parameter names** — intersection 432/432, verified
across all eight `bf16_zero_pp_rank_*` shards. **The number in the name is the position within that
GPU, not which expert it is.**

#### What 432 counts — it is not "432 experts"

`[M]` Decomposing GPU 0's name set:

```
distinct transformer layers   : 27    (layers 1..27; layer 0 is dense, --first-k-dense-replace 1)
distinct LOCAL expert ids     :  8    (0..7)
distinct matrices per expert  :  2    (dense_h_to_4h, dense_4h_to_h  -- SwiGLU)

27 × 8 × 2 = 432        measured: 432
```

Two facts that are easy to conflate:

- **"64 experts" is per layer, not per model.** Each of the 27 MoE layers has its own 64 experts, so
  the model holds `27 × 64 = 1728` expert modules — matching the 1728 `layer_<L>_expert_<E>_*.pt`
  weight files on disk `[M]`.
- **Each expert is 2 tensors, not 1** — SwiGLU gives `dense_h_to_4h` `[2816, 2048]` and
  `dense_4h_to_h` `[2048, 1408]`.

So model-wide there are `27 × 64 × 2 = 3456` expert tensors — exactly the 3456 expert atoms a correct
conversion produces (§9), and 432 of them (one eighth) live on each GPU.

The layer index is already distinct in the name, so a single directory such as
`tmp/…layers.1.…deepspeed_experts.0.dense_h_to_4h.weight/` receives **one file per GPU** — 8 — not one
per layer.

### 16.3 What that does to the converter

Extract (§5.3) writes each parameter into a directory **named after the parameter**
(`ds_to_universal.py:148` → `dump_param_fragment:179`). Walk it GPU by GPU:

```
GPU 0 reports "…deepspeed_experts.0"  →  writes expert  0  into  tmp/…deepspeed_experts.0/
GPU 1 reports "…deepspeed_experts.0"  →  writes expert  8  into  tmp/…deepspeed_experts.0/  ← same dir
GPU 2 reports "…deepspeed_experts.0"  →  writes expert 16  into  tmp/…deepspeed_experts.0/  ← same dir
 …
GPU 7 reports "…deepspeed_experts.0"  →  writes expert 56  into  tmp/…deepspeed_experts.0/
```

`[M]` The resulting tree:

```
tmp/…deepspeed_experts.0/0/   8 files, 23,070,249 B each   ← experts 0, 8, 16, 24, 32, 40, 48, 56
tmp/…deepspeed_experts.1/0/   8 files                      ← experts 1, 9, 17, …, 57
 …
tmp/…deepspeed_experts.7/0/   8 files                      ← experts 7, 15, 23, …, 63

tmp/…deepspeed_experts.8/     does not exist
 …                            56 of the 64 experts have no directory at all
```

`2816 × 2048 × 4 B = 23,068,672` — so each of those files is a **whole expert's `dense_h_to_4h`**, not
a piece of one. `[M]` Distinct expert ids present anywhere in `tmp/`: **8, out of 64.**

Counting directories confirms the loss precisely: a correct conversion needs
`27 layers × 64 experts × 2 matrices = 3456` expert directories. This run produced
`27 × 8 × 2 = 432` — **one eighth** — each holding 8 files that should have been in 8 different
directories.

### 16.4 Why merge then crashes

Stage 2a (§6.1) reads `tmp/…deepspeed_experts.0/0/`, finds 8 files, and does what §6.1 derived: **8
files in one directory means 8 pieces of one tensor**, so glue them in rank order and reshape.

```
8 experts × 5,767,168 elements = 46,137,344
reshape to [2816, 2048]        =  5,767,168
```

```
ds_to_universal.py:226   slice = torch.cat(shards, dim=0).reshape(slice_shape)
RuntimeError: shape '[2816, 2048]' is invalid for input of size 46137344
```

`[M]` It fires at merge item **198 of 630** — `630 = 198 non-expert + 432 expert`, so every
non-expert parameter merged correctly and the **first routed expert** failed.

### 16.5 Why that assumption is right everywhere else

For an ordinary parameter, multiple files in one directory really are pieces of one tensor. `[M]` The
embedding (§6.1):

```
tmp/…word_embeddings.weight/0/  file from GPU 0 = the first 177,209,344 numbers
                                file from GPU 1 = the next   27,787,264 numbers
                                                  (2 files; GPUs 2-7 hold none of it)
```

Gluing those gives back `100096 × 2048`. **Correct.**

The directory layout is identical in both cases, and the converter cannot tell them apart:

| Multiple files in one directory | Means | Correct action |
|---|---|---|
| embedding, layernorm, attention… | *n* **pieces of one** tensor, split by ZeRO | glue them |
| routed experts | *n* **separate whole** tensors, one per GPU | must **not** glue them |

That is the entire bug. **The name is being used as the address, and for experts the name is not
unique.**

### 16.6 Where the fix already exists — a pipeline this walkthrough never traced

Expert **weights** do not have this problem, and the reason is one line of code you have not seen yet,
because it lives in a pipeline §3 explicitly set aside:

> §3: *"Note what `main()` does **not** do: it never looks at `layer_*_expert_*` files."*

There are **two** independent pipelines for expert data:

```
EXPERT WEIGHTS (bf16)                        EXPERT OPTIMIZER STATE (fp32 + Adam m,v)
─────────────────────────────────            ────────────────────────────────────────
save  engine.py:_save_moe_checkpoint         save  engine.py:_save_zero_checkpoint
        :3283  RENAMES local → global                 (no rename)
        → layer_<L>_expert_<E>_*.pt                   → bf16_zero_pp_rank_<R>_*.pt

load  engine.py:load_moe_state_dict          convert  ds_to_universal.py
        :2656  renames global → local                  :148  (no rename)   ← THE BUG
                                             load     base_optimizer.py:47 (no rename)
```

**Everything in Parts I and II of this document is the right-hand column.** The left-hand column is the
weight path, and it solved this in 2021.

`[C]` The rename, `engine.py:3283`, inside `_save_moe_checkpoint`:

```python
global_expert_id = expp_rank * num_local_experts + int(local_expert_id)
expert_key = key.replace(f'{moe_str_prefix}{local_expert_id}',
                         f'{moe_str_prefix}{global_expert_id}')
experts_state_dict[str(global_expert_id)][expert_key] = truncated
```

`expp_rank` is this GPU's index within the expert-parallel group; `num_local_experts` is 8. So GPU 1's
local expert 0 is written as global expert `1 × 8 + 0 = 8`. `[C]` The inverse runs on load,
`engine.py:2656` in `load_moe_state_dict`.

`[M]` That is why the weight file `layer_0_expert_63_mp_rank_00_model_states.pt` genuinely contains
`…deepspeed_experts.63…`, while every optimizer shard still says `…deepspeed_experts.0…7`.

**The optimizer path was simply never given the same line.**

### 16.7 The fix

Apply the identical transform where the optimizer path assigns an address:

```
global_id = expp_rank × num_local_experts + local_id
```

Then GPU 1 writes into `tmp/…deepspeed_experts.8/`, all 64 directories exist with exactly one file
each, and **every stage downstream works unmodified**: stage 2a glues one file to nothing (a no-op),
no pattern fires (each expert is the paper's `Unique`), and the loader resolves a distinct directory
per GPU.

Two call sites, mirroring the weight path:

| Where | Direction | Line |
|---|---|---|
| convert · before `dump_param_fragment` | local → global | `ds_to_universal.py:148` |
| load · resolving the atom directory | global → local | `base_optimizer.py:47` |

Three supporting pieces, all forced by the converter being **offline with no process group** (§2):

1. **One shared helper** in `deepspeed/checkpoint/`, importable without a process group — the two
   call sites must be bit-symmetric, so they cannot be two implementations.
2. **`expp_rank` and `num_local_experts` recorded at save time.** They cannot be re-derived offline:
   `[C]` `groups.py` has two group-construction functions selected by
   `enable_expert_tensor_parallelism`, each with two rank interleavings selected by
   `use_data_before_expert_parallel_`, and they use *different* world sizes. **Record, never infer.**
3. **`param_shapes` expanded 8 → 64 experts.** `[M]` It is written by one rank, so it lists only that
   rank's 8 local experts (§4.4) — the `630 → 3654` expansion.

### 16.8 The load side is the dangerous half

The convert side fails **loudly** — the `RuntimeError` above. Its mirror does not.
`[C]` `base_optimizer.py:47` resolves `os.path.join(checkpoint_dir, self.param_names[lp])`, which for
**every** GPU is the same `…deepspeed_experts.0/` directory. All 8 would load GPU 0's experts, and
`universal_checkpoint.py:82`'s only guard is a numel check that a correctly-shaped wrong tensor passes
(§13).

**The crash is the lucky branch.** A fix that touched only the converter would produce a checkpoint
that loads silently wrong.

### 16.9 Why UCP was built this way

An atom is addressed by flat parameter name (§0), assumed topology-invariant. That holds for TP, PP and
DP: all three **cut a tensor up**, so identity is fixed and only layout varies — which is why all seven
of the paper's patterns are layout transforms (§7.2). EP does not cut anything; it distributes
*different tensors*, and the local `ModuleList` makes their names collide.

`[P]` The paper's MoE case does not hit this, because its expert weight is a **fused** matrix
`[n_experts × hidden_out, hidden_in]` — expert identity is an offset *inside* one tensor, so one name
still means one tensor. `[P]` Table 1's columns are DP, TP, PP, SP, ZeRO; **there is no EP column.**
X-MoE uses separate per-expert matrices, the branch §5.3.2 explicitly sets aside.

**So this is a naming problem, not a maths problem** — no new tensor operation is required, and no
existing pattern changes.

### 16.10 Every file involved

| File | Line | Role | Traced in this doc? |
|---|---|---|---|
| `deepspeed/moe/experts.py` | 40 | builds the local `ModuleList` — the origin of the local numbering | no (model code) |
| `deepspeed/runtime/engine.py` | 3283 | **weight** save: local → global rename | no (§3 excludes it) |
| `deepspeed/runtime/engine.py` | 2656 | **weight** load: global → local rename | no (§3 excludes it) |
| `deepspeed/checkpoint/ds_to_universal.py` | 148 | optimizer convert: passes `name` unchanged — **the defect** | §5.3 |
| `deepspeed/checkpoint/ds_to_universal.py` | 179 | writes `tmp/<name>/<tp>/<state>.<dp>` | §5.3 |
| `deepspeed/checkpoint/ds_to_universal.py` | 226 | glues the fragments — **where it crashes** | §6.1 |
| `deepspeed/runtime/base_optimizer.py` | 47 | optimizer load: resolves the atom directory — **the silent mirror** | §12 |
| `deepspeed/checkpoint/universal_checkpoint.py` | 82 | the numel-only guard | §13 |

Design and implementation: `UCP_EP_INCORPORATION_PLAN.md`, `UCP_MOE_EP_INTEGRATION.md`.
The live run log: `0804_2026_UCP_BLOCKERS_ENCOUNTERED.md` Blocker 5.

## 17. Landmines

Each of these is a direct consequence of a mechanism above, not a list of unrelated gotchas. The
section that explains it is named in brackets.

1. **[§6.2, §13] A wrong pattern's merge and split cancel exactly at unchanged degree.** A same-topology
   save→convert→resume test passes vacuously. **Any pattern test must change a degree.**
2. **`:82` is a numel check.** Row-parallel merged on the wrong dim, and swiglu halves interleaved,
   both preserve numel. Silent, catastrophic, and invisible to every guard in the system. Discriminate
   with `w[1,0]`, not with shapes.
3. **`--no_strict` is exactly the flag that turns those two from loud to silent.** Do not reach for it
   to make a conversion "work".
4. **Stale `tmp/` silently corrupts the merge.** `_merge_zero_shards` globs whatever fragments it
   finds; leftovers from an aborted run are indistinguishable from real ones. Stock UCP does not clear
   it at start — delete it manually before a retry.
5. **`_get_zero_stage` reads a whole shard without `mmap`**, and extraction holds one shard *per
   worker* (~23 GiB each here). Bound it with `--num_extract_workers`. **Do not use `mmap` on Lustre**
   — it turns one sequential read into 4 KB demand-paged round trips and the workers wedge in
   uninterruptible sleep.
6. **`--finetune` defeats a universal resume.** It forces `load_module_only=True`, skips optimizer and
   LR state, and resets `iteration=0`. Keep it only when loading a weights-only HF bootstrap.

---

## 18. Suggested reading order

This document went in *logical* order — each mechanism after its motivation. Reading the **source** in
that same order does not work, because the vocabulary comes last. Use this order instead:

1. **`constants.py`** (87 lines) — read in full, first. It defines every key UCP uses; nothing else
   makes sense without it.
2. **`ds_to_universal.py:469-545`** — `main()`. The five phases, in order. Ignore the `zero_stage > 2`
   branch.
3. **`ds_to_universal.py:112-150`** — `extract_zero_shards`, then `dump_param_fragment:179` for the
   `tmp/<name>/<tp>/<state>.<dp>` layout. This is where the atom's *address* is decided.
4. **`ds_to_universal.py:198-230`** — `_merge_zero_shards`. Stage 2a only. Short, and always correct.
5. **`ds_to_universal.py:232-336`** — `merge_tp_slices`. Stage 2b, the pattern branch tree. The one
   function where model semantics enter.
6. **`universal_checkpoint.py:22-121`** — `load_hp_checkpoint_state`. Read it side-by-side with step 5
   and check that each branch is the exact inverse.
7. **`base_optimizer.py:20-62`** — 63 lines, the whole load-side orchestration.
8. **`engine.py:2799-2890`** — skim for the four `load_universal_checkpoint()` switches; note `:2933`.
9. `examples_deepspeed/universal_checkpointing/{README.md, run_bf16.sh}` — the dense reference that
   works out of the box, useful as a control.

**Companions:** `0804_2026_UCP_ON_PRISTINE_TREE.md` (why stock UCP does not cover X-MoE) ·
`0804_2026_UCP_BLOCKERS_ENCOUNTERED.md` (the run log) · `UCP_EP_INCORPORATION_PLAN.md` (the design) ·
`UCP_MOE_EP_INTEGRATION.md` (what X-MoE changed).
