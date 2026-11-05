# ft-sys-v2 heartbeat system

Finds failures in a running training job within seconds, agrees on one failure list, and keeps a record for people. It does not
recover yet: the recovery inside the step (stop, rebuild, move the state, redo) is the next stage of ft-sys-v2.

```
 every 5 s   each rank's monitor thread --its line--> host memory of its node (/dev/shm/ftv2_<job>/rank_<r>.json)
 every 5 s   each node's agent --lease renewal + small beat--> etcd (3 members)       fast heartbeat, for the system
 every 60 s  each agent --its ranks' full lines--> <job>/ft/nodes/<node>.log         slow heartbeat, for people
 always      one agent is the coordinator: compares all nodes, asks for py-spy stacks, writes the failure list failed/<g>
 on a list   every agent copies it to host memory (failed.json); every rank's monitor thread reads it
```

## Files

| File | Runs in | What it does | Why it is a file of its own |
|---|---|---|---|
| `hb_monitor.py` | each training process | wraps torch.distributed's communication calls once (+1 per call); the monitor thread writes the rank's line every 5 s and checks for failed.json every 0.5 s; test injections (FTV2_INJECT) | the only code inside training; memory work only, no GPU call |
| `hb_agent.py` | one process per node | fast beat to etcd, slow beat to Lustre, processes gone (with the cause), py-spy on request, failed.json, takes the coordinator role when it is free | a separate process: a training crash cannot take it down |
| `hb_coordinator.py` | inside one agent | reads all node beats, applies the rules, asks for stacks, writes the fenced failure list, the summary line on Lustre; continues from etcd after a takeover | the role moves between agents; kept apart it is tested alone |
| `hb_rules.py` | (pure functions) | the decisions: hang (counts, then stacks), lost node (lease, then the others still), crash and its cause, out of memory | no input or output: every case is a unit test |
| `hb_stacks.py` | (pure functions) | sorts py-spy stacks into inside and outside a communication wait (ft-sys-v1's name_culprit.py rule) | reused from a tested rule |
| `hb_etcd.py` | agents | etcd v3 over its HTTP gateway: lease, put, range, txn (create-if-absent, fenced), watch with resume | standard library only: no new package |
| `hb_beat.py` | everywhere | the fields of the rank line and the node beat; atomic writes | one definition for writer and reader |
| `hb_start.sh` | the batch script | starts etcd on the first 3 nodes and one agent per node, each its own `srun --overlap` step | one place for the launch |
| `get_etcd.sh` | once, on a login node | downloads etcd 3.5.17 and checks its checksum | the binary is not kept in the repository |
| `tests/` | login node / batch | `test_rules.py` (decisions), `test_coordinator.py` (two events at once), `test_agent.py` (a gone rank), `login_e2e.py` (whole system on CPU), `gpu_e2e.py` (whole system on 4 GPU nodes), `hb_toy_trainer.py` | each decision becomes a check |

The trainer side is one switch: `XMOE_FT_V2=1` makes `megatron/training.py` install the counter before initialization, start the
monitor after it, and set the step and the save phase (search for `[ft-sys-v2`). Unset, training is unchanged.

## Decisions (design map, reviews page)

- Two heartbeats: fast through etcd for the system, slow to Lustre every minute for people (the user, 11-04).
- etcd on 3 nodes: the record and its reader survive the loss of any one node (the user's rule, 11-04).
- A replacement only on two signals (the user, 11-04): a hang needs the counts AND the stacks; a lost node needs its lease ended AND
  the others stopped, or ranks elsewhere died of communication errors. A rank is gone when its process is gone or when it has
  recorded an uncaught exception (its exit can block for good: Cray MPICH's finalize barrier, HB3 2026-11-05), reported with its
  cause; a communication error is a consequence, not a root.

## Run

    bash get_etcd.sh /path/to/tools                  # once; prints FTV2_ETCD_BIN
    # in the batch script, before the training srun:
    export FTV2_ETCD_BIN=... FTV2_PY=$(command -v python) FTV2_OUT=<job folder>
    . examples_elmoe/scripts-frontier/ft-sys-v2/heartbeat-sys/hb_start.sh
    # the training: XMOE_FT_V2=1 in its environment

hb_start.sh starts etcd without http_proxy/https_proxy: etcd's peer transport follows them, and a site proxy cannot reach the
nodes' internal addresses (HB2, 2026-11-05: the members never formed a cluster). If etcd takes no write in about 60 s, it prints
`[hb_start] ERROR` in the job's output.

Settings (environment): FTV2_FAST_BEAT_S (5), FTV2_LEASE_S (15), FTV2_RANK_BEAT_S (5), FTV2_SLOW_BEAT_S (60), FTV2_HANG_STILL_S (15),
FTV2_SAVE_STILL_S (300), FTV2_LOST_CONFIRM_S (10), FTV2_CRASH_SETTLE_S (25), FTV2_EQUAL_STILL_S (120).
