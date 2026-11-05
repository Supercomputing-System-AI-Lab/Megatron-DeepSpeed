# ft-sys-v2 pre-run gate

ft-sys-v1's gate (examples_elmoe/scripts-frontier/resilience/bin/, commit 7fb97edf), copied here for ft-sys-v2. Its tests, rules and
judge are v1's. Only the software-stack paths are settings now, with the new stack as the default (ROCm 7.1.1, the torch 2.13 venv,
aws-ofi-nccl 1.22.0): GATE_PYTHON, GATE_OFI_DIR, ROCM_PATH. Each copied file's first lines say so.

```
 hb_start.sh, FTV2_GATE=quick|full
   -> ftv2_gate.sh  -> gate.sh (v1): the node step (gate_node.sh: rocm-smi, RAS, xGMI, versions, /dev/shm, dmesg; full: GEMM per
                       GCD, the xGMI all-to-all), the re-probe, the network step (full: gate_net.py pair and switch tests on the
                       training's network path), the judge (gate_judge.py: calibrated thresholds, the majority rule, the pair
                       test's naming rules, two tiers: dead / degraded)
                    -> the layout: dead nodes out, a standby that passed takes a dead training node's place, degraded kept
                    -> ft/gate/ (gate.log, result.json), one GATE line in ft/record.log, etcd key gate/result
```

| File | Source |
|---|---|
| `ftv2_gate.sh` | new: runs the gate, makes the layout, records it; FTV2_GATE_STRICT=1 (default) ends the job when a training node is dead, because the 0623 launcher trains on every node of the allocation (ft-sys-v2's own sbatch will apply the layout instead) |
| `gate.sh`, `gate_node.sh`, `gate_net.sh`, `gate_net.py`, `gate_gemm.py`, `gate_judge.py`, `net_launch_retry.sh` | v1, stack paths as settings; gate.sh's default baseline is `gate_baseline_v2.tsv` here |
| `tests/test_gate_judge.py`, `tests/test_gate_net_unit.py` | v1's tests, pointed at the copies; both pass |
| `gate_baseline_v2.tsv` | not yet: written by the first calibration run on the new stack (`FTV2_GATE=full FTV2_GATE_CALIBRATE=1` on nodes known to be good). Until then every test is informational and excludes nobody (v1's rule 1) |

A calibration at a few nodes applies only to jobs within 4x of its node count (v1's scale guard); production needs its own.
