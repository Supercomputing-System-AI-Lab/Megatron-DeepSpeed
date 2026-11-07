#!/usr/bin/env python3
"""gpu_event_barrier_test.py (2026-11-07, after P1b 5873196): does a CUDA event recorded on the current stream right after a
barrier block, with RCCL's default algorithms (normal mode) or with Ring / Simple (determinism mode)?

P1b: at Megatron's fused-kernel step (rank 0 relinks and loads the kernels while every other rank waits in a barrier), 4 of the 7
other ranks of rank 0's node stayed for minutes in torch.cuda.Event.record (hipEventRecord, rocr InterruptSignal::WaitRelaxed),
called by the C tracking (hb_monitor._track) right after that barrier returned; every earlier C run had the Ring / Simple pins.

One srun step per environment (the sbatch file runs it twice). Every rank, per round: barrier (rank 0 first sleeps `delay` s),
then torch.cuda.Event().record() on the current stream, then a closing barrier; each call's seconds printed. A watchdog prints
every thread's Python stack and exits after 120 s in one round. Pass: every record returns in under 1 s in every round."""
import faulthandler
import os
import sys
import time

import torch
import torch.distributed as dist


def main():
    rank, world, local = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS']), int(os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', init_method=f'tcp://{os.environ["FTV2_TEST_ADDR"]}:{os.environ["FTV2_TEST_PORT"]}',
                            rank=rank, world_size=world)     # no device_id: as Megatron's init (the barrier guesses the device)
    tag = os.environ.get('FTV2_TEST_TAG', '?')
    bad = 0
    for name, delay, op in (('barrier, rank 0 late 10 s', 10, 'barrier'), ('barrier', 0, 'barrier'),
                            ('all_reduce, rank 0 late 10 s', 10, 'all_reduce'), ('barrier, rank 0 late 30 s', 30, 'barrier')):
        faulthandler.dump_traceback_later(120, exit=True)
        if rank == 0 and delay:
            time.sleep(delay)
        t0 = time.time()
        if op == 'barrier':
            dist.barrier()
        else:
            dist.all_reduce(torch.ones(1, device='cuda'))
        t1 = time.time()
        ev = torch.cuda.Event()
        ev.record()
        t2 = time.time()
        dist.barrier()
        t3 = time.time()
        faulthandler.cancel_dump_traceback_later()
        ok = (t2 - t1) < 1.0
        bad += not ok
        print(f'{"PASS" if ok else "FAIL"} [{tag}] rank {rank}: {name}: {op} {t1 - t0:.3f} s, record {t2 - t1:.4f} s, '
              f'closing barrier {t3 - t2:.3f} s', flush=True)
    dist.destroy_process_group()
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
