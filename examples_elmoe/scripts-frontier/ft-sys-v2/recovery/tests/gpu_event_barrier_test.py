#!/usr/bin/env python3
"""gpu_event_barrier_test.py (2026-11-07, after P1b 5873196): does a CUDA event recorded on the current stream right after a
barrier block, with RCCL's default algorithms (normal mode) or with Ring / Simple (determinism mode)?

P1b: at Megatron's fused-kernel step (rank 0 relinks and loads the kernels while every other rank waits in a barrier), 4 of the 7
other ranks of rank 0's node stayed for minutes in torch.cuda.Event.record (hipEventRecord, rocr InterruptSignal::WaitRelaxed),
called by the C tracking (hb_monitor._track) right after that barrier returned; every earlier C run had the Ring / Simple pins.

One srun step per environment (the sbatch file runs it twice). Every rank, per round: barrier (rank 0 first sleeps `delay` s),
then torch.cuda.Event().record() on the current stream, then a closing barrier; each call's seconds printed. A watchdog prints
every thread's Python stack and exits after 120 s in one round. Pass: every record returns in under 1 s in every round.
M3E1 5873502 ran it on torch's bundled RCCL 2.27.7 / HIP 7.1 (no preload), not the training's TheRock RCCL 2.30.4 / HIP 7.14; the
sbatch file now preloads TheRock's libraries as the stop tests did (FTV2_TEST_PRELOAD). Three ways per round (2026-11-07):
fresh = a new torch.cuda.Event() recorded (what the tracking did until now: the HIP event is created at its first record);
pool = an event created and recorded once at start, recorded again; track = the call made through the C tracking (hb_monitor
installed, S['c_track'] on: the synchronous call made asynchronous and waited for, no event)."""
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
    pool = torch.cuda.Event()
    pool.record()                                   # created now (its first record), before any collective
    torch.cuda.synchronize()
    rounds = [(way, name, delay, op) for way in ('fresh', 'pool', 'track')
              for name, delay, op in (('barrier, rank 0 late 10 s', 10, 'barrier'), ('barrier', 0, 'barrier'),
                                      ('all_reduce, rank 0 late 10 s', 10, 'all_reduce'), ('barrier, rank 0 late 30 s', 30, 'barrier'))]
    for way, name, delay, op in rounds:
        if way == 'track' and not getattr(main, 'tracking', False):
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                            'heartbeat-sys'))
            import hb_monitor as hb
            hb.install_counter()
            hb.S['c_track'] = True
            main.tracking = True
        faulthandler.dump_traceback_later(120, exit=True)
        if rank == 0 and delay:
            time.sleep(delay)
        t0 = time.time()
        if op == 'barrier':
            dist.barrier()
        else:
            dist.all_reduce(torch.ones(1, device='cuda'))
        t1 = time.time()
        if way == 'fresh':
            ev = torch.cuda.Event()
            ev.record()
        elif way == 'pool':
            pool.record()
        t2 = time.time()
        dist.barrier()
        t3 = time.time()
        faulthandler.cancel_dump_traceback_later()
        ok = (t2 - t1) < 1.0
        bad += not ok
        print(f'{"PASS" if ok else "FAIL"} [{tag}] rank {rank}: {way}: {name}: {op} {t1 - t0:.3f} s, after it {t2 - t1:.4f} s, '
              f'closing barrier {t3 - t2:.3f} s', flush=True)
    dist.destroy_process_group()
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
