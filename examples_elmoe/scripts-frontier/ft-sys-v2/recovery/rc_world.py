"""rc_world.py (ft-sys-v2 recovery, 2026-11-06): the new world after a stop. Framework-free: torch.distributed and etcd.

The lowest surviving rank hosts a fresh TCPStore on its node's high-speed address and publishes host:port in etcd as world/<g>;
every rank (survivors and the standby) connects to it and calls torch.distributed.init_process_group under the prefix ftv2/world<g>/
(the toy's make_groups_for_generation, toy_ft5.py:212-216). torch is called directly: DeepSpeed's init_distributed skips the call
when a world exists (deepspeed/comm/torch.py:140-147). The first world of the job is the template's (MASTER_ADDR on node 0); a
recovery never reuses it, because its store lives in rank 0's process."""
import fcntl
import socket
import struct
import time
from datetime import timedelta

import torch
import torch.distributed as dist

_STORES = []        # the store of every generation stays referenced for the life of the process (the host serves it)


def node_address(ifname='hsn0'):
    """The IPv4 address of the high-speed interface (the template's MASTER_ADDR rule), else the hostname's address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifname[:15].encode()))[20:24]    # SIOCGIFADDR
        return socket.inet_ntoa(addr)
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def make_world(etcd, g, rank, world_size, host_rank, timeout_s, log, backend='nccl', wait_s=300.0):
    """Join generation g's world as `rank` of `world_size`. Returns (seconds to the store, seconds to init_process_group)."""
    t0 = time.time()
    key = f'world/{g}'
    if rank == host_rank:
        ip = node_address()
        store = dist.TCPStore(ip, 0, world_size, is_master=True, timeout=timedelta(seconds=timeout_s), wait_for_workers=False)
        etcd.put(key, f'{ip}:{store.port}')
        log(f'world {g}: hosting the store at {ip}:{store.port}')
    else:
        v = None
        while v is None:
            v, _ = etcd.get(key)
            if v is None:
                if time.time() - t0 > wait_s:
                    raise RuntimeError(f'world {g}: no store address in etcd after {wait_s:.0f} s')
                time.sleep(0.1)
        ip, port = v.rsplit(':', 1)
        store = dist.TCPStore(ip, int(port), world_size, is_master=False, timeout=timedelta(seconds=timeout_s))
    _STORES.append(store)
    t1 = time.time()
    dist.init_process_group(backend, store=dist.PrefixStore(f'ftv2/world{g}/', store), rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=timeout_s))
    t2 = time.time()
    log(f'world {g}: rank {rank} of {world_size} joined (store {t1 - t0:.2f} s, init {t2 - t1:.2f} s)')
    return round(t1 - t0, 3), round(t2 - t1, 3)


def world_barrier(device):
    """One collective on the whole world, so its communicator exists on every rank before any point-to-point round (a batched
    send/receive uses the group's communicator, which every rank of the group must create together)."""
    if device.type == 'cuda':
        t = torch.ones(1, device=device)
        dist.all_reduce(t)
        torch.cuda.synchronize(device)
    else:
        dist.barrier()
