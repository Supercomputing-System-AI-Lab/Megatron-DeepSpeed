"""hb_stacks.py (ft-sys-v2 heartbeat system, 2026-11-05): signal 2 of a hang, from py-spy stacks.

The test, taken from ft-sys-v1's name_culprit.py (X-MoE-resilience sandbox, 2026-10-16; it named the hung rank 260 right at 671B
after the call count alone had named a healthy rank): a hung rank's main thread is OUTSIDE any device or communication wait, while
the ranks that wait for it sit INSIDE one. WAIT is that file's pattern, unchanged.
What changed (HB2b 5855002, X-MoE 10B, 2026-11-05): v1 called a rank waiting when ANY frame matched, so the hung rank 13 (asleep in
the injection, called from broadcast_data) counted as waiting, and its 7 peers, spinning in the HIP runtime's signal wait under
deepspeed/moe/moe_v2.py, matched nothing. Only where the thread is blocked counts now: the innermost frames down to the first
Python frame (blocking_site), matched against WAIT and against the device and communication runtimes (DEVICE).
Input: the text of `py-spy dump --native --pid P` for each rank of the suspect's node."""
import re

WAIT = re.compile(r'synchronize|memcpy_and_sync|ProcessGroupNCCL|c10d|all_to_all|alltoall|all_reduce|allreduce|all_gather|allgather|'
                  r'reduce_scatter|broadcast|barrier|irecv|isend|\brecv\b|\bsend\b|p2p\.py|\.wait\b|wait \(|hipStreamSynchronize|hipDeviceSynchronize|'
                  r'hipMemcpy|rccl|nccl', re.I)
# native frames of a thread that waits for the GPU or for a peer: the HSA and HIP runtimes (a host wait on a GPU signal, as after
# an all-to-all whose partner never came), the communication libraries, the read of a GPU scalar
DEVICE = re.compile(r'libhsa-runtime|libamdhip|librccl|libnccl|libcuda|BusyWaitSignal|InterruptSignal|hsa_signal_wait|gloo::|'
                    r'ProcessGroup|_local_scalar_dense', re.I)
PY_FRAME = re.compile(r'(?:\.py|<frozen [^>]+>):\d+\)\s*$')


def blocking_site(frames):
    """Where the thread is blocked: its innermost frames, down to and including the first Python frame."""
    site = []
    for f in frames:
        site.append(f)
        if PY_FRAME.search(f):
            break
    return site


def main_thread_frames(text):
    """The frames of the MainThread in one py-spy dump, or None if the dump has none (the process is gone or py-spy failed)."""
    m = re.search(r'Thread \S+ \((?:active|idle)[^)]*\): "MainThread"\n(.*?)(?:\n\s*\n|\nThread |\Z)', text, re.S)
    return [ln.strip() for ln in m.group(1).strip().split('\n')] if m else None


def is_waiting(frames):
    return any(WAIT.search(f) or DEVICE.search(f) for f in blocking_site(frames))


def verdict(dumps, suspect):
    """dumps: {rank: py-spy text}. Returns (decision, detail):
    'confirmed'  the suspect is the only sampled rank outside a wait, and the others wait;
    'ambiguous'  no rank outside a wait (every rank waits: a hang inside the GPU, the open case), or several, or the suspect waits;
    'no-stack'   py-spy gave the suspect no main thread (it may be gone)."""
    outside, waiting, missing = [], [], []
    for r, text in sorted(dumps.items()):
        fr = main_thread_frames(text or '')
        if fr is None:
            missing.append(r)
        elif is_waiting(fr):
            waiting.append(r)
        else:
            outside.append(r)
    detail = {'outside': outside, 'waiting': waiting, 'no_stack': missing}
    if suspect in missing:
        return 'no-stack', detail
    if outside == [suspect]:
        return 'confirmed', detail
    return 'ambiguous', detail
