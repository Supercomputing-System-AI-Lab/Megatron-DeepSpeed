#!/usr/bin/env python3
"""H1 the heartbeat (design 1009 sections 3.1, 3.1a, 5.2), the trainer-side detectors' exit path, and the fault injection.

In the trainer (XMOE_HB=1): start(rank, local_rank, ...) launches a daemon thread that writes ONE tab-separated line every
HB_PERIOD_S (10 s) to /dev/shm/hb_<job>_<attempt>/rank_<r>.tsv; local rank 0 also starts the per-node writer (this file with
--node-writer) that appends the node's rank files to <job dir>/hb/<node>.tsv every 10 s + a per-node jitter (file kept open,
no fsync). Hooks (one line each at the call sites): set_phase(), coll_enter()/coll_exit() (coll_seq += 1, phase 'all_to_all'),
end_step(). The line's fields (header HB_FIELDS): the 15 of design 3.1a, then attempt, pid, flag, epoch.

Exit path of the detectors: die(code, evidence) writes <job dir>/attempts/attempt_<k>/exit/r<rank>.json (RES_EXIT_DIR, set by the
loop per attempt) and os._exit(code).
Fault injection: XMOE_INJECT=<case>:<rank>:<step> and/or XMOE_INJECT_SCHEDULE="case:rank:step,..." (cases hang, exit, kill,
slow, oom, fshang, nan, inithang, hostmem [2026-10-11: the rank's heartbeat reads host memory as 98 percent, so D10 exits 19
without allocating any memory], realhostoom [2026-10-11, sandbox: the rank allocates real host memory until the OOM killer ends it]); each fires once per job (a marker file <job dir>/data/inject/<case>_r<rank>_s<step>.fired).
training_log_line(): one line per step into <job dir>/training.log (design 3.11), called by the trainer's writer rank.
THE FILES' HEADER (2026-10-09 late evening, the user): every rank file and every node file starts with the HB_FIELDS header line
(the node writer writes it once when it creates the node file and drops the rank files' header lines while copying); every reader
skips a line that starts with 'time' or '#'. The second header line names the source of the ECC counts.
THE ECC COUNTS (2026-10-09 late evening, the user: on the compute nodes /sys/class/drm/card<k>/device/ras/umc_err_count is not
readable by the user, so ecc_ce and ecc_ue read -1 on all 2,048 ranks of job 5645677): the heartbeat thread refreshes the two
counts through `rocm-smi -d <GCD> --showrasinfo all` every sixth beat (once a minute; one subprocess of about a second, which does
not hold the GIL while it waits), keeps the last values between refreshes, and writes -1 only when rocm-smi itself fails. The
counts are the sums of the correctable and uncorrectable columns over the ENABLED blocks of the GCD's RAS table; on Frontier the
UMC (memory) block reads DISABLED there, so the memory counters themselves are not exposed and the enabled blocks (GFX, SDMA, MMHUB)
are what the line carries; the header line says so. The sysfs file is still tried first and used when it is readable. The xGMI
count stays a sysfs read, which works.
The job folder layout (design 3.11): the machine-readable files under <job dir>/data/ (RES_DATA_DIR).
Nothing here runs unless the trainer calls it behind its flags.
"""
import json, os, socket, sys, threading, time, glob, signal, statistics, zlib

HB_FIELDS = ['time', 'node', 'rank', 'step', 'phase', 'coll_seq', 'step_time_s', 'busy_s', 'hbm_gb', 'host_mem_pct',
             'ecc_ce', 'ecc_ue', 'xgmi_err', 'proc_state', 'loss', 'attempt', 'pid', 'flag', 'epoch']
PERIOD = float(os.environ.get('HB_PERIOD_S', '10'))

S = {'step': 0, 'phase': 'init', 'coll_seq': 0, 'step_time': None, 'busy': None, 'loss': None, 'flag': '-',
     'coll_t': 0.0, '_coll_t0': None, '_phase_before': 'forward', 'save_t': 0.0}
# THE SAVE WINDOW (2026-10-09 late evening; found in the 260-node job 5645677 at 22:34): a save holds the GIL in its Python-level
# host copy and its background torch.save, so the heartbeat thread can be starved for a minute while the main thread is healthy,
# and the last stage's ranks (the largest files) finish a minute after rank 0. The window = the phase is a save phase, or the last
# save activity (save_mark / save_activity) is less than SAVE_WINDOW_S ago. The aggregator uses the same 120 s for its stillness
# and minority rules inside the window. RULE FOR FUTURE DETECTORS: any early-kill rule must check in_save_window() and never act
# inside the window (D3, the save deadline, is the save's own detector and is the only one allowed to act there).
SAVE_WINDOW_S = float(os.environ.get('XMOE_SAVE_WINDOW_S', '120'))
_io = {'fh': None, 'lock': threading.Lock()}
_ctx = {'rank': -1, 'local_rank': 0, 'job_dir': None, 'attempt': os.environ.get('RES_ATTEMPT', '0'), 'node': socket.gethostname(),
        'card': None, 'started': False, 'hist': [], 'detect': os.environ.get('XMOE_DETECT') == '1', 'save_hist': []}


def data_dir(*p):
    """<job dir>/data/<p> (RES_DATA_DIR overrides; the job folder's machine-readable half, design 3.11)."""
    d = os.environ.get('RES_DATA_DIR') or os.path.join(_ctx['job_dir'] or os.environ.get('RES_JOB_DIR') or '/tmp', 'data')
    return os.path.join(d, *p)


def ts(t=None):
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t or time.time()))


def log(msg):
    print(f'[resilience {ts()} rank {_ctx["rank"]} {_ctx["node"]}] {msg}', flush=True)


# ------------------------------------------------------------------ hooks (cheap; safe to call when the thread is off)
def set_phase(p):
    S['phase'] = p


def coll_enter(kind='all_to_all'):
    S['coll_seq'] += 1
    S['_phase_before'] = S['phase'] if S['phase'] != kind else S['_phase_before']
    S['phase'] = kind
    S['_coll_t0'] = time.perf_counter()


def coll_exit():
    t0 = S['_coll_t0']
    if t0 is not None:
        S['coll_t'] += time.perf_counter() - t0
        S['_coll_t0'] = None
    S['phase'] = S['_phase_before']


def begin_step():
    S['coll_t'] = 0.0


def end_step(step, step_time, loss=None):
    S['step'] = step
    S['step_time'] = step_time
    S['busy'] = max(0.0, step_time - S['coll_t'])
    if loss is not None:
        S['loss'] = loss
    h = _ctx['hist']
    flag = '-'
    pred = os.environ.get('XMOE_STEP_PRED_S')
    if pred and step_time > 1.3 * float(pred):
        flag = 'slow_run'                                  # D5: the step against the prediction
    if len(h) >= 3 and step_time > 2.0 * statistics.median(h[-5:]):
        flag = 'slow_step'                                 # D5: the step against the run's own median
    S['flag'] = flag if not S['flag'].startswith('host_mem') else S['flag']
    h.append(step_time)


# ------------------------------------------------------------------ the counters (design 3.1a)
def _find_card(local_rank):
    try:
        import torch
        p = torch.cuda.get_device_properties(local_rank)
        bus = getattr(p, 'pci_bus_id', None)
        if bus is None:
            return None
        for d in glob.glob('/sys/class/drm/card*/device/uevent'):
            try:
                for line in open(d):
                    if line.startswith('PCI_SLOT_NAME='):
                        slot = line.strip().split('=')[1]          # 0000:c1:00.0
                        if int(slot.split(':')[1], 16) == int(bus):
                            return os.path.dirname(d)
            except Exception:
                continue
    except Exception:
        return None
    return None


ECC_REFRESH_BEATS = int(os.environ.get('XMOE_ECC_REFRESH_BEATS', '6'))      # rocm-smi once a minute at a 10-s beat
_ecc_state = {'ce': -1, 'ue': -1, 'src': 'none', 'umc': 'unknown', 'beats': 0, 'note': ''}


def _find_smi():
    for c in (os.environ.get('ROCM_PATH', '') + '/bin/rocm-smi', '/opt/rocm/bin/rocm-smi', '/opt/rocm-6.4.1/bin/rocm-smi'):
        if c.startswith('/') and os.access(c, os.X_OK):
            return c
    import shutil
    return shutil.which('rocm-smi')


def _ecc_smi(gcd):
    """(ce, ue, umc_state, note) from `rocm-smi -d <gcd> --showrasinfo all`: the sums over the ENABLED blocks of that GPU's RAS table
    (a block line reads: Block Status [Correctable Uncorrectable]); (-1, -1, 'unknown', why) when rocm-smi fails."""
    smi = _find_smi()
    if not smi:
        return -1, -1, 'unknown', 'rocm-smi not found'
    try:
        import subprocess
        r = subprocess.run([smi, '-d', str(gcd), '--showrasinfo', 'all'], capture_output=True, text=True, timeout=20)
    except Exception as e:
        return -1, -1, 'unknown', f'rocm-smi failed: {type(e).__name__}'
    ce = ue = 0; umc = 'absent'; seen = False
    for ln in r.stdout.splitlines():
        f = ln.split()
        if len(f) >= 2 and f[1] in ('ENABLED', 'DISABLED') and f[0].isupper():
            seen = True
            if f[0] == 'UMC':
                umc = f[1]
            if f[1] == 'ENABLED' and len(f) >= 4 and f[2].isdigit() and f[3].isdigit():
                ce += int(f[2]); ue += int(f[3])
    if not seen:
        return -1, -1, 'unknown', f'rocm-smi gave no RAS table (exit {r.returncode})'
    return ce, ue, umc, ''


def _ecc_refresh(card, gcd, beat):
    """Refresh _ecc_state: the sysfs file when it is readable (every beat), else rocm-smi every ECC_REFRESH_BEATS beats."""
    ce, ue = _ecc(card)
    if ce >= 0 or ue >= 0:
        _ecc_state.update(ce=ce, ue=ue, src='sysfs umc_err_count', umc='sysfs')
        return
    if beat % ECC_REFRESH_BEATS == 0 or _ecc_state['src'] == 'none':
        ce, ue, umc, note = _ecc_smi(gcd)
        _ecc_state.update(ce=ce, ue=ue, src='rocm-smi --showrasinfo all (the enabled blocks of this GCD)', umc=umc, note=note)


def _ecc(card):
    ce = ue = -1
    if card:
        try:
            for line in open(os.path.join(card, 'ras', 'umc_err_count')):
                k, _, v = line.partition(':')
                if k.strip() == 'ce':
                    ce = int(v)
                elif k.strip() == 'ue':
                    ue = int(v)
        except Exception:
            pass
    return ce, ue


def _xgmi(card):
    if card:
        try:
            return int(open(os.path.join(card, 'xgmi_error')).read().split()[0])
        except Exception:
            pass
    return -1


def host_mem_pct():
    try:
        m = {}
        for line in open('/proc/meminfo'):
            k, v = line.split(':', 1)
            m[k] = int(v.split()[0])
        return 100.0 * (m['MemTotal'] - m['MemAvailable']) / m['MemTotal']
    except Exception:
        return -1.0


def _proc_state():
    try:
        tid = threading.main_thread().native_id
        st = open(f'/proc/{os.getpid()}/task/{tid}/stat').read()
        return st.rsplit(')', 1)[1].split()[0]
    except Exception:
        return '?'


def _hbm():
    try:
        import torch
        return torch.cuda.memory_allocated(_ctx['local_rank']) / 1e9
    except Exception:
        return -1.0


def line():
    ce, ue = _ecc_state['ce'], _ecc_state['ue']
    hm = host_mem_pct()
    f = lambda x, fmt: '-' if x is None else fmt % x
    vals = [ts(), _ctx['node'], str(_ctx['rank']), str(S['step']), S['phase'], str(S['coll_seq']), f(S['step_time'], '%.2f'),
            f(S['busy'], '%.2f'), '%.2f' % _hbm(), '%.1f' % hm, str(ce), str(ue), str(_xgmi(_ctx['card'])), _proc_state(),
            f(S['loss'], '%.4f'), str(_ctx['attempt']), str(os.getpid()), S['flag'], '%.1f' % time.time()]
    return '\t'.join(vals) + '\n', hm


def write_line_now(why=''):
    """Write one heartbeat line from the CALLING thread (the heartbeat thread may be starved by the GIL during a save). Safe when
    the heartbeat is not started (no file): then it does nothing."""
    fh = _io['fh']
    if fh is None:
        return False
    try:
        ln, _ = line()
        with _io['lock']:
            fh.write(ln); fh.flush()
        return True
    except Exception:
        return False


def save_mark(where=''):
    """Called by the save path (SaveDeadline, the async engine's host copy and its wait for the previous checkpoint): the phase
    becomes 'save' (if it is not a save phase already), the save activity is stamped, and one line is written at once."""
    if not S['phase'].startswith('save'):
        S['phase'] = 'save'
    S['save_t'] = time.time()
    write_line_now(where)


def save_activity(where=''):
    """Called from the async engine's background writer and commit thread while the main thread may already train again: the
    save activity is stamped (the window stays open), the line's flag reads save_bg so that the aggregator sees the save, and one
    line is written at once. The phase is left to the main thread."""
    S['save_t'] = time.time()
    if not S['flag'].startswith('host_mem'):
        S['flag'] = 'save_bg'
    write_line_now(where)


def in_save_window():
    """True while this rank is in a save or less than SAVE_WINDOW_S after its last save activity. Every early-kill rule must
    check this and never act inside the window."""
    return S['phase'].startswith('save') or (time.time() - S['save_t']) < SAVE_WINDOW_S


# ------------------------------------------------------------------ the exit path of the detectors
def die(code, evidence, cls=None):
    """Write the exit record and stop this process with the class code (no atexit, no teardown)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import exit_codes
        cls = cls or exit_codes.name(code)
    except Exception:
        cls = cls or str(code)
    rec = {'code': int(code), 'class': cls, 'rank': _ctx['rank'], 'node': _ctx['node'], 'step': S['step'], 'phase': S['phase'],
           'time': time.time(), 'time_str': ts(), 'attempt': _ctx['attempt'], 'evidence': str(evidence)[-2000:]}
    try:
        if _ctx['job_dir'] or os.environ.get('RES_EXIT_DIR'):
            d = os.environ.get('RES_EXIT_DIR') or data_dir('exit', f'attempt_{_ctx["attempt"]}')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f'r{_ctx["rank"]}.json'), 'w') as fh:
                json.dump(rec, fh)
    except Exception as e:
        print(f'[resilience] could not write the exit record: {e}', flush=True)
    log(f'DETECTOR EXIT code {code} ({cls}) at step {S["step"]} phase {S["phase"]}: {str(evidence)[:300]}')
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    os._exit(int(code))


def excepthook(etype, value, tb):
    """D4 / D9: an uncaught exception in the main thread -> exit 11 (GPU OOM), 16 (HIP in the text) or 17."""
    import traceback
    text = ''.join(traceback.format_exception(etype, value, tb))
    sys.stderr.write(text); sys.stderr.flush()
    if etype is SystemExit:
        os._exit(value.code if isinstance(value.code, int) else 0)
    name = getattr(etype, '__name__', '')
    if name == 'OutOfMemoryError' or 'out of memory' in str(value).lower():
        die(11, f'{name}: {str(value)[:600]} | hbm_gb={_hbm():.2f}')
    if 'HIP' in text or 'hipError' in text or 'HSA_STATUS' in text:
        die(16, f'{name}: {str(value)[:600]}')
    # [2026-10-13 15:15] a background save whose write failed on the filesystem is the filesystem class (12: restart from the last
    # complete save, the node excluded), not a code defect: job 5664539, rank 27, the step-20 optimizer file stalled 7.7 min on Lustre
    # and failed with 'unexpected pos' (the same write took 0.3 s at step 10); the engine kept 'latest' at step 10 and raised at step 30
    if '[AsyncPinned] a checkpoint write failed' in text:
        die(12, f'the background write of a save failed on the filesystem: {name}: {str(value)[:560]}')
    die(17, f'{name}: {str(value)[:600]}')


# ------------------------------------------------------------------ the thread and the node writer
def header_lines():
    """The two header lines of a heartbeat file: the field names, and the source of the counters."""
    return ('\t'.join(HB_FIELDS) + '\n' +
            f'# ecc_ce/ecc_ue: {_ecc_state["src"]}, refreshed every {ECC_REFRESH_BEATS} beats ({ECC_REFRESH_BEATS * PERIOD:.0f} s) when read through rocm-smi; '
            f'UMC (memory) block: {_ecc_state["umc"]}; -1 only when the read fails{(" (" + _ecc_state["note"] + ")") if _ecc_state["note"] else ""}. '
            f'xgmi_err: sysfs xgmi_error. One line every {PERIOD:.0f} s per rank.\n')


def _run(path):
    _ecc_refresh(_ctx['card'], _ctx['local_rank'], 0)
    fh = open(path, 'w')
    fh.write(header_lines()); fh.flush()
    _io['fh'] = fh
    beat = 0
    while True:
        try:
            if beat:
                _ecc_refresh(_ctx['card'], _ctx['local_rank'], beat)
            beat += 1
            ln, hm = line()
            with _io['lock']:
                fh.write(ln); fh.flush()
            if _inj.get('hostmem'):                         # [2026-10-11] the injected case: a forced reading, no memory is allocated
                hm = 98.0
            if _ctx['detect'] and hm > 0:                  # D10 host memory
                if hm > float(os.environ.get('XMOE_HOSTMEM_EXIT_PCT', '97')):
                    die(19, f'host memory {hm:.1f} percent on {_ctx["node"]}' + (' (an injected test reading, case hostmem)' if _inj.get('hostmem') else ''))
                elif hm > 90 and not S['flag'].startswith('host_mem'):
                    S['flag'] = 'host_mem_90'
                    log(f'WARNING host memory {hm:.1f} percent')
        except Exception as e:
            try:
                fh.write(f'# heartbeat error {e}\n'); fh.flush()
            except Exception:
                pass
        time.sleep(PERIOD)


def start(rank, local_rank):
    if _ctx['started'] or os.environ.get('XMOE_HB') != '1':
        return False
    _ctx.update(rank=rank, local_rank=local_rank, job_dir=os.environ.get('RES_JOB_DIR'), started=True)
    _ctx['card'] = _find_card(local_rank)
    job = os.environ.get('SLURM_JOB_ID', 'nojob')
    shm = f'/dev/shm/hb_{job}_a{_ctx["attempt"]}'
    os.makedirs(shm, exist_ok=True)
    threading.Thread(target=_run, args=(os.path.join(shm, f'rank_{rank}.tsv'),), name='xmoe-heartbeat', daemon=True).start()
    if local_rank == 0 and _ctx['job_dir']:
        import subprocess
        os.makedirs(data_dir('hb'), exist_ok=True)
        jitter = (zlib.crc32(_ctx['node'].encode()) % 50) / 10.0
        subprocess.Popen([sys.executable, os.path.abspath(__file__), '--node-writer', shm,
                          data_dir('hb', f'{_ctx["node"]}.tsv'), str(PERIOD), str(jitter), str(os.getpid())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=False)
    if _ctx['detect']:
        sys.excepthook = excepthook
    log(f'heartbeat on: {shm}/rank_{rank}.tsv every {PERIOD:.0f} s; card {_ctx["card"]}; detectors {"on" if _ctx["detect"] else "off"}; '
        f'the ECC counts come from the sysfs file when readable, else from rocm-smi --showrasinfo all every {ECC_REFRESH_BEATS} beats')
    return True


def node_writer(src, dst, period, jitter, ppid):
    offs = {}
    new_file = not os.path.exists(dst) or os.path.getsize(dst) == 0
    out = open(dst, 'a')
    if new_file:
        out.write('\t'.join(HB_FIELDS) + '\n'); out.flush()      # the header once, when the node file is created
    time.sleep(jitter)
    while True:
        for p in sorted(glob.glob(os.path.join(src, 'rank_*.tsv'))):
            try:
                with open(p) as fh:
                    fh.seek(offs.get(p, 0))
                    data = fh.read()
                    cut = data.rfind('\n') + 1          # whole lines only; a partial last line waits for the next copy
                    offs[p] = offs.get(p, 0) + len(data[:cut].encode())
                    data = data[:cut]
                if data:
                    # the rank files' header lines are dropped: the node file carries its own header once
                    out.write(''.join(l + '\n' for l in data.split('\n') if l and not l.startswith('time') and not l.startswith('#')))
            except Exception:
                pass
        out.flush()
        if os.getppid() != ppid:                            # the trainer is gone: copy once more and stop
            break
        time.sleep(period)


# ------------------------------------------------------------------ the save deadline (D3)
def save_deadline_s():
    # [2026-10-16 bug fix, 671B fault injection 5693855 attempt 1] the FIRST save of a process pays the pinned-buffer allocation of the
    # asynchronous engine (its pool is empty in a new process); after a restart some ranks took 64-78 s for it (frontier07559, stage 25)
    # and the save lasted 127 s, while the history used for the deadline was rank 0's fast saves of attempt 0 (5.6-12.6 s), so a healthy
    # save hit the 120-s floor and was classed filesystem_hang. Now: the first save of a process gets the first-save deadline (600 s,
    # or 3 x the slowest recorded save if larger); later saves of the process use this process's own history.
    v = os.environ.get('XMOE_SAVE_DEADLINE_S')
    if v:
        return float(v)
    first = float(os.environ.get('XMOE_SAVE_DEADLINE_FIRST_S', '600'))
    if not _ctx['save_hist']:
        h = _read_save_hist()
        return max(first, 3.0 * max(h)) if h else first
    return max(float(os.environ.get('XMOE_SAVE_DEADLINE_MIN_S', '120')), 3.0 * max(_ctx['save_hist'][-3:]))


def _extend_pg_timeouts(seconds):
    """[2026-10-16 bug fix, 5693855 attempt 4] the collectives of a save (its barriers) run in DeepSpeed's groups, whose timeout is the
    4-min hang timeout (XMOE_PG_TIMEOUT_MIN): a save that legitimately takes longer on one rank (217 s on rank 0 for step 12) made the
    others' barrier (ALLREDUCE, 1 element) time out and the attempt was classed collective_hang. Inside the save window the save
    deadline (D3) is the only detector allowed to act (header of this file), so every NCCL group of this rank gets a timeout of
    <seconds> for the save and its old timeout back after it. Returns [(group, old timedelta)]; a group that refuses is left as it is."""
    out = []
    try:
        import datetime
        import torch
        import torch.distributed.distributed_c10d as c10d
        if not c10d.is_initialized():
            return out
        new = datetime.timedelta(seconds=float(seconds))
        for pg in list(c10d._world.pg_map.keys()):
            try:
                if torch.device('cuda') not in pg._device_types:
                    continue
                be = pg._get_backend(torch.device('cuda'))
                old = getattr(getattr(be, 'options', None), '_timeout', None)
                if old is None or old >= new:
                    continue
                c10d._set_pg_timeout(new, pg)
                out.append((pg, old))
            except Exception:
                continue
    except Exception:
        pass
    return out


def _restore_pg_timeouts(saved):
    try:
        import torch.distributed.distributed_c10d as c10d
        for pg, old in saved:
            try:
                c10d._set_pg_timeout(old, pg)
            except Exception:
                pass
    except Exception:
        pass


def _read_save_hist():
    try:
        p = data_dir('save_times.tsv')
        return [float(l.split('\t')[1]) for l in open(p) if l.strip()][-5:]
    except Exception:
        return []


class SaveDeadline:
    """with SaveDeadline(step): <the save>.  A timer thread exits the process with 12 if the save is late (design D3).
    The save itself stays on the main thread (a CUDA device and the collectives are per thread); the timer is the helper thread."""
    def __init__(self, step):
        self.step = step
        self.done = threading.Event()

    def __enter__(self):
        self.t0 = time.time()
        self.deadline = save_deadline_s()
        self._pg_saved = _extend_pg_timeouts(self.deadline + 180) if os.environ.get('XMOE_SAVE_PG_TIMEOUT', '1') == '1' else []
        save_mark(f'save of step {self.step} starts')     # phase 'save' + one line written now (the heartbeat thread may be starved)
        # a rank inside a barrier of the save is WAITING for the others, not writing: mark it so that D3 and the aggregator name
        # the rank still writing (torch.distributed.barrier is looked up at call time by Megatron and DeepSpeed alike)
        self._orig = {}
        try:
            import torch.distributed as _d
            for name in ('barrier', 'all_reduce', 'all_gather', 'all_gather_object', 'all_gather_into_tensor', 'broadcast',
                         'broadcast_object_list', 'gather', 'reduce', 'reduce_scatter_tensor'):
                fn = getattr(_d, name, None)
                if fn is None:
                    continue
                self._orig[name] = fn
                def _w(*a, _fn=fn, **k):
                    coll_enter('save_barrier')          # waiting for the other ranks inside a collective of the save
                    try:
                        return _fn(*a, **k)
                    finally:
                        coll_exit()
                setattr(_d, name, _w)
        except Exception:
            pass
        if _ctx['detect']:
            threading.Thread(target=self._watch, daemon=True, name='xmoe-save-deadline').start()
        return self

    def _watch(self):
        if self.done.wait(self.deadline):
            return
        # a rank still writing names itself; a rank waiting in the closing barrier gives the writer a minute to exit first
        if S['phase'] in ('save_barrier', 'barrier') and self.done.wait(60):
            return
        die(12, f'save of step {self.step} not done after {time.time() - self.t0:.0f} s (deadline {self.deadline:.0f} s); '
                f'phase {S["phase"]}; main thread state {_proc_state()}')

    def __exit__(self, *a):
        self.done.set()
        _restore_pg_timeouts(getattr(self, '_pg_saved', []))
        if self._orig:
            import torch.distributed as _d
            for name, fn in self._orig.items():
                setattr(_d, name, fn)
        dt = time.time() - self.t0
        if a[0] is None:
            _ctx['save_hist'].append(dt)
            try:
                if _ctx['rank'] == 0 and _ctx['job_dir']:
                    with open(data_dir('save_times.tsv'), 'a') as fh:
                        fh.write(f'{self.step}\t{dt:.2f}\t{ts()}\n')
            except Exception:
                pass
        S['save_t'] = time.time()      # the save window stays open for SAVE_WINDOW_S after the save
        S['phase'] = 'step'
        return False


# ------------------------------------------------------------------ training.log (design 3.11): one line per step across attempts
# the column-name row, aligned with the step lines below (the user, 2026-10-10: "the header ... is not shown, so unknown what each value
# mean"); written before the first step line of every attempt and again every TL_HEAD_EVERY step lines, so a tail always shows it.
# Readers skip it: its step column is not a number.
TL_HEAD = (f'{"time":<19} | {"att":>3} | {"step":>6} | {"loss":>11} | {"TFLOP/s/GCD":>11} | {"step s":>7} | {"HBM max GB":>10} | note'
           '   (att = attempt: 0 the first run, k the run after the k-th failure)\n')
TL_HEAD_EVERY = 25


def training_log_line(step, loss, tflops, s_per_step, hbm_max_gb, save_follows=False):
    """Append one line to <job dir>/training.log; called on the writer rank only (rank 0 at PP 1, else the last rank).
    The loss is the logged lm loss; the note names the first step of an attempt, a save that follows, and a D7 flag."""
    jd = _ctx['job_dir'] or os.environ.get('RES_JOB_DIR')
    if not jd:
        return
    note = []
    if not _ctx.get('tl_first_done'):
        _ctx['tl_first_done'] = True
        rf = _ctx.get('resumed_from')
        note.append(f'first step of attempt {_ctx["attempt"]}' + (f', resumed from the save of step {rf}' if rf else ', from scratch'))
    if S['flag'] in ('slow_step', 'slow_run'):
        note.append('D5 slow step (> 2 x the median of the last 5)' if S['flag'] == 'slow_step' else 'D5 slower than 1.3 x the prediction')
    if _loss['bad'] > 0:
        note.append(f'D7 bad loss read on this rank ({S["loss"]}; {_loss["bad"]} of {os.environ.get("XMOE_LOSS_BAD_MAX", "3")} in succession)')
    if save_follows:
        note.append(f'save of step {step} follows')
    f = lambda x, fmt: '-' if x is None else fmt % x
    n = _ctx.get('tl_lines', 0); _ctx['tl_lines'] = n + 1
    ln = (TL_HEAD if n % TL_HEAD_EVERY == 0 else '') + (f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {str(_ctx["attempt"]):>3} | {step:6d} | {f(loss, "%.6f"):>11} | {f(tflops, "%.2f"):>11} | '
          f'{f(s_per_step, "%.2f"):>7} | {f(hbm_max_gb, "%.1f"):>10} | {"; ".join(note)}\n')
    try:
        with open(os.path.join(jd, 'training.log'), 'a') as fh:
            fh.write(ln)
    except Exception as e:
        log(f'training.log not written: {e}')


# ------------------------------------------------------------------ D7 the loss rule
_loss = {'hist': [], 'bad': 0}


def loss_rule(step, loss):
    """Return True when this step's loss is bad (NaN/inf or a spike); after 3 in succession exit 13."""
    import math
    h = _loss['hist']
    bad = (loss is None) or math.isnan(loss) or math.isinf(loss)
    if not bad and len(h) >= 5:
        med = statistics.median(h[-10:])
        sd = statistics.pstdev(h[-10:]) or 1e-3
        bad = loss > med + 5 * max(sd, 0.05 * abs(med))
    if bad:
        _loss['bad'] += 1
        S['flag'] = f'loss_bad_{_loss["bad"]}'
        log(f'D7 loss rule: step {step} loss {loss} is bad ({_loss["bad"]} in succession); the step is counted as skipped')
        if _ctx['detect'] and _loss['bad'] >= int(os.environ.get('XMOE_LOSS_BAD_MAX', '3')):
            die(13, f'{_loss["bad"]} bad losses in succession, last {loss} at step {step}')
    else:
        _loss['bad'] = 0
        h.append(loss)
    return bad


# ------------------------------------------------------------------ fault injection
_inj = {'cases': None, 'slow_until': None}


def _cases():
    if _inj['cases'] is None:
        txt = ','.join(x for x in (os.environ.get('XMOE_INJECT_SCHEDULE', ''), os.environ.get('XMOE_INJECT', '')) if x)
        out = []
        for item in txt.replace('+', ',').split(','):
            item = item.strip()
            if item:
                c, r, s = item.split(':')
                out.append((c, int(r), int(s)))
        _inj['cases'] = out
    return _inj['cases']


def _marker(c, r, s):
    d = data_dir('inject')
    return os.path.join(d, f'{c}_r{r}_s{s}.fired')


def _fire(c, r, s, step):
    m = _marker(c, r, s)
    os.makedirs(os.path.dirname(m), exist_ok=True)
    with open(m, 'w') as fh:
        fh.write(json.dumps({'case': c, 'rank': r, 'step_planned': s, 'step': step, 'time': time.time(), 'time_str': ts(),
                             'node': _ctx['node'], 'attempt': _ctx['attempt']}) + '\n')
    log(f'INJECT {c} fires on rank {r} ({_ctx["node"]}) at step {step} (planned {s}), attempt {_ctx["attempt"]}')


def inject(where, step):
    """Called by the trainer at 'loop_start' (before the first step of an attempt), 'step_start', 'micro', 'save', 'loss'.
    Returns a multiplier for 'loss' (nan or 1.0); otherwise None."""
    cases = _cases()
    if not cases:
        return None
    me = _ctx['rank']
    for (c, r, s) in cases:
        if r != me:
            continue
        fired = os.path.exists(_marker(c, r, s))
        if where == 'loop_start' and c == 'inithang' and not fired and step >= s:
            _fire(c, r, s, step); S['phase'] = 'init'
            time.sleep(1200)
        if where == 'step_start' and not fired and step >= s:
            if c == 'hang':
                _fire(c, r, s, step)
                while True:
                    time.sleep(60)
            if c == 'exit':
                _fire(c, r, s, step); os._exit(3)
            if c == 'kill':
                _fire(c, r, s, step); os.kill(os.getpid(), signal.SIGKILL)
            if c == 'oom':
                _fire(c, r, s, step)
                import torch
                hold = []
                while True:                                 # raises torch.cuda.OutOfMemoryError -> D4 -> exit 11
                    hold.append(torch.empty(1 << 30, dtype=torch.uint8, device=f'cuda:{_ctx["local_rank"]}'))
            if c == 'realhostoom':                      # [2026-10-11, sandbox] a REAL host OOM for the test: 4 GiB of written pages every 0.5 s
                _fire(c, r, s, step)                     # until the kernel's/cgroup's OOM killer ends this process (run the test with
                hold = []                                # XMOE_HOSTMEM_EXIT_PCT above 100 so that the 97 % guard does not exit first)
                while True:
                    hold.append(b'\x01' * (4 << 30))
                    time.sleep(0.5)
            if c == 'hostmem':
                _fire(c, r, s, step); _inj['hostmem'] = True     # the heartbeat thread's next beat (<= 10 s) exits 19
            if c in ('slow', 'nan'):
                _fire(c, r, s, step)
                _inj[c + '_from'] = step
        if where == 'micro' and c == 'slow' and fired:
            f0 = _inj.get('slow_from')
            if f0 is not None and step < f0 + 3:
                time.sleep(2.0)
        if where == 'save' and c == 'fshang' and not fired and step >= s:
            _fire(c, r, s, step)
            S['phase'] = 'save'
            time.sleep(600)
        if where == 'loss' and c == 'nan' and fired:
            f0 = _inj.get('nan_from')
            if f0 is not None and step < f0 + 3:
                return float('nan')
    return 1.0 if where == 'loss' else None


if __name__ == '__main__':
    if len(sys.argv) >= 7 and sys.argv[1] == '--node-writer':
        node_writer(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
    else:
        print('\t'.join(HB_FIELDS))
