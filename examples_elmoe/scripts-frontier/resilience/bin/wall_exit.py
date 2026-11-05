#!/usr/bin/env python3
"""wall_exit.py: the arithmetic of the WALL-AWARE EXIT (2026-10-10), used by megatron/training.py when XMOE_WALL_AWARE_EXIT=1.

THE PROBLEM IT SOLVES (the 260-node job 5645677, 2026-10-09, wall 2 h, steps of 43 min): the trainer's exit was
--exit-duration-in-mins = wall - 20 (100 min), which Megatron checks only at the END of an iteration against the trainer's own
start time. The steps ended at minute 51 and 94, so the third step started at minute 94 and the Slurm wall killed it at
minute 120: about 22 minutes of 256 nodes were lost, and a hard stop at the wall can land in the middle of a save. The user's
instruction (2026-10-09 23:50): the exit must follow the job's real Slurm end time, not a hand-set number, and a save must never
be cut by the wall.

THE RULE IN PLAIN WORDS. After every iteration the trainer asks: "can one more step, and the save that must follow it, finish
before the job's end time?" It predicts the cost of one more step as
    the slowest of the last 3 iterations x (1 + XMOE_WALL_MARGIN_FRAC)        (default margin 10 percent)
  + the cost of a save (the measured synchronous part of the last save + the measured time from the asynchronous engine's
    commit to 'latest' of the last complete save; before the first save XMOE_SAVE_EST_S, default 120 s)
  + a fixed margin XMOE_WALL_MARGIN_S                                            (default 120 s).
If the seconds that remain before the end time are fewer than that cost, the trainer exits NOW: it saves (unless the save of
this step is already in flight), waits until that save is complete on every rank and 'latest' names it, logs one sentence
with the arithmetic, and exits with code 0. Rank 0 decides and the decision is shared with every rank through an all-reduce,
so no rank continues alone.

THE END TIME. Slurm exports SLURM_JOB_END_TIME (the UNIX timestamp of the job's projected end; sbatch(1), Slurm 25.11 here)
into the job's environment: that is the first source. When it is missing, `scontrol show job <id>` gives EndTime once. When
neither exists, the old --exit-duration-in-mins is turned into an end time (the trainer's start + the minutes), so that the
prediction still protects the save. The source that was used is logged.

This file holds only the arithmetic and the parsing (no torch), so that tests/test_wall_exit.py can check it on a login node.
"""
import os
import re
import subprocess
import time

DEFAULT_MARGIN_FRAC = 0.1      # XMOE_WALL_MARGIN_FRAC: the step prediction is the slowest of the last 3 steps times (1 + this)
DEFAULT_MARGIN_S = 120.0       # XMOE_WALL_MARGIN_S: the fixed margin in seconds, added once
DEFAULT_SAVE_EST_S = 120.0     # XMOE_SAVE_EST_S: the save cost assumed before any save of this run was measured
STEP_HISTORY = 3               # the step prediction looks at this many recent iterations


def fmt(x):
    """1300 -> '1,300'; 2584.4 -> '2,584'; None -> '-'."""
    if x is None:
        return '-'
    return f'{float(x):,.0f}'


def hms(epoch):
    """An epoch -> 'HH:MM:SS' in the local time zone (the Slurm end time as a person reads it in squeue)."""
    return time.strftime('%H:%M:%S', time.localtime(float(epoch)))


# ------------------------------------------------------------------------------------------------------ the end time
def parse_scontrol_endtime(text):
    """The EndTime=YYYY-MM-DDTHH:MM:SS field of `scontrol show job` -> epoch seconds (local time), or None.
    'Unknown' and 'N/A' (a pending job) give None."""
    m = re.search(r'EndTime=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', text or '')
    if not m:
        return None
    try:
        return float(time.mktime(time.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S')))
    except (ValueError, OverflowError):
        return None


def end_time_from_env(env=None):
    """(epoch, 'SLURM_JOB_END_TIME') when Slurm exported the end time into the job, else (None, None)."""
    env = os.environ if env is None else env
    v = env.get('SLURM_JOB_END_TIME', '').strip()
    if v.isdigit() and int(v) > 0:
        return float(v), 'SLURM_JOB_END_TIME'
    return None, None


def end_time_from_scontrol(job_id, run=None, timeout_s=10):
    """(epoch, 'scontrol show job EndTime') from one scontrol call, else (None, None). `run` is injectable for the test."""
    if not job_id:
        return None, None
    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s).stdout)
    try:
        t = parse_scontrol_endtime(run(['scontrol', 'show', 'job', str(job_id)]))
    except Exception:
        t = None
    return (t, 'scontrol show job EndTime') if t else (None, None)


def resolve_end_time(env=None, job_id=None, train_start=None, exit_duration_min=None, run=None):
    """The job's end time and where it came from: (epoch or None, source, one explaining sentence).
    Order: SLURM_JOB_END_TIME, then scontrol, then the old --exit-duration-in-mins as (train start + minutes), else none."""
    env = os.environ if env is None else env
    t, src = end_time_from_env(env)
    if t:
        return t, src, f'the end time {hms(t)} comes from SLURM_JOB_END_TIME, which Slurm exported into the job'
    t, src = end_time_from_scontrol(job_id or env.get('SLURM_JOB_ID'), run=run)
    if t:
        return t, src, f'the end time {hms(t)} comes from `scontrol show job {job_id or env.get("SLURM_JOB_ID")}` (SLURM_JOB_END_TIME was not in the environment)'
    if exit_duration_min and train_start:
        t = float(train_start) + 60.0 * float(exit_duration_min)
        return t, '--exit-duration-in-mins', (f'the end time {hms(t)} is the trainer\'s start plus --exit-duration-in-mins {exit_duration_min} '
                                               f'(neither SLURM_JOB_END_TIME nor scontrol gave an end time): the fallback')
    return None, None, 'no end time is known (no SLURM_JOB_END_TIME, no scontrol EndTime, no --exit-duration-in-mins): the wall-aware exit is off'


# ------------------------------------------------------------------------------------------------------ the save-time files
def read_save_files(data_dir):
    """The last measured save times of this job from the job folder's data/ files, for an attempt that has not saved yet:
    (synchronous seconds from save_times.tsv, written by hb.py's SaveDeadline on rank 0; asynchronous commit-to-'latest'
    seconds from save_async_times.tsv, written by the async engine on rank 0). Either is None when its file has no line."""
    def last_value(name):
        try:
            lines = [l for l in open(os.path.join(data_dir, name)) if l.strip() and not l.startswith('#')]
            return float(lines[-1].split('\t')[1]) if lines else None
        except Exception:
            return None
    if not data_dir:
        return None, None
    return last_value('save_times.tsv'), last_value('save_async_times.tsv')


# ------------------------------------------------------------------------------------------------------ the predictor
class Predictor:
    """Keeps the last step times and the last save times; predicts the cost of one more step; decides."""

    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.margin_frac = float(env.get('XMOE_WALL_MARGIN_FRAC', DEFAULT_MARGIN_FRAC))
        self.margin_s = float(env.get('XMOE_WALL_MARGIN_S', DEFAULT_MARGIN_S))
        self.save_est_s = float(env.get('XMOE_SAVE_EST_S', DEFAULT_SAVE_EST_S))
        self.steps = []            # the last STEP_HISTORY iteration times, seconds
        self.sync_save_s = None    # the synchronous part of the last save of this process (the save-checkpoint timer)
        self.async_save_s = None   # the asynchronous part: from the engine's commit to 'latest' (rank 0 knows it)
        self.save_src = None       # where the save cost came from, for the log

    def record_step(self, seconds):
        self.steps.append(float(seconds))
        del self.steps[:-STEP_HISTORY]

    def record_save(self, sync_s=None, async_s=None, src=None):
        if sync_s is not None:
            self.sync_save_s = float(sync_s)
        if async_s is not None:
            self.async_save_s = float(async_s)
        if src:
            self.save_src = src

    def seed_from_files(self, data_dir):
        """Before the first save of an attempt: take the previous attempt's measured times from the job folder, if any."""
        s, a = read_save_files(data_dir)
        if s is not None or a is not None:
            self.record_save(sync_s=s, async_s=a, src='the job folder\'s save_times.tsv / save_async_times.tsv (a previous attempt)')

    def step_basis(self):
        """(the step time used, 'the slowest of the last k iterations') or (None, note) before the first step."""
        if not self.steps:
            return None, 'no iteration measured yet'
        return max(self.steps), f'the slowest of the last {len(self.steps)} iteration(s)'

    def save_cost(self):
        """(seconds, source). The measured sync + async parts when any save was measured, else the estimate."""
        if self.sync_save_s is None and self.async_save_s is None:
            return self.save_est_s, f'XMOE_SAVE_EST_S (no save measured yet)'
        s = (self.sync_save_s or 0.0) + (self.async_save_s or 0.0)
        return s, self.save_src or ('the last save: %s s synchronous + %s s from the commit to \'latest\'' % (fmt(self.sync_save_s or 0), fmt(self.async_save_s or 0)))

    def decide(self, now, end):
        """The decision for the step that just ended. Returns a dict with every number of the sentence; 'done' is True when
        the remaining time is shorter than the predicted cost of one more step. With no end time: never done."""
        remaining = None if end is None else float(end) - float(now)
        step_s, basis = self.step_basis()
        save_s, save_src = self.save_cost()
        step_pred = None if step_s is None else step_s * (1.0 + self.margin_frac)
        cost = None if step_pred is None else step_pred + save_s + self.margin_s
        done = bool(end is not None and cost is not None and remaining < cost)
        return dict(done=done, remaining=remaining, step_s=step_s, step_basis=basis, factor=1.0 + self.margin_frac,
                    step_pred=step_pred, save_s=save_s, save_src=save_src, margin_s=self.margin_s, cost=cost)


def arithmetic(d):
    """'2,584 x 1.1 + 68 + 120 = 3,030 s' from a decision dict (the user's example sentence)."""
    if d.get('cost') is None:
        return 'no prediction yet (no iteration measured)'
    return f'{fmt(d["step_s"])} x {d["factor"]:.1f} + {fmt(d["save_s"])} + {fmt(d["margin_s"])} = {fmt(d["cost"])} s'


def exit_sentence(step, d, end, source, world, latest_tag, save_ok, wait_s, started_save):
    """The one full-sentence log line of the planned exit."""
    if latest_tag is None and save_ok:
        save_txt = 'no checkpoint is written in this run (no --save), so there is no save to wait for'
    else:
        save_txt = (f"the save of step {step} is complete on all {world} ranks ('latest' names {latest_tag}, {fmt(wait_s)} s after the "
                f"{'save started' if started_save else 'step ended'})" if save_ok else
                f"the save of step {step} is NOT complete after {fmt(wait_s)} s ('latest' names {latest_tag or 'nothing'}; the next link resumes from it)")
    return (f'wall-aware exit at step {step}: {fmt(d["remaining"])} s remain before the Slurm end time {hms(end)} (source {source}), '
            f'one more step would need {arithmetic(d)}; {save_txt}; exiting with code 0')


def continue_sentence(step, d, end, source):
    """A short line for the rank-0 log at a step where training goes on (XMOE_WALL_VERBOSE=1 prints it every step)."""
    return (f'wall-aware check at step {step}: {fmt(d["remaining"])} s remain before the Slurm end time {hms(end)} ({source}); '
            f'one more step needs {arithmetic(d)}; training continues')


def record_line(attempt, step, d, end, source, latest_tag, save_ok, wait_s, sentence):
    """One tab-separated line for <job>/data/wall_exit.tsv (the restart loop, results.py and STATUS.md read it)."""
    cols = [f'{time.time():.0f}', str(attempt), str(step), fmt(d['remaining']).replace(',', ''), fmt(d['step_s']).replace(',', ''),
            f'{d["factor"]:.2f}', fmt(d['save_s']).replace(',', ''), fmt(d['margin_s']).replace(',', ''), fmt(d['cost']).replace(',', ''),
            f'{float(end):.0f}' if end else '-', source or '-', latest_tag or '-', 'yes' if save_ok else 'no', fmt(wait_s).replace(',', ''),
            sentence.replace('\t', ' ')]
    return '\t'.join(cols) + '\n'


RECORD_HEADER = ('# wall_exit.tsv: one line per wall-aware exit (megatron/training.py, XMOE_WALL_AWARE_EXIT=1). Columns: epoch, attempt, step, '
                 'remaining_s, step_s, factor, save_s, margin_s, cost_s, end_epoch, end_source, latest_tag, save_complete, wait_s, sentence\n')
