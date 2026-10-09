#!/usr/bin/env python3
"""render_branches.py — group eval results by TRAINING-DATA BRANCH, not by run dir.

This is the ASSIGNMENT OF RECORD for the 4-branch showcase: the untrained
DeepSeek-MoE-16B base, the Nemotron continued-pretrain, and the two dolci
instruction branches (masked SFT vs full-loss packed). A run dir changes when a
job resumes, but the training lineage does not, so branches continue across run
dirs; BRANCHES below records which run dir owns each step -- edit here if a
continuation is attributed to the wrong branch. Every branch table shares the
same step columns so the four tables read as one comparison; each branch is
plotted on ITS OWN training-step axis (see STEP_ALIAS);
steps a branch never reached render as empty cells.
"""
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
# THE experiment rule (experiment_of): shared with the done-check, report.py and the
# dashboard, so no selector in this file carries its own copy.
sys.path.insert(0, os.path.join(EVAL_DIR, os.pardir, os.pardir, os.pardir,
                                'tasks', 'eval_harness'))
import result_registry  # noqa: E402

STEPS = ['0', '20', '40', '50', '60', '80', '100', '130']
BRANCHES = {
    'Base (DeepSeek-MoE-16B)': [
        ('deepseek_16b_ds', '0', 'base'),
    ],
    # This run's global step counter carries over from its initialisation, so its
    # on-disk checkpoints 80/100/120 are steps 20/40/60 of NEMOTRON training. The
    # table shows each branch's own training steps, so this branch is plotted at
    # 20/40/60 and STEP_ALIAS maps those back to the stored checkpoint names.
    'Nemotron continued-pretrain': [
        ('2026-08-09_5213097', '20', ''),
        ('2026-08-09_5213097', '40', ''),
        ('2026-08-09_5213097', '60', ''),
    ],
    'Masked-SFT (dolci-sft)': [
        ('2026-08-14_5261740', '20', ''),
        ('2026-08-14_5261740', '50', ''),
        ('2026-08-14_5266786', '80', ''),
        ('2026-08-15_5276906', '100', ''),
        ('2026-08-15_5276906', '130', ''),
    ],
    'Full-loss packed (dolci-fullloss)': [
        ('2026-08-14_5264971', '20', ''),
        ('2026-08-14_5264971', '50', ''),
        ('2026-08-14_5266788', '80', ''),
        ('2026-08-14_5266788', '100', ''),
        ('2026-08-15_5276907', '130', ''),
    ],
}
# task -> (registry task name, EXACT metric name, display name).
# Exact, never a substring: 'acc' is a substring of 'acc_norm', and because both are
# written by the same job with the same timestamp the substring form silently rendered
# acc_norm while results/tables/*.md rendered acc -- two tables, one checkpoint, 0.15
# apart (audit 2026-08-16). hellaswag reports acc_norm, the metric the DeepSeek-MoE
# paper's ~77-79 numbers use, and the display name says so.
METRICS = [
    ('gsm8k', 'exact_match/flexible-extract', 'gsm8k (flexible)'),
    ('ifeval', 'prompt_level_strict_acc', 'ifeval (prompt-strict)'),
    ('bbh', 'exact_match/get-answer', 'bbh'),
    ('minerva_math', 'exact_match', 'minerva_math'),
    ('triviaqa', 'exact_match/remove_whitespace', 'triviaqa'),
    ('humaneval_chat', 'pass_at_1', 'humaneval (chat, fixed)'),
    ('mbpp_chat', 'pass_at_1', 'mbpp (chat)'),
    ('hellaswag', 'acc_norm', 'hellaswag (canary, acc_norm)'),
    ('gsm8k', 'exact_match/strict-match', 'gsm8k (strict, format-only)†'),
]

# Display step -> stored checkpoint step, for runs whose global counter continues
# from their initialisation. Keeps every branch on ITS OWN training-step axis, which
# is the only axis on which a step number means the same thing across the table.
STEP_ALIAS = {
    '2026-08-09_5213097': {'20': '80', '40': '100', '60': '120'},
}


# The committed instruct-short subsets (task_table.py *_s rows). A row that does not
# carry EXACTLY this (limit, num_fewshot) is a different measurement wearing the same
# lm-eval task name -- e.g. the 8-shot n=32 gsm8k probes on 2026-08-09_5213105 -- and
# must never win a cell. Enforced, not assumed.
COMMITTED = {
    'gsm8k': (100, 0), 'ifeval': (40, 0), 'bbh': (216, 3), 'minerva_math': (49, 4),
    'triviaqa': (500, 5), 'humaneval_chat': (80, 0), 'mbpp_chat': (80, 0),
    'hellaswag': (100, 0),
}

# Cells that measure the EVALUATION BUDGET rather than the model, keyed by
# (run, DISPLAY step, task) with the measurement that established each one.
#
# Keyed per CELL, not per (run, task). The earlier (run, task) key painted all three
# Nemotron ifeval cells with one brush, which was wrong in both directions (audit
# 2026-08-17): step 20 ifeval had 1 of its own 40 replies affected -- 2.5%, so the flag
# was hiding a genuine 0.325 -> 0.150 drop behind a disclaimer its own cache
# contradicted -- while step 60 gsm8k, at 64 of 100, carried the branch's only apparent
# gsm8k gain and was not flagged at all.
#
# Criterion: >= 20% of the replies for THAT cell open `<think>` and never close it.
# Counted per reply, not as (total opens - total closes): the difference form reported
# -8 on Nemotron step-60 minerva_math, where the checkpoint emits `</think>` with no
# opener. Where one response-cache file serves two tasks (a cache is keyed by run,
# step, code_fp, seq_len and shot count, so tasks sharing those share a file), replies
# are attributed by rebuilding each task's own request keys -- the step-40 seq4096
# cache holds 120 replies, 80 humaneval (100% unclosed) and 40 ifeval (22.5%), and
# counting the file as a whole would have reported 74.2% for both.
#
# Deliberately per task and per step: on the very checkpoints flagged below, bbh
# (0 of 216 replies open `<think>`; the replies are coherent multi-step derivations),
# minerva_math (0 of 49) and triviaqa (0 of 500) are unaffected. Blanket-flagging them
# would understate the model as badly as publishing the truncated cells overstates it.
GEN_ARTIFACT = {
    ('2026-08-09_5213097', '20', 'humaneval_chat'): '80 of 80 replies open `<think>` and none close it within max_gen_toks=512',
    ('2026-08-09_5213097', '40', 'humaneval_chat'): '80 of 80 replies open `<think>` and none close it within max_gen_toks=512',
    ('2026-08-09_5213097', '60', 'humaneval_chat'): '80 of 80 replies open `<think>` and none close it within max_gen_toks=512',
    ('2026-08-09_5213097', '40', 'mbpp_chat'): '80 of 80 replies open `<think>` and none close it within max_gen_toks=512',
    ('2026-08-09_5213097', '60', 'mbpp_chat'): '80 of 80 replies open `<think>` and none close it within max_gen_toks=512',
    ('2026-08-09_5213097', '40', 'ifeval'): '9 of 40 replies open `<think>` and never close it (22.5%)',
    ('2026-08-09_5213097', '60', 'ifeval'): '10 of 40 replies open `<think>` and never close it (25.0%)',
    ('2026-08-09_5213097', '60', 'gsm8k'): '64 of 100 replies open `<think>` and never close it within max_gen_toks=256 (64.0%)',
}

# Cells measured and deliberately NOT flagged, recorded so the absence of a flag is a
# result rather than an oversight. Same measurement, same caches, same date.
GEN_ARTIFACT_CLEARED = {
    ('2026-08-09_5213097', '20', 'ifeval'): '1 of 40 replies (2.5%) -- the 0.325 -> 0.150 drop is the model, not the budget',
    ('2026-08-09_5213097', '20', 'gsm8k'): '1 of 100 replies (1.0%)',
    ('2026-08-09_5213097', '40', 'gsm8k'): '15 of 100 replies (15.0%)',
}


def load_rows():
    """Every PRODUCTION registry row.

    Rows written by a run declared `experiment=<tag>` are dropped here, at the door, by
    the shared rule (result_registry.experiment_of): they are measurements made to
    compare, never to publish, and every cell selector below would otherwise treat one
    as an ordinary -- and usually the newest -- candidate.
    """
    rows = []
    reg = os.path.join(EVAL_DIR, 'results', 'registry')
    for fn in sorted(os.listdir(reg)):     # sorted: shard order must not decide ties
        if fn.endswith('.jsonl'):
            for line in open(os.path.join(reg, fn)):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if result_registry.experiment_of(row):
                    REJECTED.append((row.get('ckpt_run'), row.get('step'),
                                     row.get('task'),
                                     'experiment=%s (never a published value)'
                                     % result_registry.experiment_of(row)))
                    continue
                rows.append(row)
    return rows


REJECTED = []      # (run, step, task, why) for rows excluded -- printed, never silent


def cell(rows, run, step, task, metric):
    """Winning row for one cell, with provenance enforced.

    Ranking is (complete-before-partial, timestamp) -- matching render_results.py's
    documented policy. Ranking on timestamp alone let a re-run of score_partial.py
    override a finished evaluation (audit 2026-08-16).

    `step` is the branch's OWN training step; STEP_ALIAS translates it to the step
    number the checkpoint is stored under for runs whose global counter continues
    from their initialisation.
    """
    disp_step = step
    step = STEP_ALIAS.get(run, {}).get(step, step)
    want = COMMITTED.get(task)
    best = None
    for r in rows:
        if (r.get('ckpt_run') != run or str(r.get('step')) != step
                or r.get('task') != task or str(r.get('metric', '')) != metric):
            continue
        partial = bool(r.get('partial'))
        if not partial and want is not None:
            # Check the number of documents ACTUALLY scored (sample_len), not the
            # job-level `limit` bookkeeping field. limit is the manifest budget and was
            # over-reported for solo runs of tasks that share a manifest -- a 100-doc
            # hellaswag scored under short_q512_fs0.json recorded limit=200, and this
            # guard then discarded a correct measurement (2026-08-17). sample_len is
            # what the grader saw, so it is the honest provenance signal; limit remains
            # the fallback for rows written before sample_len existed.
            #
            # The sample_len row must come from THE SAME WRITE as the row under test --
            # same job_id and same timestamp. Matching only (run, step, task, partial)
            # takes whichever sample_len happens to appear first in file order, which is
            # borrowing a number from a different job. Measured while auditing this file
            # (2026-08-17): on deepseek_16b_ds@0/gsm8k the guard evaluated the 8-shot
            # n=1319 full-set rows as got=(100, 8) -- it had borrowed sample_len=100 from
            # job 5278458 and paired it with fewshot=8 from the row under test. Only the
            # shot-count half rejected them. Any future full-set run at the committed
            # shot count would have passed the document-count check on a borrowed number.
            n_scored = next(
                (x.get('value') for x in rows
                 if x.get('ckpt_run') == run and str(x.get('step')) == step
                 and x.get('task') == task and x.get('metric') == 'sample_len'
                 and bool(x.get('partial')) == partial
                 and x.get('job_id') == r.get('job_id') and x.get('ts') == r.get('ts')),
                None)
            got = (int(n_scored) if n_scored is not None else r.get('limit'),
                   r.get('num_fewshot'))
            if got != want:
                REJECTED.append((run, step, task,
                                 f'docs/fewshot {got} != committed {want}'))
                continue
        key = (not partial, r.get('ts') or '')
        if best is None or key > best[0]:
            best = (key, r.get('value'), partial, r.get('limit'), r.get('sample_len'))
    if best is None:
        return ''
    _, v, partial, lim, _ = best
    if partial:
        # A partial cell's own coverage: score_partial's `partial` fraction is against
        # the FULL task, so the honest denominator is its scored-doc count.
        n = next((r.get('value') for r in rows
                  if r.get('ckpt_run') == run and str(r.get('step')) == step
                  and r.get('task') == task and r.get('metric') == 'sample_len'
                  and bool(r.get('partial'))), None)
        n_txt = f'{int(n)}' if n else '?'
        return f'{v:.3f}*[n={n_txt}]'
    if (run, disp_step, task) in GEN_ARTIFACT:
        return f'{v:.3f}‡'
    return f'{v:.3f}'


def main():
    rows = load_rows()
    out = ['# Branch comparison — grouped by training data',
           '',
           'Every number is one fixed, committed document subset per task, identical across',
           'checkpoints, so differences are paired comparisons rather than sampling noise:',
           'gsm8k n=100 (0-shot), ifeval n=40 (0-shot), bbh n=216 (3-shot),',
           'minerva_math n=49 (4-shot), triviaqa n=500 (5-shot), humaneval-chat n=80,',
           'mbpp-chat n=80, hellaswag n=100. A cell is rendered only if its stored run',
           'matches that exact subset size and shot count.',
           '',
           '`*[n=N]` = provisional: only N of the subset\'s documents were scored.',
           '`‡` = generation-budget artifact, NOT a model-quality number (see the footnote).',
           'Step 0 = the untrained DeepSeek-MoE-16B base.',
           '',
           '**Read down a column, not across branches at the same step.** The branches run',
           'different learning rates, token counts per step, and schedule lengths, so equal',
           'step numbers are not equal amounts of training (details in the showcase report).',
           '']
    for branch, points in BRANCHES.items():
        pts = {s: (run, f) for run, s, f in points}
        out += [f'## {branch}', '',
                '| task | ' + ' | '.join(
                    f'step {s}{" ("+pts[s][1]+")" if s in pts and pts[s][1] else ""}'
                    for s in STEPS) + ' |',
                '|---|' + '---|' * len(STEPS)]
        for task, metric, disp in METRICS:
            vals = [cell(rows, pts[s][0], s, task, metric) if s in pts else ''
                    for s in STEPS]
            if any(vals):
                out.append(f'| {disp} | ' + ' | '.join(vals) + ' |')
        out.append('')
    out += ['† gsm8k strict requires the exact `#### <answer>` output format; it is',
            'reported for completeness, not as a measure of ability (use the flexible',
            'row for that).', '']
    if GEN_ARTIFACT:
        out += ['‡ This cell measures the EVALUATION BUDGET, not the model. The checkpoint',
                '  answers with a `<think>` reasoning monologue that the generation limit cuts',
                '  off before an answer appears, so the grader scores a truncated reply no',
                '  matter how capable the model is. Flag criterion: at least 20% of the replies',
                '  for THIS cell open `<think>` and never close it, counted per reply in that',
                "  cell's own response cache and attributed by rebuilding each task's request",
                '  keys where one cache file serves two tasks. Measured:', '']
        for (run, step, task), why in sorted(GEN_ARTIFACT.items(),
                                             key=lambda kv: (kv[0][2], int(kv[0][1]))):
            out.append(f'  - `{run}` @ step {step} / {task}: {why}.')
        out += ['',
                '  Measured and deliberately NOT flagged, so that a missing flag is a result',
                '  rather than an oversight:', '']
        for (run, step, task), why in sorted(GEN_ARTIFACT_CLEARED.items(),
                                             key=lambda kv: (kv[0][2], int(kv[0][1]))):
            out.append(f'  - `{run}` @ step {step} / {task}: {why}.')
        out += ['',
                '  Also unaffected on these same checkpoints and therefore unflagged: bbh',
                '  (0 of 216 replies open `<think>`; they are coherent multi-step derivations),',
                '  minerva_math (0 of 49) and triviaqa (0 of 500).',
                '  A re-measurement of the flagged cells with a larger budget is in the queue.', '']
    dest = os.path.join(EVAL_DIR, 'results', 'tables', 'BRANCH_COMPARISON.md')
    open(dest, 'w').write('\n'.join(out) + '\n')
    print(f'wrote {dest}')
    if REJECTED:
        print(f'  [provenance] rejected {len(REJECTED)} row(s) that did not match the '
              f'committed subset:')
        for run, step, task, why in sorted(set(REJECTED)):
            print(f'    {run}@{step} {task}: {why}')


if __name__ == '__main__':
    main()
