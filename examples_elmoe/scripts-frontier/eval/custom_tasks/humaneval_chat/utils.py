"""humaneval_chat: HumanEval scored through the model's TRAINED chat format.

WHY THIS EXISTS: the standard humaneval task feeds the bare function signature as a
raw completion prompt. Models trained on chat-formatted data (all our SFT/CPT
checkpoints) emit only whitespace for such prompts -- measured 80/80 degenerate on
four checkpoints (see 0815_2026_INSTRUCT_SHORT_SAMPLES.md) -- so the standard task
measured prompt format, not code ability. Here the problem is posed inside the
trained `User: ...\\n\\nAssistant:` wrapper, and the grader extracts the code from a
conversational reply before running the official tests.

Scoring = execute the candidate against the dataset's own `test` + `check(entry_point)`
in a subprocess with a timeout (same trust model as HF's code_eval: we run
model-generated code deliberately; compute nodes have no outbound network).
"""
import os
import re
import subprocess
import sys
import tempfile
import textwrap


def doc_to_text(doc):
    return ('User: Complete the following Python function. Reply with the complete '
            'function definition only, inside a python code block.\n\n'
            '```python\n' + doc['prompt'].rstrip() + '\n```\n\nAssistant:')


def _extract_code(gen):
    """The reply may be a fenced block, a bare definition, or prose+code.

    The unfenced branch must tolerate leading whitespace -- identical defect and
    identical fix to custom_tasks/mbpp_chat/utils.py (see the long note there).
    Until 2026-08-17 a reply that opened with a space before `def` matched the
    anchor `(?:^|\\n)` nowhere, the raw reply was executed, and Python raised
    `IndentationError: unexpected indent` before any test ran. On this task the
    bug was masked wherever the model emitted a ``` fence (the fenced branch
    strips the indent), which is why the masked-SFT column was correct and the
    full-loss column read 0.000 at every step. Both are fixed here.
    """
    m = re.search(r'```(?:python)?\s*\n(.*?)```', gen, re.S)
    if m:
        return m.group(1)
    m = re.search(r'(?:^|\n)[ \t]*((?:from |import |def |@|class ).*)', gen, re.S)
    return textwrap.dedent(m.group(1)) if m else gen


def process_results(doc, results):
    gen = results[0] if results else ''
    code = _extract_code(gen)
    # If the model rewrote the whole function, use its version; if it wrote only a
    # body continuation, graft it onto the official prompt.
    candidate = code if re.search(rf'def\s+{re.escape(doc["entry_point"])}\s*\(', code) \
        else doc['prompt'] + code
    program = candidate + '\n\n' + doc['test'] + f'\n\ncheck({doc["entry_point"]})\n'
    passed = 0.0
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        passed = 1.0 if r.returncode == 0 else 0.0
    except Exception:
        passed = 0.0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {'pass_at_1': passed, 'nonempty': 1.0 if gen.strip() else 0.0}
