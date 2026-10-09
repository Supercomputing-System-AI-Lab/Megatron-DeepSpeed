"""mbpp_chat: MBPP scored through the model's TRAINED chat format.

WHY THIS EXISTS: same reason as humaneval_chat (see its utils.py) -- chat-formatted
checkpoints emit whitespace for raw-completion prompts, so the standard mbpp task
measured prompt format, not code ability. Here the problem is posed inside the trained
`User: ...\\n\\nAssistant:` wrapper. The standard MBPP 3-prompt convention includes the
test cases so the model knows the expected function NAME (there is no entry_point
field); we include test_list[0] as the signature hint.

Dataset: google-research-datasets/mbpp, config 'full' (the only config in our offline
cache -- 'sanitized' is not cached), test split = 500 docs. Fields: 'text' (problem
statement), 'code' (reference, unused), 'test_list' (assert statements),
'test_setup_code' (non-empty for exactly one test doc, task_id 367 -- prepended, else
that doc can never pass).

Scoring = execute the candidate + '\\n'.join(test_list) in a subprocess with a timeout
(same trust model as HF's code_eval: we run model-generated code deliberately; compute
nodes have no outbound network). Unlike humaneval there is no official prompt stub to
graft a body continuation onto: the extracted code IS the candidate.
"""
import os
import re
import subprocess
import sys
import tempfile
import textwrap


def doc_to_text(doc):
    return ('User: Write a Python function for the following task: '
            + doc['text'].strip() + '\n\n'
            'Your function must pass this test:\n\n'
            '```python\n' + doc['test_list'][0].strip() + '\n```\n\n'
            'Reply with the complete function definition only, inside a python '
            'code block.\n\nAssistant:')


def _extract_code(gen):
    """The reply may be a fenced block, a bare definition, or prose+code.

    The unfenced branch must tolerate leading whitespace. Before 2026-08-17 the
    anchor was `(?:^|\\n)((?:from |import |def |...)` with no re.M, so `^` meant
    start-of-string only and a reply that opened with a single space before `def`
    matched NOTHING: the function fell through to `return gen`, the temp program
    began with an indented `def`, and Python raised `IndentationError: unexpected
    indent` at line 1 before a single assert ran. That scored the reply 0.0 while
    `nonempty` still reported 1.0, so the sanity metric never fired. Measured on
    the cached replies: 59 of 80 masked-SFT step-130 generations open with exactly
    one space, and the cell read 0.013 instead of 0.288. `[ \\t]*` after the anchor
    admits them; textwrap.dedent then removes the common indent so the extracted
    block is a valid top-level definition.
    """
    m = re.search(r'```(?:python)?\s*\n(.*?)```', gen, re.S)
    if m:
        return m.group(1)
    m = re.search(r'(?:^|\n)[ \t]*((?:from |import |def |@|class ).*)', gen, re.S)
    return textwrap.dedent(m.group(1)) if m else gen


def process_results(doc, results):
    gen = results[0] if results else ''
    candidate = _extract_code(gen)
    setup = doc.get('test_setup_code') or ''
    program = (setup + '\n\n' if setup.strip() else '') + candidate + '\n\n' \
        + '\n'.join(doc['test_list']) + '\n'
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
