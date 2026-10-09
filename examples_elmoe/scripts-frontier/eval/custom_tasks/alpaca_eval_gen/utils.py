"""alpaca_eval_gen helpers.

Generation TRANSPORT for AlpacaEval 2: the model answers the 805 official
instructions; the LC win-rate judging happens OFFLINE later (alpaca_eval CLI +
API judge) on the file exported by eval/export_alpacaeval.py. The metrics here
are mechanical transport indicators only, so the run integrates with the
registry/cache like any other row without pretending to be a quality score.

Prompt format is EXACTLY the trained SFT wrapper (tools/sft_chat_format.py):
    <bos>User: {q}\n\nAssistant: {a}<eos>
The prompt ends at 'Assistant:' WITHOUT the trailing space: whole-string BPE
would encode a dangling lone-space token the model never saw in training
(training encoded the 'Assistant: ' prefix incrementally, so its space fuses
with the prefix); ending at the colon lets the model emit the natural
space-prefixed first content token.
"""


def doc_to_text(doc):
    return 'User: ' + doc['instruction'] + '\n\nAssistant:'


def process_results(doc, results):
    text = results[0] if results else ''
    return {
        'gen_len': float(len(text)),
        'nonempty': 1.0 if text.strip() else 0.0,
    }
