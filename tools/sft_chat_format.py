"""The chat format and (token_ids, answer_mask) renderer for SFT data.

Single source of truth. Import this everywhere a conversation is rendered or a
mask is built -- the same literals used to live in three places
(measure_sft_corpus.py, prepare_nemotron_deepseek.sh twice) and had already
drifted between two of them.

The DeepSeek chat format, written out literally rather than pulled from the
-chat model's tokenizer_config, because the paper has to state the exact format
we trained on and the base model's tokenizer_config carries no chat_template:

    <bos>User: {q}\n\nAssistant: {a}<eos>User: {q}\n\nAssistant: {a}<eos>...

Three properties are load-bearing, and each is a silent training bug if broken:

  * Tokenize INCREMENTALLY, never tok(A) + tok(B). BPE merges across a
    boundary, so concatenating independent encodings misaligns the mask by a
    few positions per turn -- which trains the model on a shifted target and is
    nearly impossible to spot afterwards.
  * The EOS after each assistant turn IS supervised (mask 1). It is the only
    token that teaches the model to stop; mask it and the model generates until
    the length cap. Invisible in the loss curve.
  * The mask is built BY POSITION, never by token id. On the DeepSeek tokenizer
    pad == eos == 100001, so any `ids == pad_id` test would mask every stop
    token in the corpus.
"""

USER_PREFIX = 'User: '
ASSISTANT_PREFIX = '\n\nAssistant: '
TURN_SEP = '\n\n'


def render(msgs, tok, bos_id, eos_id):
    """Conversation -> (token_ids, answer_mask). Mask is 1 on assistant tokens
    and each turn's EOS, 0 everywhere else (BOS, system, user turns).

    `msgs` is [{'role': 'system'|'user'|'assistant', 'content': str}, ...].
    `tok` needs a transformers-style `encode(text, add_special_tokens=False)`.
    """
    ids, mask = [bos_id], [0]
    for m in msgs:
        role, content = m['role'], (m['content'] or '')
        if role == 'system':
            piece = content + TURN_SEP
            ids += tok.encode(piece, add_special_tokens=False)
            mask += [0] * (len(ids) - len(mask))
        elif role == 'user':
            piece = USER_PREFIX + content + ASSISTANT_PREFIX
            ids += tok.encode(piece, add_special_tokens=False)
            mask += [0] * (len(ids) - len(mask))
        elif role == 'assistant':
            new = tok.encode(content, add_special_tokens=False)
            ids += new + [eos_id]
            mask += [1] * (len(new) + 1)   # the EOS is supervised -- see above
        else:
            continue
    assert len(ids) == len(mask), f'{len(ids)} != {len(mask)}'
    return ids, mask


def pad_to(ids, mask, seq_length, pad_id):
    """Pad one rendered conversation to exactly `seq_length` tokens.

    Every document must be emitted at exactly seq_length: GPTDataset packs by
    concatenating documents and cutting every seq_length+1 tokens, so equal-
    length documents are what makes one sample == one conversation (plus a
    1-token, mask-0 spill of the next document's BOS). Padding is masked out by
    POSITION -- the pad id happens to equal EOS on DeepSeek, and that is fine
    precisely because nothing here tests token values.

    Returns (ids, mask), both exactly seq_length long. Raises if the
    conversation is longer than seq_length -- callers drop those, never
    truncate (a severed answer supervises stopping mid-thought).
    """
    n = len(ids)
    assert len(mask) == n
    assert n <= seq_length, f'conversation of {n} tokens exceeds {seq_length}'
    fill = seq_length - n
    return ids + [pad_id] * fill, mask + [0] * fill


def normalize_messages(msgs):
    """Normalize the field names of public SFT sets to role/content.

    Covers the two conventions that appear in practice: messages-style
    ({'role','content'}) and ShareGPT-style ({'from','value'} with
    human/gpt role names).
    """
    norm = []
    for m in msgs:
        role = m.get('role') or m.get('from') or ''
        content = m.get('content') or m.get('value') or ''
        role = {'human': 'user', 'gpt': 'assistant'}.get(role, role)
        norm.append({'role': role, 'content': content})
    return norm
