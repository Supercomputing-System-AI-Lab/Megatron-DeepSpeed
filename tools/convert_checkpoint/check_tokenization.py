#!/usr/bin/env python
# coding=utf-8
"""Sanity-check that the framework's pre-tokenized .bin matches the DeepSeek tokenizer, and
show whether the HF-reference path (AutoTokenizer) and the preprocess path (raw tokenizers)
agree -- e.g. on the BOS token. Answers "is the 2.38-vs-4.8 gap a tokenization/BOS problem?".

Runs on CPU (login node), seconds. Needs: tokenizers, transformers, and the megatron repo on
the path (run from the repo root).

USAGE
    cd Megatron-DeepSpeed-X-MoE
    python tools/convert_checkpoint/check_tokenization.py \
        --oscar examples_elmoe/data/oscar-1GB.jsonl \
        --bin   examples_elmoe/data/oscar-deepseek_text_document \
        --tokenizer-json examples_elmoe/data/deepseek_tokenizer/tokenizer.json \
        --hf-model-path ../../models/deepseek-moe-16b-base
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscar", required=True)
    ap.add_argument("--bin", required=True, help="prefix WITHOUT .bin/.idx (Megatron indexed dataset)")
    ap.add_argument("--tokenizer-json", required=True, help="the tokenizer.json used by preprocess_data")
    ap.add_argument("--hf-model-path", required=True, help="for AutoTokenizer (the HF-reference path)")
    ap.add_argument("--n", type=int, default=60, help="how many leading tokens to compare")
    args = ap.parse_args()

    # ---- 1. first oscar document (the raw text) ----
    with open(args.oscar) as f:
        doc0 = json.loads(f.readline())["text"]
    print(f"first oscar doc: {len(doc0)} chars, starts: {doc0[:80]!r}\n")

    # ---- 2. how PREPROCESS tokenized it: raw tokenizers.Tokenizer.encode (== _HFTokenizer) ----
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(args.tokenizer_json)
    raw_ids = raw.encode(doc0).ids
    print(f"[preprocess path]  tokenizers.encode : first {args.n}: {raw_ids[:args.n]}")
    print(f"                   vocab_size={raw.get_vocab_size()}  starts_with_BOS(100000)={raw_ids[0]==100000}\n")

    # ---- 3. how HF-REFERENCE tokenized it: transformers AutoTokenizer ----
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    hf_ids = hf.encode(doc0)
    print(f"[hf-ref path]      AutoTokenizer.encode: first {args.n}: {hf_ids[:args.n]}")
    print(f"                   starts_with_BOS(100000)={hf_ids[0]==100000}\n")

    # ---- 4. what actually landed in the .bin the framework trains on ----
    sys.path.insert(0, os.getcwd())
    from megatron.data.indexed_dataset import make_dataset, infer_dataset_impl
    ds = make_dataset(args.bin, infer_dataset_impl(args.bin))
    bin_doc0 = list(map(int, ds[0][:args.n]))
    print(f"[.bin on disk]     doc[0]             : first {args.n}: {bin_doc0}")
    print(f"                   total docs={len(ds)}  starts_with_BOS(100000)={bin_doc0[0]==100000}\n")

    # ---- verdicts ----
    def head_eq(a, b, k):
        return list(a[:k]) == list(b[:k])

    print("=" * 70)
    print("VERDICTS")
    print(f"  .bin == preprocess(tokenizers)   : {head_eq(bin_doc0, raw_ids, args.n)}"
          "   <- MUST be True (else the .bin was made differently than expected)")
    print(f"  preprocess == hf-ref (AutoTok)   : {head_eq(raw_ids, hf_ids, args.n)}"
          "   <- if False, the two paths tokenize differently (likely BOS/special tokens)")
    if raw_ids[0] != hf_ids[0]:
        print(f"  NOTE: leading token differs: preprocess={raw_ids[0]} vs hf-ref={hf_ids[0]} "
              "-> a per-document BOS difference (boundary effect, NOT a 2.4-nat cause).")
    print()
    print("INTERPRETATION")
    print("  - If '.bin == preprocess' is True and 'preprocess == hf-ref' is True: tokenization")
    print("    is identical; the 2.38-vs-4.8 gap is NOT the data -> it's MoE token dropping")
    print("    (confirm with the --disable-moe-token-dropping relaunch).")
    print("  - If they differ only at the leading BOS: still a boundary effect, not the gap.")
    print("  - If they differ THROUGHOUT: the .bin was tokenized wrong -> re-run preprocess_data.")


if __name__ == "__main__":
    main()
