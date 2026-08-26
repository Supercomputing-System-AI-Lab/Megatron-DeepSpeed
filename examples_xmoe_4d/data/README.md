### instructions of creating a sample dataset

```bash 
$ wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
$ xz -d oscar-1GB.jsonl.xz
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
$ python ../../tools/preprocess_data.py \                                                                                
    --input oscar-1GB.jsonl \
    --output-prefix my-xmoe \
    --vocab-file gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --workers 8
```