#!/bin/bash
# ftv2_check_env.sh (ft-sys-v2, 2026-11-06): check every path an ft-sys-v2 run reads, with the same variables the run uses
# (env/ftv2_paths.sh: a variable set in the shell wins, else the default). One line per path:
#   OK | MISSING | NOT READABLE | NOT WRITABLE   <variable>   <path>   <what it is>
# and, after OK, "(private: <folder> ...)" when another user could not read it (a folder on the way lets no one but its owner in):
# the hint for a run by someone else. Exit 1 if any path is MISSING, NOT READABLE or NOT WRITABLE; 0 otherwise (private paths
# are reported, not counted: they are fine for their owner).
#   bash ftv2_check_env.sh              (from any folder; nothing is written, nothing is imported)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_paths.sh"
FT=$FTV2_TREE/examples_elmoe/scripts-frontier/ft-sys-v2
SHARE_GROUP=${FTV2_SHARE_GROUP:-gen150}       # the project group a colleague shares with
bad=0; priv=0

# private <path>: print the first folder on the way (or the path itself) that a user of SHARE_GROUP other than the owner cannot pass
private() {
  local p=$1 cur="" part m g
  local IFS=/
  for part in $p; do
    [ -z "$part" ] && continue
    cur="$cur/$part"
    [ -e "$cur" ] || return 1
    m=$(stat -L -c '%A' "$cur" 2>/dev/null) || return 1; g=$(stat -L -c '%G' "$cur" 2>/dev/null)
    if [ -d "$cur" ]; then
      # others may pass with o+x, or the share group with g+x
      [ "${m:9:1}" = x ] || [ "${m:9:1}" = t ] && continue
      { [ "${m:6:1}" = x ] || [ "${m:6:1}" = s ]; } && [ "$g" = "$SHARE_GROUP" ] && continue
      echo "$cur is $m $(stat -L -c '%U:%G' "$cur")"; return 0
    else
      [ "${m:7:1}" = r ] && continue
      [ "${m:4:1}" = r ] && [ "$g" = "$SHARE_GROUP" ] && continue
      echo "$cur is $m $(stat -L -c '%U:%G' "$cur")"; return 0
    fi
  done
  return 1
}

# absent <path>: why a path does not show: below a folder this user cannot pass (NOT READABLE), or not there (MISSING)
absent() {
  local q; q=$(dirname "$1"); while [ ! -e "$q" ] && [ "$q" != / ]; do q=$(dirname "$q"); done
  if [ -d "$q" ] && [ ! -x "$q" ]; then echo "NOT READABLE"; else echo MISSING; fi
}

# check <kind> <variable> <path> <what>; kind: f (file), x (program), d (folder), w (writable folder, or creatable), p (data prefix)
# SHOW_PRIVATE=1: an OK path also gets the private note (the paths outside the repositories, which another user would share)
SHOW_PRIVATE=0
check() {
  local kind=$1 var=$2 p=$3 what=$4 st=OK note="" q
  case $kind in
    p) for q in "$p.bin" "$p.idx"; do
         if [ -e "$q" ]; then [ -r "$q" ] || st="NOT READABLE"; elif [ "$st" = OK ]; then st=$(absent "$q"); fi
       done ;;
    w) q=$p; while [ ! -e "$q" ] && [ "$q" != / ]; do q=$(dirname "$q"); done
       if [ ! -d "$q" ] || [ ! -w "$q" ] || [ ! -x "$q" ]; then st="NOT WRITABLE"; fi ;;
    *) if [ -e "$p" ]; then
         case $kind in
           f) [ -r "$p" ] || st="NOT READABLE" ;;
           x) [ -r "$p" ] && [ -x "$p" ] || st="NOT READABLE" ;;
           d) [ -r "$p" ] && [ -x "$p" ] || st="NOT READABLE" ;;
         esac
       else st=$(absent "$p"); fi ;;
  esac
  if [ "$st" != OK ]; then
    bad=$((bad + 1))
  elif [ "$SHOW_PRIVATE" = 1 ] && [ "$kind" != w ]; then   # a writable cache is each user's own: private is right there
    local pp=$p; [ "$kind" = p ] && pp="$p.bin"
    if note=$(private "$pp"); then note="  (private: $note)"; priv=$((priv + 1)); else note=""; fi
  fi
  printf '%-13s %-22s %s  -- %s%s\n' "$st" "$var" "$p" "$what" "$note"
}

echo "== ft-sys-v2 run paths ($(date '+%F %T'), user $(id -un), share group $SHARE_GROUP)"
echo "-- this clone"
check d FTV2_TREE "$FTV2_TREE/megatron" "the Megatron tree (megatron/ is imported from here)"
check f FTV2_SF "$FTV2_SF/local_autorun_frontier_0623_2026.sh" "the 0623 launcher"
check f FTV2_SF "$FTV2_SF/frontier_elmoe.slurm_0623_2026.template" "the job template"
check f FTV2_SF "$FTV2_SF/eval/submit_env.sh" "the modules and PATH of the submitting shell"
check f FTV2_SF "$FTV2_SF/cache_setup.sh" "the Triton cache warm-up"
check f FTV2_SF "$FTV2_SF/../ds_config_gpt_TEMPLATE.json" "the DeepSpeed json template"
check f FTV2_SF "$FTV2_SF/../ELMoE_launch.py" "the training entry"
check f FTV2_SF "$FTV2_SF/../utils/model_registry.py" "the model sizes"
check w FTV2_SF "$FTV2_SF/temp_slurm" "job scripts and job folders are written here"
check f FTV2_ENV "$FTV2_ENV" "the new stack's env file"
check f FT "$FT/heartbeat-sys/hb_start.sh" "the heartbeat's start"
check f FT "$FT/recovery/launch/ftv2_srun.sh" "the recovery's srun hook"
echo "-- outside the repositories"
SHOW_PRIVATE=1
check x NEWSTACK_V "$NEWSTACK_V/bin/python" "the Python venv (torch 2.13.0+rocm7.1, apex, flash-attn 2.8.3)"
home=$(sed -n 's/^home *= *//p' "$NEWSTACK_V/pyvenv.cfg" 2>/dev/null)
[ -n "$home" ] && check x NEWSTACK_V "$home/python3" "the venv's base interpreter (pyvenv.cfg home)"
check f NEWSTACK_DEEPSPEED "$NEWSTACK_DEEPSPEED/deepspeed/__init__.py" "the X-MoE DeepSpeed (this branch's root)"
check f NEWSTACK_R714 "$NEWSTACK_R714/lib/librccl.so" "ROCm 7.14.1 gfx90a: RCCL 2.30.4"
check f NEWSTACK_R714 "$NEWSTACK_R714/lib/libamdhip64.so" "ROCm 7.14.1 gfx90a: HIP runtime"
check f NEWSTACK_R714 "$NEWSTACK_R714/lib/libhsa-runtime64.so" "ROCm 7.14.1 gfx90a: HSA runtime"
check f NEWSTACK_PLUGIN "$NEWSTACK_PLUGIN/librccl-net.so" "aws-ofi-nccl 1.22.0 (RCCL's network plugin)"
check f NEWSTACK_LIBFABRIC "$NEWSTACK_LIBFABRIC/libfabric.so" "libfabric 2.3.1 (site)"
check d /opt/rocm-7.1.1 /opt/rocm-7.1.1 "ROCm 7.1.1 for JIT builds (module rocm/7.1.1, site)"
check w NEWSTACK_JIT "$NEWSTACK_JIT" "JIT build caches (torch extensions, Megatron fused kernels, Triton)"
check f NEWSTACK_OVERLAY "$NEWSTACK_OVERLAY/triton_deepbind/sitecustomize.py" "Triton RTLD_DEEPBIND at start-up (a copy: $FT/env/overlay)"
if [ "${NEWSTACK_FA:-ck}" = triton ]; then
  check d NEWSTACK_OVERLAY "$NEWSTACK_OVERLAY/fa_triton_amd/flash_attn" "flash-attn Triton backend (NEWSTACK_FA=triton)"
fi
check x FTV2_ETCD_BIN "$FTV2_ETCD_BIN" "etcd 3.5.17 (heartbeat-sys/get_etcd.sh fetches one)"
pyspy=$(command -v py-spy 2>/dev/null || echo "$HOME/.local/bin/py-spy")
SHOW_PRIVATE=0   # each user's own
check x py-spy "$pyspy" "py-spy for the hang's stacks (PATH, else ~/.local/bin; pip install --user py-spy)"
SHOW_PRIVATE=1
tmpl="$FTV2_SF/frontier_elmoe.slurm_0623_2026.template"
data=${DATA_PATH:-$(sed -n 's/^export DATA_PATH=\${DATA_PATH:-\(.*\)}$/\1/p' "$tmpl" 2>/dev/null)}
check p DATA_PATH "$data" "training data prefix (.bin + .idx)$([ -z "${DATA_PATH:-}" ] && echo ', the template default')"
vocab=${FTV2_VOCAB_FILE:-$(sed -n 's/^export VOCAB_FILE=//p' "$tmpl" 2>/dev/null)}
if [ "${AUTORUN_GPT2_TOKENIZER:-0}" = 1 ]; then   # the launcher's GPT-2 rider replaces the template's tokenizer (2026-11-13)
  check f AUTORUN_GPT2_TOKENIZER "$FTV2_SF/../data/gpt2-vocab.json" "GPT-2 vocabulary (the launcher's rider reads ../data/gpt2-vocab.json)"
  check f AUTORUN_GPT2_TOKENIZER "$FTV2_SF/../data/gpt2-merges.txt" "GPT-2 merges"
else
  check f FTV2_VOCAB_FILE "$vocab" "tokenizer$([ -z "$FTV2_VOCAB_FILE" ] && echo ', the template VOCAB_FILE')"
fi
if [ -d "$NEWSTACK_MEGATRON_COPY" ] && [ -x "$NEWSTACK_MEGATRON_COPY" ]; then st=present; else st=absent; fi
printf '%-13s %-22s %s  -- %s\n' "unused" NEWSTACK_MEGATRON_COPY "$NEWSTACK_MEGATRON_COPY" "$st; on PYTHONPATH after FTV2_TREE, so megatron is not imported from it"

echo "-- where the job's python finds deepspeed and megatron (the job's PYTHONPATH order; find_spec only, nothing imported)"
if [ -x "$NEWSTACK_V/bin/python" ]; then
  pp="$FTV2_TREE:$NEWSTACK_MEGATRON_COPY:$NEWSTACK_OVERLAY/triton_deepbind:$NEWSTACK_DEEPSPEED"
  for mod in deepspeed:NEWSTACK_DEEPSPEED:$NEWSTACK_DEEPSPEED megatron:FTV2_TREE:$FTV2_TREE; do
    IFS=: read -r name var want <<< "$mod"
    got=$(FTV2_CHECK_PP="$pp" "$NEWSTACK_V/bin/python" -I -B -c "
import os, sys, importlib.util
sys.path[:0] = [p for p in os.environ['FTV2_CHECK_PP'].split(':') if p]
try:
    print(importlib.util.find_spec('$name').origin)
except Exception as e:
    print('ERROR', repr(e))" 2>&1 | tail -1)
    if [[ "$got" == "$want/"* ]]; then st=OK; else st=WRONG; bad=$((bad + 1)); fi
    printf '%-13s %-22s %s  -- import %s\n' "$st" "$var" "$got" "$name"
  done
else
  echo "SKIPPED       (no python at $NEWSTACK_V/bin/python)"
fi

echo "== $bad problem(s); $priv path(s) private to their owner (another user sets the variable to a copy of their own)"
[ "$bad" -eq 0 ]
