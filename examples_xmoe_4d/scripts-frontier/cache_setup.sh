#!/bin/bash
PERSISTENT_CACHE=$1
TMP_TRITON_DIR=$2
mkdir -p ${TMP_TRITON_DIR}
echo "[cache_setup] $(hostname): copying 371MB cache to /tmp..."
cp -r ${PERSISTENT_CACHE}/. ${TMP_TRITON_DIR}/
echo "[cache_setup] $(hostname): done at $(date '+%F %T')"