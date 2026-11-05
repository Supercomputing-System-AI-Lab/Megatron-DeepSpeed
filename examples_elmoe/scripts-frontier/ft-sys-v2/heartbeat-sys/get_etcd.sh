#!/bin/bash
# get_etcd.sh (ft-sys-v2 heartbeat system, 2026-11-05): fetch the etcd release the heartbeat system was tested with, check its
# SHA-256 against the release's own SHA256SUMS, and unpack it. The binary is not kept in the repository (about 56 MB, and a
# downloaded program should come from its publisher with its checksum); the agents find it through FTV2_ETCD_BIN.
#   bash get_etcd.sh [<folder>]     default folder: ./_tools ; prints the line to export
set -euo pipefail
V=${ETCD_VERSION:-v3.5.17}
DIR=${1:-$(pwd)/_tools}/etcd-$V
mkdir -p "$DIR" && cd "$DIR"
URL=https://github.com/etcd-io/etcd/releases/download/$V
curl -sSL -o etcd-$V-linux-amd64.tar.gz $URL/etcd-$V-linux-amd64.tar.gz
curl -sSL -o SHA256SUMS $URL/SHA256SUMS
grep "etcd-$V-linux-amd64.tar.gz" SHA256SUMS | sha256sum -c -
tar -xzf etcd-$V-linux-amd64.tar.gz
"$DIR/etcd-$V-linux-amd64/etcd" --version | head -1
echo "export FTV2_ETCD_BIN=$DIR/etcd-$V-linux-amd64/etcd"
