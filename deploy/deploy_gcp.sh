#!/usr/bin/env bash
# Builds the engine image on this machine, uploads it with the trained model to a GCP VM and
# (re)starts live paper trading there. Run from the repository root after training:
#
#   deploy/deploy_gcp.sh VM ZONE [MODEL] [-- extra engine flags]
#   deploy/deploy_gcp.sh quant-vm us-east1-b ensemble_model -- --online-lr 0.001
#
# MODEL is the name under models/ (default ensemble_model); use what compare_methods.py
# selected, and its gamma / online learning rate as extra flags. MODEL_ONLY=1 skips the image
# (after retraining, when the C++ code has not changed).
#
# Needs here: docker and the gcloud CLI (gcloud auth login; gcloud config set project ID).
# Needs on the VM, once: docker and ~/quant/alpaca.env (see README, "Deploying to a GCP VM").
# Your keys are never uploaded by this script; the image holds no keys, data or models.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,14p' "$0"; exit 1; }
VM=$1 ZONE=$2
shift 2
MODEL=ensemble_model
if [ $# -gt 0 ] && [ "$1" != "--" ]; then MODEL=$1; shift; fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
EXTRA=""
[ $# -gt 0 ] && EXTRA=$(printf '%q ' "$@")

die() { echo "[-] $*" >&2; exit 1; }
[ -f Dockerfile ] && [ -d src ] || die "run from the repository root"
[ -f "models/$MODEL.weights" ] && [ -f "models/${MODEL}_config.txt" ] || die "models/$MODEL.weights or its _config.txt is missing (train first)"
command -v gcloud >/dev/null || die "install the gcloud CLI: https://cloud.google.com/sdk/docs/install"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FILES=("$TMP/model.tar.gz" deploy/run_live.sh)
tar czf "$TMP/model.tar.gz" -C models "$MODEL.weights" "${MODEL}_config.txt"
if [ "${MODEL_ONLY:-0}" != 1 ]; then
    echo "[+] Building the image (compiles and runs the engine tests)"
    docker build -t quant-engine:latest .
    docker save quant-engine:latest | gzip > "$TMP/quant-engine.tar.gz"
    FILES+=("$TMP/quant-engine.tar.gz")
fi
echo "[+] Uploading $(du -ch "${FILES[@]}" | tail -1 | cut -f1) to $VM"
SSH=(gcloud compute ssh "$VM" --zone "$ZONE" --command)
"${SSH[@]}" "mkdir -p ~/quant/models ~/quant/out"
gcloud compute scp --zone "$ZONE" "${FILES[@]}" "$VM:~/quant/"
"${SSH[@]}" "set -e; cd ~/quant
    if [ -f quant-engine.tar.gz ]; then gunzip -c quant-engine.tar.gz | docker load; rm quant-engine.tar.gz; fi
    tar xzf model.tar.gz -C models && rm model.tar.gz
    chmod +x run_live.sh && ./run_live.sh $MODEL $EXTRA"
echo "[+] Done. Watch it with: gcloud compute ssh $VM --zone $ZONE --command 'docker logs -f --tail 60 quant-live'"
