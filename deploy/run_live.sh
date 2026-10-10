#!/usr/bin/env bash
# (Re)starts live paper trading on this machine (the GCP VM) in the background. It keeps
# running until stopped: it sleeps while the market is closed and Docker restarts it after a
# crash or a reboot. deploy/deploy_gcp.sh copies this file to ~/quant/ on the VM and runs it.
#
#   ~/quant/run_live.sh [MODEL] [extra engine flags]
#   ~/quant/run_live.sh ensemble_model --online-lr 0.001
#
# Needs ~/quant/alpaca.env with your Alpaca *paper* keys (never copied anywhere else):
#   APCA_API_KEY_ID=PK...
#   APCA_API_SECRET_KEY=...
#
# Watch:  docker logs -f --tail 60 quant-live   (a status summary every 5 minutes)
# Stop:   docker stop -t 30 quant-live          (flattens if the market is open, prints the report)
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${1:-ensemble_model}
[ $# -gt 0 ] && shift

die() { echo "[-] $*" >&2; exit 1; }
[ -f alpaca.env ] || die "create $PWD/alpaca.env with APCA_API_KEY_ID=... and APCA_API_SECRET_KEY=... (your paper keys)"
chmod 600 alpaca.env
[ -f "models/$MODEL.weights" ] && [ -f "models/${MODEL}_config.txt" ] || die "models/$MODEL.weights and models/${MODEL}_config.txt are missing"
grep -qw avx2 /proc/cpuinfo || die "this CPU has no AVX2; build the image with --build-arg QUANT_ARCH=x86-64-v2"
docker image inspect quant-engine:latest >/dev/null 2>&1 || die "no quant-engine image; run deploy/deploy_gcp.sh from your machine"
systemctl is-enabled docker >/dev/null 2>&1 || echo "[!] docker does not start at boot: sudo systemctl enable docker"
mkdir -p out

ENGINE=(--model "/app/models/$MODEL.weights")
MOUNTS=(-v "$PWD/models:/app/models:ro" -v "$PWD/out:/app/out")

echo "[+] Checking the keys and connections"
docker run --rm --env-file alpaca.env --user "$(id -u):$(id -g)" "${MOUNTS[@]}" quant-engine "${ENGINE[@]}" --live-check 5 "$@" ||
    die "live check failed (keys in alpaca.env? network?)"

# Graceful stop of the previous run: flattens if the market is open, saves online learning
docker stop -t 30 quant-live >/dev/null 2>&1 || true
docker rm quant-live >/dev/null 2>&1 || true

docker run -d --name quant-live --restart unless-stopped --cpus=2 --memory=512m \
    --log-opt max-size=20m --log-opt max-file=5 \
    --user "$(id -u):$(id -g)" --env-file alpaca.env "${MOUNTS[@]}" \
    quant-engine --live --paper-orders "${ENGINE[@]}" --trades /app/out/live_trades.csv \
    --online-state /app/out/online_state.weights "$@" >/dev/null

echo "[+] quant-live started with models/$MODEL.weights $*"
echo "    watch: docker logs -f --tail 60 quant-live"
sleep 3
docker logs --tail 20 quant-live
