# Slim runtime image for the C++ engine (no Python, no libtorch).
#
#   docker build -t quant-engine .
#
# Backtest (needs no network):
#   docker run --rm --network=none --cpus=2 --memory=1g --user "$(id -u):$(id -g)" \
#     -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out" \
#     quant-engine
#
# Live paper trading on Alpaca data (runs until stopped; sleeps while the market is closed;
# `docker stop` flattens the paper positions and prints the report):
#   docker run -d --name quant-live --restart unless-stopped --cpus=2 --memory=1g \
#     --user "$(id -u):$(id -g)" -e APCA_API_KEY_ID -e APCA_API_SECRET_KEY \
#     -v "$PWD/models:/app/models:ro" -v "$PWD/out:/app/out" \
#     quant-engine --live --model /app/models/ensemble_model.weights --trades /app/out/live_trades.csv
#   docker logs -f quant-live
# If the build fails with "failed to add
# the host (veth...) <=> sandbox (veth...) pair interfaces: operation not supported", the
# host kernel cannot create Docker's virtual network interfaces (on Arch: the kernel was
# upgraded without a reboot). Reboot, or build with `docker build --network=host ...`.
#
# Models come from python/train_and_export.py (run outside the container). Any engine flag
# can be appended, e.g. `quant-engine --data /app/data/alpaca_ticks.csv --period val`;
# `--entrypoint model_runner` runs the offline evaluator instead.

# trixie: its libcurl (8.14) supports WebSockets, which the live feed needs
FROM debian:trixie-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake make libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
COPY tests ./tests
# x86-64-v3 (AVX2 + FMA) runs on any current x86 cloud CPU; -march=native would tie the
# binary to the build machine and can crash with "illegal instruction" on the VM.
ARG QUANT_ARCH=x86-64-v3
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQUANT_ARCH=${QUANT_ARCH} \
 && cmake --build build -j"$(nproc)" \
 && ./build/engine_tests tests/fixtures \
 && ./build/quant_engine --help | grep -q -- --live \
 && strip build/quant_engine build/model_runner

FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends libcurl4t64 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/quant_engine /src/build/model_runner /usr/local/bin/
WORKDIR /app
USER 65534:65534
ENTRYPOINT ["quant_engine"]
CMD ["--data", "/app/data/market_ticks.csv", "--model", "/app/models/ensemble_model.weights", "--trades", "/app/out/trades.csv"]
