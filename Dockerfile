# syntax=docker/dockerfile:1.7
FROM rust:1.88-bookworm AS build

WORKDIR /workspace
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src

RUN cargo build --release --locked

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system dcgaimodel \
    && useradd --system --gid dcgaimodel --no-create-home dcgaimodel

WORKDIR /app

COPY --from=build /workspace/target/release/dcgaimodel /usr/local/bin/dcgaimodel
COPY data/inference/frozen-v9/policy-packs-v5-compositional.json /app/data/inference/frozen-v9/policy-packs-v5-compositional.json
COPY data/experiments/v9-multiclass-cpu-100e/models/seed-20260826-normal-family-split.json /app/data/experiments/v9-multiclass-cpu-100e/models/seed-20260826-normal-family-split.json
COPY data/experiments/v9-multiclass-cpu-100e/models/seed-20260827-normal-family-split.json /app/data/experiments/v9-multiclass-cpu-100e/models/seed-20260827-normal-family-split.json
COPY data/experiments/v9-multiclass-cpu-100e/models/seed-20260828-normal-family-split.json /app/data/experiments/v9-multiclass-cpu-100e/models/seed-20260828-normal-family-split.json

RUN chown -R dcgaimodel:dcgaimodel /app /usr/local/bin/dcgaimodel

USER dcgaimodel
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/dcgaimodel", "serve-shadow-inference", "--artifact-root", "/app", "--bind", "0.0.0.0:8080"]
