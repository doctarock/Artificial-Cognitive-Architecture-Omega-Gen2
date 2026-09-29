# Artificial Cognitive Architecture (ACA)

ACA is an experimental cognitive-architecture prototype. It has now been run through practical testing and passes many mechanical checks, including focused regression tests around its loop, memory, tier routing, and store behavior.

This repository is being made public for educational purposes: to show the shape of the experiment, the engineering tradeoffs, and the lessons learned while building it.

## Current Status

This build should be read as a tested research prototype, not a finished production system. While many mechanical checks pass, some real-world systems require more architectural complexity than this version currently carries. That additional complexity is being addressed in a new build.

In particular, expect rough edges around scale, long-running operation, model orchestration, external service integration, and operational hardening. The source is useful for study, comparison, and experimentation, but it should not be treated as a drop-in autonomous agent platform.

## Running Locally

With Rust installed:

```powershell
cargo test
cargo run -p omega-acad
```

The daemon can run with no model endpoints configured. In that mode it creates a fresh local database and uses fallback behavior where possible, which is useful for inspecting startup and the API surface before wiring in local or hosted model services.

Optional integrations, such as voice, video, Ollama-compatible endpoints, OpenAI-compatible endpoints, and the Godot visualization, are configured through environment variables.

# Artificial-Cognitive-Architecture-Omega-Gen2

A different kind of LLM, evolved.

## Omega Cognitive Cluster Skeleton

Milestone 1 lives in `crates/omega-cluster`. It provides the v0.1 cognitive
protocol, OCL operation vocabulary, shared cognitive state, HTTP processor
registry/client, mock executive/attention/memory services, and a runtime demo
that drives a cognitive event to `WAIT`.

Run the skeleton demo:

```powershell
cargo run -p omega-cluster --bin omega-cluster-demo
```

Run the persistent runtime workbench:

```powershell
cargo run -p omega-cluster --bin omega-runtime -- "new observation for Omega"
```

That command writes inspectable state and trace files under `dist/omega-cluster`.

Inspect the trace:

```powershell
cargo run -p omega-cluster --bin omega-trace
```

Check configured processor health/capabilities:

```powershell
cargo run -p omega-cluster --bin omega-cluster-check
```

Run the focused tests:

```powershell
cargo test -p omega-cluster
```

Run one mock processor as an independent service:

```powershell
$env:OMEGA_PROCESSOR="attention"
$env:OMEGA_PROCESSOR_PORT="9102"
cargo run -p omega-cluster --bin omega-mock-processor
```

Run the whole mock cluster as long-lived services:

```powershell
cargo run -p omega-cluster --bin omega-mock-cluster
```

Milestone 2 research notes are in `research/integration-audit.md`.

Downloaded research model/data artifacts are documented in `models/README.md`
and validated in `models/research-artifacts.json`.

The event-driven cognitive substrate and its current implementation boundaries
are documented in `docs/event-driven-cognitive-substrate.md`.
