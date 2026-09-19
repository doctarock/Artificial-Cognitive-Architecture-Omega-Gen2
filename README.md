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
