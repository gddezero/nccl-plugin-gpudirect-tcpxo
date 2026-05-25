# nccl-plugin-gpudirect-tcpxo (EPv2 GIN, from-scratch)

A minimal, single-file **NCCL GIN PROXY** plugin that lets [DeepEP_v2](https://github.com/deepseek-ai/DeepEP)'s `tests/elastic/` suite pass end-to-end across **two H100 `a3-megagpu-8g`** nodes on GCP, using TCP over a GPUDirect NIC (`eth1`) plus **[GDRCopy](https://github.com/NVIDIA/gdrcopy)** as the data plane.

- ABI: NCCL plugin v11 + v13 (`NCCL_NET_DEVICE_GIN_PROXY`), tested against NCCL **2.30.4-1**.
- ~1.5 K lines of C++ in a single file (`plugin/src/tcpxo_gin_plugin.cc`).
- Built from scratch in roughly one engineering day; no code reused from earlier PR#521 / EPv1 attempts.

> **Goal of this repo:** prove that the DeepEP_v2 elastic test suite can run unmodified on top of NCCL 2.30.4's GIN PROXY interface on a3-mega + TCPXO, and document the non-obvious gotchas (especially the `cudaMemcpyAsync` deadlock under busy-wait kernels) for anyone attempting the same.

For a deep-dive on architecture, wire protocol, debugging timeline, and what was deliberately deferred, read **[RESULTS.md](RESULTS.md)**. The full plugin contract is in **[SPEC_M0_epv2_gin_contract.md](SPEC_M0_epv2_gin_contract.md)**.

---

## Quickstart

```bash
# 1. Build (on either node, inside a container that has CUDA 12.x + GDRCopy headers/lib)
cd plugin && make clean && make
# → plugin/build_deploy/libnccl-net-tcpxo-gin.so

# 2. Deploy the .so to the other node (use scp, NOT cat | ssh — cat-pipe truncates ELFs)
scp build_deploy/libnccl-net-tcpxo-gin.so peer:/work/epv2_gin_plugin/build_deploy/

# 3. Run a test (e.g. test_pp) — on each node:
cd plugin/tests
RANK=<0|1> WORLD_SIZE=2 MASTER_ADDR=<rank0 eth1 IP> MASTER_PORT=29550 N_LOCAL=1 \
  bash run_test_pp_d1.sh
```

Prerequisites on each node:
- `a3-megagpu-8g` (or any 8× H100 SXM box) with TCPXO GPUDirect NICs (`eth1`..`eth8`)
- `gdrdrv` kernel module loaded; `/dev/gdrdrv` present
- `libgdrapi.so` reachable (`ldconfig -p | grep libgdrapi`)
- DeepEP_v2 installed in the container under test (`tests/elastic/test_*.py` will be invoked)
- NCCL 2.30.4-1 — either stock, or the version patched by [`d1_nccl_patch.patch`](d1_nccl_patch.patch) if you want `[D1]` PROXY-chain logging

---

## Status — DeepEP_v2 `tests/elastic/`

All measurements on **2 × H100 `a3-megagpu-8g`** (us-central1), 1 rank per node, NCCL 2.30.4-1, plugin commit [`2b9b15a`](../../commit/2b9b15a), data plane = TCP-over-`eth1` (single GPUDirect NIC) + GDRCopy host BAR1 alias for H↔D copies.

| Test | Result | One-liner |
|---|---|---|
| `test_barrier.py` | ✅ pass | 1000 barriers, avg **75.6 µs** per barrier |
| `test_pp.py`      | ✅ pass | 6/6 Profiling sub-iters (hide × concurrent = 2 × 3), 374 data ops |
| `test_engram.py`  | ✅ pass | 6.7 M plugin events, requires `OP_GET` wire op |
| `test_ep.py`      | ✅ pass | 445 K dispatch/combine ops (`--skip-perf-test`) |
| `test_agrs.py`    | ⊘ N/A   | DeepEP asserts `nvl_ranks == num_ranks` — intra-node only, not satisfiable with 2 nodes × 1 rank |

---

## Two-node performance numbers

### `test_pp` (16 KB tensors, 2 nodes × 1 rank, eth1, single TCP stream)

Latencies are reported per direction. `send` is the enqueue cost (kernel returns once the GFD is queued); `recv` is the actual end-to-end round-trip and is the number you should treat as throughput-meaningful.

| `hide_rdma_latency` | `concurrent` | send µs | send GB/s | recv µs | recv GB/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.21 | 3.56 | 111.4 | 0.29 |
| 1 | 2 | 8.92 | 3.67 |  77.5 | 0.42 |
| 1 | 3 | 8.71 | 3.76 |  78.5 | 0.42 |
| 0 | 1 | 9.25 | 3.54 | 107.1 | 0.15 |
| 0 | 2 | 8.89 | 3.69 |  86.6 | 0.19 |
| 0 | 3 | 8.72 | 3.76 |  81.8 | 0.20 |

**Peak observed:** ~**0.42 GB/s** real bidirectional throughput at `concurrent=2..3` with `hide_rdma_latency=1`.

### `test_pp` tensor-size sweep (hide=1, concurrent=3, per-size peak row)

Same hardware, same plugin, only `TOKENS × HIDDEN` (bf16) changes.

| message size | send µs | send GB/s | recv µs | **recv GB/s** |
|---:|---:|---:|---:|---:|
| 4 KB    | 8.92  |   0.92 |   40.66 | **0.20** |
| 16 KB   | 9.07  |   3.61 |   72.38 | **0.45** |
| 64 KB   | 9.01  |  14.55 |  194.64 | **0.67** |
| 256 KB  | 9.15  |  57.33 |  746.57 | **0.70** |
| 1 MB    | 9.70  | 216.13 | 1895.00 | **1.11** |

Small messages are dominated by a ~35–50 µs per-op overhead (GDRCopy + signal RMW + TCP syscall + recv\_thread wakeup); once that amortises, the single-NIC / single-stream TCP fabric cap takes over at **~1.1 GB/s**.

### `test_barrier` (2 nodes × 1 rank)

| Metric | Value |
|---|---|
| barriers executed | 1000 |
| avg barrier latency | **75.6 µs** |
| plugin signal-only ops on the wire | 2 × 1000 |

### `test_engram` (2 nodes × 1 rank)

| Metric | Value |
|---|---|
| plugin events processed | **6.7 M** |
| wire ops exercised | `OP_PUT`, `OP_PUT_SIGNAL`, `OP_GET` (with `req_id` matching) |
| outcome | torch.equal validation passes on every iter |

### `test_ep` (2 nodes × 1 rank, `--skip-perf-test`)

| Metric | Value |
|---|---|
| dispatch + combine ops | **445 K** |
| outcome | correctness pass; perf timing skipped by flag |

> **Why these throughput numbers are modest:** the data plane is a single TCP stream over one GPUDirect NIC (`eth1`), with a single-threaded `recv_thread` in the plugin. The a3-mega fabric can do ~**227 GB/s busbw** end-to-end via `libfastrak` (the TCPXO GPU-direct substrate used by production NCCL builds) — reaching that would require a multi-week rewrite of the data plane. **That was explicitly out of scope for this milestone**, which is about functional correctness of the GIN ABI surface. See `RESULTS.md` → "Bottleneck attribution" and "What this plugin is and is not".

---

## Repository layout

```
.
├── README.md                          ← this file
├── RESULTS.md                         ← detailed write-up: architecture, wire protocol, debug timeline
├── SPEC_M0_epv2_gin_contract.md       ← plugin contract / NCCL ABI / milestones
├── LICENSE                            ← Apache 2.0
├── NOTICE                             ← third-party attributions (NCCL, GDRCopy, CUDA, DeepEP)
├── d1_nccl_patch.patch                ← debug instrumentation for NCCL 2.30.4-1 GIN PROXY chain
└── plugin/
    ├── Makefile
    ├── include/nccl_plugin_abi/       ← NCCL plugin headers, vendored from 2.30.4-1
    ├── src/tcpxo_gin_plugin.cc        ← the plugin (~1.5K lines)
    └── tests/
        ├── gdr_smoke.c                ← GDRCopy correctness check
        ├── gdr_busy_stress.cu         ← GDRCopy latency under a busy CUDA kernel
        ├── dlopen_smoke.c             ← raw dlopen + dlsym of plugin exports
        ├── nccl_load_smoke.c          ← NCCL-side plugin load smoke (1 rank)
        ├── m1_e2e_elastic_init.py     ← M1 dual-node ElasticBuffer init
        ├── d1_analyze.py              ← parse [D1] markers from NCCL logs
        └── run_test_{barrier,pp,engram,agrs,ep}_d1.sh
```

---

## Key design choices (one-liner each)

1. **GDRCopy on the data plane.** `cudaMemcpyAsync` deadlocks for ~100 s when a user kernel is busy-waiting on signal memory, even with `cudaStreamNonBlocking` and `CUDA_DEVICE_MAX_CONNECTIONS=32`. Pinning each MR with `gdr_pin_buffer` + `gdr_map` and copying via the host BAR1 alias completes in ~8 µs regardless of GPU contention. H100 BAR1 = 128 GB.
2. **TCP over `eth1`** (a GPUDirect NIC, not the management NIC). One stream per peer, syscall-driven. Throughput is capped by this choice; correctness is not.
3. **Single 68-byte `WireMsgHdr`** for all three ops (`OP_PUT`, `OP_PUT_SIGNAL`, `OP_GET`). `OP_GET` is implemented by sending a header-only request and matching the reply by `req_id` in a `pending_gets` map.
4. **Single-file plugin, no external state.** All maps (`mrs[]`, `pending_gets`, peer connections) live inside the plugin context. Easier to read, audit, and extend than splitting across files.

---

## Licensing & attribution

- This repo: **Apache License 2.0** (see [LICENSE](LICENSE)).
- NCCL plugin ABI headers under `plugin/include/nccl_plugin_abi/` are vendored verbatim from NCCL 2.30.4-1 (Apache 2.0).
- GDRCopy and the CUDA Runtime are linked at runtime, not vendored.
- DeepEP_v2 is **not** vendored; tests invoke it from an external installation.

See [NOTICE](NOTICE) for full third-party attribution.
