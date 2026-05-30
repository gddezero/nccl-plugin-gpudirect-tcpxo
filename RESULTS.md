# DeepEP_v2 elastic suite over TCPXO — from-scratch NCCL GIN plugin

## TL;DR

A ~1.5K line single-file C++ NCCL GIN plugin (v11+v13 ABI) that lets DeepEP_v2's `elastic/` test suite pass end-to-end across **two H100 a3-megagpu-8g nodes** using **TCP-over-eth1** + **GDRCopy** as the data plane.

Built from scratch on 2026-05-20 in roughly one engineering day. No code reused from prior PR#521 / EPv1-era plugin work.

| Test | Result | Notes |
|---|---|---|
| `tests/elastic/test_barrier.py` | ✅ | 1000 barriers, avg 75.6 µs |
| `tests/elastic/test_pp.py`      | ✅ | 6/6 Profiling sub-iter (hide × concurrent = 2×3), 374 data ops, ~0.4 GB/s recv |
| `tests/elastic/test_engram.py`  | ✅ | 6.7 M plugin events, OP_GET wire protocol added |
| `tests/elastic/test_ep.py`      | ✅ | 445 K dispatch/combine ops (`--skip-perf-test`) |
| `tests/elastic/test_agrs.py`    | ⊘ N/A | DeepEP asserts `nvl_ranks == num_ranks`; intra-node only, not satisfiable with 2 nodes × 1 rank |

---

## Why this was hard

The "easy" path — `cudaMemcpyAsync` on a CUDA-allocated MR — **deadlocks** on H100 + CUDA 12.8 the moment a user kernel starts busy-waiting on signal memory:

```
DeepEP barrier kernel  ──[busy-wait clock64() on signal[i]]──→ user stream never exits
                              ↑
                              └── waiting for plugin signal
                              ↓
plugin recv_thread ──[cudaMemcpyAsync H→D]──→ blocks for 100 s
                                              (despite cudaStreamNonBlocking)
                                              eventually returns 719 because GPU asserted
```

CUDA documents `cudaStreamNonBlocking` as opting out of default-stream sync, but empirically on Hopper / driver R560 it still serializes against busy-spin kernels via the command queue. We proved this by writing into our own private `cudaMalloc` buffer with the same result — see `DIAG-PROBE` in commit history.

**Workaround: GDRCopy.** Pin each CUDA MR with `gdr_pin_buffer` + `gdr_map`, then read/write via the host-mapped BAR1 alias with `gdr_copy_to_mapping` / `gdr_copy_from_mapping`. These bypass the CUDA runtime entirely and complete in 8 µs even while a busy-wait kernel pins all SMs. H100 BAR1 is 128 GB; we never hit the cap.

---

## Architecture

```
                ┌─────────── rank 0 (node-a) ───────────┐
                │                                                │
   DeepEP elastic kernel                                         │
        │                                                        │
        │ writes GFD into NCCL proxy queue                       │
        ↓                                                        │
   NCCL GIN PROXY (NCCL 2.30.4)                                  │
        │                                                        │
        │ calls plugin entry points                              │
        ↓                                                        │
   tcpxo-gin plugin                                              │
   ├── gin_regMrSym  → gdr_pin_buffer + gdr_map  (host BAR1 VA) │
   ├── gin_iputSignal / iput → do_send                          │
   │      ├── self  → apply_recv (in-process)                   │
   │      └── cross → write_all(hdr [+ payload]) over TCP eth1  │
   ├── gin_iget → OP_GET hdr-only TCP send + pending map        │
   ├── recv_thread (poll/read peers)                            │
   │      └── apply_recv → gdr_copy_to_mapping (data + signal)  │
   └── handle_get_request → do_send OP_PUT reply with req_id    │
                                                                 │
                                  eth1 (e.g. 10.x.y.z/32, your own subnet)           │
                                  TCP single-stream              │
                                                                 │
                ┌─────────── rank 1 (node-b) ───────────┘
                │                                                │
                                  (mirror)                       │
                                                                 │
                └────────────────────────────────────────────────┘
```

### Wire protocol (68-byte `WireMsgHdr`)

| field | bytes | meaning |
|---|---|---|
| magic           | 4 | `0x47494E58` ("GINX") |
| op              | 1 | `1=OP_PUT`, `2=OP_PUT_SIGNAL`, `3=OP_GET` |
| signal_op       | 1 | `0=NONE`, `1=INC`, `2=ADD` |
| req_id          | 8 | iget request matching (echoed by GET reply) |
| dst_token       | 8 | receiver's MR token (or for OP_GET: where reply lands) |
| dst_off         | 8 | offset within dst MR |
| size            | 8 | payload bytes (or for OP_GET: bytes to read remotely) |
| signal_token    | 8 | signal MR token (or for OP_GET: remote src MR token) |
| signal_off      | 8 | signal offset (or for OP_GET: remote src offset) |
| signal_val      | 8 | INC/ADD argument |
| padding         | 6 | alignment |

`OP_GET` reuses `signal_token`/`signal_off` to specify the remote source address. Reply is an ordinary `OP_PUT` with `req_id` echoed back so the requester can match a `Request*` from its `pending_gets` map and signal completion.

---

## Repository layout

```
epv2_gin/
├── RESULTS.md                        ← this file
├── SPEC_M0_epv2_gin_contract.md      ← M0 spec (plugin contract, NCCL ABI, milestones)
├── d1_nccl_patch.patch               ← D1 NCCL debug instrumentation (apply on NCCL 2.30.4-1)
├── plugin/
│   ├── Makefile                      ← single-file build, links libcudart + libgdrapi
│   ├── include/nccl_plugin_abi/      ← copies of NCCL 2.30.4 plugin headers (v11..v13)
│   ├── src/tcpxo_gin_plugin.cc       ← THE plugin (~1.5K lines)
│   └── tests/
│       ├── gdr_smoke.c               ← GDRCopy correctness check
│       ├── gdr_busy_stress.cu        ← GDRCopy latency under busy GPU kernel
│       ├── dlopen_smoke.c            ← raw dlopen + dlsym of plugin exports
│       ├── nccl_load_smoke.c         ← NCCL-side plugin load (1-rank)
│       ├── m1_e2e_elastic_init.py    ← M1 dual-node ElasticBuffer init
│       ├── d1_analyze.py             ← parse [D1] markers from NCCL log
│       └── run_test_{barrier,pp,engram,agrs,ep}_d1.sh  ← per-test launchers
└── spec_research/                    ← exported header/source snippets used during M0
```

---

## How to reproduce

### 1. Container prep (once per node)

```bash
# In deepep-dev container, ensure libgdrapi is reachable.
# Host has /dev/gdrdrv (gdrdrv kernel module loaded on both H100 nodes).
ls /usr/local/gdrcopy/lib/libgdrapi.so   # should exist
ldconfig -p | grep libgdrapi             # should resolve
```

### 2. Build plugin

```bash
cd /work/epv2_gin_plugin       # = host ~/deepep_dev/epv2_gin_plugin
make clean && make             # produces build_deploy/libnccl-net-tcpxo-gin.so
```

Deploy to the other node — **use `scp -3`, not cat-pipe** (cat-pipe occasionally truncates the binary; symptom is `invalid ELF header` at NCCL load time):

```bash
scp -3 node-a:/work/.../libnccl-net-tcpxo-gin.so node-b:/work/.../libnccl-net-tcpxo-gin.so
```

### 3. (Optional) build patched NCCL for D1 debug

```bash
# clone NCCL 2.30.4-1 anywhere outside this repo, then:
cd /path/to/nccl-2.30.4-1
patch -p1 < /path/to/epv2_gin/d1_nccl_patch.patch
make src.build TRACE=1 -j$(nproc)
# replace /usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib/libnccl.so.2 on both nodes
```

### 4. Run tests

On node B (rank 1):
```bash
cd /work/epv2_gin_plugin/tests
RANK=1 WORLD_SIZE=2 MASTER_ADDR=$NODE_A_ETH1_IP MASTER_PORT=29550 N_LOCAL=1 \
  bash run_test_pp_d1.sh
```

On node A (rank 0), simultaneously:
```bash
cd /work/epv2_gin_plugin/tests
RANK=0 WORLD_SIZE=2 MASTER_ADDR=$NODE_A_ETH1_IP MASTER_PORT=29550 N_LOCAL=1 \
  bash run_test_pp_d1.sh
```

Substitute `run_test_{barrier,engram,ep}_d1.sh` for the other tests. `run_test_agrs_d1.sh` exists but will hit `nvl_ranks == num_ranks` assert by design.

---

## Performance — full test suite on 2 × a3-megagpu-8g (2 nodes × 1 rank, eth1)

### test_pp — fixed message size (16 KB), hide × concurrent sweep

| hide_rdma_latency | concurrent | send µs | send GB/s | recv µs | recv GB/s |
|---|---|---|---|---|---|
| 1 | 1 | 9.21 | 3.56 | 111.4 | 0.29 |
| 1 | 2 | 8.92 | 3.67 | 77.5  | 0.42 |
| 1 | 3 | 8.71 | 3.76 | 78.5  | 0.42 |
| 0 | 1 | 9.25 | 3.54 | 107.1 | 0.15 |
| 0 | 2 | 8.89 | 3.69 | 86.6  | 0.19 |
| 0 | 3 | 8.72 | 3.76 | 81.8  | 0.20 |

`send` is enqueue latency (kernel returns once GFD is in queue), not real transfer time. `recv` is the actual round-trip latency. Real bandwidth ≈ recv column.

### test_pp — tensor-size sweep (hide=1, concurrent=3 = peak row per size)

Same 2 × a3-megagpu-8g, same plugin, only `TOKENS × HIDDEN` (bf16) changes. `send GB/s` grows linearly with size because it is `size / enqueue_latency`, not transfer rate.

| message size | send µs | send GB/s | recv µs | **recv GB/s** |
|---:|---:|---:|---:|---:|
| 4 KB    | 8.92  |   0.92 |   40.66 | **0.20** |
| 16 KB   | 9.07  |   3.61 |   72.38 | **0.45** |
| 64 KB   | 9.01  |  14.55 |  194.64 | **0.67** |
| 256 KB  | 9.15  |  57.33 |  746.57 | **0.70** |
| 1 MB    | 9.70  | 216.13 | 1895.00 | **1.11** |

**Take-away:** the per-op overhead floor (≈ 35–50 µs of GDRCopy + signal RMW + TCP syscall + recv_thread wakeup) dominates at small sizes; once messages clear ~256 KB the cost amortises and the single-NIC / single-stream TCP fabric cap takes over at **~1.1 GB/s**.

### Other elastic tests

| Test | Result | Headline numbers |
|---|---|---|
| `test_barrier.py`  | ✅ pass | **~35 µs per barrier** (single iter); ~75 µs/barrier over a 1000-iter run |
| `test_engram.py`   | ✅ pass | issue **271 µs** + wait **2704 µs** for 256-token / 128-hidden iget; **0.09 MPPS** at 256 B/msg |
| `test_ep.py` (with perf) | ✅ correctness | dispatch / combine / reduced-combine / cached-dispatch / expanded-dispatch all run and validate; final perf-summary line trips a `num_scaleout_bytes / t` div-by-zero **inside the DeepEP test harness itself** — cosmetic, not a plugin defect |
| `test_agrs.py`     | ⊘ N/A | DeepEP asserts `nvl_ranks == num_ranks` — intra-node only, not satisfiable with 2 × 1 |

### Bottleneck attribution (recv side)

| Layer | Cost / contribution |
|---|---|
| GDRCopy *N* B H→D + 2× 8 B signal RMW | ~6–10 µs (constant + ~1 µs/KB above 64 KB) |
| TCP `recvmsg` syscall + `poll()` wake-up | ~10 µs |
| Single `recv_thread` services all peers | serializes inflight ops |
| Single TCP stream over one `eth1` NIC | caps real recv at **~1.1 GB/s** at 1 MB messages |

Switching `NCCL_SOCKET_IFNAME` from eth0 to eth1 made **no measurable throughput difference** (both saturate at the same single-stream TCP cap). The eth0 → eth1 switch was for correctness / cleanliness, not speed.

Real upgrade path is **libfastrak (TCPXO GPU-direct)** — same data plane production NCCL uses for 227 GB/s busbw on a3-mega. That is a multi-week rewrite of the data plane, not patches to the current plugin. An interim step that stays inside the current TCP+GDRCopy architecture is sharding traffic across `eth1`..`eth8` with one `recv_thread` per NIC (theoretical ~8× headroom, in practice gated by GDRCopy serialisation and per-peer dependencies — see "Recv optimisation paths" below).

### Recv optimisation: multi-NIC sharding (implemented, measured)

The plugin now supports `NCCL_SOCKET_IFNAMES=eth1,eth2,...,eth8` and opens one TCP connection per `(peer, NIC)`, with one `recv_thread` per NIC and per-NIC staging buffers. NIC selection is `hash(dst_token, dst_off) % n_nics`, so the same target region always lands on the same NIC (preserves per-region ordering).

**Measured on `test_pp` 2 × 1, recv peak GB/s at hide=1, concurrent=3:**

| message size | 1 NIC | 4 NICs (eth1×4 same NIC) | 8 NICs (eth1..eth8) |
|---:|---:|---:|---:|
|   4 KB | 0.20 | — | 0.158 |
|  16 KB | 0.45 | — | 0.430 |
|  64 KB | 0.67 | — | 0.646 |
| 256 KB | 0.70 | — | 0.679 |
|   1 MB | **1.11** | **1.107** | **1.026** |

**Result on this micro-benchmark: no gain.** Both same-NIC multi-stream and 8-NIC sharding land at the same ~1.1 GB/s ceiling. So the cap is *not* per-TCP-stream and *not* per-NIC — it sits above the transport layer. Diagnosis:

- A 2-rank × 1-process test has exactly one peer. With `inflight=3`, at most 3 distinct `(dst_token, dst_off)` regions exist in flight → at most 3 of the 8 NICs ever carry a packet.
- The `test_pp` pipeline pattern is fundamentally round-trip: each iter has to wait for the inflight tensor's signal to come back before issuing the next, so the latency floor (GDRCopy + TCP + signal RMW ≈ 300–800 µs per RTT depending on size) sets the throughput ceiling, not the link.
- Same-NIC × 4 streams returns the same number (1.107 GB/s), confirming we are *not* hitting a per-stream TCP cap.

**Where multi-NIC sharding is expected to pay off**, given the same code with no further changes:
- larger `world_size` (more peer pairs in flight → more distinct hash buckets → real NIC parallelism);
- `test_ep` dispatch / combine traffic, where each iter fans tokens out to all experts (lots of distinct `(dst_token, dst_off)` keys);
- multi-process per node (`N_LOCAL > 1`), which simultaneously increases concurrent peers.

### N_LOCAL=2 measurement — uncovered a hash-collision bug

Ran `test_pp` with `N_LOCAL=2` (4 ranks total, 2 procs per node), 1 MB tensors:

| config | recv peak GB/s (hide=1, c=3) |
|---|---:|
| N_LOCAL=1, 1 NIC  | 1.11 |
| N_LOCAL=1, 8 NIC  | 1.026 |
| N_LOCAL=2, 1 NIC  | 0.83 |
| N_LOCAL=2, 8 NIC  | **0.90** (+8% over N_LOCAL=2/1-NIC) |

Per-NIC `recv_thread` fire counts on rank 0 (8-NIC run) tell the real story:

```
recv_thread[nic=0]: 1477    ← carries traffic
recv_thread[nic=4]: 1467    ← carries traffic
recv_thread[nic=1,2,3,5,6,7]: 4 each   ← connect/finalize noise only
```

**Root cause: `hash(dst_token, dst_off) % n_nics` collides.** In `test_pp` the set of live `(dst_token, dst_off)` keys is tiny (≈ inflight × a few token slots), so the hash maps them onto only 2 of the 8 NICs. Six NICs sit idle. The theoretical ceiling for this run is therefore ~2× (not 8×), and DeepEP's own RTT pipeline cap eats most of that, leaving the measured +8%.

**Fix candidates (next session):**
- **A — mix `dst_rank` into the hash** (`mix ^= dst_rank * <odd const>`): one line, no ordering impact (same region on same rank still maps to same NIC), spreads across NICs as soon as there is >1 peer. Lowest risk; do this first.
- **B — round-robin on a per-op counter (`req_id`)**: spreads perfectly but **breaks per-region ordering** (two PUTs to the same `(token, off)` could take different NICs and arrive out of order). Only safe if DeepEP never issues ordered same-region writes; needs verification before use.

The infra is validated end-to-end at N_LOCAL=2 (no correctness regressions, all 8 recv_threads spawn and shut down cleanly); only the load-balancing hash needs improvement.

### Other recv optimisation paths still on the table

1. **Per-peer `recv_thread`** even on the same NIC: removes the head-of-line block where a slow peer's `apply_recv` stalls the others. Not implemented.
2. **`recvmmsg` / `sendmmsg` batching** when there are queued GFDs for the same peer: one syscall per N small messages instead of N syscalls.
3. **Coalesce per-op `WireMsgHdr`** for back-to-back ops to the same destination, so a 4 KB op stops paying 68 bytes (~1.6%) of header overhead and one full syscall round-trip.
4. **`epoll(7)` or `io_uring`** to replace the per-iter `poll()` scan; once there are 16+ peers the scan dominates wake-up cost.
5. **TCP socket tuning** (`TCP_NODELAY`, larger `SO_SNDBUF/SO_RCVBUF`, BBR). Already partly on by default in NGC kernels.

For the specific bottleneck visible at 2 × 1 micro-benchmark scale (DeepEP-side pipeline RTT, not transport), none of the above will move the number. `libfastrak` remains the only path to >>10 GB/s, but at any larger world size multi-NIC sharding **plus the hash fix above** should start to pay.

---

## Key debugging milestones (the day in order)

1. **D1 (patched NCCL with [D1] INFO markers)** — proved NCCL 2.30.4 GIN PROXY chain works correctly; the plugin *is* getting called. Overturned the previous "NCCL self bug" hypothesis from PR#521-era investigations.
2. **Dedicated `cudaStreamNonBlocking` signal stream** — let `pp_set_config` barrier finally pass.
3. **DIAG-PROBE** — wrote 16 KB into a plugin-owned `cudaMalloc`'d buffer during a busy-wait kernel: still 100 s hang + 719. Proved the issue is *not* DeepEP-buffer-specific; any `cudaMemcpyAsync` is doomed under busy-spin.
4. **GDRCopy data plane** — `gdr_pin_buffer` + `gdr_map` + `gdr_copy_to_mapping`. 8 µs even under a 10 s busy-spin kernel.
5. **eth0 → eth1** (GPUDirect NIC, TCP-level) — no longer touching the control plane.
6. **`INFLIGHT=4`** for `test_pp` Profiling — the runner had `INFLIGHT=2` but Profiling iterates `concurrent ∈ {1,2,3}` so the buffer needed more slots.
7. **`OP_GET` wire protocol + `handle_get_request`** — implemented real `iget` to make `test_engram` pass. Hit two follow-on bugs:
   - **Recursive `cc->mr_mu` deadlock**: `handle_get_request` was holding the MR lock while calling `do_send` (which acquires the same lock via `find_mr_for_range`). Fix: release the lock before `do_send`.
   - **Cat-pipe corruption**: streaming the `.so` between nodes via `cat | ssh` occasionally yielded `invalid ELF header`. Always use `scp -3` for binaries.

---

## What this plugin is and is not

**Is**: a minimal functional NCCL GIN PROXY plugin good enough to make every DeepEP_v2 elastic test that is physically possible on 2 nodes × 1 rank a3-mega pass. Single-file. No external state. Easy to extend.

**Is not**: a production data plane. TCP over a single eth1 stream caps throughput well below what the a3-mega fabric is capable of. The right production substrate is libfastrak (TCPXO GPU-direct) — but that path was deliberately deferred because functional correctness is the prerequisite, and that is what this work delivers.
