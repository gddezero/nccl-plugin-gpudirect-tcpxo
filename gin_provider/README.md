# gin_provider — TCPXO NCCL GIN PROXY-mode plugin

Implements `ncclGinPlugin_v13` (NCCL ≥ 2.30.4) on top of GCP a3-mega's
existing FasTrak / DXS / Falcon transport, so DeepEP V2 can target TCPXO
without the upstream DOCA-GPUNetIO + Mellanox CX-7 dependency chain.

## Status

| Layer | State |
|---|---|
| ABI surface (`ncclGin_v13_t`, 17 host callbacks) | ✅ exported as `ncclGinPlugin_v13` |
| Init / Devices / GetProperties | ✅ wired to fastrak NIC discovery |
| Listen | ✅ `dxs->Listen` + 128B `ListenHandle` wire format |
| Connect mesh | ✅ outbound `dxs->Connect` + Accept loop, interleaved |
| CreateContext | ✅ allocates `ProxyGpuCtxOwned` + `ScratchPool`, starts threads |
| RegMrSym / RegMrSymDmaBuf | ✅ `getDmabufFd` + `buf_mgr->RegBuf` |
| Outbound progress thread | ✅ GFD ring → WireHeader → `dxs::Send` (header + payload) |
| Inbound progress threads | ✅ one per accepted RecvSocket; demux on `WireHeader.source_rank` |
| Signal / Counter forwarding | ✅ atomic_add into `ProxyGpuCtx.signals[]` |
| Get path | 🚧 stub (M3.1) |
| Iput / IputSignal / Iget / Iflush / Test (host-side fallback) | 🚧 stubs (PROXY hot path is GPU-side GFD) |
| End-to-end NCCL 2.30.4 integration | 🚧 NCCL build in progress |
| DeepEP V2 dispatch / combine | 🚧 needs M5 (NCCL 2.30 binary + 2-host orchestration) |

## Layout

```
gin_provider/
├── BUILD
├── README.md                        (this file)
├── plugin_main.{h,cc}               17-callback dispatcher + ncclGinPlugin_v13 export
├── nccl_gin_v13_abi.h               local copy of upstream gin_v13.h with corrected include
├── listen_handle.h                  128B wire format used during Listen→Connect handshake
├── wire_protocol.h                  64B per-op WireHeader on the data plane
├── gpu_ctx_alloc.{h,cc}             cudaHostAlloc(MAPPED) for the NCCL device-visible context
├── scratch_pool.{h,cc}              cudaMalloc + RegBuf scratch slab for header staging
├── gfd_decoder.{h,cc}               decode 128B ncclGinProxyGfd_t from device kernel
├── proxy_context.{h,cc}             ListenComm / CollComm / GinCtx — three lifecycle layers
├── proxy_progress.{h,cc}            outbound + inbound progress engines
└── test/
    ├── load_plugin_test.cc          dlopen smoke test; verified working in fastrakTestContainer
    ├── two_rank_loopback_test.cc    2-process fork test (exposes DXS inter-host limitation)
    └── nccl_init_test.cc            tagged manual; runs once libnccl.so.2.30.4 is built
```

## Build

```sh
# Inside the tcpxo-build container (Ubuntu 22.04 + CUDA 12.8 + bazel 8.0.1):
cd /work/tcpxo-fork
./prepare_source.sh
bazel run webrtc:build_sctp -- $(realpath webrtc)
bazel build --compilation_mode=opt //gin_provider:libnccl-gin.so
```

Output: `bazel-bin/gin_provider/libnccl-gin.so` (~5.2 MB, ncclGinPlugin_v13 exported).

## How NCCL loads it

```sh
# Override the v7 plugin search; do NOT LD_PRELOAD the v7 .so or its absl
# flag definitions will collide with ours.
export NCCL_GIN_PLUGIN=/path/to/libnccl-gin.so
export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_FASTRAK_USE_LLCM=1
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices
unset LD_PRELOAD
```

NCCL ≥ 2.30.4 will dlopen the .so, dlsym `ncclGinPlugin_v13`, and call
`init` → `devices` → `getProperties` automatically.

## Wire protocols

### ListenHandle (128 B, exchanged once per rank during NCCL bootstrap)
```
uint32 magic = 'FGIN'   uint16 version = 1   uint16 addr_family
uint8  addr[16]         uint16 port           uint8  fastrak_idx   uint8 pad
uint64 nonce            uint64 listen_token   uint8  reserved[80]
```

### WireHeader (64 B per op, prepended to every dxs::Send on the data plane)
```
uint32 magic = 'FPUT'   uint16 op             uint16 flags
uint32 source_rank      uint32 dest_rank
uint64 seq              uint64 dst_handle     uint64 dst_off
uint64 size             uint64 signal_val
uint32 signal_id        uint32 counter_id
```

`op` is one of `Put / PutSignal / Signal / Get / GetReply / Flush`. Receiver
demuxes on `source_rank`; no per-socket source matching needed.

## Known limitations

1. **DXS does not loop back on a single host** — two NICs on the same
   physical machine cannot Connect to each other through DXS. Real
   verification needs a second host (forrest-h100-02 in our setup).
2. **Get / GetReply** not yet implemented (`ncclWarning`-level no-op).
3. **Per-op synchronous Send** in `TickOutbound` blocks on each Send's
   `Test()` before issuing the next. M6 will pipeline.
4. **One inbound thread per RecvSocket** rather than a poller pool.
5. **`peer_regs` published as raw mhandle key**; the receiver re-resolves via
   its own `MemHandle` map, which works because both sides share the rank
   space and key allocation order. A real OOB ginHandle exchange will land
   when DeepEP's actual call sequence forces the issue.
