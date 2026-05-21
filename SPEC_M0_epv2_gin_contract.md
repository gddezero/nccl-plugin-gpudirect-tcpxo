# M0 Spec — TCPXO NCCL GIN Plugin for DeepEP EPv2

**任务**：从零设计一个面向 TCPXO（a3-megagpu-8g）的 NCCL GIN plugin，让 DeepEP_v2（`main@b306af0` "Public release 26/04"）在 2 节点 × 8 H100 上的 **所有 EPv2 elastic 原语**跑通。

**终极验收口径**：`/work/DeepEP_v2/tests/elastic/` 下 5 个测试脚本（按依赖序）全部 pass：
1. `test_barrier.py` — ElasticBuffer.barrier × 1000 轮
2. `test_pp.py` — pp_send / pp_recv 点对点
3. `test_engram.py` — engram_write / engram_fetch 大 buffer
4. `test_agrs.py` — all_gather fan-out
5. `test_ep.py` — dispatch / combine full case enumeration（throughput KPI 来源）

**显式 out-of-scope**：`tests/legacy/test_low_latency.py`（走 NVSHMEM，与 NCCL GIN 路径正交，本项目不覆盖）。如未来需要 low-latency 双机能跑，需要另立 NVSHMEM-over-TCPXO transport 项目。

**本文档目的**：在写任何 plugin 代码 / EPv2 patch 前，闭环描述：
1. EPv2 期待 NCCL 提供什么 host-side + device-side 接口；
2. NCCL GIN 在 PROXY 模式下，把哪些工作做掉、把哪些转嫁给 plugin；
3. 因此 plugin 需要实现的最小契约（ABI + 语义）；
4. 已经发现的 EPv2/TCPXO 兼容性风险，以及对应的应对策略；
5. M1→M4 每个里程碑的成功判据。

**调研范围 / 信息源**（本地拷贝在 `epv2_gin/spec_research/`）：

| 来源 | 路径 | 行数 |
|---|---|---|
| NCCL 2.30.4 device-side GIN API | `nccl_device/{gin.h, core.h, gin/gin_device_common.h, gin/proxy/gin_proxy*.h}` | ~1.2 k |
| NCCL plugin host ABI (v11) | `nccl_plugin_abi/net/net_v11.h`, `nccl_net.h`, `plugin.h` | 0.3 k |
| NCCL 内部 GIN orchestration（proxy 包装层） | `nccl_gin_host/{gin_host.cc, gin_host_proxy.cc, include/gin/*.h}` | 0.9 k |
| EPv2 backend 入口 | `epv2_backend/{nccl.cu, api.cuh}` | 0.25 k |
| EPv2 elastic 算法层 host 端 | `epv2_elastic/{barrier, dispatch, combine, ...}.hpp` | ~1 k |
| EPv2 device-side handle 抽象 | `epv2_common/common/{handle.cuh, comm.cuh, layout.cuh}` | ~1 k |
| EPv2 device-side kernel impls | `epv2_impls/impls/{barrier, dispatch, combine, ...}.cuh` | ~1.5 k |

明确**未**作为参考的：
- `~/deepep_dev/tcpxo-fork/gin_provider/`（PR521 时代 EPv1 plugin 实现）；
- 之前的 memory / DEVLOG / REPORT 等历史文档（包括"PutSignal fix"、"wire_seq"、"comm0"、"Iflush bug"等过往结论，本次全部清零重新设计）。

---

## 1. EPv2 在 host 端依赖的 NCCL 公开 API

`epv2_backend/nccl.cu` 是 EPv2 接 NCCL 的唯一入口（154 行，整段都很关键）。每个 rank 创建 `Buffer` 时构造一个 `NCCLSymmetricMemoryContext`，它对 NCCL 的调用序列：

```
ncclGetUniqueId() → 一个 rank
ncclCommInitRank(num_ranks, root_id, rank_idx)
ncclCommQueryProperties(comm, &props)
    // 必须满足: (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE
ncclDevCommCreate(comm, &reqs, &dev_comm)
    // reqs.ginContextCount    = num_allocated_qps   // 见下
    // reqs.ginExclusiveContexts = true              // 每个 ctx 独占 QP
    // reqs.ginQueueDepth      = 1024
    // reqs.ginTrafficClass    = sl_idx              // 来自 Python 参数
    // reqs.ginSignalCount     = num_ranks + 4
    // reqs.ginConnectionType  = RAIL (hybrid) | FULL (非 hybrid)
ncclMemAlloc(&raw_window_ptr, size)
ncclCommWindowRegister(comm, raw_window_ptr, size, &window, NCCL_WIN_DEFAULT)
    // 集合通信调用，内部 bootstrapBarrier
ncclGetLsaDevicePointer(window, 0, lsa_rank, &nvl_window_ptrs[i])
    // 对 LSA (NVLink 内) 同伴拿 mapped peer ptr
// ...运行时...
ncclCommWindowDeregister(comm, window)
ncclMemFree(raw_window_ptr)
ncclDevCommDestroy(comm, &dev_comm)
ncclCommAbort(comm)
```

EPv2 同时从 `dev_comm.lsaSize / dev_comm.lsaRank` 拿物理域信息，假设 `num_ranks % lsaSize == 0`。

这些 host 调用**全部走 NCCL 公开符号**（`/usr/local/lib/python3.10/dist-packages/nvidia/nccl/include/nccl.h` + `nccl_device/core.h` 已确认），plugin 不直接接触；plugin 是被 NCCL 在 `ncclDevCommCreate` / `ncclCommWindowRegister` 等流程里通过 `dlsym("ncclGinPlugin_v11")` 拉起的。

---

## 2. NCCL GIN PROXY 模式的内部分工

读 `nccl_gin_host/gin/gin_host.cc` + `gin_host_proxy.cc` 后清楚：在 **`NCCL_NET_DEVICE_GIN_PROXY`** 模式下，NCCL 自己做大量工作，plugin 反而轻巧。

### 2.1 connect 阶段（`ncclGinConnectOnce`）

NCCL 按顺序调 plugin：
```
plugin->init(&ginInstance, commId, logFunc)
plugin->devices(&ndev)
plugin->getProperties(0, &props)          // NCCL 读 netDeviceType 决定 PROXY/GDAKI
for (n = 0; n < ginCommCount; n++) {
    plugin->listen(ginInstance, dev_n, handle_out, &listenComm)
    bootstrapAllGather(handle_out, NCCL_NET_HANDLE_MAXSIZE=128B)
    plugin->connect(ctx, handles[], nranks, rank, listenComm, &ginComms[n])
    // PROXY 模式下: NCCL 自己调用内部包装 ncclGinProxyCreateContext()
    //              它会在 plugin 之外分配 ncclGinProxyGpuCtx_t（queues/pis/cis/signals/counters/inlines）
    //              并通过 plugin->regMrSym 把 signals & inlines 注册成 plugin 可寻址的 MR
    plugin->closeListen(listenComm)
}
```

`ginCommCount = min(NCCL_GIN_MAX_CONTEXTS, ncclParamGinNcontexts())`，EPv2 通过 `reqs.ginContextCount` 给 NCCL hint（注：源码注释明确说 "this is a hint, the actual context count may not match"）。

### 2.2 device-side ctx：完全由 NCCL 拥有

在 PROXY 模式下，每个 context 的 GPU 端结构 `ncclGinProxyGpuCtx_t`（见 `gin/proxy/gin_proxy_device_host_common.h`）：

```c
typedef struct {
  int nranks;
  uint32_t queueSize;          // power of 2
  ncclGinProxyGfd_t *queues;   // [nranks * queueSize], 每 entry 128B
  uint32_t *pis;               // [nranks], producer idx，GPU 写
  uint32_t *cis;               // [nranks], consumer idx，CPU proxy 写
  uint64_t *counters;          // [nCounters], 64-bit
  uint64_t *signals;           // [nSignals], 64-bit；作为一个 MR 注册给 plugin
} ncclGinProxyGpuCtx_t;
```

- `queues / pis` 是 GPU 写、CPU 读；`cis / counters / signals` 用 GDRCopy 让 CPU 直接写 GPU 内存，没有 GDRCopy 时回退到 pinned host + WC。
- 这些**由 NCCL 自己分配并连到 dev_comm**，plugin 不参与。device-side kernel 用 `ncclGin.put/signal/...` 时，inline 模板生成 GFD 写到 `queues[peer][pi++]`，CPU proxy 线程把它翻译成 plugin 调用。

### 2.3 progress：NCCL 自起线程

`ncclGinConnectOnce` 检查 `devHandle->needsProxyProgress`：PROXY 模式必然为 1（见 `ncclGinProxyCreateContext` line 411）。NCCL `pthread_create("NCCL GIN Progress")` 起线程跑 `ncclGinProgress(ginState)` → `ncclGinProxyProgress(plugin, ginCtx)`：

```c
ncclGinProxyProgress:
  proxyGinPollCompletions(...)        // 调 plugin->test(request) 检完成，bump CIs，可选 counter+1
  for each peer:
    if proxyGinPollGfd(peer) returns 1:
      proxyGinProcessGfd:
        switch op:
          case Put:        plugin->iput(...)
          case Put+Signal: plugin->iputSignal(..., signalOff=signalId*8, signalMhandle=signalsGinHandle, val, op)
    plugin->ginProgress(collComm)     // 给 plugin 做内部 poll 的机会
```

**注意**：本文档读到的 NCCL 2.30 GIN proxy 包装层（`proxyGinProcessGfd`）目前**只处理 `ncclGinProxyOpPut`**，VA-signal-only / Get / Flush / 纯 SignalInc 没 dispatch 到 plugin（`switch default: assert(0)`）。这意味着：
- 任何走 `ncclGin_VASignalAdd`-only 的路径在 proxy 模式下会触发 NCCL 内部 assert；
- 只走 `ncclGin_SignalInc{signal_id}` + put 的路径是 OK 的（被包装成 `iputSignal`）；
- `Get` op 在 proxy 模式下走 plugin 还是走 NCCL 自己，需要后续 M3 时再深读。

这是一个待澄清项 — M2 设计时需先用最小 case 验证 VASignalAdd 在 proxy 是否真的不可用，再决定是绕开 EPv2 的相关路径还是在 NCCL 层 hook。

---

## 3. Plugin host-side 契约（`ncclGin_v11_t`）

PROXY 模式下，plugin 真正必须实现的子集：

| # | 函数 | 是否必需 | 责任 |
|---|---|---|---|
| 1 | `name` | 必 | 字符串名（log 用） |
| 2 | `init(void** ctx, uint64_t commId, ncclDebugLogger_t)` | 必 | 进程内单次。返回 plugin instance。 |
| 3 | `devices(int* ndev)` | 必 | 返回 TCPXO 可见 NIC 数（a3-mega 是 8） |
| 4 | `getProperties(int dev, ncclNetProperties_v11_t* props)` | 必 | 见 §3.1 |
| 5 | `listen(void* ctx, int dev, void* handle, void** listenComm)` | 必 | handle ≤ 128B，写入 NCCL 给的 buf，bootstrap 会 allgather 它 |
| 6 | `connect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm)` | 必 | **集合** — 阻塞到全员连通；建立到所有 peer 的 TCPXO data-plane 连接 |
| 7 | `regMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle, void** ginHandle)` | 必 | type ∈ {CUDA, HOST}；要支持 `mrFlags = NCCL_NET_MR_FLAG_FORCE_SO`（signal MR 用，要求强排序）；见 §3.2 |
| 8 | `regMrSymDmaBuf(... int fd, ...)` | 推荐 | DMA-BUF 路径；不支持时返回 ncclInvalidUsage，NCCL 会自动 fallback 到 `regMrSym` |
| 9 | `deregMrSym(void* collComm, void* mhandle)` | 必 | — |
| 10 | `closeColl(void* collComm)` | 必 | — |
| 11 | `closeListen(void* listenComm)` | 必 | — |
| 12 | `iput(collComm, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, **request)` | 必 | 见 §3.3 |
| 13 | `iputSignal(collComm, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff, signalMhandle, signalValue, signalOp, **request)` | 必 | 见 §3.3 |
| 14 | `test(collComm, request, int* done)` | 必 | 轮询完成；见 §3.3 |
| 15 | `ginProgress(collComm)` | 推荐 | NCCL proxy 线程每轮调一次，给 plugin 推进内部状态机的机会 |
| 16 | `queryLastError(ginCtx, bool* hasError)` | 必 | 报告致命错误（让 NCCL 早 abort） |
| 17 | `finalize(void* ctx)` | 必 | — |
| 18 | `createContext` | **不调用** | PROXY 模式被 NCCL 自己的 `ncclGinProxyCreateContext` 替代 |
| 19 | `destroyContext` | **不调用** | 同上 |
| 20 | dlopen 符号 | 必 | `extern "C" ncclGin_v11_t ncclGinPlugin_v11;` |

### 3.1 `getProperties` 必须填的字段

| 字段 | 值（M1 起步） | 说明 |
|---|---|---|
| `netDeviceType` | `NCCL_NET_DEVICE_GIN_PROXY` | 唯一让 NCCL 走 proxy wrapper 的开关 |
| `netDeviceVersion` | `NCCL_GIN_PROXY_VERSION` = 100 | 与 GFD 协议绑定 |
| `name` / `pciPath` / `guid` | 取 TCPXO NIC 的 sysfs 信息 | — |
| `ptrSupport` | `NCCL_PTR_HOST \| NCCL_PTR_CUDA`（M1 不强求 DMABUF） | 必须支持这俩 — NCCL 在 createContext 时会注册 inlines (HOST) + signals (CUDA, FORCE_SO) |
| `maxRecvs` | ≥ 1（先填 1，影响 proxy queue 上限） | `queueSize = NCCL_NET_MAX_REQUESTS * maxRecvs` = `32 * maxRecvs`，并被向上取 2 的幂 |
| `maxP2pBytes` | `1 << 30`（1 GiB，proxy 内部 `DataChunkSize` 也是 1 GiB） | iput 单笔大小上限 |
| `regIsGlobal` | 0（保守）/ 1（如果 mhandle 跨 comm 可复用） | 1 节省内存，先取 0 |
| `forceFlush` | 0 | TCPXO 自带数据落地保证（待 M3 验证） |
| `speed` | 200000 (200 Gbps) 或 NIC 实际 | log only |

### 3.2 `regMrSym` 的 ginHandle 设计

`ginHandle` 会被 GPU kernel 写进 GFD 的 `srcHandle / dstHandle 63-bit` 槽，CPU proxy 翻译时再传回 plugin 的 `iput*`。约束：
1. **必须 ≤ 63 bit**（GFD qword 顶 1 bit 是 valid flag）；
2. **必须可被 plugin 自己用来定位 MR**：plugin 收到 `dstMhandle=ginHandle`，要能找到 "peer=rank 的 MR 在 peer 上的本地起始 VA"；
3. **同一次 `regMrSym` 在所有 rank 上必须返回"可全局解析"的 handle**：通常做法是 `regMrSym` 内部做一次 bootstrap-allgather，让每个 rank 知道所有 peer 的对应 MR 起始 VA + DXS endpoint info，本地存一张 `(ginHandle, peer) → remote_VA` 的表。

推荐设计：plugin 内部维护一个 per-comm 的 `vector<MrRecord>`，`ginHandle = index<<1 | (CUDA ? 1 : 0)`（或简单 `index | gen_bit`），`MrRecord` 内部含每个 peer 的 remote VA + DXS 连接句柄。

### 3.3 `iput / iputSignal / test` 语义

* `iput(... srcOff, srcMhandle, size, dstOff, dstMhandle, rank, req)`：
  - 异步把本 rank 的 `srcMhandle + srcOff` 共 `size` 字节，写到 `rank` 的 `dstMhandle + dstOff`。
  - `test(req, done)` 标记 `done=1` 时：source buffer 可复用；**数据是否在远端可见不保证**，要靠后续 `iputSignal` / fence 锁住。
  - NCCL 的 proxy 包装会在拆 chunk 时连续 post 多个 GFD，每个 GFD 一个 plugin->iput，每个独立 test，**plugin 不必合并**。

* `iputSignal(... signalOff, signalMhandle, signalValue, signalOp, req)`：
  - 上面的 iput + 远端完成后做 64-bit 原子 op（`INC=+1` 或 `ADD=+signalValue`），目标地址 = `(signalMhandle, signalOff)` 在 peer 上。
  - **顺序约束**：data write 必须先在远端落地，再发起 signal，再标 done。这是 GIN 的 release/signal 契约的核心。
  - `signalMhandle` 是 NCCL 自己注册的 signals MR（PROXY 模式下=`signalsGinHandle`，64 KiB 量级），地址类型 CUDA + `NCCL_NET_MR_FLAG_FORCE_SO`。

* `test(collComm, req, done)`：non-blocking，返回 `done=1` 当所有上述语义满足。NCCL proxy 在 `proxyGinPollCompletions` 里每轮调一次，按 peer 的 CI 滚动 bump（要按提交顺序 done，否则 hole 不前进）。

### 3.4 plugin 内部数据流（TCPXO 数据面建议）

a3-mega 节点有 8 个 TCPXO NIC（GVNIC）；TCPXO 数据面靠 DXS + GPUDirect TCPX-O。M1→M3 我们做最小可行版本，**不**做 rail-aware 优化：

```
plugin instance
├── per-NIC TCPXO socket pool（先 1 NIC，M3 后扩 8 NIC + 简单 round-robin）
├── per-collComm:
│   ├── nranks 个 send endpoint + nranks 个 recv endpoint（DXS 连接）
│   ├── MR table: vector<MrRecord{ size, local_va, [peer]remote_va }>
│   ├── inflight request pool（NCCL_NET_MAX_REQUESTS * maxRecvs ≥ proxy queueSize）
└── 一个 worker 线程（可选）做 DXS 完成 poll，否则在 ginProgress 里 poll
```

`iputSignal` 的两种实现路径（M2 决定取哪个，先列出选项给 reviewer 看，不在 spec 阶段拍板）：

| 方案 | 实现 | 优点 | 缺点 |
|---|---|---|---|
| (a) 远端 proxy 协助 | iput 在 wire 上多发一条 "signal-trailer" 控制消息；peer 的 ginProgress 收到后做本地 `__atomic_fetch_add` 到 signals MR | TCPXO 不需要 native atomic；干净 | 多一跳 CPU 介入 |
| (b) 远端写一段 inline | iputSignal 直接在 iput 后跟一笔 8B inline 写到 `signalsGinHandle + signalOff`（不是 atomic add，但 INC 可以用 send-side seq 累计） | 不需要 peer 线程 | INC 需要 send 端维护期望值；ADD 不能做 |

倾向 (a) — 简单清晰，且 NCCL 的 proxy 线程本来就在轮询每个 ctx；GIN proxy 线程能复用做 signal 后期处理。

---

## 4. EPv2 device-side 用法 + 兼容性风险

EPv2 算法层用 `handle::NCCLGin`（`epv2_common/common/handle.cuh`）作为唯一的 device-side 抽象，包了 NCCL `ncclGin_BackendMask` 的 `signal/wait/get/flushAsync/put/putValue`。

### 4.1 EPv2 算子映射到 NCCL GIN device-side ops

| EPv2 算子 | 调用 | proxy 路径覆盖？ |
|---|---|---|
| `red_add_rel<team, T>(sym_ptr, value, dst_rank)` (跨 NIC) | `gin.signal(team, dst, ncclGin_VASignalAdd(window, offset, val))` | ⚠️ proxy wrapper switch 只处理 Put 类，**VASignal-only op 可能 assert** — M2 必须先验证 |
| `put<team>(recv, send, bytes, dst)` | `gin.put(team, dst, win, rOff, win, sOff, bytes, none, none, coop)` | ✅ → 翻译成 plugin `iput` |
| `put_value<team, T>(sym_ptr, value, dst)` (跨 NIC) | `gin.putValue(team, dst, win, off, val)` | ⚠️ proxy 内部走 inline put 路径，包装层应该走 iput-with-inline；待 M3 时验证 |
| `get<team>(src, dst, bytes, src_rank)` | `gin.get(team, src, rWin, rOff, lWin, lOff, bytes, coop)` | ⚠️ proxy `proxyGinProcessGfd` 当前**未实现 Get 分支**，可能需要 NCCL upstream patch 或 EPv2 改算法 |
| `signal<team>(dst, remote_action)` | `gin.signal(team, dst, action)` | 取决于 action：SignalInc 配 Put 经过 iputSignal OK；单独 VASignalAdd 同上面待验证 |
| `flush_async / wait` | local poll on signal/counter | proxy 在 device 端走 `waitForGfdComplete`（CPU CI bump），无需 plugin |

### 4.2 **关键风险：elastic barrier 是 GDAKI-only**

`epv2_common/common/comm.cuh:166-167`（`gin_barrier_wo_local_sync` 的 hot path）：

```cpp
const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin._ginHandle) + gin.contextId;
const auto signal_ptr = reinterpret_cast<uint64_t*>(
    __ldg(reinterpret_cast<uint64_t*>(&gdaki->signals_table.buffer))) + signal_idx;
```

`gin._ginHandle` 在 PROXY 模式下指向 `ncclGinProxyGpuCtx_t`，跟 `ncclGinGdakiGPUContext` 内存布局**完全无关**。这段直接强转 → 读垃圾 → kernel SEGV 或 hang。

**这意味着** `ElasticBuffer.barrier()`（M2 目标）在 proxy 模式下**不可能直接跑通**，必须二选一：

1. **本地 patch EPv2**：把上述 GDAKI shortcut 改成调 `gin.gin.waitSignal(coop, signal_id, target, 64, memory_order_acquire)`（公开 API，NCCL `gin__funcs.h:ncclGinWaitSignal` 已实现 proxy 分发）。改动量小（~10 行）、风险低。
2. **绕过 elastic barrier**：用 EPv2 已有的 `red_add_rel` + 自旋 wait 自己写 barrier（更接近原 NVSHMEM 路径）。改动量大、容易跑偏。

**M2 采取方案 1**，patch 路径：clone DeepEP_v2 → `deep_ep/include/deep_ep/common/comm.cuh:166-179` 改为调 `gin.waitSignal()`。patch 文件单独维护在 `epv2_gin/patches/`，build 时 apply。**绝不修改** `/work/DeepEP_v2/` 已 build 的源（用户规则）；clone 到 `epv2_gin/DeepEP_v2_patched/` 单独 build。

### 4.3 其他风险 / 待澄清项

| ID | 风险 | 影响里程碑 | 应对 |
|---|---|---|---|
| R1 | proxy wrapper 当前不分发 Get / VASignal-only / Flush 到 plugin | M2、M3 | M2 写最小 case 直接确认；如确实是 NCCL bug 上游 patch + 自带 build |
| R2 | EPv2 elastic barrier GDAKI 强转（§4.2） | M2 | 本地 patch DeepEP_v2 改用 `waitSignal` |
| R3 | TCPXO 无 native RDMA atomic | M2 | iputSignal 的 signal 用 §3.4 方案 (a) — 远端 proxy 线程 emulate atomic_add |
| R4 | `ncclGin_VASignalAdd` 在 EPv2 `red_add_rel` 的 RDMA 路径 | M3 (dispatch) | 同 R1，若 proxy 不支持，本地短路 EPv2 这条路径优先走 indexed signal |
| R5 | DMA-BUF 注册路径（`regMrSymDmaBuf`） | M3 后 | M1/M2 不实现，让 NCCL fallback 到 `regMrSym`，性能后续再补 |
| R6 | 8 NIC rail-aware flow steering | M4 性能 | M1-M3 全用 1 NIC，跑通后 round-robin 扩 8 NIC |
| R7 | `ginContextCount > 1` & `ginExclusiveContexts=true` | M3 | M1/M2 只支持 1 ctx（实际 ctx 数是"hint"，EPv2 接受不等），M3 时若 EPv2 强要多 ctx 再扩 |

---

## 5. M1→M4 里程碑与成功判据

每个 milestone 完成需提交 (a) 数据证明（log / matrix / md5），(b) 我（reviewer）签字，再进下一个。

### M1 — Hello-world minimal plugin

**Scope**：实现 `ncclGin_v11_t` 的最小可加载版本：
- `init / devices / getProperties / listen / connect / regMrSym / deregMrSym / closeColl / closeListen / finalize`
- `iput / iputSignal / test`：可以 stub（直接 `*done = 1`，不实际发数据） — 因为 M1 只验证 EPv2 buffer init 能跑过；
- `ginProgress / queryLastError`：空实现；
- 不实现 `regMrSymDmaBuf`（返回 `ncclInvalidUsage`）；
- TCPXO 数据面：在 `connect` 里**真实**建立 DXS 连接（即使 M1 不发数据，否则 M2 起步会暴露大量 connect 问题）。

**成功判据**：
1. .so 在两节点 md5sum 一致；
2. 两节点 docker 跑 `import deep_ep; b = deep_ep.Buffer(...)`，没有 fatal error，`ncclCommQueryProperties` 拿到 `ginType = NCCL_GIN_TYPE_PROXY`；
3. `ncclDevCommCreate` + `ncclCommWindowRegister` 全部返回 `ncclSuccess`；
4. 不要求任何 dispatch / barrier 跑得通。

每个 milestone 的成功判据都直接绑定 `/work/DeepEP_v2/tests/elastic/` 的某个脚本，避免抽象 "case matrix" 与官方 test 的口径漂移。

> **2026-05-19 重排**：用户约束"非必要不 patch DeepEP"。读 `utils/testing.py:152,164` 发现 EPv2 自带 escape hatch — **`EP_DISABLE_BARRIER_PROFILING=1`** 让 `bench_kineto` 跳过 `barrier=buffer.barrier()` 调用。这意味着 `test_pp / test_engram / test_agrs / test_ep` 全部可以 zero-patch 跑通（它们的 `buffer.barrier()` 使用都在 `bench_kineto` 内部）。`test_barrier.py` 本身是 barrier benchmark，必然撞 §4.2 GDAKI 强转，**重排为 M5（最后）**，只有当其他 4 个 test 都跑通后再单独决策是否 patch DeepEP（或等上游 TODO 落地）。

### M5 — `test_barrier.py` 双机过（需要 DeepEP patch 或上游 fix）

**Scope**：M5 是终点不是起点 — 仅在 M2-M4 全过、用户授权 patch / 上游修 TODO 后才动。
- Apply EPv2 patch（改 `common/comm.cuh:166-179` 用 `gin.gin.waitSignal()`）OR 等上游；
- `iput`/`iputSignal` 必须已在 M2-M4 中实现真实 TCPXO write。

**成功判据**：1000 轮 `buffer.barrier()` 全过（不 timeout）。

### M2 — `test_pp.py` 双机过

**Scope**：
- `iput` 实现真正的 TCPXO write；
- `iputSignal` 实现 data write + signal trailer（§3.4 方案 a）；
- `test` 正确反映 source-reusable + signal-emitted；
- 一对一 send/recv，验证 plugin 的 per-peer 路径独立性；
- 用 `EP_DISABLE_BARRIER_PROFILING=1` 跑（绕过 §4.2 GDAKI 强转，不需要 patch DeepEP）。

**成功判据**：
1. 2 节点 × 8 rank/节点 跑 `EP_DISABLE_BARRIER_PROFILING=1 python tests/elastic/test_pp.py` 默认配置；
2. 接收 tensor 与期望 bit-exact；
3. 连跑 3 次过。

### M3a — `test_engram.py` 双机过

**Scope**：
- `iput / get` 大数据（≥ 1 GiB engram buffer）路径；
- 若 R1（Get path 在 proxy 不被分发）确实是真问题，本里程碑解决之；
- 验证 EPv2 `engram_fetch` 的 `flushAsync + wait` 在 proxy 模式下语义正确（local visibility ordering）。

**成功判据**：
1. 2 节点 × 8 rank/节点 跑 `EP_DISABLE_BARRIER_PROFILING=1 python tests/elastic/test_engram.py` 默认配置；
2. fetch 出的 tensor 与 ref 一致；
3. 连跑 3 次过。

### M3b — `test_agrs.py` 双机过

**Scope**：
- All-gather fan-out：单 rank 同时 put 到所有 peer + 等所有 peer 完成；
- 验证 plugin 的 multi-peer 并发 iputSignal 不互相阻塞；
- 验证 `iput` chunked path 在多并发下行为正确；
- test_agrs.py 本身不调 `buffer.barrier()`，可不设 env。

**成功判据**：
1. 2 节点 × 8 rank/节点 跑 `python tests/elastic/test_agrs.py` 默认 + `--num-ops` 多组；
2. 全 round bit-exact 通过；
3. 连跑 3 次过。

### M4 — `test_ep.py` 双机过（throughput KPI）

**Scope**：
- 完整 dispatch + combine；
- 全 `enumerate_ep_modes()` 组合（do_handle_copy × expert_alignment × use_fp8_dispatch × num_bias × with_previous_event × async_with_compute_stream × allocate_on_comm_stream）；
- 多 NIC（rail-aware 简单 round-robin）；
- 性能（busbw）报告但不卡硬阈值（首版能跑 = 胜）。

**成功判据**：
1. 2 节点 × 8 rank/节点 跑 `EP_DISABLE_BARRIER_PROFILING=1 python tests/elastic/test_ep.py` 默认配置，全 case 过；
2. 所有 case bit-exact（与 `ref_dispatch / ref_combine` 比对）；
3. 报 busbw（GB/s）和 dispatch/combine 单次时延（μs）；
4. 连跑 3 次过。

**终极交付（throughput KPI）**：M2→M4 全部通过 = `test_pp / test_engram / test_agrs / test_ep` 4 个 elastic test 双机 PASS + busbw 数字归档。

**M5（可选）**：`test_barrier.py` 仅当用户授权 patch DeepEP 或上游修复了 `// TODO(NCCL)` 后再跑。

---

## 6. 工程纪律与目录布局

```
epv2_gin/                                  ← 新 git branch "epv2-gin-from-scratch" 根
├── SPEC_M0_epv2_gin_contract.md          ← 本文件
├── spec_research/                         ← M0 调研所读原文（NCCL / EPv2）
├── plugin/                                ← M1 起的 plugin 源码（CMake + 单 .so）
├── DeepEP_v2_patched/                     ← clone + apply patch（M2 起需要）
├── patches/                               ← 维护对 DeepEP_v2 的 minimal patch
└── tests/                                 ← 各 milestone 验证脚本
```

操作记录：所有改动写到项目根的 `operation_history_v2.md`（新文件）；老的 `operation_history.md` 不再追加。所有 binary 跨节点部署后必须 `md5sum` 验证两节点。**不**修改容器内已 build 的 `DeepEP_v2` 源、**不**在历史 plugin fork 上做任何改动。

---

## 7. 待 reviewer 决策（不在 spec 阶段拍板）

1. **iputSignal 实现方案**（§3.4 (a) vs (b)）：倾向 (a)，但若 reviewer 有 TCPXO 内部知识倾向 (b) 请告知。
2. **EPv2 patch 策略**（§4.2 方案 1 vs 2）：倾向方案 1（patch comm.cuh，~10 行），可接受 maintain 一个 patch 文件。
3. **是否同时维护本地 NCCL build**：若 R1 是真问题，要不要在 plugin 同分支里也维护一个 NCCL fork？倾向"先不"，遇到再说。

请就以上 7 节内容给 review 反馈或直接放行进入 M1。
