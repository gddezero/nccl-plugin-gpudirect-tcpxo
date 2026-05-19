/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/proxy_progress.h"
#include "plugin/nccl_net.h"
#include "gin_provider/signal_atomic.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>

#include "absl/log/log.h"
#include "gin_provider/gdr_helper.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "dxs/client/dxs-client-interface.h"
#include "gin_provider/gfd_decoder.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/proxy_context.h"
#include "gin_provider/scratch_pool.h"
#include "gin_provider/wire_protocol.h"

namespace fastrak::gin {

namespace {

constexpr absl::Duration kSendTimeout = absl::Seconds(60);

// v8 (kept): deadline check throttled to every 4096 spins (absl::Now() at
// every spin was a measurable hot-path overhead). v8b PAUSE-instead-of-
// yield variant was reverted: inbound threads share cores with NCCL proxy
// threads, and pure busy-spin starved the proxy thread of CPU. yield()
// gives schedulers a chance to interleave, and the deadline-check throttle
// alone shaves ~5-10ns per spin worth of absl::Now overhead.
absl::Status WaitSendDone(dxs::SendOpInterface& op, absl::string_view what) {
  auto deadline = absl::Now() + kSendTimeout;
  uint32_t spins = 0;
  while (true) {
    auto s = op.Test();
    if (s.has_value()) {
      if (!s->ok()) return *s;
      return absl::OkStatus();
    }
    if ((++spins & 4095) == 0 && absl::Now() > deadline) {
      return absl::DeadlineExceededError(
          absl::StrCat("WaitSendDone timeout: ", what));
    }
    std::this_thread::yield();
  }
}

absl::StatusOr<uint64_t> WaitRecvDone(dxs::LinearizedRecvOpInterface& op,
                                      absl::string_view what) {
  auto deadline = absl::Now() + kSendTimeout;
  uint32_t spins = 0;
  while (true) {
    auto s = op.Test();
    if (s.has_value()) return *s;
    if ((++spins & 4095) == 0 && absl::Now() > deadline) {
      return absl::DeadlineExceededError(
          absl::StrCat("WaitRecvDone timeout: ", what));
    }
    std::this_thread::yield();
  }
}

}  // namespace

ProxyProgress::ProxyProgress(GinCtx* ctx) : ctx_(ctx) {
  if (ctx_ != nullptr && ctx_->gpu_ctx() != nullptr) {
    peer_out_.resize(ctx_->gpu_ctx()->nranks);
  }
}

ProxyProgress::~ProxyProgress() { Stop(); }

void ProxyProgress::Start() {
  if (outbound_thread_.joinable()) return;
  stop_.store(false, std::memory_order_release);
  outbound_thread_ = std::thread(&ProxyProgress::RunOutbound, this);
  // Spawn one inbound thread per accepted RecvSocket.
  if (ctx_ != nullptr && ctx_->coll() != nullptr) {
    size_t n = ctx_->coll()->num_inbound();
    for (size_t i = 0; i < n; ++i) {
      inbound_threads_.emplace_back(&ProxyProgress::RunInbound, this, i);
    }
  }
}

void ProxyProgress::Stop() {
  stop_.store(true, std::memory_order_release);
  if (outbound_thread_.joinable()) outbound_thread_.join();
  for (auto& t : inbound_threads_) {
    if (t.joinable()) t.join();
  }
  inbound_threads_.clear();
}

void ProxyProgress::RunOutbound() {
  while (!stop_.load(std::memory_order_acquire)) {
    TickOutbound();
    std::this_thread::sleep_for(std::chrono::microseconds(2));
  }
}

// v6 (R8 / cleanup): TickOutbound is the legacy GFD-ring drain path. PROXY
// mode (the only mode this plugin builds today) never writes to ctx_->gpu_ctx
// queues — IputCommon dxs::Sends directly. Both the spawned outbound_thread_
// and the GinProgress ABI poll this; in PROXY mode every iteration breaks at
// the first pi==ci check, so the cost is just the per-peer load. We keep the
// body for the (currently unused) non-shim path, but it intentionally never
// runs in PROXY mode and has not been audited for the same concurrency
// fixes the inbound path got (S3/S4); add a mutex if PROXY mode ever starts
// dispatching through here.
void ProxyProgress::TickOutbound() {
  auto* gpu = ctx_ ? ctx_->gpu_ctx() : nullptr;
  auto* cc = ctx_ ? ctx_->coll() : nullptr;
  auto* scratch = ctx_ ? ctx_->scratch() : nullptr;
  if (gpu == nullptr || cc == nullptr || scratch == nullptr) return;

  // PR3c-iii (β port, 2026-05-16): emit async ACK to each peer per
  // progress tick. CollComm::EmitAck is best-effort + idempotent (skips
  // when no new info to ACK, when peer not connected, when in-flight
  // ACK quota reached). This out-of-band ACK channel breaks the PR3c-ii
  // Test() gate circular deadlock — runs even when wrapper credit is
  // stuck because GinProgress (which calls TickOutbound) is invoked by
  // NCCL's polling regardless of in-flight request count.
  int nranks_total = (cc->nranks() > 0) ? cc->nranks() : gpu->nranks;
  for (int p = 0; p < nranks_total; ++p) {
    if (p == cc->rank()) continue;
    auto s = cc->EmitAck(p, scratch);
    if (!s.ok()) {
      // Best-effort; log once per peer-error pair (TLS counter to avoid spam).
      thread_local int ack_err_tls = 0;
      if (ack_err_tls < 4) {
        ++ack_err_tls;
        LOG(WARNING) << "TickOutbound: EmitAck(" << p << ") failed: " << s;
      }
    }
  }

  for (int p = 0; p < gpu->nranks; ++p) {
    if (p == cc->rank()) continue;
    auto* peer = cc->peer(p);
    if (peer == nullptr || peer->send_socks.empty()) continue;
    // Lane 0 only — TickOutbound is the legacy GFD-ring path which PROXY
    // mode never reaches; not worth striping.
    auto* sock = peer->send_socks[0].get();
    if (sock == nullptr) continue;

    auto* pi_atomic =
        reinterpret_cast<std::atomic<uint32_t>*>(&gpu->host_pis[p]);
    auto* ci_atomic =
        reinterpret_cast<std::atomic<uint32_t>*>(&gpu->host_cis[p]);

    while (true) {
      uint32_t pi = pi_atomic->load(std::memory_order_acquire);
      uint32_t ci = peer_out_[p].local_ci;
      if (ci == pi) break;
      uint32_t slot = ci & (gpu->queue_size - 1);
      auto* gfd = &gpu->host_queues[p * gpu->queue_size + slot];
      if (!GfdReady(*gfd)) break;

      DecodedGfd dec = DecodeGfd(*gfd);

      // Translate DecodedGfd into a WireHeader. The op-bit set determines
      // which WireOp we publish. All ops share the same dst_handle / dst_off
      // / signal slots and we set fields conservatively.
      WireHeader hdr{};
      hdr.magic = kWireMagic;
      hdr.source_rank = static_cast<uint32_t>(cc->rank());
      hdr.dest_rank = static_cast<uint32_t>(p);
      // M6: seq removed from WireHeader to make room for signal_handle.
      // peer_out_[p].next_seq still bumped for local debug.
      ++peer_out_[p].next_seq;
      hdr.signal_handle = 0;  // PROXY mode never reaches this path; legacy.
      hdr.dst_handle = dec.dst_handle;
      hdr.dst_off = dec.dst_off;
      hdr.size = dec.size;
      hdr.signal_val = dec.signal_val;
      // TickOutbound consumes our own GpuCtx queue, which NCCL PROXY mode
      // never writes to (the shim allocates its own queues and calls
      // IputCommon directly). signal_off stays 0 here — this code is
      // kept only as a fallback for non-shim test paths.
      hdr.signal_off = 0;
      const uint16_t op_mask = dec.op;
      constexpr uint16_t kOpVASignal = 1u << 5;
      constexpr uint16_t kOpGet = 1u << 6;
      constexpr uint16_t kOpFlush = 1u << 7;
      constexpr uint16_t kOpWithSignalInc = 1u << 3;
      constexpr uint16_t kOpWithSignalAdd = 1u << 4;
      if (op_mask & kOpFlush) {
        hdr.op = kWireOpFlush;
      } else if (op_mask & kOpGet) {
        hdr.op = kWireOpGet;
      } else if (op_mask & kOpVASignal) {
        hdr.op = kWireOpSignal;
      } else if (op_mask & (kOpWithSignalInc | kOpWithSignalAdd)) {
        hdr.op = kWireOpPutSignal;
      } else {
        hdr.op = kWireOpPut;
      }

      // Stage the header into the per-peer TX ring.
      uint32_t tx_slot = peer_out_[p].tx_slot++;
      size_t hdr_off = scratch->TxSlotOffset(p, tx_slot);
      void* hdr_dev = static_cast<uint8_t*>(scratch->device_ptr) + hdr_off;
      cudaError_t cerr = cudaMemcpy(hdr_dev, &hdr, sizeof(hdr),
                                    cudaMemcpyHostToDevice);
      if (cerr != cudaSuccess) {
        LOG(ERROR) << "TickOutbound: cudaMemcpy(header) failed: "
                   << cudaGetErrorString(cerr);
        break;
      }

      // 1) Send the header.
      auto hdr_send_or = sock->Send(
          hdr_off, sizeof(WireHeader), scratch->reg_handle);
      if (!hdr_send_or.ok()) {
        LOG(ERROR) << "TickOutbound: header Send failed: "
                   << hdr_send_or.status();
        break;
      }
      auto hdr_send = std::move(*hdr_send_or);
      if (auto s = WaitSendDone(*hdr_send, "outbound header"); !s.ok()) {
        LOG(ERROR) << "TickOutbound: header Send wait: " << s;
        break;
      }

      // 2) Send the payload (only when op carries data).
      if (hdr.size > 0 && hdr.op != kWireOpSignal && hdr.op != kWireOpFlush) {
        // src_handle is a MemHandle key produced by RegMrSym on this rank.
        MemHandle* mh = cc->lookup_memhandle(dec.src_handle);
        if (mh == nullptr || mh->local_reg == 0) {
          LOG(ERROR) << "TickOutbound: unknown src_handle " << dec.src_handle;
          break;
        }
        auto pay_send_or =
            sock->Send(dec.src_off, hdr.size, mh->local_reg);
        if (!pay_send_or.ok()) {
          LOG(ERROR) << "TickOutbound: payload Send failed: "
                     << pay_send_or.status();
          break;
        }
        auto pay_send = std::move(*pay_send_or);
        if (auto s = WaitSendDone(*pay_send, "outbound payload"); !s.ok()) {
          LOG(ERROR) << "TickOutbound: payload Send wait: " << s;
          break;
        }
      }

      ConsumeGfd(gfd);
      peer_out_[p].local_ci = ci + 1;
      ci_atomic->store(peer_out_[p].local_ci, std::memory_order_release);
    }
  }
}

namespace {
// v16: sharded mutex pool to avoid global serialization in cudaMemcpy
// signal RMW fallback. Hash (signal_handle ^ signal_off) into 64 shards
// so inbound threads with disjoint signal slots run concurrently.
constexpr size_t kV16SignalShards = 64;
absl::Mutex& V16SignalShard(uint64_t signal_handle, uint64_t signal_off) {
  static absl::Mutex shards[kV16SignalShards];
  return shards[(signal_handle ^ signal_off) & (kV16SignalShards - 1)];
}

// v18: lazy chunked GDR pin to serve PutSignal/Signal RMW for buffers
// that exceeded the upfront GDR pin cap. Keys: (mh*, chunk_off). Pins
// 1 MiB chunks on demand and caches forever (until plugin shutdown).
constexpr size_t kV18ChunkBits = 20;        // 1 MiB chunks
constexpr size_t kV18ChunkSize = 1ull << kV18ChunkBits;
constexpr size_t kV18ChunkMask = kV18ChunkSize - 1;
struct V18ChunkKey {
  MemHandle* mh;
  size_t chunk_off;
  bool operator==(const V18ChunkKey& o) const {
    return mh == o.mh && chunk_off == o.chunk_off;
  }
  template <typename H>
  friend H AbslHashValue(H h, const V18ChunkKey& k) {
    return H::combine(std::move(h), reinterpret_cast<uintptr_t>(k.mh),
                      k.chunk_off);
  }
};
absl::Mutex& V18ChunkMu() { static absl::Mutex mu; return mu; }
absl::flat_hash_map<V18ChunkKey, std::shared_ptr<GdrPinnedRegion>>&
V18ChunkMap() {
  static auto* m = new absl::flat_hash_map<
      V18ChunkKey, std::shared_ptr<GdrPinnedRegion>>;
  return *m;
}
// Returns host VA for the (mh, off) signal slot, lazily pinning a
// chunk if not yet cached. Returns nullptr on pin failure.
uint8_t* V18LazyHostAddr(MemHandle* mh, size_t off) {
  if (mh == nullptr || mh->base == nullptr) return nullptr;
  if (!GdrAvailable()) return nullptr;
  size_t chunk_off = off & ~kV18ChunkMask;
  size_t chunk_size = std::min(kV18ChunkSize, mh->bytes - chunk_off);
  std::shared_ptr<GdrPinnedRegion> chunk;
  {
    absl::MutexLock l(&V18ChunkMu());
    auto key = V18ChunkKey{mh, chunk_off};
    auto it = V18ChunkMap().find(key);
    if (it != V18ChunkMap().end()) {
      chunk = it->second;
    } else {
      void* chunk_base = static_cast<uint8_t*>(mh->base) + chunk_off;
      auto pin_or = GdrPinnedRegion::Create(chunk_base, chunk_size);
      if (pin_or.ok()) {
        chunk = std::make_shared<GdrPinnedRegion>(std::move(*pin_or));
        V18ChunkMap()[key] = chunk;
        static std::atomic<int> dbg{0};
        int n = dbg.fetch_add(1);
        if (n < 8) {
          LOG(INFO) << "v18: lazy chunk pin OK mh=" << mh
                    << " chunk_off=" << chunk_off
                    << " size=" << chunk_size << " (count #" << n << ")";
        }
      } else {
        static std::atomic<int> dbg_f{0};
        if (dbg_f.fetch_add(1) < 4) {
          LOG(WARNING) << "v18: lazy chunk pin FAILED mh=" << mh
                       << " chunk_off=" << chunk_off
                       << " size=" << chunk_size << ": "
                       << pin_or.status();
        }
        return nullptr;
      }
    }
  }
  return static_cast<uint8_t*>(chunk->host_map()) + (off - chunk_off);
}
}  // namespace

void ProxyProgress::RunInbound(size_t inbound_idx) {
  auto* cc = ctx_ ? ctx_->coll() : nullptr;
  auto* gpu = ctx_ ? ctx_->gpu_ctx() : nullptr;
  auto* scratch = ctx_ ? ctx_->scratch() : nullptr;
  if (cc == nullptr || gpu == nullptr || scratch == nullptr) return;
  // v13b: this thread is started by std::thread without inheriting the CUDA
  // device context; cudaMemcpy on inline-source paths and the existing
  // device->host header memcpy both require that current device == cc->dev().
  if (cc->dev() >= 0) {
    cudaSetDevice(cc->dev());
    cudaGetLastError();
  }
  auto* recv_sock = cc->inbound(inbound_idx);
  if (recv_sock == nullptr) return;
  // v9: this inbound thread's recv socket was accepted on a specific local
  // NIC; use that NIC's reg handle for both the header recv and the
  // payload recv (the dst MemHandle's per_nic_regs[nic_idx] must have
  // been populated by RegMrSym for the corresponding bufmgr).
  const int inbound_nic = cc->inbound_nic_idx(inbound_idx);
  dxs::Reg hdr_recv_reg = scratch->per_nic_reg_handles[inbound_nic];
  if (hdr_recv_reg == 0) hdr_recv_reg = scratch->reg_handle;

  // M6.2: each inbound thread owns a disjoint stride of the rx slot
  // ring so concurrent threads never DMA into the same 64B header slot.
  // Slice = floor(kRxSlots / num_inbound), starting at inbound_idx*slice.
  // (Per review S4: previously stride=1 caused thread 0 and thread 1 to
  // collide on slot 1 from the second iteration.)
  size_t num_in = cc->num_inbound();
  if (num_in == 0) num_in = 1;
  size_t slice = kRxSlots / num_in;
  if (slice == 0) slice = 1;
  size_t base = inbound_idx * slice;
  size_t rx_local = 0;

  while (!stop_.load(std::memory_order_acquire)) {
    size_t rx_slot = base + rx_local;
    if (rx_slot >= kRxSlots) rx_slot %= kRxSlots;
    size_t rx_off = scratch->RxSlotOffset(rx_slot);
    rx_local = (rx_local + 1) % slice;

    auto recv_or = recv_sock->RecvLinearized(rx_off, sizeof(WireHeader),
                                             hdr_recv_reg);
    if (!recv_or.ok()) {
      LOG(ERROR) << "RunInbound[" << inbound_idx
                 << "]: RecvLinearized(header) failed: " << recv_or.status();
      // v6 (fault-tolerance, NARROWED): only set g_has_error on hard
      // socket faults (not on the deadline-exceeded teardown noise we
      // get when a peer closes naturally). RecvLinearized failing
      // synchronously means the socket is wedged.
      SetGinError("RunInbound:RecvLinearized");
      break;
    }
    auto recv = std::move(*recv_or);
    auto sz_or = WaitRecvDone(*recv, "inbound header");
    if (!sz_or.ok()) {
      if (stop_.load(std::memory_order_acquire)) break;
      // v6 (fault-tolerance, NARROWED): WaitRecvDone failing is the
      // common teardown signal — peer closed its end of the socket
      // and our recv times out. Logging it is enough; setting
      // g_has_error here would make NCCL abort multi-config tests
      // (DeepEP test_pp recreates contexts between configs and reads
      // QueryLastError; a stale teardown error trips it).
      LOG(ERROR) << "RunInbound[" << inbound_idx
                 << "]: header recv wait: " << sz_or.status();
      break;
    }
    // Read the header back. Prefer GDR-mapped host VA (no cudaMemcpy
    // serialization). The same write-combining caveats apply: we just read
    // a 64-byte header that the NIC has already DMA'd in, so a load fence
    // is enough.
    WireHeader hdr;
    if (scratch->host_ptr != nullptr) {
      __asm__ __volatile__("lfence" ::: "memory");
      std::memcpy(&hdr, static_cast<uint8_t*>(scratch->host_ptr) + rx_off,
                  sizeof(hdr));
    } else {
      void* hdr_dev =
          static_cast<uint8_t*>(scratch->device_ptr) + rx_off;
      cudaError_t cerr =
          cudaMemcpy(&hdr, hdr_dev, sizeof(hdr), cudaMemcpyDeviceToHost);
      if (cerr != cudaSuccess) {
        LOG(ERROR) << "RunInbound: cudaMemcpy(header) failed: "
                   << cudaGetErrorString(cerr);
        SetGinError("RunInbound:cudaMemcpy-hdr");
        continue;
      }
    }
    if (hdr.magic != kWireMagic) {
      LOG(ERROR) << "RunInbound: bad magic 0x" << std::hex << hdr.magic;
      SetGinError("RunInbound:bad-magic");
      continue;
    }
    // PR3c-ii (β port, 2026-05-16): every inbound hdr piggybacks the
    // sender's view of "what I (the sender) have processed of YOUR ops"
    // in hdr.piggy_ack_high_water (sender wrote it in IputCommon or via
    // EmitAck). atomic_max into peer_acked_high_water[source_rank] gives
    // Test() the ack signal it needs to gate done=1. Skip if 0 sentinel.
    if (hdr.piggy_ack_high_water != 0) {
      if (auto* atom = cc->peer_acked_high_water(
              static_cast<int>(hdr.source_rank))) {
        uint64_t cur = atom->load(std::memory_order_relaxed);
        while (hdr.piggy_ack_high_water > cur &&
               !atom->compare_exchange_weak(cur, hdr.piggy_ack_high_water,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
          // cur was updated by another thread; loop and retry.
        }
      }
    }
    // v5 (M6.6): TLS dbg counter (was static std::atomic<int> rx_dbg).
    // The atomic fetch_add cost a cross-thread cache-line bounce on every
    // inbound header even after the first 50 firings, contended across
    // num_inbound() threads. TLS counters are zero-contention; each thread
    // independently logs its first 50 hdrs.
    thread_local int rx_dbg_tls = 0;
    if (rx_dbg_tls < 50) {
      ++rx_dbg_tls;
      LOG(INFO) << "RunInbound DBG #" << rx_dbg_tls
                << " op=" << hdr.op << " src=" << hdr.source_rank
                << " dst=" << hdr.dest_rank << " size=" << hdr.size
                << " sig_off=" << hdr.signal_off
                << " sig_val=" << hdr.signal_val;
    }

    // v11: cross-lane commit-order gate. Every IputCommon op stamps
    // hdr.wire_seq from a per-peer monotonic counter. For ops where order
    // matters w.r.t. signal visibility (PutSignal / Signal), we must wait
    // until ALL prior ops from the same source rank have committed before
    // we apply the signal RMW — otherwise a faster lane can leak a later
    // op's signal increment ahead of an earlier op's data write, and the
    // consumer reads stale memory. Pure Iput (no signal) needs to bump the
    // counter too so subsequent signals don't get stuck. Track the highest
    // wire_seq this thread has fully processed locally (not strictly
    // needed, but lets us short-circuit when the seq is in order).
    //
    // wire_seq == 0 means the sender did not assign a seq (legacy
    // TickOutbound path); skip ordering for those.
    auto wait_for_commit_turn = [cc](uint32_t src, uint64_t seq) {
      if (seq == 0) return;
      auto* atom = cc->recv_commit_seq(static_cast<int>(src));
      if (atom == nullptr) return;
      uint32_t spins = 0;
      // v25: wedge detector — log once per stuck wait so we can see if
      // dispatch-2 deadlock is rooted in a missed bump_commit_seq from a
      // prior op (sender's wire_seq is monotonic across dispatch calls;
      // any skipped bump permanently wedges all later ops on this src).
      auto t0 = std::chrono::steady_clock::now();
      bool warned = false;
      while (atom->load(std::memory_order_acquire) != seq) {
        if ((++spins & 4095) == 0) std::this_thread::yield();
        if (!warned) {
          auto dt = std::chrono::steady_clock::now() - t0;
          if (dt > std::chrono::seconds(5)) {
            LOG(WARNING) << "v25 wait_for_commit_turn WEDGED src=" << src
                         << " expected_seq=" << seq
                         << " atom=" << atom->load(std::memory_order_acquire);
            warned = true;
          }
        }
      }
    };
    auto bump_commit_seq = [cc](uint32_t src, uint64_t seq) {
      if (seq == 0) return;
      auto* atom = cc->recv_commit_seq(static_cast<int>(src));
      if (atom == nullptr) return;
      // Sanity: only this thread sees this exact wire_seq (sender allocates
      // them monotonically and routes one to a single lane), so a plain
      // store to seq+1 is enough.
      atom->store(seq + 1, std::memory_order_release);
    };

    switch (hdr.op) {
      case kWireOpPut:
      case kWireOpPutSignal: {
        // Validate dst only if there's a payload to land. SignalInc/VA
        // signal paths arrive with dst_handle=0 and size=0 — validation
        // would (incorrectly) reject them and skip the signal write below.
        if (hdr.size > 0) {
          // v6 (S8): lookup_memhandle masks bit 0 internally for the
          // shim-packed handle. v6 fault tolerance: a missing dst_handle
          // is now an error visible via QueryLastError instead of a
          // silent dropped op.
          MemHandle* dst = cc->lookup_memhandle(hdr.dst_handle);
          // v9: per-NIC reg lookup. dst->per_nic_regs[inbound_nic] must be
          // non-zero (RegMrSym populates every provisioned NIC).
          dxs::Reg dst_reg_for_nic =
              (dst != nullptr) ? dst->per_nic_regs[inbound_nic] : 0;
          if (dst == nullptr || dst_reg_for_nic == 0) {
            LOG(ERROR) << "RunInbound: bad dst_handle " << hdr.dst_handle
                       << " for size=" << hdr.size << " op=" << hdr.op
                       << " nic=" << inbound_nic;
            SetGinError("RunInbound:bad-dst-handle");
            // v25: drop op cleanly — gate then bump so later wire_seqs
            // from this src don't wedge in wait_for_commit_turn forever.
            wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
            bump_commit_seq(hdr.source_rank, hdr.wire_seq);
            break;
          }

          if ((hdr.flags & kWireFlagInlineSrc) && hdr.size > 0) {
            void* dst_dev =
                static_cast<uint8_t*>(dst->base) + hdr.dst_off;
            cudaError_t cerr =
                cudaMemcpy(dst_dev, hdr.inline_data, hdr.size,
                           cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) {
              LOG(ERROR) << "RunInbound: inline cudaMemcpy failed: "
                         << cudaGetErrorString(cerr) << " size=" << hdr.size
                         << " dev=" << cc->dev();
              SetGinError("RunInbound:inline-cudaMemcpy");
              // v25: drop op cleanly (see comment above).
              wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
              bump_commit_seq(hdr.source_rank, hdr.wire_seq);
              break;
            }
            wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
            if (hdr.op == kWireOpPutSignal && hdr.signal_handle != 0) {
              uint8_t* slot_b =
                  cc->signal_host_addr(hdr.signal_handle, hdr.signal_off);
              if (slot_b != nullptr) {
                auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(slot_b);
                absl::MutexLock l(cc->signal_mu());
                *slot_u64 += hdr.signal_val;
                // v19: clflushopt + mfence to flush WC store buffer to PCIe
            __asm__ __volatile__("clflushopt (%0); mfence" :: "r"(reinterpret_cast<volatile void*>(slot_u64)) : "memory");
              }
            }
            bump_commit_seq(hdr.source_rank, hdr.wire_seq);
            break;
          }
          auto p_or = recv_sock->RecvLinearized(hdr.dst_off, hdr.size,
                                                dst_reg_for_nic);
          if (!p_or.ok()) {
            LOG(ERROR) << "RunInbound: payload RecvLinearized failed: "
                       << p_or.status();
            SetGinError("RunInbound:payload-RecvLinearized");
            // v25: drop op cleanly (see comment above).
            wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
            bump_commit_seq(hdr.source_rank, hdr.wire_seq);
            break;
          }
          auto p = std::move(*p_or);
          auto p_sz = WaitRecvDone(*p, "inbound payload");
          if (!p_sz.ok()) {
            LOG(ERROR) << "RunInbound: payload recv wait: " << p_sz.status();
            SetGinError("RunInbound:payload-wait");
            // v25: drop op cleanly (see comment above).
            wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
            bump_commit_seq(hdr.source_rank, hdr.wire_seq);
            break;
          }
          // v5 (M6.6): TLS dbg counter — see rx_dbg_tls comment.
          thread_local int pl_dbg_tls = 0;
          if (pl_dbg_tls < 16) {
            ++pl_dbg_tls;
            LOG(INFO) << "RunInbound payload OK off=" << hdr.dst_off
                      << " size=" << hdr.size << " got=" << *p_sz
                      << " dst_handle=0x" << std::hex << hdr.dst_handle
                      << std::dec;
          }
        }
        // v11: cross-lane gate AFTER payload Recv, BEFORE signal RMW (or
        // before bumping the seq for pure Puts). Lanes complete payloads
        // in parallel; only the post-commit step is serialized.
        wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
        if (hdr.op == kWireOpPutSignal) {
          // v7 (S3 fix): per-CollComm signal_mu_ serialises the
          // load+store+sfence RMW. The slot lives in GDR write-
          // combining memory; LOCK XADD does not commit visibly to the
          // GPU view (v6 attempt failed). Mutex makes multi-writer
          // accumulation correct (DeepEP dispatch's N->1 reduction).
          uint8_t* slot_b =
              cc->signal_host_addr(hdr.signal_handle, hdr.signal_off);
          if (slot_b != nullptr) {
            auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(slot_b);
            absl::MutexLock l(cc->signal_mu());
            uint64_t prev = *slot_u64;
            *slot_u64 = prev + hdr.signal_val;
            // v19: clflushopt + mfence to flush WC store buffer to PCIe
            __asm__ __volatile__("clflushopt (%0); mfence" :: "r"(reinterpret_cast<volatile void*>(slot_u64)) : "memory");
          } else {
            // v20: GPU atomicAdd kernel — bypasses all host-side RMW
            // emulation (GDR pin, cudaMemcpy fallback, lazy chunk pin).
            // Hardware-atomic; no mutex; no BAR1 contention.
            MemHandle* sig_mh = cc->lookup_memhandle(hdr.signal_handle);
            if (sig_mh != nullptr && (sig_mh->ptr_type & NCCL_PTR_CUDA) &&
                sig_mh->base != nullptr &&
                hdr.signal_off + sizeof(uint64_t) <= sig_mh->bytes) {
              void* sig_dev =
                  static_cast<uint8_t*>(sig_mh->base) + hdr.signal_off;
              cudaStream_t stream = cc->signal_stream();
              cudaError_t cerr =
                  LaunchSignalAdd(sig_dev, hdr.signal_val, stream);
              if (cerr == cudaSuccess) {
                cudaStreamSynchronize(stream);
                static std::atomic<int> dbg_v20{0};
                int n = dbg_v20.fetch_add(1);
                if (n < 4) {
                  LOG(INFO) << "v20: PutSignal atomicAdd kernel OK "
                            << "sig_h=" << hdr.signal_handle
                            << " off=" << hdr.signal_off
                            << " val+=" << hdr.signal_val
                            << " (count #" << n << ")";
                }
                bump_commit_seq(hdr.source_rank, hdr.wire_seq);
                break;
              } else {
                static std::atomic<int> dbg_v20_f{0};
                if (dbg_v20_f.fetch_add(1) < 4) {
                  LOG(WARNING) << "v20: PutSignal atomicAdd kernel launch "
                               << "failed: " << cudaGetErrorString(cerr);
                }
                // Fall through to old fallbacks if kernel launch fails.
              }
            }
            // (legacy v18+v14 fallbacks kept for non-CUDA signal targets)
            MemHandle* sig_mh_legacy = cc->lookup_memhandle(hdr.signal_handle);
            sig_mh = sig_mh_legacy;
            if (sig_mh != nullptr) {
              uint8_t* lazy = V18LazyHostAddr(sig_mh, hdr.signal_off);
              if (lazy != nullptr) {
                auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(lazy);
                absl::MutexLock l(&V16SignalShard(hdr.signal_handle,
                                                  hdr.signal_off));
                uint64_t prev = *slot_u64;
                *slot_u64 = prev + hdr.signal_val;
                // v19: clflushopt + mfence to flush WC store buffer to PCIe
            __asm__ __volatile__("clflushopt (%0); mfence" :: "r"(reinterpret_cast<volatile void*>(slot_u64)) : "memory");
                bump_commit_seq(hdr.source_rank, hdr.wire_seq);
                break;
              }
            }
            if (sig_mh != nullptr && sig_mh->base != nullptr &&
                hdr.signal_off + sizeof(uint64_t) <= sig_mh->bytes) {
              void* sig_dev = static_cast<uint8_t*>(sig_mh->base) +
                              hdr.signal_off;
              absl::MutexLock l(&V16SignalShard(hdr.signal_handle,
                                                hdr.signal_off));
              uint64_t prev = 0;
              cudaError_t cerr1 = cudaMemcpy(&prev, sig_dev,
                                             sizeof(uint64_t),
                                             cudaMemcpyDeviceToHost);
              if (cerr1 != cudaSuccess) {
                LOG(ERROR) << "v14 PutSignal D2H cudaMemcpy: "
                           << cudaGetErrorString(cerr1);
                SetGinError("RunInbound:PutSignal:cudaMemcpy-D2H");
              } else {
                uint64_t newv = prev + hdr.signal_val;
                cudaError_t cerr2 = cudaMemcpy(sig_dev, &newv,
                                               sizeof(uint64_t),
                                               cudaMemcpyHostToDevice);
                if (cerr2 != cudaSuccess) {
                  LOG(ERROR) << "v14 PutSignal H2D cudaMemcpy: "
                             << cudaGetErrorString(cerr2);
                  SetGinError("RunInbound:PutSignal:cudaMemcpy-H2D");
                } else {
                  static std::atomic<int> dbg_v14_ps{0};
                  int n = dbg_v14_ps.fetch_add(1);
                  if (n < 4) {
                    LOG(INFO) << "v14: PutSignal cudaMemcpy fallback OK "
                              << "sig_h=" << hdr.signal_handle
                              << " off=" << hdr.signal_off
                              << " val+=" << hdr.signal_val
                              << " (count #" << n << ")";
                  }
                }
              }
            } else {
              LOG(ERROR) << "RunInbound: PutSignal no signal map "
                            "(sig_h=" << hdr.signal_handle
                         << " off=" << hdr.signal_off
                         << " mh=" << (sig_mh ? "found" : "null")
                         << " primary_size=" << cc->signal_size_bytes()
                         << ")";
              SetGinError("RunInbound:PutSignal:no-map");
            }
          }
        }
        bump_commit_seq(hdr.source_rank, hdr.wire_seq);
        break;
      }
      case kWireOpSignal: {
        // v11: standalone Signal carries no payload; gate before RMW.
        wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
        // v7 (S3 fix): same per-CollComm signal_mu_ as PutSignal above.
        uint8_t* slot_b =
            cc->signal_host_addr(hdr.signal_handle, hdr.signal_off);
        if (slot_b != nullptr) {
          auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(slot_b);
          absl::MutexLock l(cc->signal_mu());
          uint64_t prev = *slot_u64;
          *slot_u64 = prev + hdr.signal_val;
          // v19: clflushopt + mfence to flush WC store buffer to PCIe
            __asm__ __volatile__("clflushopt (%0); mfence" :: "r"(reinterpret_cast<volatile void*>(slot_u64)) : "memory");
        } else {
          // v20: GPU atomicAdd kernel — bypasses all host-side RMW.
          MemHandle* sig_mh = cc->lookup_memhandle(hdr.signal_handle);
          if (sig_mh != nullptr && (sig_mh->ptr_type & NCCL_PTR_CUDA) &&
              sig_mh->base != nullptr &&
              hdr.signal_off + sizeof(uint64_t) <= sig_mh->bytes) {
            void* sig_dev =
                static_cast<uint8_t*>(sig_mh->base) + hdr.signal_off;
            cudaStream_t stream = cc->signal_stream();
            cudaError_t cerr =
                LaunchSignalAdd(sig_dev, hdr.signal_val, stream);
            if (cerr == cudaSuccess) {
              cudaStreamSynchronize(stream);
              static std::atomic<int> dbg_v20s{0};
              int n = dbg_v20s.fetch_add(1);
              if (n < 4) {
                LOG(INFO) << "v20: Signal atomicAdd kernel OK "
                          << "sig_h=" << hdr.signal_handle
                          << " off=" << hdr.signal_off
                          << " (count #" << n << ")";
              }
              bump_commit_seq(hdr.source_rank, hdr.wire_seq);
              break;
            }
          }
          // (legacy fallbacks below for non-CUDA targets / launch failure)
          if (sig_mh != nullptr) {
            uint8_t* lazy = V18LazyHostAddr(sig_mh, hdr.signal_off);
            if (lazy != nullptr) {
              auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(lazy);
              absl::MutexLock l(&V16SignalShard(hdr.signal_handle,
                                                hdr.signal_off));
              uint64_t prev = *slot_u64;
              *slot_u64 = prev + hdr.signal_val;
              // v19: clflushopt + mfence to flush WC store buffer to PCIe
            __asm__ __volatile__("clflushopt (%0); mfence" :: "r"(reinterpret_cast<volatile void*>(slot_u64)) : "memory");
              bump_commit_seq(hdr.source_rank, hdr.wire_seq);
              break;
            }
          }
          if (sig_mh != nullptr && sig_mh->base != nullptr &&
              hdr.signal_off + sizeof(uint64_t) <= sig_mh->bytes) {
            void* sig_dev = static_cast<uint8_t*>(sig_mh->base) +
                            hdr.signal_off;
            absl::MutexLock l(&V16SignalShard(hdr.signal_handle,
                                              hdr.signal_off));
            uint64_t prev = 0;
            cudaError_t cerr1 = cudaMemcpy(&prev, sig_dev,
                                           sizeof(uint64_t),
                                           cudaMemcpyDeviceToHost);
            if (cerr1 != cudaSuccess) {
              LOG(ERROR) << "v14 Signal D2H cudaMemcpy: "
                         << cudaGetErrorString(cerr1);
              SetGinError("RunInbound:Signal:cudaMemcpy-D2H");
            } else {
              uint64_t newv = prev + hdr.signal_val;
              cudaError_t cerr2 = cudaMemcpy(sig_dev, &newv,
                                             sizeof(uint64_t),
                                             cudaMemcpyHostToDevice);
              if (cerr2 != cudaSuccess) {
                LOG(ERROR) << "v14 Signal H2D cudaMemcpy: "
                           << cudaGetErrorString(cerr2);
                SetGinError("RunInbound:Signal:cudaMemcpy-H2D");
              } else {
                static std::atomic<int> dbg_v14_sg{0};
                int n = dbg_v14_sg.fetch_add(1);
                if (n < 4) {
                  LOG(INFO) << "v14: Signal cudaMemcpy fallback OK "
                            << "sig_h=" << hdr.signal_handle
                            << " off=" << hdr.signal_off
                            << " val+=" << hdr.signal_val
                            << " (count #" << n << ")";
                }
              }
            }
          } else {
            LOG(ERROR) << "RunInbound: Signal no signal map (sig_h="
                       << hdr.signal_handle
                       << " off=" << hdr.signal_off
                       << " mh=" << (sig_mh ? "found" : "null")
                       << " primary_size=" << cc->signal_size_bytes()
                       << ")";
            SetGinError("RunInbound:Signal:no-map");
          }
        }
        bump_commit_seq(hdr.source_rank, hdr.wire_seq);
        break;
      }
      case kWireOpAck:
        // PR3c-iii (β port, 2026-05-16): standalone async ACK from peer.
        // hdr.piggy_ack_high_water was already consumed before the switch
        // (atomic_max into peer_acked_high_water). No payload, no signal
        // RMW, no commit_seq bump (sender stamped wire_seq=0). Done.
        break;
      case kWireOpFlush:
        // Marker only — flush is initiated by source's progress thread.
        // Still bump seq so subsequent ops aren't stuck.
        wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
        bump_commit_seq(hdr.source_rank, hdr.wire_seq);
        break;
      case kWireOpGet:
      case kWireOpGetReply:
        // TODO(M3.1): Get path. Skipped for the first MVP.
        LOG(WARNING) << "RunInbound: WireOpGet not yet implemented";
        wait_for_commit_turn(hdr.source_rank, hdr.wire_seq);
        bump_commit_seq(hdr.source_rank, hdr.wire_seq);
        break;
      default:
        LOG(ERROR) << "RunInbound: unknown WireOp " << hdr.op;
        SetGinError("RunInbound:unknown-op");
    }
  }
}

}  // namespace fastrak::gin
