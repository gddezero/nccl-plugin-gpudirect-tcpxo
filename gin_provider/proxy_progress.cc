/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/proxy_progress.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>

#include "absl/log/log.h"
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

// Spin until SendOp completes or timeout.
absl::Status WaitSendDone(dxs::SendOpInterface& op, absl::string_view what) {
  auto deadline = absl::Now() + kSendTimeout;
  while (true) {
    auto s = op.Test();
    if (s.has_value()) {
      if (!s->ok()) return *s;
      return absl::OkStatus();
    }
    if (absl::Now() > deadline) {
      return absl::DeadlineExceededError(
          absl::StrCat("WaitSendDone timeout: ", what));
    }
    std::this_thread::yield();
  }
}

absl::StatusOr<uint64_t> WaitRecvDone(dxs::LinearizedRecvOpInterface& op,
                                      absl::string_view what) {
  auto deadline = absl::Now() + kSendTimeout;
  while (true) {
    auto s = op.Test();
    if (s.has_value()) return *s;
    if (absl::Now() > deadline) {
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

void ProxyProgress::RunInbound(size_t inbound_idx) {
  auto* cc = ctx_ ? ctx_->coll() : nullptr;
  auto* gpu = ctx_ ? ctx_->gpu_ctx() : nullptr;
  auto* scratch = ctx_ ? ctx_->scratch() : nullptr;
  if (cc == nullptr || gpu == nullptr || scratch == nullptr) return;
  auto* recv_sock = cc->inbound(inbound_idx);
  if (recv_sock == nullptr) return;

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
                                             scratch->reg_handle);
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
          if (dst == nullptr || dst->local_reg == 0) {
            LOG(ERROR) << "RunInbound: bad dst_handle " << hdr.dst_handle
                       << " for size=" << hdr.size << " op=" << hdr.op;
            SetGinError("RunInbound:bad-dst-handle");
            break;
          }
          auto p_or = recv_sock->RecvLinearized(hdr.dst_off, hdr.size,
                                                dst->local_reg);
          if (!p_or.ok()) {
            LOG(ERROR) << "RunInbound: payload RecvLinearized failed: "
                       << p_or.status();
            SetGinError("RunInbound:payload-RecvLinearized");
            break;
          }
          auto p = std::move(*p_or);
          auto p_sz = WaitRecvDone(*p, "inbound payload");
          if (!p_sz.ok()) {
            LOG(ERROR) << "RunInbound: payload recv wait: " << p_sz.status();
            SetGinError("RunInbound:payload-wait");
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
            __asm__ __volatile__("sfence" ::: "memory");
          } else {
            LOG(ERROR) << "RunInbound: PutSignal no signal map "
                          "(sig_h=0x" << std::hex << hdr.signal_handle
                       << std::dec << " off=" << hdr.signal_off
                       << " primary_size=" << cc->signal_size_bytes() << ")";
            SetGinError("RunInbound:PutSignal:no-map");
          }
        }
        break;
      }
      case kWireOpSignal: {
        // v7 (S3 fix): same per-CollComm signal_mu_ as PutSignal above.
        uint8_t* slot_b =
            cc->signal_host_addr(hdr.signal_handle, hdr.signal_off);
        if (slot_b != nullptr) {
          auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(slot_b);
          absl::MutexLock l(cc->signal_mu());
          uint64_t prev = *slot_u64;
          *slot_u64 = prev + hdr.signal_val;
          __asm__ __volatile__("sfence" ::: "memory");
        } else {
          LOG(ERROR) << "RunInbound: Signal no signal map (sig_h=0x"
                     << std::hex << hdr.signal_handle << std::dec
                     << " off=" << hdr.signal_off
                     << " primary_size=" << cc->signal_size_bytes() << ")";
          SetGinError("RunInbound:Signal:no-map");
        }
        break;
      }
      case kWireOpFlush:
        // Marker only — flush is initiated by source's progress thread.
        break;
      case kWireOpGet:
      case kWireOpGetReply:
        // TODO(M3.1): Get path. Skipped for the first MVP.
        LOG(WARNING) << "RunInbound: WireOpGet not yet implemented";
        break;
      default:
        LOG(ERROR) << "RunInbound: unknown WireOp " << hdr.op;
        SetGinError("RunInbound:unknown-op");
    }
  }
}

}  // namespace fastrak::gin
