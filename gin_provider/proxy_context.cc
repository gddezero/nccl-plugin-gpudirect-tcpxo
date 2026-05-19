/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/proxy_context.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "dxs/client/oss/status_macros.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/proxy_progress.h"
#include "gin_provider/scratch_pool.h"
#include "gin_provider/wire_protocol.h"

namespace fastrak::gin {

// v6 (S5): plugin-wide error flag. Storage lives here so both
// plugin_main.cc and proxy_progress.cc can SetGinError() without
// circular includes. Read by QueryLastError; cleared on Init().
std::atomic<bool> g_has_error{false};

// v9: multi-NIC opt-in. See header comment.
bool MultiNicEnabled() {
  static bool cached = []() {
    const char* v = std::getenv("NCCL_GIN_MULTI_NIC");
    if (v == nullptr || *v == 0) return false;
    bool on = (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' ||
               v[0] == 't' || v[0] == 'T');
    LOG(INFO) << "NCCL_GIN_MULTI_NIC=" << v << " -> "
              << (on ? "ENABLED (multi-NIC fan-out)"
                     : "disabled (single-NIC behaviour)");
    return on;
  }();
  return cached;
}

// M6.2: env-overridable per-peer fan-out. Cached so we only parse once.
int FanoutPerPeer() {
  static int cached = []() {
    const char* v = std::getenv("NCCL_GIN_FANOUT");
    if (v == nullptr || *v == 0) return kDefaultFanout;
    int n = std::atoi(v);
    if (n <= 0) return kDefaultFanout;
    if (n > 32) n = 32;  // sanity clamp
    LOG(INFO) << "NCCL_GIN_FANOUT=" << n << " (per-peer parallel send sockets)";
    return n;
  }();
  return cached;
}

// ---- CollComm ----

CollComm::~CollComm() {
  // v6 (S7 attempted, REVERTED): the original review flagged a UAF
  // window between CloseColl and DestroyContext where the progress
  // thread can still touch destroyed peers_/sockets/memhandles_. We
  // tried to fix it by tracking attached GinCtx and synchronously
  // joining their progress threads here — but inbound/outbound
  // WaitRecvDone has a 60s deadline that doesn't observe stop_, so
  // ~CollComm started blocking CloseColl for up to 60s and broke
  // multi-config NCCL flows (DeepEP test_pp crashed with CUDA
  // launch failure after the first config when CloseColl took too
  // long). Proper fix needs either (a) a way to abort an in-flight
  // dxs::RecvLinearized at teardown time or (b) separating "signal
  // stop" from "join" so ~CollComm only signals. For now we keep the
  // v5 best-effort behaviour: progress threads see destroyed memory
  // briefly and may log/crash, but CloseColl returns promptly.
  // ctxs_ list is kept for future use (currently unread).
  peers_.clear();
}

void CollComm::register_ctx(GinCtx* ctx) {
  absl::MutexLock l(&ctx_mu_);
  ctxs_.push_back(ctx);
}

void CollComm::unregister_ctx(GinCtx* ctx) {
  absl::MutexLock l(&ctx_mu_);
  ctxs_.erase(std::remove(ctxs_.begin(), ctxs_.end(), ctx), ctxs_.end());
}

absl::Status CollComm::Init(
    int dev, uint8_t fastrak_idx, std::string nic_ip, int nranks, int rank,
    dxs::DxsClientInterface* absl_nonnull dxs,
    tcpdirect::BufferManagerClientInterface* absl_nonnull buf,
    const std::array<dxs::DxsClientInterface*, kMaxNics>& per_nic_dxs,
    const std::array<tcpdirect::BufferManagerClientInterface*, kMaxNics>&
        per_nic_bufmgr,
    const std::array<std::string, kMaxNics>& nic_ips_by_idx) {
  dev_ = dev;
  fastrak_idx_ = fastrak_idx;
  nic_ip_ = std::move(nic_ip);
  nranks_ = nranks;
  rank_ = rank;
  dxs_ = dxs;
  buf_ = buf;
  per_nic_dxs_ = per_nic_dxs;
  per_nic_bufmgr_ = per_nic_bufmgr;
  nic_ips_by_idx_ = nic_ips_by_idx;
  n_provisioned_nics_ = 0;
  for (int i = 0; i < kMaxNics; ++i) {
    if (per_nic_dxs_[i] != nullptr && per_nic_bufmgr_[i] != nullptr) {
      ++n_provisioned_nics_;
    }
  }
  peers_.resize(nranks);
  // v11: per-source-rank receive-side commit-order counters, 1-based to
  // match PeerConn::wire_seq_next start. Allocated as a unique_ptr array
  // because std::atomic isn't move/copy.
  recv_commit_seq_n_ = static_cast<size_t>(nranks);
  recv_commit_seq_ =
      std::make_unique<std::atomic<uint64_t>[]>(recv_commit_seq_n_);
  for (size_t i = 0; i < recv_commit_seq_n_; ++i) {
    recv_commit_seq_[i].store(1, std::memory_order_relaxed);
  }
  // PR3c-ii (β port, 2026-05-16): per-dst peer ack high water. Init 0
  // = sentinel "no ack from this dst yet"; Test() bypasses gate then.
  peer_acked_high_water_n_ = static_cast<size_t>(nranks);
  peer_acked_high_water_ =
      std::make_unique<std::atomic<uint64_t>[]>(peer_acked_high_water_n_);
  for (size_t i = 0; i < peer_acked_high_water_n_; ++i) {
    peer_acked_high_water_[i].store(0, std::memory_order_relaxed);
  }
  LOG(INFO) << "CollComm::Init dev=" << dev << " fastrak_idx=" << (int)fastrak_idx
            << " nic_ip=" << nic_ip_ << " rank=" << rank << "/" << nranks
            << " n_provisioned_nics=" << n_provisioned_nics_;
  return absl::OkStatus();
}

void CollComm::set_peer(int peer_rank, PeerConn conn) {
  peers_.at(peer_rank) = std::move(conn);
}

PeerConn* CollComm::peer(int peer_rank) {
  if (peer_rank < 0 || static_cast<size_t>(peer_rank) >= peers_.size()) {
    return nullptr;
  }
  return &peers_[peer_rank];
}

uint64_t CollComm::register_memhandle(MemHandle h) {
  absl::MutexLock l(&mh_mu_);
  // Use ordinal (registration sequence) as the key. Both ranks invoke
  // regMrSym in the same logical order for the same set of buffers within
  // a CollComm (NCCL's symmetric memory init is deterministic, and DeepEP
  // calls pp/dispatch buffer registration symmetrically across ranks), so
  // ordinal N on rank A maps to the corresponding buffer registered as
  // ordinal N on rank B.  This avoids the need to OOB-exchange peer
  // mhandles for non-symmetric (separately cudaMalloc'd) buffers like the
  // ElasticBuffer pp scratch (different VA per rank).
  // Shift by 1 to leave bit 0 free (NCCL packs srcHandle/dstHandle as a
  // 63-bit field with bit 0 used as a flag).
  uint64_t key = (next_mh_key_++) << 1;
  void* base = h.base;
  size_t bytes = h.bytes;
  uint64_t local_reg = h.local_reg;
  memhandles_.insert_or_assign(key, std::move(h));
  static std::atomic<int> reg_dbg{0};
  if (reg_dbg.fetch_add(1) < 32) {
    LOG(INFO) << "register_memhandle: rank=" << rank_
              << " key=0x" << std::hex << key
              << " base=" << base
              << " bytes=" << std::dec << bytes
              << " local_reg=" << local_reg
              << " total_keys=" << memhandles_.size();
  }
  return key;
}

MemHandle* CollComm::lookup_memhandle(uint64_t key) {
  absl::MutexLock l(&mh_mu_);
  auto it = memhandles_.find(key);
  if (it != memhandles_.end()) return &it->second;
  // Range-based fallback: NCCL may pass a pointer INTO a registered region
  // (e.g. base + offset), not the exact base. Find an entry whose
  // [base, base + bytes) contains `key`.
  for (auto& kv : memhandles_) {
    uint64_t base_k = kv.first;
    uint64_t end_k = base_k + kv.second.bytes;
    if (key >= base_k && key < end_k) return &kv.second;
  }
  static std::atomic<int> lk_dbg{0};
  if (lk_dbg.fetch_add(1) < 16) {
    LOG(WARNING) << "lookup_memhandle MISS rank=" << rank_
                 << " key=0x" << std::hex << key
                 << " (decimal=" << std::dec << key << ")"
                 << " known_keys=" << memhandles_.size();
    int n = 0;
    for (auto& kv : memhandles_) {
      if (n++ >= 8) { LOG(WARNING) << "  ... more keys omitted"; break; }
      LOG(WARNING) << "  known: key=0x" << std::hex << kv.first
                   << " base=" << kv.second.base
                   << " bytes=" << std::dec << kv.second.bytes;
    }
  }
  return nullptr;
}

void CollComm::erase_memhandle(uint64_t key) {
  absl::MutexLock l(&mh_mu_);
  memhandles_.erase(key);
}

void CollComm::set_signal_buffer(GdrPinnedRegion region) {
  signal_region_ = std::move(region);
  signal_host_map_ = static_cast<uint64_t*>(signal_region_.host_map());
  signal_size_bytes_ = signal_region_.size();
  LOG(INFO) << "CollComm: signal buffer attached via GDR, host_map="
            << signal_host_map_ << " size=" << signal_size_bytes_ << " bytes";
}

uint8_t* CollComm::signal_host_addr(uint64_t signal_handle,
                                    uint64_t signal_off) {
  // M6: prefer per-MemHandle GDR map when sender provided an explicit
  // signal_handle. This is what DeepEP dispatch needs — its scratch
  // signal area is a regular cudaMalloc'd buffer (not the FORCE_SO
  // signalsDev), so the buffer-specific GDR pin we created in RegMrSym
  // is the only host-writable view of it.
  if (signal_handle != 0) {
    MemHandle* mh = lookup_memhandle(signal_handle);
    if (mh != nullptr && mh->bytes >= signal_off + sizeof(uint64_t)) {
      void* hm = mh->gdr_host_map();
      if (hm != nullptr) {
        return static_cast<uint8_t*>(hm) + signal_off;
      }
      static std::atomic<int> dbg{0};
      if (dbg.fetch_add(1) < 8) {
        LOG(WARNING) << "v21 signal_host_addr: mhandle 0x" << std::hex
                     << signal_handle
                     << " has no GDR pin (size=" << std::dec << mh->bytes
                     << " base=" << mh->base
                     << "); returning nullptr to force GPU atomicAdd path "
                     << "(was v20 buggy fallthrough to signal_host_map_)";
      }
    }
    // v21: do NOT fall through to signal_host_map_ here. That region is
    // the NCCL-internal barrier signal area; writing DeepEP signal RMW
    // there silently corrupts the wrong memory and DeepEP's
    // net.readSignal() reads the real device slot which stays zero.
    // Return nullptr -> caller routes to v20 GPU atomicAdd kernel which
    // writes directly to mh->base + signal_off (the actual signal slot).
    return nullptr;
  }
  // signal_handle == 0: pure NCCL-internal barrier signal path.
  if (signal_host_map_ != nullptr &&
      signal_off + sizeof(uint64_t) <= signal_size_bytes_) {
    return reinterpret_cast<uint8_t*>(signal_host_map_) + signal_off;
  }
  return nullptr;
}



cudaStream_t CollComm::signal_stream() {
  // v20: lazy-init under signal_mu_; CUDA streams are cheap.
  absl::MutexLock l(&signal_mu_);
  if (signal_stream_ == nullptr) {
    if (dev_ >= 0) cudaSetDevice(dev_);
    cudaError_t cerr = cudaStreamCreateWithFlags(&signal_stream_,
                                                  cudaStreamNonBlocking);
    if (cerr != cudaSuccess) {
      LOG(ERROR) << "v20: cudaStreamCreate failed: "
                 << cudaGetErrorString(cerr);
      signal_stream_ = nullptr;
    }
  }
  return signal_stream_;
}

// ---- GinCtx ----

GinCtx::~GinCtx() {
  StopProgress();
  // v6 (S7 attempted, REVERTED): no unregister_ctx call since Init no
  // longer registers. See CollComm::~CollComm comment.
}

absl::Status GinCtx::Init(uint32_t queue_size, int n_counters, int n_signals) {
  if (coll_ == nullptr) {
    return absl::FailedPreconditionError("GinCtx::Init: coll_ is null");
  }
  ASSIGN_OR_RETURN(
      gpu_ctx_, AllocateProxyGpuCtx(coll_->nranks(), queue_size, n_counters,
                                    n_signals));
  // v9: scratch pool registers on every provisioned NIC so sender lanes
  // spread across NICs each have a valid header reg.
  ASSIGN_OR_RETURN(
      scratch_,
      AllocateScratchPool(coll_->nranks(), coll_->per_nic_bufmgr_array(),
                          static_cast<int>(coll_->fastrak_idx())));
  // v6 (S7 attempted, REVERTED): coll_->register_ctx(this) removed —
  // see CollComm::~CollComm comment for why the synchronous-join fix
  // had to be backed out. The function is kept declared for the future
  // attempt that needs an abortable WaitRecvDone.
  return absl::OkStatus();
}

// PR3c-iii (β port, 2026-05-16): emit a standalone kWireOpAck to peer.
// Best-effort, idempotent. Called by ProxyProgress::TickOutbound on every
// progress tick. Only emits when recv_commit_seq[peer] > last_ack_emitted_to_peer
// (avoid ACK flood). Prunes pending_ack_sops of completed Sends.
//
// This out-of-band timer ACK channel is what makes the PR3c-ii Test()
// gate viable: even if both sides' Test() stall waiting for peer ACK,
// the progress thread keeps emitting ACKs independent of payload traffic,
// eventually unblocking both Tests. AWS uses a fancier flush_stale_acks
// with GIN_ACK_MAX_AGE timer; we just emit-on-change which is simpler and
// rate-limited naturally by recv_commit_seq advancement.
absl::Status CollComm::EmitAck(int peer_rank, ScratchPool* sp) {
  if (peer_rank < 0 || peer_rank >= static_cast<int>(peers_.size())) {
    return absl::OkStatus();  // ignore bad rank
  }
  if (sp == nullptr) return absl::OkStatus();
  PeerConn* pc = &peers_[peer_rank];

  // PR3c-iii fix (2026-05-16): serialize all EmitAck access to per-peer
  // pending_ack_sops + last_ack_emitted_to_peer. Multiple GinCtx per
  // CollComm (4 in HT EP=16 test) means 4 progress threads call this
  // concurrently on the same peer; without lock vector erase + push_back
  // race causes SIGSEGV. Hold for whole function (small body, low contention).
  absl::MutexLock ack_lock(&ack_mu_);

  // Prune completed in-flight ACKs first.
  if (!pc->pending_ack_sops.empty()) {
    auto& vec = pc->pending_ack_sops;
    vec.erase(std::remove_if(vec.begin(), vec.end(),
                              [](std::unique_ptr<dxs::SendOpInterface>& op) {
                                if (op == nullptr) return true;
                                auto r = op->Test();
                                // r.has_value() means Test returned (done
                                // or error). Either way, sop completed.
                                return r.has_value();
                              }),
              vec.end());
  }

  if (pc->send_socks.empty()) return absl::OkStatus();  // not connected
  auto* sock = pc->send_socks[0].get();
  if (sock == nullptr) return absl::OkStatus();

  // Check if we have new info to ACK.
  auto* atom = recv_commit_seq(peer_rank);
  if (atom == nullptr) return absl::OkStatus();
  uint64_t hw = atom->load(std::memory_order_acquire);
  uint64_t last = pc->last_ack_emitted_to_peer.load(std::memory_order_acquire);
  // Initial recv_commit_seq is 1 (1-based); skip if hw==1 (no progress) and
  // last==0 (never emitted before).
  if (hw <= last) return absl::OkStatus();

  // Cap pending in-flight ACKs to avoid unbounded growth in extreme tail.
  if (pc->pending_ack_sops.size() > 16) return absl::OkStatus();

  // Build a standalone ACK WireHeader. wire_seq=0 → skip cross-lane
  // ordering (kWireOpAck has no commit semantics). piggy_ack_high_water
  // carries the high water; receiver's RunInbound atomic_max'es it into
  // its peer_acked_high_water[my rank].
  WireHeader hdr{};
  hdr.magic = kWireMagic;
  hdr.op = static_cast<uint16_t>(kWireOpAck);
  hdr.source_rank = static_cast<uint32_t>(rank_);
  hdr.dest_rank = static_cast<uint32_t>(peer_rank);
  hdr.size = 0;
  hdr.wire_seq = 0;
  hdr.piggy_ack_high_water = hw;

  // Stage in TX scratch (use a per-thread slot ring to avoid conflict
  // with IputCommon's tls_tx_seq).
  static thread_local uint32_t tls_ack_tx_seq[64] = {0};
  uint32_t slot_idx =
      tls_ack_tx_seq[peer_rank % 64]++ &
      static_cast<uint32_t>(kTxSlotsPerPeer - 1);
  size_t hdr_off = sp->TxSlotOffset(peer_rank, slot_idx);
  if (sp->host_ptr != nullptr) {
    std::memcpy(static_cast<uint8_t*>(sp->host_ptr) + hdr_off, &hdr,
                sizeof(hdr));
    __asm__ __volatile__("sfence" ::: "memory");
  } else {
    cudaError_t cerr = cudaMemcpy(
        static_cast<uint8_t*>(sp->device_ptr) + hdr_off, &hdr, sizeof(hdr),
        cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
      return absl::InternalError("EmitAck: cudaMemcpy failed");
    }
  }

  // Use lane 0's NIC reg (consistent with TickOutbound legacy + ACK is
  // small so single-lane is fine).
  int lane_nic_idx =
      pc->local_nic_idx_for_lane.empty() ? 0 : pc->local_nic_idx_for_lane[0];
  dxs::Reg hdr_reg = sp->per_nic_reg_handles[lane_nic_idx];
  if (hdr_reg == 0) hdr_reg = sp->reg_handle;
  auto sop_or = sock->Send(hdr_off, sizeof(WireHeader), hdr_reg);
  if (!sop_or.ok()) {
    return sop_or.status();
  }

  // Park the SendOp so it stays alive until Send completes.
  pc->pending_ack_sops.push_back(std::move(*sop_or));
  pc->last_ack_emitted_to_peer.store(hw, std::memory_order_release);
  return absl::OkStatus();
}

void GinCtx::StartProgress() {
  if (progress_ == nullptr) {
    progress_ = std::make_unique<ProxyProgress>(this);
  }
  progress_->Start();
}

void GinCtx::StopProgress() {
  if (progress_ != nullptr) {
    progress_->Stop();
    progress_.reset();
  }
}

}  // namespace fastrak::gin
