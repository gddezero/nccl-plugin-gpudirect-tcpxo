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
        LOG(WARNING) << "signal_host_addr: mhandle 0x" << std::hex
                     << signal_handle
                     << " has no GDR pin (size=" << std::dec << mh->bytes
                     << " base=" << mh->base << "); falling back";
      }
    }
  }
  // Fallback: primary per-context signal buffer (NCCL barrier path).
  if (signal_host_map_ != nullptr &&
      signal_off + sizeof(uint64_t) <= signal_size_bytes_) {
    return reinterpret_cast<uint8_t*>(signal_host_map_) + signal_off;
  }
  return nullptr;
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
