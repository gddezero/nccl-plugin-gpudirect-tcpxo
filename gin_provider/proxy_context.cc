/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/proxy_context.h"

#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "dxs/client/oss/status_macros.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/proxy_progress.h"

namespace fastrak::gin {

// ---- CollComm ----

CollComm::~CollComm() {
  // Destructors of held unique_ptrs handle teardown (DXS sockets close in
  // their own destructors via SendMessage(CloseDataSockMessage)).
  peers_.clear();
}

absl::Status CollComm::Init(
    int dev, uint8_t fastrak_idx, std::string nic_ip, int nranks, int rank,
    dxs::DxsClientInterface* absl_nonnull dxs,
    tcpdirect::BufferManagerClientInterface* absl_nonnull buf) {
  dev_ = dev;
  fastrak_idx_ = fastrak_idx;
  nic_ip_ = std::move(nic_ip);
  nranks_ = nranks;
  rank_ = rank;
  dxs_ = dxs;
  buf_ = buf;
  peers_.resize(nranks);
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
  uint64_t key = next_mh_key_++;
  memhandles_.emplace(key, std::move(h));
  return key;
}

MemHandle* CollComm::lookup_memhandle(uint64_t key) {
  absl::MutexLock l(&mh_mu_);
  auto it = memhandles_.find(key);
  if (it == memhandles_.end()) return nullptr;
  return &it->second;
}

void CollComm::erase_memhandle(uint64_t key) {
  absl::MutexLock l(&mh_mu_);
  memhandles_.erase(key);
}

// ---- GinCtx ----

GinCtx::~GinCtx() { StopProgress(); }

absl::Status GinCtx::Init(uint32_t queue_size, int n_counters, int n_signals) {
  if (coll_ == nullptr) {
    return absl::FailedPreconditionError("GinCtx::Init: coll_ is null");
  }
  ASSIGN_OR_RETURN(
      gpu_ctx_, AllocateProxyGpuCtx(coll_->nranks(), queue_size, n_counters,
                                    n_signals));
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
