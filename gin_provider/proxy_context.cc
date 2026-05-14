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

ProxyContext::~ProxyContext() { Shutdown(); }

absl::Status ProxyContext::Init(int nranks, int rank, uint32_t queue_size,
                                int n_counters, int n_signals) {
  rank_ = rank;
  nranks_ = nranks;
  ASSIGN_OR_RETURN(gpu_ctx_, AllocateProxyGpuCtx(nranks, queue_size,
                                                 n_counters, n_signals));
  peers_.resize(nranks);
  // Progress thread is created lazily by the plugin host once peer
  // sockets are wired (in Connect / Accept). Init alone does not start it.
  return absl::OkStatus();
}

void ProxyContext::Shutdown() {
  if (progress_ != nullptr) {
    progress_.reset();
  }
  peers_.clear();
  gpu_ctx_.reset();
}

void ProxyContext::register_peer(int rank_idx, PeerConn conn) {
  peers_.at(rank_idx) = std::move(conn);
}

uint64_t ProxyContext::register_memhandle(MemHandle h) {
  absl::MutexLock l(&mh_mu_);
  uint64_t key = next_mh_key_++;
  memhandles_.emplace(key, std::move(h));
  return key;
}

MemHandle* ProxyContext::lookup_memhandle(uint64_t key) {
  absl::MutexLock l(&mh_mu_);
  auto it = memhandles_.find(key);
  if (it == memhandles_.end()) return nullptr;
  return &it->second;
}

}  // namespace fastrak::gin
