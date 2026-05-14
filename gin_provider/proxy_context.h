/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Per-collective-comm state for the FasTrak GIN PROXY provider.
 *
 * One ProxyContext is created per ncclGin_v13_t::createContext call. It owns:
 *   - the GPU-visible ProxyGpuCtxOwned (pis/cis/queues/...)
 *   - the per-peer dxs::SendSocket / dxs::RecvSocket
 *   - the per-buffer dma-buf registration state (peer_reg_handle map)
 *   - the host progress thread that drains the GFD ring
 */

#ifndef GIN_PROVIDER_PROXY_CONTEXT_H_
#define GIN_PROVIDER_PROXY_CONTEXT_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "gin_provider/gpu_ctx_alloc.h"

namespace fastrak::gin {

class ProxyProgress;

// One per-peer connection lives in here.
struct PeerConn {
  std::unique_ptr<dxs::SendSocketInterface> send_sock;
  std::unique_ptr<dxs::RecvSocketInterface> recv_sock;
};

// Memory registration: one local Reg + the peer's published Reg keys for the
// same window. NCCL exchanges these out-of-band via OOB during connect/regMr.
struct MemHandle {
  uint64_t local_va_base = 0;
  size_t   bytes = 0;
  dxs::Reg local_reg = 0;
  std::vector<dxs::Reg> peer_regs;  // size = nranks
};

class ProxyContext {
 public:
  ProxyContext() = default;
  ~ProxyContext();

  // Initialize after dxs/buf-mgr clients are wired in.
  absl::Status Init(int nranks, int rank, uint32_t queue_size,
                    int n_counters, int n_signals);

  // Stop progress thread. Idempotent.
  void Shutdown();

  ProxyGpuCtxOwned* gpu_ctx() { return gpu_ctx_.get(); }

  PeerConn& peer(int rank_idx) { return peers_.at(rank_idx); }
  size_t num_peers() const { return peers_.size(); }

  void register_peer(int rank_idx, PeerConn conn);

  uint64_t register_memhandle(MemHandle h);
  MemHandle* lookup_memhandle(uint64_t key);

 private:
  std::unique_ptr<ProxyGpuCtxOwned> gpu_ctx_;
  std::unique_ptr<ProxyProgress> progress_;
  std::vector<PeerConn> peers_;

  absl::Mutex mh_mu_;
  absl::flat_hash_map<uint64_t, MemHandle> memhandles_;
  uint64_t next_mh_key_ = 1;

  int rank_ = -1;
  int nranks_ = 0;
};

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_PROXY_CONTEXT_H_
