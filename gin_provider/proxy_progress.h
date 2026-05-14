/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Host progress thread that drains the GFD ring per peer.
 *
 * For each peer we maintain a local consumer cursor `local_ci`. Each tick:
 *   for each peer p in 0..nranks:
 *     while local_ci[p] != atomic_load(pis[p]):
 *       gfd = queues[p * queueSize + (local_ci[p] & mask)]
 *       if !GfdReady(gfd) break
 *       decoded = DecodeGfd(gfd)
 *       if decoded.op contains Put:
 *         dxs::Send(peer p, src_off, src_handle->reg, size)
 *       elif Get: ...
 *       elif Signal: ...
 *       wait for op completion (or post async and complete via callback)
 *       atomic_store(cis[p], local_ci[p] + 1)  // unlocks the GPU
 *       ConsumeGfd(&gfd)
 *       local_ci[p] += 1
 *
 * For now we run a single thread that round-robins over peers. If a single
 * peer's traffic dominates, future work spawns one thread per NIC.
 */

#ifndef GIN_PROVIDER_PROXY_PROGRESS_H_
#define GIN_PROVIDER_PROXY_PROGRESS_H_

#include <atomic>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"

namespace fastrak::gin {

class ProxyContext;

class ProxyProgress {
 public:
  explicit ProxyProgress(ProxyContext* ctx);
  ~ProxyProgress();

  void Start();
  void Stop();

  // Single tick of the polling loop. Exposed so plugin->ginProgress can
  // invoke it directly if NCCL chooses cooperative progress.
  void Tick();

 private:
  void Run();

  ProxyContext* ctx_;
  std::vector<uint32_t> local_ci_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_PROXY_PROGRESS_H_
