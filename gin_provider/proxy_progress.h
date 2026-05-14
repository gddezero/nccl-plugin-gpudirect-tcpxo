/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Host progress thread that drains the GFD ring per peer.
 *
 * For each peer p we keep a local consumer cursor `local_ci_[p]`. Each tick:
 *   pi = atomic_load(host_pis[p])
 *   while local_ci_[p] != pi:
 *     gfd = host_queues[p * queueSize + (local_ci_[p] & (queueSize-1))]
 *     if !GfdReady(gfd) break
 *     dec = DecodeGfd(gfd)
 *     dispatch(dec):
 *       Put         -> dxs::Send(peer p, src_off, src_handle->reg, size)
 *       PutInline   -> stash inline value into a per-peer scratch reg, Send
 *       Get         -> queue a request to the peer's host proxy (out of scope M3)
 *       Signal      -> remote atomic add via a special control message
 *       Flush       -> wait for inflight ops then post a marker
 *     wait for ack (or queue async completion)
 *     ConsumeGfd(&gfd)
 *     atomic_store(host_cis[p], local_ci_[p] + 1)  // unblocks GPU device
 *
 * For now this thread round-robins over peers. We can later spawn one thread
 * per NIC if a single peer's traffic dominates.
 */

#ifndef GIN_PROVIDER_PROXY_PROGRESS_H_
#define GIN_PROVIDER_PROXY_PROGRESS_H_

#include <atomic>
#include <thread>
#include <vector>

namespace fastrak::gin {

class GinCtx;

class ProxyProgress {
 public:
  explicit ProxyProgress(GinCtx* ctx);
  ~ProxyProgress();

  void Start();
  void Stop();

  // Single tick of the polling loop. Exposed so plugin->ginProgress can
  // invoke it directly when NCCL chooses cooperative progress.
  void Tick();

 private:
  void Run();

  GinCtx* ctx_;
  std::vector<uint32_t> local_ci_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_PROXY_PROGRESS_H_
