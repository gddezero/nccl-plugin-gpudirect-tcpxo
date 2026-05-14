/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/proxy_progress.h"

#include <atomic>
#include <chrono>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "gin_provider/gfd_decoder.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/proxy_context.h"

namespace fastrak::gin {

ProxyProgress::ProxyProgress(ProxyContext* ctx) : ctx_(ctx) {
  if (ctx_ != nullptr && ctx_->gpu_ctx() != nullptr) {
    local_ci_.assign(ctx_->gpu_ctx()->nranks, 0);
  }
}

ProxyProgress::~ProxyProgress() { Stop(); }

void ProxyProgress::Start() {
  if (thread_.joinable()) return;
  stop_.store(false, std::memory_order_release);
  thread_ = std::thread(&ProxyProgress::Run, this);
}

void ProxyProgress::Stop() {
  stop_.store(true, std::memory_order_release);
  if (thread_.joinable()) thread_.join();
}

void ProxyProgress::Run() {
  while (!stop_.load(std::memory_order_acquire)) {
    Tick();
    // TODO(perf): replace with a monotonic spin-wait that backs off only
    // after consecutive empty ticks.
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
}

void ProxyProgress::Tick() {
  auto* gpu = ctx_ ? ctx_->gpu_ctx() : nullptr;
  if (gpu == nullptr) return;

  for (int p = 0; p < gpu->nranks; ++p) {
    auto* pi_atomic = reinterpret_cast<std::atomic<uint32_t>*>(&gpu->host_pis[p]);
    auto* ci_atomic = reinterpret_cast<std::atomic<uint32_t>*>(&gpu->host_cis[p]);
    uint32_t pi = pi_atomic->load(std::memory_order_acquire);

    while (local_ci_[p] != pi) {
      uint32_t slot = local_ci_[p] & (gpu->queue_size - 1);
      auto* gfd = &gpu->host_queues[p * gpu->queue_size + slot];
      if (!GfdReady(*gfd)) break;

      DecodedGfd dec = DecodeGfd(*gfd);

      // TODO: dispatch dec.op to dxs::Send / dxs::RecvLinearized / signal
      // wait for completion (or queue async and complete later)
      // For now this is a no-op that just acks the GFD so we can validate
      // the decoder + ring loop.
      (void)dec;

      ConsumeGfd(gfd);
      local_ci_[p] += 1;
      ci_atomic->store(local_ci_[p], std::memory_order_release);
    }
  }
}

}  // namespace fastrak::gin
