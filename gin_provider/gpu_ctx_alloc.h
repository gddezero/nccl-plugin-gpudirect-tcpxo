/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Allocator for ncclGinProxyGpuCtx_t and its inner ring memory.
 *
 * NCCL device kernels write into pis[], read from cis[], and stream 128-byte
 * GFDs into queues[]. The host progress thread reads pis[], increments cis[],
 * and parses queues[]. Both views must reach the same physical memory.
 *
 * Strategy:
 *   - cudaHostAlloc(cudaHostAllocMapped) for the entire context blob.
 *   - cudaHostGetDevicePointer to obtain the GPU view of every field.
 *   - The GPU view is what we publish on ncclNetDeviceHandle_v11_t::handle.
 *
 * This file abstracts the allocation so the rest of the plugin stays
 * independent of CUDA driver specifics.
 */

#ifndef GIN_PROVIDER_GPU_CTX_ALLOC_H_
#define GIN_PROVIDER_GPU_CTX_ALLOC_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"

namespace fastrak::gin {

// One allocation that is mapped into both host and device address spaces.
struct MappedRegion {
  void* host_ptr = nullptr;
  void* dev_ptr = nullptr;
  size_t bytes = 0;
};

// Per-comm-context GPU/host shared state.
struct ProxyGpuCtxOwned {
  // The struct passed to NCCL device code (lives in MappedRegion below).
  ncclGinProxyGpuCtx_t* host_view = nullptr;  // host VA of the struct itself
  ncclGinProxyGpuCtx_t* dev_view = nullptr;   // device VA of the struct itself

  // Backing storage (one big slab; struct + arrays packed inside).
  MappedRegion ctx_blob;

  // Convenience host pointers into the blob.
  uint32_t* host_pis = nullptr;  // size = nranks
  uint32_t* host_cis = nullptr;  // size = nranks
  ncclGinProxyGfd_t* host_queues = nullptr;  // size = nranks * queueSize
  uint64_t* host_counters = nullptr;  // size = nCounters
  uint64_t* host_signals = nullptr;   // size = nSignals

  int nranks = 0;
  uint32_t queue_size = 0;
  int n_counters = 0;
  int n_signals = 0;
};

// Allocate a single mapped slab holding the ncclGinProxyGpuCtx_t plus
// its pis/cis/queues/counters/signals arrays.
//
// queue_size MUST be a power of two (NCCL device code uses idx & (queueSize-1)).
// Returns owned state; deallocator runs on destruction.
absl::StatusOr<std::unique_ptr<ProxyGpuCtxOwned>> AllocateProxyGpuCtx(
    int nranks, uint32_t queue_size, int n_counters, int n_signals);

// Free the underlying mapped region. Called by destructor; exposed for tests.
void FreeProxyGpuCtx(ProxyGpuCtxOwned* owned);

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_GPU_CTX_ALLOC_H_
