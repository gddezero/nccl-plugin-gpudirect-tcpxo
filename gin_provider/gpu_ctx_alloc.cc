/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/gpu_ctx_alloc.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace fastrak::gin {
namespace {

// Round up size to a CUDA-friendly alignment so each sub-array sits on a
// cacheline boundary (64B). Some arrays (queues) are 128B per element so
// they're naturally aligned, but pis/cis are 4B each and we want them
// on separate cachelines from the queue base.
constexpr size_t kAlign = 64;

inline size_t RoundUp(size_t v) {
  return (v + kAlign - 1) & ~(kAlign - 1);
}

}  // namespace

absl::StatusOr<std::unique_ptr<ProxyGpuCtxOwned>> AllocateProxyGpuCtx(
    int nranks, uint32_t queue_size, int n_counters, int n_signals) {
  if (nranks <= 0 || queue_size == 0 ||
      (queue_size & (queue_size - 1)) != 0) {
    return absl::InvalidArgumentError(
        "AllocateProxyGpuCtx: nranks>0 and queue_size must be power-of-two");
  }

  // Layout the slab: header (sizeof ncclGinProxyGpuCtx_t), then arrays.
  // We pre-pad each segment to kAlign bytes.
  const size_t hdr_bytes = RoundUp(sizeof(ncclGinProxyGpuCtx_t));
  const size_t pis_bytes = RoundUp(sizeof(uint32_t) * nranks);
  const size_t cis_bytes = RoundUp(sizeof(uint32_t) * nranks);
  const size_t queues_bytes =
      RoundUp(sizeof(ncclGinProxyGfd_t) * nranks * queue_size);
  const size_t counters_bytes =
      RoundUp(sizeof(uint64_t) * (n_counters > 0 ? n_counters : 1));
  const size_t signals_bytes =
      RoundUp(sizeof(uint64_t) * (n_signals > 0 ? n_signals : 1));
  const size_t total =
      hdr_bytes + pis_bytes + cis_bytes + queues_bytes +
      counters_bytes + signals_bytes;

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(
      &host_ptr, total,
      cudaHostAllocMapped | cudaHostAllocWriteCombined);
  if (err != cudaSuccess) {
    return absl::InternalError(
        absl::StrCat("cudaHostAlloc failed: ", cudaGetErrorString(err)));
  }
  std::memset(host_ptr, 0, total);

  void* dev_ptr = nullptr;
  err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    return absl::InternalError(absl::StrCat(
        "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err)));
  }

  auto owned = std::make_unique<ProxyGpuCtxOwned>();
  owned->ctx_blob.host_ptr = host_ptr;
  owned->ctx_blob.dev_ptr = dev_ptr;
  owned->ctx_blob.bytes = total;
  owned->nranks = nranks;
  owned->queue_size = queue_size;
  owned->n_counters = n_counters;
  owned->n_signals = n_signals;

  // Compute host views.
  auto* base_h = static_cast<uint8_t*>(host_ptr);
  size_t off = 0;
  owned->host_view = reinterpret_cast<ncclGinProxyGpuCtx_t*>(base_h + off);
  off += hdr_bytes;
  owned->host_pis = reinterpret_cast<uint32_t*>(base_h + off);
  off += pis_bytes;
  owned->host_cis = reinterpret_cast<uint32_t*>(base_h + off);
  off += cis_bytes;
  owned->host_queues = reinterpret_cast<ncclGinProxyGfd_t*>(base_h + off);
  off += queues_bytes;
  owned->host_counters = reinterpret_cast<uint64_t*>(base_h + off);
  off += counters_bytes;
  owned->host_signals = reinterpret_cast<uint64_t*>(base_h + off);

  // Compute device views (same offsets relative to dev_ptr).
  auto* base_d = static_cast<uint8_t*>(dev_ptr);
  off = 0;
  owned->dev_view = reinterpret_cast<ncclGinProxyGpuCtx_t*>(base_d + off);
  off += hdr_bytes;
  uint32_t* d_pis = reinterpret_cast<uint32_t*>(base_d + off);
  off += pis_bytes;
  uint32_t* d_cis = reinterpret_cast<uint32_t*>(base_d + off);
  off += cis_bytes;
  ncclGinProxyGfd_t* d_queues =
      reinterpret_cast<ncclGinProxyGfd_t*>(base_d + off);
  off += queues_bytes;
  uint64_t* d_counters = reinterpret_cast<uint64_t*>(base_d + off);
  off += counters_bytes;
  uint64_t* d_signals = reinterpret_cast<uint64_t*>(base_d + off);

  // Populate header pointers using DEVICE addresses — that's what the GPU
  // kernel will dereference.
  owned->host_view->nranks = nranks;
  owned->host_view->queueSize = queue_size;
  owned->host_view->queues = d_queues;
  owned->host_view->pis = d_pis;
  owned->host_view->cis = d_cis;
  owned->host_view->counters = d_counters;
  owned->host_view->signals = d_signals;

  return owned;
}

void FreeProxyGpuCtx(ProxyGpuCtxOwned* owned) {
  if (owned == nullptr) return;
  if (owned->ctx_blob.host_ptr != nullptr) {
    cudaError_t err = cudaFreeHost(owned->ctx_blob.host_ptr);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaFreeHost failed: " << cudaGetErrorString(err);
    }
    owned->ctx_blob.host_ptr = nullptr;
    owned->ctx_blob.dev_ptr = nullptr;
  }
}

}  // namespace fastrak::gin
