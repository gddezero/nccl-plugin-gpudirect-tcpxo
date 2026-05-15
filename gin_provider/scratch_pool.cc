/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/scratch_pool.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nccl_cuda/cuda_common.h"

namespace fastrak::gin {

absl::StatusOr<std::unique_ptr<ScratchPool>> AllocateScratchPool(
    int nranks,
    const std::array<tcpdirect::BufferManagerClientInterface*, kMaxNics>&
        bufmgrs,
    int primary_nic_idx) {
  if (nranks <= 0) {
    return absl::InvalidArgumentError("AllocateScratchPool: bad args");
  }
  if (primary_nic_idx < 0 || primary_nic_idx >= kMaxNics ||
      bufmgrs[primary_nic_idx] == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("AllocateScratchPool: bad primary_nic_idx=",
                     primary_nic_idx));
  }
  auto p = std::make_unique<ScratchPool>();
  p->nranks = nranks;
  p->tx_per_peer_bytes = kTxSlotsPerPeer * kWireHeaderSize;
  p->tx_base_off = 0;
  p->rx_base_off = static_cast<size_t>(nranks) * p->tx_per_peer_bytes;
  p->rx_total_bytes = kRxSlots * kWireHeaderSize;
  p->total_bytes = p->rx_base_off + p->rx_total_bytes;

  // Round up to a page boundary so cuMemGetHandleForAddressRange (used by
  // getDmabufFd) is happy.
  size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t alloc_bytes = (p->total_bytes + page - 1) & ~(page - 1);
  if (alloc_bytes < p->total_bytes) alloc_bytes = p->total_bytes;

  cudaError_t err = cudaMalloc(&p->device_ptr, alloc_bytes);
  if (err != cudaSuccess || p->device_ptr == nullptr) {
    return absl::InternalError(
        absl::StrCat("cudaMalloc(scratch) failed: ", cudaGetErrorString(err)));
  }

  // Zero the tx region so headers we never wrote remain identifiable as 0.
  err = cudaMemset(p->device_ptr, 0, alloc_bytes);
  if (err != cudaSuccess) {
    cudaFree(p->device_ptr);
    return absl::InternalError(
        absl::StrCat("cudaMemset(scratch) failed: ", cudaGetErrorString(err)));
  }

  auto fd_or = fastrak::getDmabufFd(p->device_ptr, alloc_bytes, /*pci=*/"");
  if (!fd_or.ok()) {
    cudaFree(p->device_ptr);
    return fd_or.status();
  }
  p->dmabuf_fd = *fd_or;

  // v9: register the SAME dma-buf with EVERY NIC's BufferManagerClient so
  // sender lanes spread across NICs each have a valid reg.
  int n_regged = 0;
  for (int i = 0; i < kMaxNics; ++i) {
    if (bufmgrs[i] == nullptr) continue;
    auto reg_or = bufmgrs[i]->RegBuf(p->dmabuf_fd, alloc_bytes);
    if (!reg_or.ok()) {
      // Roll back any earlier successful regs and fail.
      for (int j = 0; j < i; ++j) {
        if (bufmgrs[j] != nullptr && p->per_nic_reg_handles[j] != 0) {
          (void)bufmgrs[j]->DeregBuf(p->per_nic_reg_handles[j]);
          p->per_nic_reg_handles[j] = 0;
        }
      }
      close(p->dmabuf_fd);
      cudaFree(p->device_ptr);
      return reg_or.status();
    }
    p->per_nic_reg_handles[i] = *reg_or;
    ++n_regged;
  }
  p->reg_handle = p->per_nic_reg_handles[primary_nic_idx];

  // Pin via GDRCopy so the proxy thread can write WireHeaders from CPU
  // without going through cudaMemcpy (which serializes with other compute
  // kernels via the legacy default stream and stalls when the barrier
  // kernel is mid-flight).
  if (GdrAvailable()) {
    auto pin_or = GdrPinnedRegion::Create(p->device_ptr, alloc_bytes);
    if (pin_or.ok()) {
      p->gdr_region = std::move(*pin_or);
      p->host_ptr = p->gdr_region.host_map();
    } else {
      LOG(WARNING) << "AllocateScratchPool: GDR pin failed: "
                   << pin_or.status() << " — falling back to cudaMemcpy";
    }
  }

  LOG(INFO) << "AllocateScratchPool: " << alloc_bytes
            << " bytes, tx_per_peer=" << p->tx_per_peer_bytes
            << " rx_total=" << p->rx_total_bytes
            << " n_regged=" << n_regged
            << " primary_reg=" << static_cast<unsigned long long>(p->reg_handle)
            << " host_ptr=" << p->host_ptr;
  return p;
}

void FreeScratchPool(ScratchPool* p,
                     const std::array<tcpdirect::BufferManagerClientInterface*,
                                      kMaxNics>& bufmgrs) {
  if (p == nullptr) return;
  for (int i = 0; i < kMaxNics; ++i) {
    if (p->per_nic_reg_handles[i] != 0 && bufmgrs[i] != nullptr) {
      auto s = bufmgrs[i]->DeregBuf(p->per_nic_reg_handles[i]);
      if (!s.ok()) {
        LOG(ERROR) << "FreeScratchPool DeregBuf nic=" << i << ": " << s;
      }
      p->per_nic_reg_handles[i] = 0;
    }
  }
  if (p->dmabuf_fd >= 0) close(p->dmabuf_fd);
  if (p->device_ptr != nullptr) cudaFree(p->device_ptr);
  *p = ScratchPool{};
}

}  // namespace fastrak::gin
