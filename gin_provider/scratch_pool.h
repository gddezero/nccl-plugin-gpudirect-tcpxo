/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Per-CollComm CUDA scratch buffer used to stage WireHeaders for outbound
 * dxs::Send and to receive incoming WireHeaders via RecvLinearized.
 *
 * Layout:
 *   region 0 (TX, size = nranks * K * 64): outbound headers
 *     slot at offset r * K * 64 + slot * 64 for peer r, slot in [0..K)
 *   region 1 (RX, size = K_in * 64):       inbound headers, shared pool
 *     slot at offset slot * 64 for slot in [0..K_in)
 *
 * The whole buffer is one cudaMalloc, registered as a single dma-buf with
 * buf_mgr->RegBuf, so DXS can Send / RecvLinearized into any sub-region.
 */

#ifndef GIN_PROVIDER_SCRATCH_POOL_H_
#define GIN_PROVIDER_SCRATCH_POOL_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "gin_provider/gdr_helper.h"

namespace fastrak::gin {

constexpr size_t kWireHeaderSize = 64;
constexpr size_t kTxSlotsPerPeer = 32;   // outbound header slots per peer
constexpr size_t kRxSlots = 256;         // inbound header pool slots

struct ScratchPool {
  void*       device_ptr = nullptr;     // cudaMalloc base
  int         dmabuf_fd = -1;
  size_t      total_bytes = 0;
  dxs::Reg    reg_handle = 0;
  int         nranks = 0;

  // GDR-pinned host VA over the scratch device memory. Lets us write
  // WireHeaders directly from CPU (no cudaMemcpy round-trip that would
  // serialize with other kernels via the default stream).
  GdrPinnedRegion gdr_region;
  void*       host_ptr = nullptr;       // host VA == device_ptr's GDR map

  // Region offsets (bytes from device_ptr).
  size_t      tx_base_off = 0;          // = 0
  size_t      tx_per_peer_bytes = 0;    // = kTxSlotsPerPeer * kWireHeaderSize
  size_t      rx_base_off = 0;          // = nranks * tx_per_peer_bytes
  size_t      rx_total_bytes = 0;       // = kRxSlots * kWireHeaderSize

  // Helpers.
  size_t TxSlotOffset(int peer, size_t slot) const {
    return tx_base_off + peer * tx_per_peer_bytes +
           (slot % kTxSlotsPerPeer) * kWireHeaderSize;
  }
  size_t RxSlotOffset(size_t slot) const {
    return rx_base_off + (slot % kRxSlots) * kWireHeaderSize;
  }
};

// Allocate + register the scratch pool. `buf` is the per-NIC buffer manager
// already obtained from NicClientRouter. Caller owns the returned ScratchPool
// and must call FreeScratchPool on teardown.
absl::StatusOr<std::unique_ptr<ScratchPool>> AllocateScratchPool(
    int nranks, tcpdirect::BufferManagerClientInterface* buf);

void FreeScratchPool(ScratchPool* p,
                     tcpdirect::BufferManagerClientInterface* buf);

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_SCRATCH_POOL_H_
