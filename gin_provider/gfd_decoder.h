/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Decoder for the 128-byte GFD wire format that NCCL uses to ferry GIN
 * operations from the device kernel to the host proxy thread.
 *
 * Layout (from src/include/nccl_device/gin/proxy/gin_proxy_device_host_common.h):
 *   16 packed qwords, each is a tagged union (ncclGinProxyQword_t).
 *   qword[0] = header (size + version + flag bit)
 *   qword[1..2] = either inline data, src offset/handle, or VA signal off/handle
 *   qword[3] = dst offset
 *   qword[4] = dst handle
 *   qword[5] = completion (counterId, signalId, signalVal low)
 *   qword[6] = signal val mid/high
 *   qword[7] = headerExt (op type)
 *   qword[8..15] reserved
 *
 * The flag bit in each qword is set to 1 by the device when a slot is
 * fully constructed; the host MUST wait for all 8 used qword flag bits
 * before reading the entry, then clear them on consume.
 */

#ifndef GIN_PROVIDER_GFD_DECODER_H_
#define GIN_PROVIDER_GFD_DECODER_H_

#include <cstddef>
#include <cstdint>

#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"

namespace fastrak::gin {

// Decoded view of a single GFD entry. Only the fields appropriate for the
// detected op are valid; check `op` first.
struct DecodedGfd {
  uint16_t op = 0;            // ncclGinProxyOp_t bitmask
  uint64_t size = 0;          // bytes to transfer
  uint64_t src_off = 0;
  uint64_t src_handle = 0;
  uint64_t dst_off = 0;
  uint64_t dst_handle = 0;
  uint64_t va_signal_off = 0;
  uint64_t va_signal_handle = 0;
  uint64_t signal_val = 0;
  uint32_t counter_id = 0;
  uint32_t signal_id = 0;
  uint64_t inline_val = 0;       // up to 96 bits but we only use 64
  bool has_inline = false;
};

// Returns true if every used qword has the flag bit set (i.e. the device has
// finished constructing this slot). Caller should retry on false.
bool GfdReady(const ncclGinProxyGfd_t& gfd);

// Decode the GFD into a friendlier struct. Caller must already have observed
// GfdReady() == true.
DecodedGfd DecodeGfd(const ncclGinProxyGfd_t& gfd);

// Clear all flag bits to 0 so the device can reuse this slot.
void ConsumeGfd(ncclGinProxyGfd_t* gfd);

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_GFD_DECODER_H_
