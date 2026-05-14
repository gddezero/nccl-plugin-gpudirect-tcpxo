/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/gfd_decoder.h"

#include <atomic>
#include <cstdint>

namespace fastrak::gin {
namespace {

// The qword index constants come from
// nccl_device/gin/proxy/gin_proxy_device_host_common.h:
//   header = 0, inlineLow/srcOff/vaSignalOff = 1,
//   inlineHigh/srcHandle/vaSignalHandle = 2,
//   dstOff = 3, dstHandle = 4, completion = 5,
//   signalVal = 6, headerExt = 7
constexpr int kHeader = 0;
constexpr int kSlot1 = 1;          // inlineLow / srcOff / vaSignalOff
constexpr int kSlot2 = 2;          // inlineHigh / srcHandle / vaSignalHandle
constexpr int kDstOff = 3;
constexpr int kDstHandle = 4;
constexpr int kCompletion = 5;
constexpr int kSignalVal = 6;
constexpr int kHeaderExt = 7;
constexpr int kFirstUnused = 8;

// Op-bit decoding: the device sets exactly one of Put / Get / Flush / VASignal
// in qword[7].headerExt.op. WithSignal/WithCounter/WithInline are additive bits.
constexpr uint16_t kOpVASignal = 1u << 5;
constexpr uint16_t kOpGet = 1u << 6;
constexpr uint16_t kOpFlush = 1u << 7;
constexpr uint16_t kOpWithInline = 1u << 1;

inline uint64_t flag_bit(const ncclGinProxyQword_t& q) {
  return q.flag.v;
}

}  // namespace

bool GfdReady(const ncclGinProxyGfd_t& gfd) {
  // Use atomic_acquire on the trailing qword as the publication barrier:
  // the device sets headerExt last, so observing its flag bit guarantees
  // every preceding store is visible. We still spot-check kHeader for paranoia.
  const auto* q = gfd.qword;
  std::atomic_thread_fence(std::memory_order_acquire);
  return flag_bit(q[kHeaderExt]) && flag_bit(q[kHeader]);
}

DecodedGfd DecodeGfd(const ncclGinProxyGfd_t& gfd) {
  const auto* q = gfd.qword;
  DecodedGfd d;
  d.op = q[kHeaderExt].headerExt.op;
  d.size = q[kHeader].header.size;
  d.dst_off = q[kDstOff].dstOff.dstOff;
  d.dst_handle = q[kDstHandle].dstHandle.dstHandle;
  d.signal_id = q[kCompletion].completion.signalId;
  d.counter_id = q[kCompletion].completion.counterId;

  // Reassemble the 64-bit signal value from its split storage.
  uint64_t lo = q[kCompletion].completion.signalValLow;
  uint64_t mid = q[kSignalVal].signalVal.signalValLow2;
  uint64_t hi = q[kSignalVal].signalVal.signalValHigh;
  d.signal_val = lo | (mid << 16) | (hi << 32);

  if (d.op & kOpFlush) {
    return d;  // Flush carries no further fields.
  }

  if (d.op & kOpWithInline) {
    d.has_inline = true;
    uint64_t lo32 = q[kSlot1].inlineLow.inlineValLow;
    uint64_t lo16 = q[kSlot1].inlineLow.inlineValLow2;
    uint64_t hi16 = q[kSlot2].inlineHigh.inlineValHigh;
    d.inline_val = lo32 | (lo16 << 32) | (hi16 << 48);
  } else if (d.op & kOpVASignal) {
    d.va_signal_off = q[kSlot1].vaSignalOff.vaSignalOff;
    d.va_signal_handle = q[kSlot2].vaSignalHandle.vaSignalHandle;
  } else {
    d.src_off = q[kSlot1].srcOff.srcOff;
    d.src_handle = q[kSlot2].srcHandle.srcHandle;
  }

  return d;
}

void ConsumeGfd(ncclGinProxyGfd_t* gfd) {
  // Clear the flag bit on every qword so the device may reuse this slot.
  for (int i = 0; i < ncclGinProxyGfdQwords; ++i) {
    gfd->qword[i].flag.v = 0;
  }
  std::atomic_thread_fence(std::memory_order_release);
}

}  // namespace fastrak::gin
