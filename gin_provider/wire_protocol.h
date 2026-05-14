/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Wire protocol for the FasTrak GIN PROXY plugin.
 *
 * DXS sends are anonymous bulk pipes ((src_off, size, src_reg) -> stream).
 * They carry no destination address and no per-message metadata. So every
 * GIN op is fronted by a fixed-size WireHeader that the receiver pulls
 * first to learn what to do with the bytes that follow.
 *
 *   layout per op on the wire:
 *     [ 64-byte WireHeader ]  -- always
 *     [ size-byte payload  ]  -- only for ops with size > 0
 *
 * Source-side progress thread:
 *   for each consumable GFD:
 *     hdr = build WireHeader(op, src_rank, dst_handle, dst_off, size, ...)
 *     if size > 0:
 *       memcpy hdr into the per-peer scratch reg, then dxs::Send the header
 *       dxs::Send the payload (src_reg, src_off, size)
 *     else:
 *       memcpy hdr; dxs::Send the header only (signal, flush, ...)
 *     wait for both Sends to ack -> bump cis on GPU side
 *
 * Receiver-side recv-loop (one per accepted RecvSocket):
 *   loop:
 *     RecvLinearized(0, 64, hdr_reg) -> wait
 *     parse hdr -> pick action by op
 *       PUT: RecvLinearized(dst_off, size, dst_reg) -> wait
 *            then optional signal: atomic_add signal_val to signals[signal_id]
 *       SIGNAL: atomic_add signal_val to signals[signal_id] (no payload)
 *       GET:    issue dxs::Send back to peer with the requested data
 *       FLUSH:  no-op (acknowledgement-only marker)
 *
 * Because every header carries source_rank, the receiver does not need to
 * pre-match accepted sockets to source ranks; any worker can pull a header
 * off any RecvSocket and demultiplex.
 */

#ifndef GIN_PROVIDER_WIRE_PROTOCOL_H_
#define GIN_PROVIDER_WIRE_PROTOCOL_H_

#include <cstdint>

namespace fastrak::gin {

constexpr uint32_t kWireMagic = 0x46505554u;  // 'F','P','U','T'

enum WireOp : uint16_t {
  kWireOpInvalid = 0,
  kWireOpPut = 1,
  kWireOpPutSignal = 2,    // Put + atomic add signal at end
  kWireOpSignal = 3,       // bare atomic add (no payload)
  kWireOpGet = 4,          // request peer to send back (size, reg, off)
  kWireOpGetReply = 5,     // peer's reply to a Get
  kWireOpFlush = 6,
};

constexpr uint16_t kWireFlagHasCounter = 1u << 0;

struct WireHeader {
  uint32_t magic;          // = kWireMagic
  uint16_t op;             // WireOp
  uint16_t flags;          // bitmask of kWireFlag*
  uint32_t source_rank;
  uint32_t dest_rank;      // sanity check
  // Peer mhandle key identifying the SIGNAL buffer on the destination rank.
  // Both ranks register the same logical buffer with identical ordinal keys
  // (see proxy_context.cc::register_memhandle), so the sender's local key
  // for a buffer is also the receiver's key for that buffer.
  // 0 means "use the per-context primary signal buffer attached via
  // FORCE_SO" (NCCL barrier path back-compat).
  uint64_t signal_handle;
  uint64_t dst_handle;     // peer's MemHandle key for dst window
  uint64_t dst_off;
  uint64_t size;           // payload bytes after this header
  uint64_t signal_val;
  // Byte offset within the destination rank's signal buffer (either the
  // signal_handle one or, if signal_handle==0, the primary per-context
  // FORCE_SO buffer). NCCL proxy shim encodes:
  //   signal_off = (signal_id + ctx_id * nSignalsPerCtx) * sizeof(uint64_t).
  // Receiver does atomic_add at <signal_buffer_host_map>[signal_off / 8].
  uint64_t signal_off;
};
static_assert(sizeof(WireHeader) == 64,
              "WireHeader must be exactly 64 bytes");

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_WIRE_PROTOCOL_H_
