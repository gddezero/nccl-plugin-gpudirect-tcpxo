/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * The host-side progress engine for the FasTrak GIN PROXY plugin.
 *
 * Two kinds of work happen here:
 *
 *   - Outbound: a single thread polls every peer's slice of the GFD ring,
 *     decodes each consumable GFD into a WireHeader, copies the header into
 *     the per-peer scratch ring, dxs::Sends the header, optionally dxs::Sends
 *     the payload, waits for both Sends to ack, then advances the GPU-visible
 *     consumer cursor (cis[peer]).
 *
 *   - Inbound: one thread per accepted RecvSocket. Each pre-posts a 64-byte
 *     RecvLinearized into its slice of the rx scratch ring, parses the
 *     resulting WireHeader, then either:
 *       * Put / PutSignal: pre-posts a payload RecvLinearized into the
 *         destination MemHandle's reg at dst_off, optionally bumps a signal.
 *       * Signal:           bumps signals[signal_id] += signal_val.
 *       * Get:              issues a GetReply send back to the peer.
 *       * Flush:            no-op marker (used as a barrier).
 *
 * Performance posture: this is the simplest correct implementation —
 * per-op synchronous send/recv and one thread per recv socket. M6 will
 * pipeline and consolidate threads.
 */

#ifndef GIN_PROVIDER_PROXY_PROGRESS_H_
#define GIN_PROVIDER_PROXY_PROGRESS_H_

#include <atomic>
#include <memory>
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

  // One step of the outbound loop. Exposed so plugin->ginProgress can drive
  // it cooperatively when NCCL prefers that over a dedicated thread.
  void TickOutbound();

 private:
  void RunOutbound();
  void RunInbound(size_t inbound_idx);

  GinCtx* ctx_ = nullptr;

  // Per-peer outbound state.
  struct PeerOut {
    uint32_t local_ci = 0;          // last consumed GFD index for this peer
    uint64_t next_seq = 0;
    uint32_t tx_slot = 0;           // next ring slot to use (mod kTxSlotsPerPeer)
  };
  std::vector<PeerOut> peer_out_;

  std::atomic<bool> stop_{false};
  std::thread outbound_thread_;
  std::vector<std::thread> inbound_threads_;
};

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_PROXY_PROGRESS_H_
