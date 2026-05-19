/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Per-collective comm state for the FasTrak GIN PROXY provider.
 *
 * The GIN v13 ABI introduces three layers of state:
 *
 *   ListenComm  -- one per (dev) per `listen()` call. Owns a dxs::ListenSocket.
 *   CollComm    -- one per `connect()` call. Owns the n*(n-1) Send/Recv mesh.
 *   GinCtx      -- one per `createContext()`. Owns the ncclGinProxyGpuCtx_t
 *                  and the host progress thread that drains the GFD ring.
 *
 * RegMrSym handles live on the CollComm (NCCL exchanges peer reg keys before
 * calling createContext, but for our purposes a single CollComm pairs with
 * one or more GinCtx).
 */

#ifndef GIN_PROVIDER_PROXY_CONTEXT_H_
#define GIN_PROVIDER_PROXY_CONTEXT_H_

#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <cuda_runtime.h>
#include "absl/synchronization/mutex.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "gin_provider/gdr_helper.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/scratch_pool.h"

namespace fastrak::gin {

class ProxyProgress;
class GinCtx;

// v6 (S5): plugin-wide error flag exposed to NCCL via QueryLastError.
// Set by any fatal error path in plugin_main.cc / proxy_progress.cc;
// cleared on Init() (re-)entry. Use SetGinError() so call sites are
// greppable and future tooling can hook them.
extern std::atomic<bool> g_has_error;
inline void SetGinError(const char* /*where*/ = nullptr) {
  g_has_error.store(true, std::memory_order_release);
}

// v9: one ListenComm now owns up to kMaxNics ListenSockets — one per
// fastrak NIC — so peers can fan out across N receiver NICs. The primary
// (NCCL-driven) NIC is the dev NCCL called Listen(dev=...) on; others are
// added best-effort and entries with a null socket stay null. Indexing is
// by fastrak NIC index (NOT a dense [0..N) range).
struct ListenComm {
  int dev = -1;
  uint8_t primary_fastrak_idx = 0;  // NCCL's chosen dev's fastrak idx
  std::string primary_nic_ip;
  // listen_socks[i] is the ListenSocket bound to fastrak NIC i (or null if
  // we couldn't / chose not to open one on that NIC).
  std::array<std::unique_ptr<dxs::ListenSocketInterface>, kMaxNics>
      listen_socks;
  std::array<uint16_t, kMaxNics> listen_ports = {};   // port for slot i
  std::array<std::string, kMaxNics> nic_ips = {};     // IP for slot i
  uint64_t nonce = 0;
  uint64_t listen_token = 0;
};

// Per-peer DXS connection inside a CollComm.
//
// M6.2 fan-out: each (rank, peer) keeps N parallel send sockets so a
// single stream of pp_send / dispatch ops can be striped across multiple
// dxs Sends in parallel. Each socket still resolves to the same NIC
// (set by the CollComm's fastrak_idx_), so this is "multi-stream / same-
// NIC", but it lifts the per-socket queue-depth limit and lets the dxs
// server worker fan-out absorb pipelining the inline hdr-DONE wait can't.
// Receivers keep one inbound thread per accepted RecvSocket — total
// count is N*(nranks-1).
//
// PR1 (β port, 2026-05-16): in-flight bitmap ring size for the AWS-style
// per-peer ACK tracking added in PR3 (PeerConn::active_put_signal). 12-bit
// modular seq chosen for 16x headroom over the upstream-required
// NCCL_NET_MAX_REQUESTS * maxRecvs (= 32 * 1 = 32) concurrency floor and
// 4x headroom over plugin queueDepth=1024 + kTxSlotsPerPeer=1024 caps.
// AWS uses 8192 (13-bit) for fewer-rail TCP; our PROXY queueDepth caps
// in-flight strictly below 1024, so 4096 leaves comfortable margin
// without ballooning bitset memory (4096 bits = 512 B per peer). Distinct
// from wire_seq_next which is plugin-private cross-lane FIFO ordering
// (B1 finding, do not conflate).
constexpr size_t kAckRingSize = 4096;

struct PeerConn {
  std::vector<std::unique_ptr<dxs::SendSocketInterface>> send_socks;
  // v9: parallel to send_socks. local_nic_idx_for_lane[lane] is the local
  // fastrak NIC index that the lane's send socket was opened from. The
  // sender uses this to look up MemHandle::per_nic_regs / ScratchPool::
  // per_nic_reg_handles for that lane.
  std::vector<int> local_nic_idx_for_lane;
  std::atomic<uint64_t> tx_seq{0};  // round-robin lane selector
  // v11: per-peer monotonic outbound op counter, written into
  // WireHeader.wire_seq for EVERY IputCommon op. Receiver uses this to
  // serialize signal-commit ordering across fanout lanes (a later op's
  // signal cannot surface until earlier ops on other lanes have committed
  // their data writes). 1-based on the wire (0 means "no seq, no ordering").
  std::atomic<uint64_t> wire_seq_next{1};
  // PR3a (β port, 2026-05-16): AWS-style ACK protocol state.
  // Distinct layer from wire_seq_next above:
  //   - wire_seq_next        = v11 cross-lane FIFO ordering (never resets).
  //   - next_target_seq_num  = AWS modular sender cursor, mod kAckRingSize.
  //   - active_put_signal[s] = sender thinks seq `s` is awaiting peer ACK.
  //   - next_delivered_signal_seq_num = receiver strict in-order cursor;
  //                            signal raise advances by 1 per retire.
  //   - consecutive_puts_without_ack  = force-ACK throttle counter (PUT-only
  //                            batches force ACK every GIN_ACK_INTERVAL).
  //   - pending_ack          = per-peer bundled-range ACK accumulator
  //                            (piggybacked into next PutSignal/Put hdr or
  //                            flushed as standalone kWireOpAck after
  //                            GIN_ACK_MAX_AGE progress ticks).
  // All wired in PR3b (sender) and PR3c (receiver + Test() gate); PR3a is
  // declarative-only and must keep runtime behavior identical to PR1.
  std::atomic<uint16_t> next_target_seq_num{0};
  std::bitset<kAckRingSize> active_put_signal;
  std::atomic<uint16_t> next_delivered_signal_seq_num{0};
  uint32_t consecutive_puts_without_ack = 0;
  struct PendingAck {
    uint16_t seq_num   = 0;
    uint16_t ack_count = 0;
    uint32_t age_ticks = 0;
    bool     linked    = false;
  } pending_ack;
  // PR3c-iii (β port, 2026-05-16): async ACK emission state used by
  // CollComm::EmitAck (called from TickOutbound). last_ack_emitted_to_peer
  // tracks the high water we last sent to this peer so we only emit when
  // there is new info (avoid ACK flood). pending_ack_sops holds in-flight
  // ACK SendOps so the unique_ptrs stay alive until Send completes;
  // pruned every TickOutbound call. This out-of-band timer ACK channel
  // is what breaks the PR3c-ii circular deadlock — sender's Test() gate
  // gets ACK updates even when wrapper credit is stuck.
  std::atomic<uint64_t> last_ack_emitted_to_peer{0};
  std::vector<std::unique_ptr<dxs::SendOpInterface>> pending_ack_sops;
  PeerConn() = default;
  PeerConn(const PeerConn&) = delete;
  PeerConn& operator=(const PeerConn&) = delete;
  PeerConn(PeerConn&& o) noexcept
      : send_socks(std::move(o.send_socks)),
        local_nic_idx_for_lane(std::move(o.local_nic_idx_for_lane)),
        tx_seq(o.tx_seq.load(std::memory_order_relaxed)),
        wire_seq_next(o.wire_seq_next.load(std::memory_order_relaxed)),
        next_target_seq_num(
            o.next_target_seq_num.load(std::memory_order_relaxed)),
        active_put_signal(o.active_put_signal),
        next_delivered_signal_seq_num(
            o.next_delivered_signal_seq_num.load(std::memory_order_relaxed)),
        consecutive_puts_without_ack(o.consecutive_puts_without_ack),
        pending_ack(o.pending_ack),
        last_ack_emitted_to_peer(
            o.last_ack_emitted_to_peer.load(std::memory_order_relaxed)),
        pending_ack_sops(std::move(o.pending_ack_sops)) {}
  PeerConn& operator=(PeerConn&& o) noexcept {
    send_socks = std::move(o.send_socks);
    local_nic_idx_for_lane = std::move(o.local_nic_idx_for_lane);
    tx_seq.store(o.tx_seq.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    wire_seq_next.store(o.wire_seq_next.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    next_target_seq_num.store(
        o.next_target_seq_num.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    active_put_signal = o.active_put_signal;
    next_delivered_signal_seq_num.store(
        o.next_delivered_signal_seq_num.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    consecutive_puts_without_ack = o.consecutive_puts_without_ack;
    pending_ack = o.pending_ack;
    last_ack_emitted_to_peer.store(
        o.last_ack_emitted_to_peer.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    pending_ack_sops = std::move(o.pending_ack_sops);
    return *this;
  }
};

// PR3a (β port, 2026-05-16): per-outstanding-(peer, seq) receiver state for
// AWS-style in-order signal raise. Map key = (uint64_t(peer_rank) << 16) |
// seq_num (mirrors AWS nccl_ofi_gin.cpp:739-742). When num_seg_completions
// == total_segments AND metadata_received, the op becomes "deliverable";
// retire_completed_peer_iput_ops (added in PR3c) walks the per-peer
// next_delivered_signal_seq_num cursor in strict order and raises signals
// only when contiguous. Out-of-order seqs wait. Wired in PR3c.
struct RecvReqState {
  uint16_t total_segments      = 0;
  uint16_t num_seg_completions = 0;
  bool     metadata_received   = false;
  bool     is_ack_requested    = false;
  uint64_t signal_handle       = 0;
  uint64_t signal_off          = 0;
  uint64_t signal_val          = 0;
  uint16_t signal_op           = 0;  // 0 = none, 1 = INC, 2 = ADD
};

// Default fan-out per peer; can be overridden via NCCL_GIN_FANOUT env var.
// v10 investigation (2026-05-15): tried raising default 1 -> 3 to capture
// the +54% PP 4096x7168 conc=3 hide=1 bandwidth gain (~35 -> ~54 GB/s).
// REVERTED at v10 because num_stress_iterations=100 reproducibly corrupted
// data at fanout > 1: v10 fanout=3 mismatch at seed=6, fanout=2 at seed=8,
// v9 fanout=3 at seed=14. Root cause was a cross-lane signal-ordering
// race: with fanout > 1, three concurrent IputSignal ops stripe across
// three lanes; each receiver inbound thread does
// `data RecvLinearized -> signal RMW (signal++)` independently, so a
// faster lane can publish a later op's signal increment before an earlier
// op on a slower lane has finished its data DMA. The DeepEP/NCCL consumer
// reads a slot indexed by `recv_count % inflight` and trusts that signal
// == recv_count + 1 means slot N is filled — but the signal has advanced
// by a different lane that filled a DIFFERENT slot.
// v11 fix (this iteration): WireHeader gains a `wire_seq` field stamped
// monotonically per peer at the sender, and CollComm gains a per-source
// atomic `recv_commit_seq` array. RunInbound spin-waits on
// recv_commit_seq[src] == hdr.wire_seq before applying signal RMW (and
// bumps it after), so signal commits are serialized across lanes per
// source rank. Payload Recv still happens in parallel — only the
// post-Recv commit step is ordered. PP 4096x7168 conc=3 hide=1
// stress=100 now passes at fanout=3 (v11.md). With ordering safe,
// kDefaultFanout flipped 1 -> 3 to capture the +54% gain by default.
constexpr int kDefaultFanout = 3;
int FanoutPerPeer();

// v9: multi-NIC fan-out is OFF by default. When ON, Listen opens kMaxNics
// ListenSockets and Connect spreads lanes across local NICs; each MemHandle
// is registered with every NIC's BufferManagerClient. Empirically the
// multi-NIC path on a3-mega regresses test_pp 4096x7168 conc=3 fanout=3 vs
// single-NIC v7 (~42 GB/s vs ~55 GB/s) because of extra inbound thread
// contention and per-NIC reg duplication; keep it as opt-in until the
// receiver-side cost is amortized. Set NCCL_GIN_MULTI_NIC=1 to enable.
bool MultiNicEnabled();

// Memory registration: holds the local DXS Reg plus the array of peer Regs
// gathered out-of-band by NCCL after RegMrSym (peer regs are stored in the
// `ginHandle` that NCCL distributes to peers; we look them up by mhandle).
//
// v9: per_nic_regs[i] is the local DXS Reg this buffer received from NIC i's
// BufferManagerClient (or 0 if not registered there). Sender lane targeting
// local NIC i uses per_nic_regs[i]; receiver inbound on NIC j uses
// per_nic_regs[j]. local_reg is kept as an alias of per_nic_regs
// [primary_nic] for legacy callers and logs.
struct MemHandle {
  void*    base = nullptr;
  size_t   bytes = 0;
  int      ptr_type = 0;     // NCCL_PTR_HOST / CUDA / DMABUF
  int      dmabuf_fd = -1;
  uint64_t dmabuf_offset = 0;
  std::array<dxs::Reg, kMaxNics> per_nic_regs = {};
  dxs::Reg local_reg = 0;    // alias of per_nic_regs[primary_nic]
  std::vector<dxs::Reg> peer_regs;  // size = nranks; peer_regs[r] is rank r's view

  // Optional GDR pin: when this MemHandle was registered as
  // NCCL_PTR_CUDA, the plugin tries to GDRCopy-pin the underlying GPU
  // memory so the host proxy thread can do CPU-side atomic_add for
  // PutSignal/Signal landing in arbitrary CUDA buffers (M6: DeepEP
  // dispatch scratch, not just NCCL's small FORCE_SO signalsDev).
  // shared_ptr keeps MemHandle copyable through flat_hash_map ops.
  std::shared_ptr<GdrPinnedRegion> gdr_pin;
  void* gdr_host_map() const { return gdr_pin ? gdr_pin->host_map() : nullptr; }
};

class CollComm {
 public:
  CollComm() = default;
  ~CollComm();

  // v9: per_nic_dxs[i] / per_nic_bufmgr[i] are non-null for every NIC the
  // CollComm will send through; index i is the NIC's fastrak index. The
  // primary (listen) NIC's slot must also be populated. nic_ips_by_idx[i]
  // is the local IP for NIC i (used only for log lines).
  absl::Status Init(
      int dev, uint8_t fastrak_idx, std::string nic_ip, int nranks, int rank,
      dxs::DxsClientInterface* absl_nonnull dxs,
      tcpdirect::BufferManagerClientInterface* absl_nonnull buf,
      const std::array<dxs::DxsClientInterface*, kMaxNics>& per_nic_dxs,
      const std::array<tcpdirect::BufferManagerClientInterface*, kMaxNics>&
          per_nic_bufmgr,
      const std::array<std::string, kMaxNics>& nic_ips_by_idx);

  // Set per-peer connection (called after dxs::Connect / Accept handshake).
  void set_peer(int peer_rank, PeerConn conn);

  // Inbound socket pool. Receivers don't pre-match by source rank — they pull
  // a WireHeader off any of these and demux on header.source_rank.
  // v9: each pushed recv_sock is tagged with the local NIC index it was
  // accepted on, so the inbound thread knows which per_nic_reg to use.
  void push_inbound_recv_sock(std::unique_ptr<dxs::RecvSocketInterface> r,
                              int nic_idx) {
    inbound_recv_socks_.push_back(std::move(r));
    inbound_nic_idx_.push_back(nic_idx);
  }
  size_t num_inbound() const { return inbound_recv_socks_.size(); }
  dxs::RecvSocketInterface* inbound(size_t i) {
    return i < inbound_recv_socks_.size() ? inbound_recv_socks_[i].get()
                                          : nullptr;
  }
  int inbound_nic_idx(size_t i) const {
    return i < inbound_nic_idx_.size() ? inbound_nic_idx_[i] : 0;
  }

  // v9: per-NIC accessors. Returns nullptr if NIC i is not provisioned for
  // this CollComm.
  dxs::DxsClientInterface* per_nic_dxs(int i) const {
    return (i >= 0 && i < kMaxNics) ? per_nic_dxs_[i] : nullptr;
  }
  tcpdirect::BufferManagerClientInterface* per_nic_bufmgr(int i) const {
    return (i >= 0 && i < kMaxNics) ? per_nic_bufmgr_[i] : nullptr;
  }
  const std::array<tcpdirect::BufferManagerClientInterface*, kMaxNics>&
  per_nic_bufmgr_array() const {
    return per_nic_bufmgr_;
  }
  const std::string& nic_ip_by_idx(int i) const {
    static const std::string empty;
    return (i >= 0 && i < kMaxNics) ? nic_ips_by_idx_[i] : empty;
  }
  // Number of NICs actually provisioned (non-null dxs entries).
  int n_provisioned_nics() const { return n_provisioned_nics_; }

  PeerConn* peer(int peer_rank);
  size_t num_peers() const { return peers_.size(); }
  int rank() const { return rank_; }
  int dev() const { return dev_; }
  int nranks() const { return nranks_; }
  uint8_t fastrak_idx() const { return fastrak_idx_; }
  const std::string& nic_ip() const { return nic_ip_; }
  dxs::DxsClientInterface* dxs() { return dxs_; }
  tcpdirect::BufferManagerClientInterface* buffer_mgr() { return buf_; }

  // Memory registration.
  uint64_t register_memhandle(MemHandle h);
  MemHandle* lookup_memhandle(uint64_t key);
  void erase_memhandle(uint64_t key);

  // GDRCopy-mapped signal buffer (NCCL proxy shim's per-context signalsDev).
  // Lifetime: pinned on the FORCE_SO+CUDA RegMrSym call right after our
  // CreateContext, released on the matching DeregMrSym (or CollComm dtor).
  // Read by ProxyProgress inbound thread to do host-side atomic_add.
  void set_signal_buffer(GdrPinnedRegion region);
  uint64_t* signal_host_map() const { return signal_host_map_; }
  size_t signal_size_bytes() const { return signal_size_bytes_; }

  // Resolve a (signal_handle, signal_off) pair to a CPU-writable byte
  // address. Returns nullptr if no map can be found or if the offset is
  // out of range. signal_handle == 0 falls back to the primary FORCE_SO
  // signal buffer (for NCCL barrier path back-compat).
  uint8_t* signal_host_addr(uint64_t signal_handle, uint64_t signal_off);

  // v7 (S3): mutex serialising load+store+sfence RMW on signal slots.
  // The slots live in GDR write-combining memory where __atomic_fetch_add
  // does not become visible to the GPU view (v6 attempt failed). Mutex
  // makes multi-writer accumulation correct (DeepEP dispatch's N->1
  // reduction) without needing PCIe atomics. ~30ns per call.
  absl::Mutex* signal_mu() { return &signal_mu_; }

  // v20: dedicated stream for GPU atomicAdd kernel signal RMW. Lazy-init
  // on first call under signal_mu_. Per-CollComm so launches serialize
  // in wire_seq order; GPU atomicAdd handles cross-source atomicity.
  cudaStream_t signal_stream();

  // PR3c-ii (β port, 2026-05-16): accessor for the per-dst peer ack
  // high water array. Returns nullptr for out-of-range or before Init.
  // Read by Test() to gate; written by RunInbound via atomic_max.
  std::atomic<uint64_t>* peer_acked_high_water(int dst_rank) {
    if (dst_rank < 0 ||
        dst_rank >= static_cast<int>(peer_acked_high_water_n_) ||
        peer_acked_high_water_ == nullptr)
      return nullptr;
    return &peer_acked_high_water_[dst_rank];
  }

  // PR3c-iii (β port, 2026-05-16): emit a standalone kWireOpAck wire op
  // to `peer_rank` carrying our current recv_commit_seq[peer_rank] in
  // hdr.piggy_ack_high_water. Best-effort: returns OK if no new info to
  // ACK (idempotent), or if peer not yet connected. Called from
  // TickOutbound on every progress tick. This is the async out-of-band
  // ACK channel that breaks PR3c-ii circular deadlock — does NOT depend
  // on bidirectional payload traffic; works even when wrapper credit
  // stuck and no GFD posted.
  absl::Status EmitAck(int peer_rank, ScratchPool* sp);

  // v11: per-source-rank "next seq to commit" counter. The receiver uses
  // this to enforce cross-lane post-payload ordering when fanout > 1: an
  // inbound thread receives op with hdr.wire_seq==S, finishes the payload
  // RecvLinearized, then spin-waits until next_commit_seq[src]==S before
  // writing the signal RMW (or, for pure Iput, just bumping the counter).
  // This ensures a faster lane cannot let a later op's signal surface
  // before earlier ops on other lanes have landed their data. Counter is
  // 1-based; matches the sender's PeerConn::wire_seq_next start value.
  // Returns nullptr for out-of-range src.
  std::atomic<uint64_t>* recv_commit_seq(int src_rank) {
    if (src_rank < 0 || src_rank >= static_cast<int>(recv_commit_seq_n_) ||
        recv_commit_seq_ == nullptr)
      return nullptr;
    return &recv_commit_seq_[src_rank];
  }

  // v6 (S7): GinCtx instances register themselves so ~CollComm can stop
  // their progress threads BEFORE the CollComm's sockets / mhandle map
  // are destroyed. Without this the progress thread can dereference
  // stale CollComm state (use-after-free) when CloseColl runs before
  // DestroyContext.
  void register_ctx(GinCtx* ctx);
  void unregister_ctx(GinCtx* ctx);

 private:
  int dev_ = -1;
  uint8_t fastrak_idx_ = 0;
  std::string nic_ip_;
  int nranks_ = 0;
  int rank_ = -1;
  dxs::DxsClientInterface* dxs_ = nullptr;
  tcpdirect::BufferManagerClientInterface* buf_ = nullptr;

  std::vector<PeerConn> peers_;
  std::vector<std::unique_ptr<dxs::RecvSocketInterface>> inbound_recv_socks_;
  std::vector<int> inbound_nic_idx_;  // parallel to inbound_recv_socks_

  // v9: per-NIC handles. Slot i is non-null iff NIC index i is provisioned
  // for this CollComm. The primary (listen) NIC has its slot populated too.
  std::array<dxs::DxsClientInterface*, kMaxNics> per_nic_dxs_ = {};
  std::array<tcpdirect::BufferManagerClientInterface*, kMaxNics>
      per_nic_bufmgr_ = {};
  std::array<std::string, kMaxNics> nic_ips_by_idx_;
  int n_provisioned_nics_ = 0;

  absl::Mutex mh_mu_;
  absl::flat_hash_map<uint64_t, MemHandle> memhandles_ ABSL_GUARDED_BY(mh_mu_);
  uint64_t next_mh_key_ ABSL_GUARDED_BY(mh_mu_) = 1;

  // GDR-pinned signal buffer for the most recent GinCtx attached to this
  // CollComm. Access is read-mostly after attach; ProxyProgress inbound
  // reads signal_host_map_ (raw pointer) for hot-path atomic_add. The
  // GdrPinnedRegion holds the pin/map ownership.
  GdrPinnedRegion signal_region_;
  uint64_t* signal_host_map_ = nullptr;
  size_t signal_size_bytes_ = 0;

  // v6 (S7): list of GinCtx instances backed by this CollComm. Used by
  // ~CollComm to stop their progress threads before tearing down sockets.
  absl::Mutex ctx_mu_;
  std::vector<GinCtx*> ctxs_ ABSL_GUARDED_BY(ctx_mu_);

  // v7 (S3): see signal_mu() comment above.
  absl::Mutex signal_mu_;

  // PR3c-iii fix (2026-05-16): protect EmitAck access to per-peer
  // pending_ack_sops + last_ack_emitted_to_peer. Multiple GinCtx per
  // CollComm (test shows 4) -> 4 progress threads -> 4 concurrent EmitAck
  // on same peer -> race on vector erase + push_back -> SIGSEGV.
  // Single CollComm-wide mutex (vs per-peer) keeps it simple; EmitAck
  // body is small + only called from progress threads, so contention
  // negligible.
  absl::Mutex ack_mu_;
  cudaStream_t signal_stream_ = nullptr;  // v20

  // v11: per-source-rank receive-side commit-order counter. Sized to
  // nranks_ in Init. std::atomic isn't move/copy, so we hold them in a
  // unique_ptr array. Inbound threads read recv_commit_seq_[src_rank].
  std::unique_ptr<std::atomic<uint64_t>[]> recv_commit_seq_;
  size_t recv_commit_seq_n_ = 0;

  // PR3c-ii (β port, 2026-05-16): per-dst peer ack high water. Init 0
  // = sentinel "no ack from this dst yet"; Test() bypasses gate then.
  std::unique_ptr<std::atomic<uint64_t>[]> peer_acked_high_water_;
  size_t peer_acked_high_water_n_ = 0;

  // PR3a (β port, 2026-05-16): per-CollComm map of outstanding
  // (peer_rank, seq_num) -> RecvReqState. Key encoding mirrors AWS
  // nccl_ofi_gin.cpp:739-742: (uint64_t(peer_rank) << 16) | seq_num.
  // Inserted by inbound RX handlers (PR3c), removed by
  // retire_completed_peer_iput_ops (PR3c). Guarded by outstanding_reqs_mu_
  // because the receiver progress thread (inserting) and ack-emit code
  // (potentially erasing) run on different threads.
  absl::Mutex outstanding_reqs_mu_;
  absl::flat_hash_map<uint64_t, RecvReqState> outstanding_iput_signal_recv_reqs
      ABSL_GUARDED_BY(outstanding_reqs_mu_);
};

// Per createContext() instance: owns the GPU-visible proxy context and the
// host progress thread.
class GinCtx {
 public:
  explicit GinCtx(CollComm* coll) : coll_(coll) {}
  ~GinCtx();

  absl::Status Init(uint32_t queue_size, int n_counters, int n_signals);

  ProxyGpuCtxOwned* gpu_ctx() { return gpu_ctx_.get(); }
  CollComm* coll() { return coll_; }
  ProxyProgress* progress() { return progress_.get(); }
  ScratchPool* scratch() { return scratch_.get(); }

  void StartProgress();
  void StopProgress();

  // v6 (S7): called by ~CollComm when CollComm is being destroyed before
  // this GinCtx. After detach_coll() returns, coll() == nullptr; further
  // ABI calls on this GinCtx that need the CollComm should fail cleanly.
  void detach_coll() { coll_ = nullptr; }

 private:
  CollComm* coll_ = nullptr;          // not owned
  std::unique_ptr<ProxyGpuCtxOwned> gpu_ctx_;
  std::unique_ptr<ProxyProgress> progress_;
  std::unique_ptr<ScratchPool> scratch_;
};

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_PROXY_CONTEXT_H_
