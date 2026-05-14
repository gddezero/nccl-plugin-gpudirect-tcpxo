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

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "gin_provider/gdr_helper.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/scratch_pool.h"

namespace fastrak::gin {

class ProxyProgress;

// One per `plugin->listen(dev, ...)` call.
struct ListenComm {
  int dev = -1;
  uint8_t fastrak_idx = 0;
  std::string nic_ip;
  std::unique_ptr<dxs::ListenSocketInterface> listen_sock;
  uint64_t nonce = 0;       // randomized for the wire ListenHandle
  uint64_t listen_token = 0;
};

// Per-peer DXS connection inside a CollComm.
struct PeerConn {
  std::unique_ptr<dxs::SendSocketInterface> send_sock;
  std::unique_ptr<dxs::RecvSocketInterface> recv_sock;
};

// Memory registration: holds the local DXS Reg plus the array of peer Regs
// gathered out-of-band by NCCL after RegMrSym (peer regs are stored in the
// `ginHandle` that NCCL distributes to peers; we look them up by mhandle).
struct MemHandle {
  void*    base = nullptr;
  size_t   bytes = 0;
  int      ptr_type = 0;     // NCCL_PTR_HOST / CUDA / DMABUF
  int      dmabuf_fd = -1;
  uint64_t dmabuf_offset = 0;
  dxs::Reg local_reg = 0;
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

  absl::Status Init(int dev, uint8_t fastrak_idx, std::string nic_ip,
                    int nranks, int rank,
                    dxs::DxsClientInterface* absl_nonnull dxs,
                    tcpdirect::BufferManagerClientInterface* absl_nonnull buf);

  // Set per-peer connection (called after dxs::Connect / Accept handshake).
  void set_peer(int peer_rank, PeerConn conn);

  // Inbound socket pool. Receivers don't pre-match by source rank — they pull
  // a WireHeader off any of these and demux on header.source_rank.
  void push_inbound_recv_sock(std::unique_ptr<dxs::RecvSocketInterface> r) {
    inbound_recv_socks_.push_back(std::move(r));
  }
  size_t num_inbound() const { return inbound_recv_socks_.size(); }
  dxs::RecvSocketInterface* inbound(size_t i) {
    return i < inbound_recv_socks_.size() ? inbound_recv_socks_[i].get()
                                          : nullptr;
  }

  PeerConn* peer(int peer_rank);
  size_t num_peers() const { return peers_.size(); }
  int rank() const { return rank_; }
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

 private:
  CollComm* coll_ = nullptr;          // not owned
  std::unique_ptr<ProxyGpuCtxOwned> gpu_ctx_;
  std::unique_ptr<ProxyProgress> progress_;
  std::unique_ptr<ScratchPool> scratch_;
};

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_PROXY_CONTEXT_H_
