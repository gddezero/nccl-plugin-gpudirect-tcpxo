/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * GIN PROXY mode plugin entrypoint. Implements ncclGinPlugin_v13 backed by
 * the existing FasTrak DXS / Falcon transport.
 */

#include "gin_provider/plugin_main.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"
#include "gin_provider/gdr_helper.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/listen_handle.h"
#include "gin_provider/proxy_context.h"
#include "gin_provider/proxy_progress.h"
#include "gin_provider/scratch_pool.h"
#include "gin_provider/wire_protocol.h"
#include <optional>
#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include "nccl_cuda/cuda_common.h"
#include "plugin/nccl_net.h"
#include "plugin/net/net_v12.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/init.h"
#include "tcpdirect_plugin/fastrak_offload/nic_client_router.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"

namespace fastrak::gin {
namespace {

constexpr const char* kPluginName = "fastrak-gin-proxy";
constexpr int kSocketReadyTimeoutMs = 10000;

std::atomic<bool> g_initialized{false};
// v6 (S5): g_has_error storage moved to proxy_context.cc so both this
// translation unit and proxy_progress.cc can SetGinError() via the
// inline helper in proxy_context.h. QueryLastError below reads it.

ncclResult_t StatusToNccl(const absl::Status& s) {
  if (s.ok()) return ncclSuccess;
  switch (s.code()) {
    case absl::StatusCode::kInvalidArgument:
      return ncclInvalidArgument;
    case absl::StatusCode::kUnimplemented:
      return ncclSystemError;
    default:
      return ncclInternalError;
  }
}

// Spin-wait for an async DXS socket to reach the connected/listening state.
template <typename SockT>
absl::Status WaitSocketReady(SockT& sock, absl::string_view what) {
  auto deadline = absl::Now() + absl::Milliseconds(kSocketReadyTimeoutMs);
  while (true) {
    auto status = sock.SocketReady();
    if (status.has_value()) return *status;
    if (absl::Now() > deadline) {
      return absl::DeadlineExceededError(
          absl::StrCat(what, ": SocketReady timed out"));
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

// Pick a uint64 nonce. Process-local randomness; we mix in the address of a
// stack variable to make collisions across listens within the same process
// vanishingly unlikely.
uint64_t MakeNonce() {
  static std::atomic<uint64_t> ctr{0};
  uint64_t local;
  uint64_t addr = reinterpret_cast<uint64_t>(&local);
  return (ctr.fetch_add(0x9E3779B97F4A7C15ULL, std::memory_order_relaxed) ^
          addr ^ static_cast<uint64_t>(absl::ToUnixNanos(absl::Now())));
}

// Map an NCCL device index (0..kNcclNetIfs-1, post-IFNAME filtering) to the
// FasTrak NIC index that we expose on the wire. The v7 plugin keys this off
// the GPU PCI addr (because each GPU pairs with one NIC), but the GIN ABI
// gives us only the NIC dev — we just use the dev index directly.
absl::StatusOr<uint8_t> DeviceToFastrakIdx(int dev) {
  if (dev < 0 || dev >= kNcclNetIfs) return absl::InvalidArgumentError("dev OOR");
  return static_cast<uint8_t>(dev);
}

// ----- 17 ABI callbacks -----

ncclResult_t Init(void** ctx, uint64_t commId,
                  ncclDebugLogger_t logFunction) {
  if (ctx == nullptr) return ncclInvalidArgument;
  absl::Status s = fastrak::PluginCoreInit(logFunction);
  if (!s.ok()) {
    LOG(ERROR) << "FasTrak GIN init failed: " << s;
    return StatusToNccl(s);
  }
  static int sentinel = 0;
  *ctx = &sentinel;
  // v6 (S5): clear plugin-wide error flag at every (re)init.
  g_has_error.store(false, std::memory_order_release);
  g_initialized.store(true, std::memory_order_release);
  LOG(INFO) << absl::StrFormat(
      "FasTrak GIN provider (PROXY mode) init: commId=%lu, ndev=%d", commId,
      fastrak::kNcclNetIfs);
  return ncclSuccess;
}

ncclResult_t Devices(int* ndev) {
  if (ndev == nullptr) return ncclInvalidArgument;
  *ndev = fastrak::kNcclNetIfs;
  return ncclSuccess;
}

ncclResult_t GetProperties(int dev, ncclNetProperties_v12_t* props) {
  if (props == nullptr) return ncclInvalidArgument;
  if (dev < 0 || dev >= fastrak::kNcclNetIfs) return ncclInvalidArgument;
  std::memset(props, 0, sizeof(*props));
  const auto& d = fastrak::kNcclSocketDevs[dev];
  props->name = const_cast<char*>(d.dev_name);
  props->pciPath = const_cast<char*>(d.pci_path);
  props->guid = static_cast<uint64_t>(dev);
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 200000;
  props->port = 0;
  props->latency = 5.0f;
  props->maxComms = 64;
  props->maxRecvs = 1;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  props->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_NET_SIZE;
  props->maxMultiRequestSize = 1;
  props->railId = static_cast<int16_t>(dev);
  props->planeId = static_cast<int16_t>(dev / 4);
  return ncclSuccess;
}

ncclResult_t Listen(void* /*ctx*/, int dev, void* handle,
                    void** listenComm) {
  if (handle == nullptr || listenComm == nullptr) return ncclInvalidArgument;
  if (dev < 0 || dev >= fastrak::kNcclNetIfs) return ncclInvalidArgument;
  const auto& d = fastrak::kNcclSocketDevs[dev];
  if (d.pci_path == nullptr) {
    LOG(ERROR) << "GIN Listen on ctrl/non-fastrak dev " << dev;
    return ncclInvalidArgument;
  }

  auto idx_or = DeviceToFastrakIdx(dev);
  if (!idx_or.ok()) return StatusToNccl(idx_or.status());
  const std::string nic_ip = d.ip_addr;

  auto dxs_or = fastrak::GetNicClientRouter().GetDxsClient(nic_ip);
  if (!dxs_or.ok()) return StatusToNccl(dxs_or.status());
  auto* dxs = *dxs_or;

  auto listen_or = dxs->Listen();
  if (!listen_or.ok()) return StatusToNccl(listen_or.status());
  auto listen_sock = std::move(*listen_or);

  if (auto s = WaitSocketReady(*listen_sock, "GIN Listen"); !s.ok()) {
    return StatusToNccl(s);
  }

  auto lc = std::make_unique<ListenComm>();
  lc->dev = dev;
  lc->fastrak_idx = *idx_or;
  lc->nic_ip = nic_ip;
  lc->nonce = MakeNonce();
  lc->listen_token = MakeNonce();

  // Encode the wire handle.
  ListenHandle h;
  std::memset(&h, 0, sizeof(h));
  h.magic = kListenHandleMagic;
  h.version = kListenHandleVersion;
  // For a3-mega, dxs addresses are IPv4 strings. Convert to packed bytes.
  in_addr in;
  if (inet_pton(AF_INET, nic_ip.c_str(), &in) == 1) {
    h.addr_family = AF_INET;
    std::memcpy(h.addr, &in.s_addr, 4);
  } else {
    in6_addr in6;
    if (inet_pton(AF_INET6, nic_ip.c_str(), &in6) == 1) {
      h.addr_family = AF_INET6;
      std::memcpy(h.addr, &in6.s6_addr, 16);
    } else {
      LOG(ERROR) << "GIN Listen: cannot parse NIC ip " << nic_ip;
      return ncclInternalError;
    }
  }
  h.port = static_cast<uint16_t>(listen_sock->Port());
  h.fastrak_idx = lc->fastrak_idx;
  h.nonce = lc->nonce;
  h.listen_token = lc->listen_token;
  EncodeListenHandle(handle, h);

  lc->listen_sock = std::move(listen_sock);

  LOG(INFO) << absl::StrFormat(
      "GIN Listen: dev=%d (%s) ip=%s port=%u fastrak_idx=%u",
      dev, d.dev_name, nic_ip, h.port, h.fastrak_idx);

  *listenComm = lc.release();
  return ncclSuccess;
}

ncclResult_t Connect(void* /*ctx*/, void* handles[], int nranks, int rank,
                     void* listenComm, void** collComm) {
  if (handles == nullptr || listenComm == nullptr || collComm == nullptr ||
      nranks <= 0 || rank < 0 || rank >= nranks) {
    return ncclInvalidArgument;
  }
  auto* lc = static_cast<ListenComm*>(listenComm);
  auto dxs_or = fastrak::GetNicClientRouter().GetDxsClient(lc->nic_ip);
  if (!dxs_or.ok()) return StatusToNccl(dxs_or.status());
  auto* dxs = *dxs_or;
  auto buf_or = fastrak::GetNicClientRouter().GetBufferManagerClient(lc->nic_ip);
  if (!buf_or.ok()) return StatusToNccl(buf_or.status());

  auto cc = std::make_unique<CollComm>();
  if (auto s = cc->Init(lc->dev, lc->fastrak_idx, lc->nic_ip, nranks, rank,
                        dxs, *buf_or);
      !s.ok()) {
    return StatusToNccl(s);
  }

  // M6.2 fan-out: open `fanout` parallel sockets per peer instead of one.
  // Each peer is doing the same concurrently; issuing them all up front
  // before draining ready-ness avoids n*(n-1) handshake deadlocks.
  const int fanout = FanoutPerPeer();
  // pending[r] holds `fanout` SendSocket unique_ptrs for peer r (indices
  // not yet ready are non-null; entries become null as they are moved into
  // PeerConn.send_socks).
  std::vector<std::vector<std::unique_ptr<dxs::SendSocketInterface>>> pending(
      nranks);
  // accumulate ready sockets per peer here, only flush into PeerConn once
  // all `fanout` lanes for that peer have come up (set_peer takes a value).
  std::vector<std::vector<std::unique_ptr<dxs::SendSocketInterface>>> ready(
      nranks);
  std::vector<int> ready_count(nranks, 0);
  for (int r = 0; r < nranks; ++r) {
    if (r == rank) continue;
    if (handles[r] == nullptr) {
      LOG(ERROR) << "GIN Connect: handles[" << r << "] is null";
      return ncclInvalidArgument;
    }
    ListenHandle h = DecodeListenHandle(handles[r]);
    if (h.magic != kListenHandleMagic || h.version != kListenHandleVersion) {
      LOG(ERROR) << absl::StrFormat(
          "GIN Connect: handles[%d] invalid magic=0x%x version=%u", r, h.magic,
          h.version);
      return ncclInvalidArgument;
    }
    char addr_str[INET6_ADDRSTRLEN] = {0};
    if (h.addr_family == AF_INET) {
      inet_ntop(AF_INET, h.addr, addr_str, sizeof(addr_str));
    } else if (h.addr_family == AF_INET6) {
      inet_ntop(AF_INET6, h.addr, addr_str, sizeof(addr_str));
    } else {
      LOG(ERROR) << "GIN Connect: handles[" << r << "] bad addr_family";
      return ncclInvalidArgument;
    }
    pending[r].resize(fanout);
    ready[r].reserve(fanout);
    for (int lane = 0; lane < fanout; ++lane) {
      auto sock_or = dxs->Connect(addr_str, h.port);
      if (!sock_or.ok()) {
        LOG(ERROR) << "GIN Connect: dxs->Connect lane=" << lane << " to rank "
                   << r << " failed: " << sock_or.status();
        return StatusToNccl(sock_or.status());
      }
      pending[r][lane] = std::move(*sock_or);
    }
  }

  // Interleaved progress loop: drain pending outbound + accept inbound
  // until both sides have all (nranks-1) * fanout sockets.
  const size_t inbound_target =
      static_cast<size_t>(nranks - 1) * static_cast<size_t>(fanout);
  size_t accepted = 0;
  auto deadline = absl::Now() + absl::Seconds(120);
  while (true) {
    // Drain pending outbound (per peer per lane).
    for (int r = 0; r < nranks; ++r) {
      if (r == rank) continue;
      bool peer_all_done = true;
      for (int lane = 0; lane < fanout; ++lane) {
        if (pending[r][lane] == nullptr) continue;
        auto status = pending[r][lane]->SocketReady();
        if (!status.has_value()) {
          peer_all_done = false;
          continue;
        }
        if (!status->ok()) {
          LOG(ERROR) << "GIN Connect outbound to rank " << r << " lane "
                     << lane << " failed: " << *status;
          return StatusToNccl(*status);
        }
        ready[r].push_back(std::move(pending[r][lane]));
        pending[r][lane] = nullptr;
        ++ready_count[r];
      }
      if (peer_all_done && ready_count[r] == fanout && !ready[r].empty()) {
        PeerConn pc;
        pc.send_socks = std::move(ready[r]);
        cc->set_peer(r, std::move(pc));
        ready[r].clear();
      }
    }

    // Drain inbound accepts.
    if (accepted < inbound_target) {
      auto sock_or = lc->listen_sock->Accept();
      if (!sock_or.ok()) {
        LOG(ERROR) << "GIN Connect: listen->Accept failed: " << sock_or.status();
        return StatusToNccl(sock_or.status());
      }
      if (*sock_or != nullptr) {
        auto sock = std::move(*sock_or);
        if (auto s = WaitSocketReady(*sock, "GIN Connect inbound"); !s.ok()) {
          LOG(ERROR) << "GIN inbound recv socket not ready: " << s;
          return StatusToNccl(s);
        }
        cc->push_inbound_recv_sock(std::move(sock));
        accepted++;
      }
    }

    bool all_outbound_done = true;
    for (int r = 0; r < nranks; ++r) {
      if (r == rank) continue;
      if (ready_count[r] != fanout) {
        all_outbound_done = false;
        break;
      }
    }
    if (all_outbound_done && accepted >= inbound_target) break;
    if (absl::Now() > deadline) {
      LOG(ERROR) << "GIN Connect: handshake timed out, accepted=" << accepted
                 << " of " << inbound_target;
      return ncclSystemError;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  LOG(INFO) << absl::StrFormat(
      "GIN Connect: dev=%d rank=%d/%d mesh established (fanout=%d, "
      "out=%d, in=%zu)",
      lc->dev, rank, nranks, fanout, (nranks - 1) * fanout, accepted);

  *collComm = cc.release();
  return ncclSuccess;
}

ncclResult_t CreateContext(void* collComm, ncclGinConfig_v13_t* config,
                           void** ginCtx,
                           ncclNetDeviceHandle_v11_t** devHandle) {
  if (collComm == nullptr || config == nullptr || ginCtx == nullptr) {
    return ncclInvalidArgument;
  }
  // devHandle may be NULL — NCCL passes nullptr when wrapping us with its
  // own gin_host_proxy that builds the device-visible blob itself.
  auto* cc = static_cast<CollComm*>(collComm);

  // Round queue depth up to the next power of two so device-side mask works.
  uint32_t qd = config->queueDepth > 0 ? config->queueDepth : 1024;
  uint32_t qd_pow2 = 1;
  while (qd_pow2 < qd) qd_pow2 <<= 1;

  auto gctx = std::make_unique<GinCtx>(cc);
  if (auto s = gctx->Init(qd_pow2, config->nCounters, config->nSignals);
      !s.ok()) {
    return StatusToNccl(s);
  }

  auto* gpu = gctx->gpu_ctx();
  // Only fill devHandle when NCCL actually wants one. In gin_host_proxy mode
  // it passes devHandle=nullptr because NCCL builds its own device blob.
  if (devHandle != nullptr) {
    static_assert(sizeof(ncclNetDeviceHandle_v11_t) <= 256, "devHandle small");
    auto* dh = new (std::nothrow) ncclNetDeviceHandle_v11_t{};
    if (dh == nullptr) return ncclSystemError;
    dh->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    dh->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
    dh->handle = gpu->dev_view;
    dh->size = sizeof(*gpu->dev_view);
    dh->needsProxyProgress = 1;
    *devHandle = dh;
  }

  // Start the host-side proxy progress thread which will drain the GFD ring.
  // No-op until M3 wires actual dxs::Send dispatching.
  gctx->StartProgress();

  *ginCtx = gctx.release();

  LOG(INFO) << absl::StrFormat(
      "GIN CreateContext: dev=%d rank=%d nranks=%d queueDepth=%u nC=%d nS=%d",
      cc->fastrak_idx(), cc->rank(), cc->nranks(), qd_pow2,
      config->nCounters, config->nSignals);
  return ncclSuccess;
}

ncclResult_t RegMrSym(void* collComm, void* data, size_t size, int type,
                      uint64_t mrFlags, void** mhandle, void** ginHandle) {
  if (collComm == nullptr || mhandle == nullptr) return ncclInvalidArgument;
  auto* cc = static_cast<CollComm*>(collComm);

  MemHandle mh;
  mh.base = data;
  mh.bytes = size;
  mh.ptr_type = type;

  if (type & NCCL_PTR_CUDA) {
    auto fd_or = fastrak::getDmabufFd(data, size, /*pci_addr=*/"");
    if (!fd_or.ok()) {
      LOG(ERROR) << "RegMrSym: getDmabufFd failed: " << fd_or.status();
      return StatusToNccl(fd_or.status());
    }
    mh.dmabuf_fd = *fd_or;
    auto reg_or = cc->buffer_mgr()->RegBuf(mh.dmabuf_fd, size);
    if (!reg_or.ok()) {
      LOG(ERROR) << "RegMrSym: RegBuf failed: " << reg_or.status();
      return StatusToNccl(reg_or.status());
    }
    mh.local_reg = *reg_or;
  } else {
    // Host memory: nothing to register with DXS for now.
    mh.local_reg = 0;
  }
  mh.peer_regs.resize(cc->nranks(), 0);
  mh.peer_regs[cc->rank()] = mh.local_reg;

  // M6: GDR-pin every CUDA registration so PutSignal/Signal targeting
  // arbitrary cudaMalloc'd scratch (e.g. DeepEP dispatch signal slots)
  // can be served via host-side atomic writes. Skip large buffers to
  // avoid eating BAR1 unnecessarily — bulk dispatch payload buffers
  // never receive signals, so we cap at 64 MiB. If a buffer above the
  // cap actually gets a signal write, we'll log and the wire path
  // falls back to the primary FORCE_SO map (or fails noisily).
  constexpr size_t kGdrPinSizeCap = 64ull * 1024 * 1024;
  constexpr uint64_t kForceSO = 1ull << 0;  // NCCL_NET_MR_FLAG_FORCE_SO
  bool is_force_so = (mrFlags & kForceSO) != 0;
  if ((type & NCCL_PTR_CUDA) && GdrAvailable() &&
      (is_force_so || size <= kGdrPinSizeCap)) {
    auto pin_or = GdrPinnedRegion::Create(data, size);
    if (pin_or.ok()) {
      auto pin_sp = std::make_shared<GdrPinnedRegion>(std::move(*pin_or));
      // For FORCE_SO (NCCL barrier signalsDev), also publish to the
      // primary slot for the legacy fallback path.
      if (is_force_so) {
        // Cache before we hand pin_sp off into the MemHandle: the
        // primary slot keeps an independent copy of the host map by
        // re-pinning is wasteful, so we instead share via the
        // shared_ptr (set_signal_buffer takes ownership of a copy).
        // Easiest: pin twice for FORCE_SO (small buffer, ~192B) — keeps
        // primary slot self-contained.
        auto pin2_or = GdrPinnedRegion::Create(data, size);
        if (pin2_or.ok()) {
          cc->set_signal_buffer(std::move(*pin2_or));
        } else {
          LOG(WARNING) << "RegMrSym: 2nd GDR pin of FORCE_SO failed: "
                       << pin2_or.status();
        }
      }
      mh.gdr_pin = std::move(pin_sp);
      static std::atomic<int> pin_dbg{0};
      if (pin_dbg.fetch_add(1) < 16) {
        LOG(INFO) << "RegMrSym: GDR-pinned base=" << data
                  << " size=" << size
                  << " host_map=" << mh.gdr_pin->host_map()
                  << " force_so=" << is_force_so;
      }
    } else {
      static std::atomic<int> pf_dbg{0};
      if (pf_dbg.fetch_add(1) < 8) {
        LOG(WARNING) << "RegMrSym: GDR pin failed for base=" << data
                     << " size=" << size << ": " << pin_or.status();
      }
    }
  } else if ((type & NCCL_PTR_CUDA) && is_force_so && !GdrAvailable()) {
    LOG(WARNING) << "RegMrSym: FORCE_SO buffer registered but GDRCopy "
                    "unavailable — signal writes will fail";
  }

  uint64_t key = cc->register_memhandle(std::move(mh));
  *mhandle = reinterpret_cast<void*>(key);
  if (ginHandle != nullptr) {
    // For now publish the local Reg key directly. Real impl will encode a
    // {rank, reg, base, size} tuple and let NCCL exchange it OOB.
    *ginHandle = reinterpret_cast<void*>(key);
  }
  return ncclSuccess;
}

ncclResult_t RegMrSymDmaBuf(void* collComm, void* data, size_t size, int type,
                            uint64_t /*offset*/, int fd, uint64_t mrFlags,
                            void** mhandle, void** ginHandle) {
  if (collComm == nullptr || mhandle == nullptr || fd < 0) {
    return ncclInvalidArgument;
  }
  auto* cc = static_cast<CollComm*>(collComm);
  MemHandle mh;
  mh.base = data;
  mh.bytes = size;
  mh.ptr_type = type;
  mh.dmabuf_fd = fd;
  auto reg_or = cc->buffer_mgr()->RegBuf(fd, size);
  if (!reg_or.ok()) return StatusToNccl(reg_or.status());
  mh.local_reg = *reg_or;
  mh.peer_regs.resize(cc->nranks(), 0);
  mh.peer_regs[cc->rank()] = mh.local_reg;

  // Same M6 GDR-pin policy as RegMrSym above. DMA-BUF is the path NCCL
  // takes when our ptrSupport advertises NCCL_PTR_DMABUF (this is our
  // primary path on a3-mega; gin_host_proxy.cc::ncclGinProxyRegMrSym
  // routes through here when fd >= 0).
  constexpr size_t kGdrPinSizeCap = 64ull * 1024 * 1024;
  constexpr uint64_t kForceSO = 1ull << 0;  // NCCL_NET_MR_FLAG_FORCE_SO
  bool is_force_so = (mrFlags & kForceSO) != 0;
  if ((type & NCCL_PTR_CUDA) && GdrAvailable() &&
      (is_force_so || size <= kGdrPinSizeCap)) {
    auto pin_or = GdrPinnedRegion::Create(data, size);
    if (pin_or.ok()) {
      auto pin_sp = std::make_shared<GdrPinnedRegion>(std::move(*pin_or));
      if (is_force_so) {
        auto pin2_or = GdrPinnedRegion::Create(data, size);
        if (pin2_or.ok()) {
          cc->set_signal_buffer(std::move(*pin2_or));
        } else {
          LOG(WARNING) << "RegMrSymDmaBuf: 2nd GDR pin of FORCE_SO failed: "
                       << pin2_or.status();
        }
      }
      mh.gdr_pin = std::move(pin_sp);
      static std::atomic<int> pin_dbg{0};
      if (pin_dbg.fetch_add(1) < 16) {
        LOG(INFO) << "RegMrSymDmaBuf: GDR-pinned base=" << data
                  << " size=" << size
                  << " host_map=" << mh.gdr_pin->host_map()
                  << " force_so=" << is_force_so;
      }
    } else {
      static std::atomic<int> pf_dbg{0};
      if (pf_dbg.fetch_add(1) < 8) {
        LOG(WARNING) << "RegMrSymDmaBuf: GDR pin failed for base=" << data
                     << " size=" << size << ": " << pin_or.status();
      }
    }
  } else if ((type & NCCL_PTR_CUDA) && is_force_so && !GdrAvailable()) {
    LOG(WARNING) << "RegMrSymDmaBuf: FORCE_SO buffer registered but "
                    "GDRCopy unavailable — signal writes will fail";
  }

  uint64_t key = cc->register_memhandle(std::move(mh));
  *mhandle = reinterpret_cast<void*>(key);
  if (ginHandle != nullptr) *ginHandle = reinterpret_cast<void*>(key);
  return ncclSuccess;
}

ncclResult_t DeregMrSym(void* collComm, void* mhandle) {
  if (collComm == nullptr) return ncclInvalidArgument;
  auto* cc = static_cast<CollComm*>(collComm);
  uint64_t key = reinterpret_cast<uint64_t>(mhandle);
  auto* mh = cc->lookup_memhandle(key);
  if (mh == nullptr) return ncclInvalidArgument;
  if (mh->local_reg != 0) {
    auto s = cc->buffer_mgr()->DeregBuf(mh->local_reg);
    if (!s.ok()) {
      LOG(ERROR) << "DeregMrSym: DeregBuf failed: " << s;
    }
  }
  cc->erase_memhandle(key);
  return ncclSuccess;
}

ncclResult_t DestroyContext(void* ginCtx) {
  if (ginCtx != nullptr) delete static_cast<GinCtx*>(ginCtx);
  return ncclSuccess;
}

ncclResult_t CloseColl(void* collComm) {
  if (collComm != nullptr) delete static_cast<CollComm*>(collComm);
  return ncclSuccess;
}

ncclResult_t CloseListen(void* listenComm) {
  if (listenComm != nullptr) delete static_cast<ListenComm*>(listenComm);
  return ncclSuccess;
}

// Per-iput request handle: holds the in-flight DXS SendOps. Test() polls
// them for completion.
//
// v6 (R1): sig_op was a dead field — never assigned, but Test() polled it
// (nullptr → success). Dropped to avoid future maintainer confusion. If
// signal-as-separate-Send is ever wanted, add it back with a clear owner.
struct GinRequest {
  std::unique_ptr<dxs::SendOpInterface> hdr_op;
  std::unique_ptr<dxs::SendOpInterface> pay_op;
  CollComm* coll = nullptr;
  uint32_t  scratch_slot = UINT32_MAX;
};

// v7: per-thread TX scratch ring slot counter, indexed by `rank % 64`.
// Threads with rank index aliasing each other share the same ring; modulo
// kTxSlotsPerPeer=1024 keeps wrap safe.
//
// History (Task A attempt): v7 first tried promoting this to a per-peer
// atomic to close the cross-thread same-slot race exposed by the v5 edge
// stability sweep (SIGBUS at FANOUT>=2). That change broke data
// integrity at op 0 in the FANOUT=3 PP test (rank 1 mismatch) — likely
// because re-deriving lane from the same atomic shifted lane assignment
// in a way that re-tripped a latent dxs ordering quirk. Reverted.
// Investigation is logged in deepep_perf_v7.md "Task A status"; v6
// fanout=2/3 was empirically not tripping the SIGBUS in re-runs, so
// the v5 report's race may be conditional on additional state we
// haven't reproduced. Filed as known issue.
constexpr size_t kTlsTxSeqLen = 64;
static thread_local uint32_t tls_tx_seq[kTlsTxSeqLen] = {0};

// v6 (R2): signal_op_arg dropped from the IputCommon signature — it was
// received from IputSignal/Iget/Iflush callers and discarded. The wire
// has no field for it; if NCCL ever stops always meaning ADD here we
// should add a WireHeader.signal_op byte and route it. Until then,
// signalOp is silently treated as ADD.
static ncclResult_t IputCommon(void* ginCtx, int /*context*/,
                               uint64_t srcOff, void* srcMhandle, size_t size,
                               uint64_t dstOff, void* dstMhandle,
                               uint32_t rank, void** request,
                               WireOp wire_op, uint64_t signal_off,
                               void* signalMhandle, uint64_t signal_val) {
  if (ginCtx == nullptr || request == nullptr) {
    LOG(ERROR) << "IputCommon: ginCtx or request null";
    return ncclInvalidArgument;
  }
  auto* gctx = static_cast<GinCtx*>(ginCtx);
  auto* cc = gctx->coll();
  auto* sp = gctx->scratch();
  // v5 (M6.6): TLS dbg counter — replaces static std::atomic<int> dbg_count.
  // IputCommon is the hot Iput path; called by every NCCL proxy thread per
  // chunk. Contended atomic fetch_add added per-call cache-line bounces.
  thread_local int dbg_count_tls = 0;
  if (dbg_count_tls < 50) {
    ++dbg_count_tls;
    LOG(INFO) << "IputCommon DBG #" << dbg_count_tls
              << " op=" << wire_op << " rank=" << rank
              << " size=" << size << " sig_off=" << signal_off
              << " sig_val=" << signal_val
              << " my_rank=" << (cc ? cc->rank() : -1);
  }
  if (cc == nullptr || sp == nullptr) {
    LOG(ERROR) << "IputCommon: coll or scratch null";
    return ncclInternalError;
  }

  // NCCL proxy shim's gin_host_proxy.cc:107 iterates `for (int targetRank=0;
  // targetRank < ctx->nRanks; targetRank++)` — so the rank param is the
  // FULL rank, including self. Self-signal: handle locally (atomic-add to
  // our own signal_host_map), no socket send needed.
  int my_rank = cc->rank();
  int global_rank = static_cast<int>(rank);
  if (global_rank == my_rank) {
    // Self-signal short-circuit. Only Signal/PutSignal need handling here;
    // self-Put without signal would target our own memory directly.
    if (wire_op == kWireOpSignal || wire_op == kWireOpPutSignal) {
      uint64_t sig_h = reinterpret_cast<uint64_t>(signalMhandle);
      uint8_t* slot_b = cc->signal_host_addr(sig_h, signal_off);
      if (slot_b != nullptr) {
        // v7 (S3 fix): per-CollComm signal_mu_ serialises this self-
        // signal RMW with concurrent inbound writes from peers (DeepEP
        // dispatch reduction lands here too).
        auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(slot_b);
        absl::MutexLock l(cc->signal_mu());
        uint64_t prev = *slot_u64;
        uint64_t next = prev + signal_val;
        *slot_u64 = next;
        __asm__ __volatile__("sfence" ::: "memory");
        // v5: TLS — see dbg_count_tls.
        thread_local int ss_dbg_tls = 0;
        if (ss_dbg_tls < 4) {
          ++ss_dbg_tls;
          LOG(INFO) << "self-signal: sig_h=0x" << std::hex << sig_h
                    << " off=" << std::dec << signal_off
                    << " prev=" << prev << " new=" << next;
        }
      } else {
        LOG(ERROR) << "IputCommon self-signal: no signal map (sig_h=0x"
                   << std::hex << sig_h << std::dec
                   << " off=" << signal_off
                   << " primary_size=" << cc->signal_size_bytes() << ")";
        SetGinError("Iput:self-signal:no-map");
      }
    }
    // Fabricate a no-op request that reports done immediately.
    auto req = std::make_unique<GinRequest>();
    req->coll = cc;
    req->scratch_slot = 0;
    *request = req.release();
    return ncclSuccess;
  }
  PeerConn* peer = cc->peer(global_rank);
  if (peer == nullptr || peer->send_socks.empty()) {
    LOG(ERROR) << "IputCommon: bad peer rank=" << rank
               << " my_rank=" << my_rank
               << " num_peers=" << cc->num_peers();
    return ncclInvalidArgument;
  }
  // M6.2: pick a lane round-robin so successive Iputs to the same peer
  // stripe across the per-peer socket pool. Both hdr Send and payload
  // Send must use the SAME socket (TCP/dxs guarantees in-order on a
  // single sock; receiver pulls hdr-then-payload off the same RecvSock).
  //
  // Size guard: small ops (signal/tiny payloads) are control-plane bound
  // and benefit from co-locating on lane 0 — fan-out adds receiver-side
  // multi-thread overhead that overwhelms the wire saving when each op
  // is sub-millisecond on the wire. Threshold tunable via NCCL_GIN_FANOUT_MIN
  // (default 1 MiB; small-tensor PP at 512KB stays on lane 0).
  // v8: lowered default 1 MiB -> 16 KiB. The receiver has FANOUT inbound
  // threads each owning a recv socket; with the old 1 MiB threshold,
  // sub-MiB ops + signal/PutSignal traffic all funneled into lane 0,
  // leaving lanes 1..N-1 idle on small-msg-heavy phases (DeepEP combine).
  // 16 KiB is below the per-token chunk size for typical PP/dispatch and
  // safely above any single-cacheline signal: signals (size==0) still take
  // lane 0 due to the size==0 check, so no race on the GDR signal slot.
  static const size_t kFanoutMinBytes = []() {
    const char* v = std::getenv("NCCL_GIN_FANOUT_MIN");
    if (v == nullptr || *v == 0) return size_t{16ull << 10};  // 16 KiB
    long n = std::atol(v);
    return n > 0 ? static_cast<size_t>(n) : size_t{16ull << 10};
  }();
  const size_t fanout = peer->send_socks.size();
  size_t lane = 0;
  if (size >= kFanoutMinBytes && fanout > 1) {
    uint64_t lane_seq = peer->tx_seq.fetch_add(1, std::memory_order_relaxed);
    lane = lane_seq % fanout;
  }
  dxs::SendSocketInterface* sock = peer->send_socks[lane].get();
  if (sock == nullptr) {
    LOG(ERROR) << "IputCommon: lane " << lane << " send_sock null";
    return ncclInternalError;
  }
  // v6 (S8): NCCL packs srcHandle/dstHandle into a 63-bit field with bit
  // 0 reserved as a flag. We pass the RAW key on the wire so the receiver
  // can decide masking policy itself (lookup_memhandle tries both
  // masked-and-raw to support legacy callers that don't set the flag).
  uint64_t src_key = reinterpret_cast<uint64_t>(srcMhandle);
  uint64_t dst_key = reinterpret_cast<uint64_t>(dstMhandle);
  uint64_t sig_key = reinterpret_cast<uint64_t>(signalMhandle);
  // Only resolve src_mh when there's actually a payload to send. Signal
  // / flush ops legitimately call with srcMhandle=NULL → key=0, no need
  // to log a MISS.
  bool has_payload =
      (size > 0 && wire_op != kWireOpSignal && wire_op != kWireOpFlush);
  MemHandle* src_mh = has_payload ? cc->lookup_memhandle(src_key) : nullptr;
  if (has_payload && (src_mh == nullptr || src_mh->local_reg == 0)) {
    // v6 (fault tolerance): a registration mismatch is a usage error
    // — make it visible via QueryLastError so NCCL can fail-fast its
    // proxy progress, instead of waiting forever on Test().
    LOG(ERROR) << "IputCommon: bad src_mh key=0x" << std::hex << src_key
               << " size=" << std::dec << size << " wire_op=" << wire_op;
    SetGinError("IputCommon:bad-src-mh");
    return ncclInvalidArgument;
  }

  WireHeader hdr{};
  hdr.magic = kWireMagic;
  hdr.op = static_cast<uint16_t>(wire_op);
  hdr.source_rank = static_cast<uint32_t>(cc->rank());
  hdr.dest_rank = rank;
  // v6 (S8): write masked keys on the wire so receiver lookup hits our
  // ordinal map. (lookup_memhandle also masks for defence in depth.)
  hdr.signal_handle = sig_key;
  hdr.dst_handle = dst_key;
  hdr.dst_off = dstOff;
  hdr.size = size;
  hdr.signal_val = signal_val;
  hdr.signal_off = signal_off;

  // Stage header in TX scratch for this peer.
  uint32_t slot_idx =
      tls_tx_seq[rank % kTlsTxSeqLen]++ & static_cast<uint32_t>(kTxSlotsPerPeer - 1);
  size_t hdr_off = sp->TxSlotOffset(static_cast<int>(rank), slot_idx);

  // Stage the WireHeader. Prefer the GDR-mapped host VA (no kernel
  // serialization). Fall back to cudaMemcpy only when GDR pin was missing,
  // which would just be on systems without GDRCopy installed.
  if (sp->host_ptr != nullptr) {
    void* dst = static_cast<uint8_t*>(sp->host_ptr) + hdr_off;
    std::memcpy(dst, &hdr, sizeof(hdr));
    __asm__ __volatile__("sfence" ::: "memory");
  } else {
    static thread_local bool tls_cuda_inited = false;
    if (!tls_cuda_inited) {
      cudaSetDevice(0);
      cudaGetLastError();
      tls_cuda_inited = true;
    }
    cudaError_t cerr = cudaMemcpy(
        static_cast<uint8_t*>(sp->device_ptr) + hdr_off, &hdr, sizeof(hdr),
        cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
      LOG(ERROR) << "IputCommon: cudaMemcpy failed: "
                 << cudaGetErrorString(cerr);
      SetGinError("IputCommon:cudaMemcpy");
      return ncclInternalError;
    }
  }

  auto req = std::make_unique<GinRequest>();
  req->coll = cc;
  req->scratch_slot = slot_idx;

  // v5: TLS — see dbg_count_tls.
  thread_local int snd_pre_tls = 0;
  if (snd_pre_tls < 4) {
    ++snd_pre_tls;
    LOG(INFO) << "IputCommon BEFORE-Send peer=" << global_rank
              << " hdr_off=" << hdr_off
              << " size=" << sizeof(WireHeader)
              << " reg=" << sp->reg_handle;
  }
  auto hdr_or = sock->Send(hdr_off, sizeof(WireHeader), sp->reg_handle);
  // v5: TLS — see dbg_count_tls.
  thread_local int snd_post_tls = 0;
  if (snd_post_tls < 4) {
    ++snd_post_tls;
    LOG(INFO) << "IputCommon AFTER-Send peer=" << global_rank
              << " ok=" << hdr_or.ok()
              << (hdr_or.ok() ? "" : (": " + std::string(hdr_or.status().message())));
  }
  if (!hdr_or.ok()) {
    LOG(ERROR) << "IputCommon: header Send failed: " << hdr_or.status();
    // v6 (fault tolerance): peer disconnect / DXS BadConnection — surface
    // via QueryLastError. The caller drops the in-flight request; NCCL
    // will fail-fast on its next progress tick.
    SetGinError("IputCommon:hdr-Send");
    return ncclInternalError;
  }
  req->hdr_op = std::move(*hdr_or);

  // M6.4 (iter4): drop the inline hdr Send DONE wait. Both hdr and payload
  // Sends go to the same dxs SendSocket lane; dxs+TCP guarantees in-order
  // delivery on a single socket so the receiver always pulls hdr-then-
  // payload off the corresponding RecvSocket. The previous wait was a
  // single-thread serialisation point: the calling NCCL proxy thread spun
  // up to ~few-hundred-microseconds per Iput on hdr DONE before issuing
  // the payload Send, killing the pipelining we built fan-out for.
  //
  // Scratch slot recycling safety: hdr_off is allocated from per-peer TX
  // scratch ring (kTxSlotsPerPeer = 1024). A slot is reused only after
  // 1024 Iputs to the same peer have completed; a fresh hdr Send for slot
  // i sees the prior Send for slot i wrapped 1024 ops earlier, far longer
  // than any in-flight Send takes. NCCL Test() drains both hdr_op and
  // pay_op so the unique_ptrs (and therefore the dxs in-flight refs) are
  // released in order.
  //
  // Measured PP 4096x7168 conc=3 hide=1 NCCL_GIN_FANOUT=3 over 3-run
  // medians: rank0 44.3->54.0 GB/s (+22%), rank1 49.3->53.3 GB/s (+8%)
  // vs v3 (commit b4aec08, fanout=4 with the inline wait).

  if (has_payload) {
    if (src_mh == nullptr || src_mh->local_reg == 0) {
      LOG(ERROR) << "IputCommon: src_mh missing for size=" << size;
      return ncclInvalidArgument;
    }
    auto pay_or = sock->Send(srcOff, size, src_mh->local_reg);
    if (!pay_or.ok()) {
      LOG(ERROR) << "IputCommon: payload Send failed: " << pay_or.status();
      SetGinError("IputCommon:pay-Send");
      return ncclInternalError;
    }
    req->pay_op = std::move(*pay_or);
    // Don't wait for payload here — let NCCL Test() drive completion.
    // Only the hdr needs to be drained before scratch slot recycling.
  }

  *request = req.release();
  return ncclSuccess;
}

ncclResult_t Iput(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
                  size_t size, uint64_t dstOff, void* dstMhandle,
                  uint32_t rank, void** request) {
  return IputCommon(ginCtx, context, srcOff, srcMhandle, size, dstOff,
                    dstMhandle, rank, request, kWireOpPut, 0,
                    /*signalMhandle=*/nullptr, 0);
}

ncclResult_t IputSignal(void* ginCtx, int context, uint64_t srcOff,
                        void* srcMhandle, size_t size, uint64_t dstOff,
                        void* dstMhandle, uint32_t rank, uint64_t signalOff,
                        void* signalMhandle, uint64_t signalValue,
                        uint32_t /*signalOp*/, void** request) {
  // v6 (R2): signalOp dropped — wire only carries ADD today.
  return IputCommon(ginCtx, context, srcOff, srcMhandle, size, dstOff,
                    dstMhandle, rank, request, kWireOpPutSignal, signalOff,
                    signalMhandle, signalValue);
}

ncclResult_t Iget(void* ginCtx, int context, uint64_t remoteOff,
                  void* remoteMhandle, size_t size, uint64_t localOff,
                  void* localMhandle, uint32_t rank, void** request) {
  // Stub: emit a Get header; receiver-side Get reply not yet implemented.
  return IputCommon(ginCtx, context, /*srcOff=*/0, /*srcMhandle=*/localMhandle,
                    /*size=*/0, remoteOff, remoteMhandle, rank, request,
                    kWireOpGet, 0, /*signalMhandle=*/nullptr, 0);
}

ncclResult_t Iflush(void* ginCtx, int context, void* mhandle, uint32_t rank,
                    void** request) {
  return IputCommon(ginCtx, context, 0, mhandle, 0, 0, mhandle, rank, request,
                    kWireOpFlush, 0, /*signalMhandle=*/nullptr, 0);
}

ncclResult_t Test(void* /*collComm*/, void* request, int* done) {
  if (request == nullptr || done == nullptr) return ncclInvalidArgument;
  auto* req = static_cast<GinRequest*>(request);
  *done = 0;

  auto poll = [](dxs::SendOpInterface* op) -> std::optional<ncclResult_t> {
    if (op == nullptr) return ncclSuccess;
    auto s = op->Test();
    if (!s.has_value()) return std::nullopt;
    if (!s->ok()) return ncclInternalError;
    return ncclSuccess;
  };

  // v6 (R1): sig_op dropped from GinRequest.
  for (auto* op : {req->hdr_op.get(), req->pay_op.get()}) {
    auto r = poll(op);
    if (!r.has_value()) return ncclSuccess;  // not done
    if (*r != ncclSuccess) {
      // v6 (fault tolerance): a SendOp returning a hard error means the
      // peer connection is down; surface to NCCL.
      SetGinError("Test:op-fail");
      return *r;
    }
  }
  *done = 1;
  delete req;
  return ncclSuccess;
}

ncclResult_t GinProgress(void* ginCtx) {
  if (ginCtx != nullptr) {
    auto* ctx = static_cast<GinCtx*>(ginCtx);
    if (ctx->progress() != nullptr) ctx->progress()->TickOutbound();
  }
  return ncclSuccess;
}

ncclResult_t QueryLastError(void* /*ginCtx*/, bool* hasError) {
  if (hasError == nullptr) return ncclInvalidArgument;
  *hasError = g_has_error.load(std::memory_order_acquire);
  return ncclSuccess;
}

ncclResult_t Finalize(void* /*ctx*/) {
  fastrak::GetNicClientRouter().Shutdown();
  g_initialized.store(false, std::memory_order_release);
  return ncclSuccess;
}

}  // namespace
}  // namespace fastrak::gin

extern "C" ncclGin_v13_t ncclGinPlugin_v13 = {
    .name = "fastrak-gin-proxy",
    .init = fastrak::gin::Init,
    .devices = fastrak::gin::Devices,
    .getProperties = fastrak::gin::GetProperties,
    .listen = fastrak::gin::Listen,
    .connect = fastrak::gin::Connect,
    .createContext = fastrak::gin::CreateContext,
    .regMrSym = fastrak::gin::RegMrSym,
    .regMrSymDmaBuf = fastrak::gin::RegMrSymDmaBuf,
    .deregMrSym = fastrak::gin::DeregMrSym,
    .destroyContext = fastrak::gin::DestroyContext,
    .closeColl = fastrak::gin::CloseColl,
    .closeListen = fastrak::gin::CloseListen,
    .iput = fastrak::gin::Iput,
    .iputSignal = fastrak::gin::IputSignal,
    .iget = fastrak::gin::Iget,
    .iflush = fastrak::gin::Iflush,
    .test = fastrak::gin::Test,
    .ginProgress = fastrak::gin::GinProgress,
    .queryLastError = fastrak::gin::QueryLastError,
    .finalize = fastrak::gin::Finalize,
};
