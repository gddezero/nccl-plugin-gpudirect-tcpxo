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
std::atomic<bool> g_has_error{false};

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

  // Issue all outbound connects up front. Each peer is doing the same
  // concurrently, so if we waited for one connect to be ready before
  // issuing the next, we would risk deadlock during the (n*(n-1)) handshake.
  std::vector<std::unique_ptr<dxs::SendSocketInterface>> pending(nranks);
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
    auto sock_or = dxs->Connect(addr_str, h.port);
    if (!sock_or.ok()) {
      LOG(ERROR) << "GIN Connect: dxs->Connect to rank " << r << " failed: "
                 << sock_or.status();
      return StatusToNccl(sock_or.status());
    }
    pending[r] = std::move(*sock_or);
  }

  // Interleaved progress loop: poll each pending outbound for ready, and
  // poll listen->Accept(), until all outbound + (nranks-1) inbound are done.
  size_t accepted = 0;
  auto deadline = absl::Now() + absl::Seconds(120);
  while (true) {
    // Drain pending outbound.
    for (int r = 0; r < nranks; ++r) {
      if (pending[r] == nullptr) continue;
      auto status = pending[r]->SocketReady();
      if (!status.has_value()) continue;     // still pending
      if (!status->ok()) {
        LOG(ERROR) << "GIN Connect outbound to rank " << r
                   << " failed: " << *status;
        return StatusToNccl(*status);
      }
      PeerConn pc;
      pc.send_sock = std::move(pending[r]);
      cc->set_peer(r, std::move(pc));
      pending[r] = nullptr;
    }

    // Drain inbound accepts.
    if (accepted < static_cast<size_t>(nranks - 1)) {
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
      if (pending[r] != nullptr) {
        all_outbound_done = false;
        break;
      }
    }
    if (all_outbound_done &&
        accepted >= static_cast<size_t>(nranks - 1)) {
      break;
    }
    if (absl::Now() > deadline) {
      LOG(ERROR) << "GIN Connect: handshake timed out, accepted=" << accepted
                 << " of " << (nranks - 1);
      return ncclSystemError;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  LOG(INFO) << absl::StrFormat(
      "GIN Connect: dev=%d rank=%d/%d mesh established "
      "(out=%d, in=%zu)",
      lc->dev, rank, nranks, nranks - 1, accepted);

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

// Per-iput request handle: holds the in-flight DXS SendOp (and an optional
// signal SendOp for IputSignal). Test() polls them for completion.
struct GinRequest {
  std::unique_ptr<dxs::SendOpInterface> hdr_op;
  std::unique_ptr<dxs::SendOpInterface> pay_op;
  std::unique_ptr<dxs::SendOpInterface> sig_op;
  CollComm* coll = nullptr;
  uint32_t  scratch_slot = UINT32_MAX;
};

// Lazily-resolved offsets used to find a free TX scratch slot per peer.
static thread_local uint32_t tls_tx_seq[64] = {0};

static ncclResult_t IputCommon(void* ginCtx, int /*context*/,
                               uint64_t srcOff, void* srcMhandle, size_t size,
                               uint64_t dstOff, void* dstMhandle,
                               uint32_t rank, void** request,
                               WireOp wire_op, uint64_t signal_off,
                               void* signalMhandle, uint64_t signal_val,
                               uint32_t signal_op_arg) {
  if (ginCtx == nullptr || request == nullptr) {
    LOG(ERROR) << "IputCommon: ginCtx or request null";
    return ncclInvalidArgument;
  }
  auto* gctx = static_cast<GinCtx*>(ginCtx);
  auto* cc = gctx->coll();
  auto* sp = gctx->scratch();
  static std::atomic<int> dbg_count{0};
  if (dbg_count.fetch_add(1) < 50) {
    LOG(INFO) << "IputCommon DBG #" << dbg_count.load()
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
        // GDRCopy maps GPU memory as write-combining; std::atomic
        // ops are not guaranteed coherent there. Use plain RMW with
        // explicit sfence (single-writer guaranteed for barrier case).
        auto* slot_u64 = reinterpret_cast<volatile uint64_t*>(slot_b);
        uint64_t prev = *slot_u64;
        uint64_t next = prev + signal_val;
        *slot_u64 = next;
        __asm__ __volatile__("sfence" ::: "memory");
        static std::atomic<int> ss_dbg{0};
        if (ss_dbg.fetch_add(1) < 4) {
          LOG(INFO) << "self-signal: sig_h=0x" << std::hex << sig_h
                    << " off=" << std::dec << signal_off
                    << " prev=" << prev << " new=" << next;
        }
      } else {
        LOG(ERROR) << "IputCommon self-signal: no signal map (sig_h=0x"
                   << std::hex << sig_h << std::dec
                   << " off=" << signal_off
                   << " primary_size=" << cc->signal_size_bytes() << ")";
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
  if (peer == nullptr || peer->send_sock == nullptr) {
    LOG(ERROR) << "IputCommon: bad peer rank=" << rank
               << " my_rank=" << my_rank
               << " num_peers=" << cc->num_peers();
    return ncclInvalidArgument;
  }
  uint64_t src_key = reinterpret_cast<uint64_t>(srcMhandle);
  uint64_t dst_key = reinterpret_cast<uint64_t>(dstMhandle);
  // Only resolve src_mh when there's actually a payload to send. Signal
  // / flush ops legitimately call with srcMhandle=NULL → key=0, no need
  // to log a MISS.
  bool has_payload =
      (size > 0 && wire_op != kWireOpSignal && wire_op != kWireOpFlush);
  MemHandle* src_mh = has_payload ? cc->lookup_memhandle(src_key) : nullptr;
  if (has_payload && (src_mh == nullptr || src_mh->local_reg == 0)) {
    LOG(ERROR) << "IputCommon: bad src_mh key=0x" << std::hex << src_key
               << " size=" << std::dec << size << " wire_op=" << wire_op;
    return ncclInvalidArgument;
  }

  WireHeader hdr{};
  hdr.magic = kWireMagic;
  hdr.op = static_cast<uint16_t>(wire_op);
  hdr.source_rank = static_cast<uint32_t>(cc->rank());
  hdr.dest_rank = rank;
  hdr.signal_handle = reinterpret_cast<uint64_t>(signalMhandle);
  hdr.dst_handle = dst_key;
  hdr.dst_off = dstOff;
  hdr.size = size;
  hdr.signal_val = signal_val;
  hdr.signal_off = signal_off;
  (void)signal_op_arg;

  // Stage header in TX scratch for this peer.
  uint32_t slot_idx =
      tls_tx_seq[rank % 64]++ & static_cast<uint32_t>(kTxSlotsPerPeer - 1);
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
      return ncclInternalError;
    }
  }

  auto req = std::make_unique<GinRequest>();
  req->coll = cc;
  req->scratch_slot = slot_idx;

  static std::atomic<int> snd_pre{0};
  if (snd_pre.fetch_add(1) < 4) {
    LOG(INFO) << "IputCommon BEFORE-Send peer=" << global_rank
              << " hdr_off=" << hdr_off
              << " size=" << sizeof(WireHeader)
              << " reg=" << sp->reg_handle;
  }
  auto hdr_or = peer->send_sock->Send(hdr_off, sizeof(WireHeader),
                                      sp->reg_handle);
  static std::atomic<int> snd_post{0};
  if (snd_post.fetch_add(1) < 4) {
    LOG(INFO) << "IputCommon AFTER-Send peer=" << global_rank
              << " ok=" << hdr_or.ok()
              << (hdr_or.ok() ? "" : (": " + std::string(hdr_or.status().message())));
  }
  if (!hdr_or.ok()) {
    LOG(ERROR) << "IputCommon: header Send failed: " << hdr_or.status();
    return ncclInternalError;
  }
  req->hdr_op = std::move(*hdr_or);

  // M6: keep the M5.5-era hdr Send DONE → payload Send sync. We tried
  // removing it (commit X) but it didn't move single-stream PP BW (the
  // benefit only shows up under deep pipelining the test doesn't drive)
  // and risked corrupting the per-peer scratch ring under EP-style
  // bursty workloads. Keep it conservative; multi-NIC fan-out (M6.x)
  // is the next BW lever rather than dropping this sync.
  {
    auto deadline = absl::Now() + absl::Seconds(5);
    bool done = false;
    while (absl::Now() < deadline) {
      auto s = req->hdr_op->Test();
      if (s.has_value()) {
        if (!s->ok()) {
          LOG(ERROR) << "IputCommon: hdr Send op error: " << *s
                     << " peer=" << global_rank << " hdr_off=" << hdr_off;
          return ncclInternalError;
        }
        done = true;
        break;
      }
      std::this_thread::yield();
    }
    if (!done) {
      LOG(ERROR) << "IputCommon hdr Send TIMEOUT(5s) peer=" << global_rank
                 << " hdr_off=" << hdr_off << " op=" << wire_op;
      return ncclInternalError;
    }
  }

  if (has_payload) {
    if (src_mh == nullptr || src_mh->local_reg == 0) {
      LOG(ERROR) << "IputCommon: src_mh missing for size=" << size;
      return ncclInvalidArgument;
    }
    auto pay_or = peer->send_sock->Send(srcOff, size, src_mh->local_reg);
    if (!pay_or.ok()) {
      LOG(ERROR) << "IputCommon: payload Send failed: " << pay_or.status();
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
                    /*signalMhandle=*/nullptr, 0, 0);
}

ncclResult_t IputSignal(void* ginCtx, int context, uint64_t srcOff,
                        void* srcMhandle, size_t size, uint64_t dstOff,
                        void* dstMhandle, uint32_t rank, uint64_t signalOff,
                        void* signalMhandle, uint64_t signalValue,
                        uint32_t signalOp, void** request) {
  return IputCommon(ginCtx, context, srcOff, srcMhandle, size, dstOff,
                    dstMhandle, rank, request, kWireOpPutSignal, signalOff,
                    signalMhandle, signalValue, signalOp);
}

ncclResult_t Iget(void* ginCtx, int context, uint64_t remoteOff,
                  void* remoteMhandle, size_t size, uint64_t localOff,
                  void* localMhandle, uint32_t rank, void** request) {
  // Stub: emit a Get header; receiver-side Get reply not yet implemented.
  return IputCommon(ginCtx, context, /*srcOff=*/0, /*srcMhandle=*/localMhandle,
                    /*size=*/0, remoteOff, remoteMhandle, rank, request,
                    kWireOpGet, 0, /*signalMhandle=*/nullptr, 0, 0);
}

ncclResult_t Iflush(void* ginCtx, int context, void* mhandle, uint32_t rank,
                    void** request) {
  return IputCommon(ginCtx, context, 0, mhandle, 0, 0, mhandle, rank, request,
                    kWireOpFlush, 0, /*signalMhandle=*/nullptr, 0, 0);
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

  for (auto* op : {req->hdr_op.get(), req->pay_op.get(), req->sig_op.get()}) {
    auto r = poll(op);
    if (!r.has_value()) return ncclSuccess;  // not done
    if (*r != ncclSuccess) return *r;
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
