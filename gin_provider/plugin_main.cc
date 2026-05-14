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
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/listen_handle.h"
#include "gin_provider/proxy_context.h"
#include "gin_provider/proxy_progress.h"
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

  // Outbound: connect to every peer rank's published listen handle.
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
    if (!sock_or.ok()) return StatusToNccl(sock_or.status());
    auto send_sock = std::move(*sock_or);
    if (auto s = WaitSocketReady(*send_sock, "GIN Connect outbound"); !s.ok()) {
      return StatusToNccl(s);
    }

    PeerConn pc;
    pc.send_sock = std::move(send_sock);
    cc->set_peer(r, std::move(pc));
  }

  // TODO(M3): inbound side. Run (nranks - 1) Accept() calls on lc->listen_sock
  //           and match each incoming RecvSocket to its source rank via a
  //           hello message. For PROXY-mode bring-up the device-side put goes
  //           outbound only, so receive matching can land in M3.

  LOG(INFO) << absl::StrFormat(
      "GIN Connect: dev=%d rank=%d/%d outbound mesh established (%d send sockets)",
      lc->dev, rank, nranks, nranks - 1);

  *collComm = cc.release();
  return ncclSuccess;
}

ncclResult_t CreateContext(void* collComm, ncclGinConfig_v13_t* config,
                           void** ginCtx,
                           ncclNetDeviceHandle_v11_t** devHandle) {
  if (collComm == nullptr || config == nullptr || ginCtx == nullptr ||
      devHandle == nullptr) {
    return ncclInvalidArgument;
  }
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
  // Allocate a devHandle blob (NCCL takes ownership via plugin->free convention,
  // but the v13 ABI actually stores it; we use plain new and rely on
  // destroyContext for teardown).
  static_assert(sizeof(ncclNetDeviceHandle_v11_t) <= 256, "devHandle small");
  auto* dh = new (std::nothrow) ncclNetDeviceHandle_v11_t{};
  if (dh == nullptr) return ncclSystemError;
  dh->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  dh->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  dh->handle = gpu->dev_view;  // device pointer
  dh->size = sizeof(*gpu->dev_view);
  dh->needsProxyProgress = 1;

  // Start the host-side proxy progress thread which will drain the GFD ring.
  // No-op until M3 wires actual dxs::Send dispatching.
  gctx->StartProgress();

  *ginCtx = gctx.release();
  *devHandle = dh;

  LOG(INFO) << absl::StrFormat(
      "GIN CreateContext: dev=%d rank=%d nranks=%d queueDepth=%u nC=%d nS=%d",
      cc->fastrak_idx(), cc->rank(), cc->nranks(), qd_pow2,
      config->nCounters, config->nSignals);
  return ncclSuccess;
}

ncclResult_t RegMrSym(void* collComm, void* data, size_t size, int type,
                      uint64_t /*mrFlags*/, void** mhandle, void** ginHandle) {
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
                            uint64_t /*offset*/, int fd, uint64_t /*mrFlags*/,
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

ncclResult_t Iput(void*, int, uint64_t, void*, size_t, uint64_t, void*,
                  uint32_t, void**) {
  return ncclInternalError;  // M3
}
ncclResult_t IputSignal(void*, int, uint64_t, void*, size_t, uint64_t, void*,
                        uint32_t, uint64_t, void*, uint64_t, uint32_t, void**) {
  return ncclInternalError;  // M3
}
ncclResult_t Iget(void*, int, uint64_t, void*, size_t, uint64_t, void*,
                  uint32_t, void**) {
  return ncclInternalError;  // M3
}
ncclResult_t Iflush(void*, int, void*, uint32_t, void**) {
  return ncclInternalError;  // M3
}
ncclResult_t Test(void*, void*, int*) { return ncclInternalError; /* M3 */ }

ncclResult_t GinProgress(void* ginCtx) {
  if (ginCtx != nullptr) {
    auto* ctx = static_cast<GinCtx*>(ginCtx);
    if (ctx->progress() != nullptr) ctx->progress()->Tick();
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
