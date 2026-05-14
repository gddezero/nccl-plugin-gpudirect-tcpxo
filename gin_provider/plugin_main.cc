/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * GIN PROXY mode plugin entrypoint. Implements ncclGinPlugin_v13 backed by
 * the existing FasTrak DXS / Falcon transport.
 *
 * STAGE M2.5a: init / devices / getProperties / finalize / queryLastError /
 *              ginProgress wired to real FasTrak NIC discovery so NCCL can
 *              load the plugin and walk all 8 NICs. Listen / Connect /
 *              CreateContext / RegMrSym* / Iput* still return ncclInternalError
 *              until M2.5b lands.
 */

#include "gin_provider/plugin_main.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "gin_provider/listen_handle.h"
#include "gin_provider/proxy_context.h"
#include "gin_provider/proxy_progress.h"
#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include "plugin/nccl_net.h"
#include "plugin/net/net_v12.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/init.h"
#include "tcpdirect_plugin/fastrak_offload/nic_client_router.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"

namespace fastrak::gin {
namespace {

constexpr const char* kPluginName = "fastrak-gin-proxy";

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

// ----- 17 ABI callbacks -----

ncclResult_t Init(void** ctx, uint64_t commId,
                  ncclDebugLogger_t logFunction) {
  if (ctx == nullptr) return ncclInvalidArgument;

  // Reuse the existing fastrak v7 plugin's one-shot init: it sets the NCCL
  // logger, waits for RxDM, then calls initializeNetIfs() to populate
  // kNcclSocketDevs / numNcclSocketDevs.
  absl::Status s = fastrak::PluginCoreInit(logFunction);
  if (!s.ok()) {
    LOG(ERROR) << "FasTrak GIN init failed: " << s;
    return StatusToNccl(s);
  }

  // We don't yet hold per-commId state; use a sentinel non-null pointer so
  // NCCL keeps a stable handle. M2.5b will allocate a real per-commId block.
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
  if (dev < 0 || dev >= fastrak::kNcclNetIfs) {
    return ncclInvalidArgument;
  }
  std::memset(props, 0, sizeof(*props));
  const auto& d = fastrak::kNcclSocketDevs[dev];

  // The struct has `char* name`; we reuse the long-lived dev_name.
  props->name = const_cast<char*>(d.dev_name);
  props->pciPath = const_cast<char*>(d.pci_path);
  props->guid = static_cast<uint64_t>(dev);
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 200000;       // 200 Gbps per a3-mega NIC
  props->port = 0;
  props->latency = 5.0f;       // µs (placeholder — measure in M6)
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

ncclResult_t Listen(void* ctx, int dev, void* handle,
                    void** listenComm) {
  // TODO(M2.5b): allocate ListenComm, call dxs->Listen(), serialize
  // ListenHandle into `handle`.
  (void)ctx; (void)dev; (void)handle; (void)listenComm;
  return ncclInternalError;
}

ncclResult_t Connect(void* ctx, void* handles[], int nranks, int rank,
                     void* listenComm, void** collComm) {
  // TODO(M2.5b): allocate CollComm, mesh = for each peer dxs->Connect, then
  // accept (nranks-1) on listenComm.
  return ncclInternalError;
}

ncclResult_t CreateContext(void* collComm, ncclGinConfig_v13_t* config,
                           void** ginCtx,
                           ncclNetDeviceHandle_v11_t** devHandle) {
  // TODO(M2.5b): GinCtx::Init, fill devHandle with device pointer.
  return ncclInternalError;
}

ncclResult_t RegMrSym(void* collComm, void* data, size_t size, int type,
                      uint64_t mrFlags, void** mhandle, void** ginHandle) {
  // TODO(M2.5b): call buffer_mgr->RegBuf via getDmabufFd path.
  return ncclInternalError;
}

ncclResult_t RegMrSymDmaBuf(void* collComm, void* data, size_t size, int type,
                            uint64_t offset, int fd, uint64_t mrFlags,
                            void** mhandle, void** ginHandle) {
  return ncclInternalError;
}

ncclResult_t DeregMrSym(void* collComm, void* mhandle) {
  return ncclInternalError;
}

ncclResult_t DestroyContext(void* ginCtx) {
  if (ginCtx != nullptr) {
    delete static_cast<GinCtx*>(ginCtx);
  }
  return ncclSuccess;
}

ncclResult_t CloseColl(void* collComm) {
  if (collComm != nullptr) {
    delete static_cast<CollComm*>(collComm);
  }
  return ncclSuccess;
}

ncclResult_t CloseListen(void* listenComm) {
  if (listenComm != nullptr) {
    delete static_cast<ListenComm*>(listenComm);
  }
  return ncclSuccess;
}

ncclResult_t Iput(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
                  size_t size, uint64_t dstOff, void* dstMhandle,
                  uint32_t rank, void** request) {
  // PROXY mode: NCCL device kernel posts GFDs directly into the ring; this
  // host-side iput is only invoked from CPU-initiated paths and is rare.
  return ncclInternalError;
}

ncclResult_t IputSignal(void* ginCtx, int context, uint64_t srcOff,
                        void* srcMhandle, size_t size, uint64_t dstOff,
                        void* dstMhandle, uint32_t rank, uint64_t signalOff,
                        void* signalMhandle, uint64_t signalValue,
                        uint32_t signalOp, void** request) {
  return ncclInternalError;
}

ncclResult_t Iget(void* ginCtx, int context, uint64_t remoteOff,
                  void* remoteMhandle, size_t size, uint64_t localOff,
                  void* localMhandle, uint32_t rank, void** request) {
  return ncclInternalError;
}

ncclResult_t Iflush(void* ginCtx, int context, void* mhandle, uint32_t rank,
                    void** request) {
  return ncclInternalError;
}

ncclResult_t Test(void* collComm, void* request, int* done) {
  return ncclInternalError;
}

ncclResult_t GinProgress(void* ginCtx) {
  // M3: tick GinCtx->progress()->Tick() once. For now proxy thread (started
  // in createContext) auto-runs in background, so this is a no-op.
  if (ginCtx != nullptr) {
    auto* ctx = static_cast<GinCtx*>(ginCtx);
    if (ctx->progress() != nullptr) {
      ctx->progress()->Tick();
    }
  }
  return ncclSuccess;
}

ncclResult_t QueryLastError(void* ginCtx, bool* hasError) {
  if (hasError == nullptr) return ncclInvalidArgument;
  *hasError = g_has_error.load(std::memory_order_acquire);
  return ncclSuccess;
}

ncclResult_t Finalize(void* ctx) {
  fastrak::GetNicClientRouter().Shutdown();
  g_initialized.store(false, std::memory_order_release);
  return ncclSuccess;
}

}  // namespace
}  // namespace fastrak::gin

// Exported plugin descriptor.
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
