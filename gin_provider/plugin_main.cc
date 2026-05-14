/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * GIN PROXY mode plugin entrypoint. Implements the 17 host-side callbacks of
 * ncclGin_v13_t, exporting symbol ncclGinPlugin_v13.
 *
 * Status: SKELETON. Most callbacks return ncclInternalError so that we can
 * validate the build / link and confirm NCCL >= 2.30.4 picks up the symbol.
 * Real DXS / Falcon wiring lands in subsequent commits.
 */

#include "gin_provider/plugin_main.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "gin_provider/gpu_ctx_alloc.h"
#include "nccl.h"
#include "nccl_common.h"
#include "plugin/nccl_net.h"
#include "plugin/net/net_v12.h"
#include "nccl_device/net_device.h"
// MAX_NET_SIZE comes from plugin/nccl_net.h above.

// proxy_context / proxy_progress are pulled in once their dxs/webrtc deps
// finish building. Until then plugin_main only references their forward
// declarations through gpu_ctx_alloc.

namespace fastrak::gin {
namespace {

// ----------------------------------------------------------------------------
// Top-level singleton state. NCCL invokes plugin->init() once per commId.
// Multiple commIds may coexist; we keep a tiny registry keyed by ctx ptr.
// ----------------------------------------------------------------------------

std::atomic<bool> g_initialized{false};
ncclDebugLogger_t g_logger = nullptr;

// Forward refs to the per-callback implementations.

ncclResult_t Init(void** ctx, uint64_t commId,
                  ncclDebugLogger_t logFunction) {
  g_logger = logFunction;
  // For now we hand back a sentinel non-null pointer. The real plugin will
  // allocate a fastrak::gin::PluginCtx that tracks per-commId state.
  static int sentinel = 0;
  *ctx = &sentinel;
  g_initialized.store(true, std::memory_order_release);
  if (g_logger) {
    g_logger(NCCL_LOG_INFO, NCCL_INIT, __FILE__, __LINE__,
             "FasTrak GIN provider (PROXY mode) init: commId=%lu", commId);
  }
  return ncclSuccess;
}

ncclResult_t Devices(int* ndev) {
  // Mirror existing FasTrak ncclNet plugin device count: 8 NICs on a3-mega.
  // Real impl will discover via /sys/.../1ae0:0084 PCI scan.
  *ndev = 8;
  return ncclSuccess;
}

ncclResult_t GetProperties(int dev, ncclNetProperties_v12_t* props) {
  if (props == nullptr) return ncclInvalidArgument;
  std::memset(props, 0, sizeof(*props));
  // PROXY mode reporting. Device-side GFD code lives in NCCL itself.
  props->name = const_cast<char*>("fastrak-gin-proxy");
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 200000;  // 200 Gbps per NIC
  props->port = 0;
  props->latency = 5.0f;  // µs, placeholder
  props->maxComms = 64;
  props->maxRecvs = 1;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  props->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_NET_SIZE;
  props->maxMultiRequestSize = 1;
  props->railId = static_cast<int16_t>(dev);
  props->planeId = static_cast<int16_t>(dev / 4);
  // pciPath / guid / vProps must be provided in the real impl.
  return ncclSuccess;
}

ncclResult_t Listen(void* ctx, int dev, void* handle,
                    void** listenComm) {
  return ncclInternalError;  // TODO: dxs::Listen + serialise into handle
}

ncclResult_t Connect(void* ctx, void* handles[], int nranks, int rank,
                     void* listenComm, void** collComm) {
  return ncclInternalError;  // TODO: per-peer SendSocket / RecvSocket mesh
}

ncclResult_t CreateContext(void* collComm, ncclGinConfig_v13_t* config,
                           void** ginCtx,
                           ncclNetDeviceHandle_v11_t** devHandle) {
  return ncclInternalError;  // TODO: alloc ProxyGpuCtx + start progress thread
}

ncclResult_t RegMrSym(void* collComm, void* data, size_t size, int type,
                      uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ncclInternalError;  // TODO: dma-buf register via buffer_mgr_client
}

ncclResult_t RegMrSymDmaBuf(void* collComm, void* data, size_t size, int type,
                            uint64_t offset, int fd, uint64_t mrFlags,
                            void** mhandle, void** ginHandle) {
  return ncclInternalError;  // TODO: register existing fd
}

ncclResult_t DeregMrSym(void* collComm, void* mhandle) {
  return ncclInternalError;
}

ncclResult_t DestroyContext(void* ginCtx) {
  return ncclInternalError;
}

ncclResult_t CloseColl(void* collComm) { return ncclInternalError; }

ncclResult_t CloseListen(void* listenComm) { return ncclInternalError; }

ncclResult_t Iput(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
                  size_t size, uint64_t dstOff, void* dstMhandle,
                  uint32_t rank, void** request) {
  // PROXY mode: device-side GFD post is the hot path; this host iput is only
  // called from CPU-initiated paths. Return InternalError until wired.
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
  // The real impl drives the progress thread tick (poll GFD ring and
  // translate each entry into dxs::Send / dxs::RecvLinearized).
  return ncclSuccess;
}

ncclResult_t QueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

ncclResult_t Finalize(void* ctx) {
  g_initialized.store(false, std::memory_order_release);
  return ncclSuccess;
}

}  // namespace
}  // namespace fastrak::gin

// ----------------------------------------------------------------------------
// Exported plugin descriptor. NCCL >= 2.30.4 dlsym's this symbol.
// ----------------------------------------------------------------------------
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
