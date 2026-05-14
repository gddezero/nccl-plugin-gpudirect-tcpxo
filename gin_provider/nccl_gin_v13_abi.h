/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Local shim for NCCL GIN v13 plugin ABI.
 *
 * NCCL upstream's plugin/gin/gin_v13.h does an unprefixed `#include
 * "nccl_net.h"`, which only resolves when src/include/plugin is on the
 * include search path. Our Bazel `plugin_lib` rule strips include prefixes
 * down to src/include, so the bare include cannot be found.
 *
 * Rather than patch /usr/local/nccl, we replicate the v13 struct here
 * verbatim from NCCL v2.30.4-1 upstream (commit 1933fdd) and pull in the
 * peer headers using the prefixed paths that DO resolve.
 *
 * Keep this file in lockstep with upstream when bumping NCCL versions.
 */

#ifndef GIN_PROVIDER_NCCL_GIN_V13_ABI_H_
#define GIN_PROVIDER_NCCL_GIN_V13_ABI_H_

#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include "plugin/net/net_v12.h"  // ncclNetProperties_v12_t
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
} ncclGinConfig_v13_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId,
                       ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank,
                          void* listenComm, void** collComm);
  ncclResult_t (*createContext)(void* collComm, ncclGinConfig_v13_t* config,
                                void** ginCtx,
                                ncclNetDeviceHandle_v11_t** devHandle);
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type,
                           uint64_t mrFlags, void** mhandle, void** ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size,
                                 int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void** mhandle,
                                 void** ginHandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  ncclResult_t (*destroyContext)(void* ginCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
  ncclResult_t (*iput)(void* ginCtx, int context, uint64_t srcOff,
                       void* srcMhandle, size_t size, uint64_t dstOff,
                       void* dstMhandle, uint32_t rank, void** request);
  ncclResult_t (*iputSignal)(void* ginCtx, int context, uint64_t srcOff,
                             void* srcMhandle, size_t size, uint64_t dstOff,
                             void* dstMhandle, uint32_t rank,
                             uint64_t signalOff, void* signalMhandle,
                             uint64_t signalValue, uint32_t signalOp,
                             void** request);
  ncclResult_t (*iget)(void* ginCtx, int context, uint64_t remoteOff,
                       void* remoteMhandle, size_t size, uint64_t localOff,
                       void* localMhandle, uint32_t rank, void** request);
  ncclResult_t (*iflush)(void* ginCtx, int context, void* mhandle,
                         uint32_t rank, void** request);
  ncclResult_t (*test)(void* collComm, void* request, int* done);
  ncclResult_t (*ginProgress)(void* ginCtx);
  ncclResult_t (*queryLastError)(void* ginCtx, bool* hasError);
  ncclResult_t (*finalize)(void* ctx);
} ncclGin_v13_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GIN_PROVIDER_NCCL_GIN_V13_ABI_H_
