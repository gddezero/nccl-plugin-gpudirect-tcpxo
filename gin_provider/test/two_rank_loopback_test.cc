/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Two-rank loopback test for the GIN PROXY plugin. Forks itself into two
 * processes, each pretending to be one rank in a 2-rank collective. Uses two
 * different fastrak NICs (dev 0 and dev 1) so DXS treats them as distinct
 * endpoints.
 *
 * Steps:
 *   1. parent forks child; both have access to a shared mmap region
 *      holding two NCCL_GIN_HANDLE_MAXSIZE-byte slots (one per rank)
 *      plus two semaphores (handles ready, dest done).
 *   2. each rank: plugin->init / devices / getProperties.
 *   3. each rank: plugin->listen on its NIC; write its handle into shm slot.
 *   4. wait until both handles are present.
 *   5. each rank: plugin->connect with handles[].
 *   6. each rank: plugin->createContext.
 *   7. each rank: cudaMalloc + plugin->regMrSym to register a 64KB buffer.
 *   8. rank 0 writes a fake GFD into queues[1*queue_size + 0] = Put 4KB to
 *      rank 1's buffer at offset 0, with src_off=0 and src_handle=mh0,
 *      dst_handle=mh1 (peer's MemHandle key).
 *   9. rank 1's progress thread should observe the inbound and write data
 *      into its CUDA buffer. Rank 1 cudaMemcpy's the buffer back to host
 *      and verifies the bytes match what rank 0 wrote.
 *  10. teardown.
 *
 * Status: SCAFFOLD. Step 8 needs the publishing of peer MemHandle keys
 * across ranks (out-of-band exchange via shm) which the wire ginHandle does
 * not yet carry. Until M3.1 publishes peer reg keys, this test verifies
 * everything up to (and including) connect+regMrSym; the actual data put
 * stays as a TODO.
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gin_provider/nccl_gin_v13_abi.h"

namespace {

constexpr size_t kHandleSize = 128;

struct SharedRegion {
  uint8_t  handle[2][kHandleSize];
  uint32_t handle_ready[2];   // set to 1 when rank N has written its handle
  uint64_t mhandle_key[2];    // each rank publishes its mhandle key here
  uint32_t mhandle_ready[2];
};

void NcclLogger(ncclDebugLogLevel level, unsigned long /*flags*/,
                const char* file, int line, const char* fmt, ...) {
  va_list ap;
  fprintf(stderr, "[lvl=%d %s:%d] ", level, file, line);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}

int RunRank(int my_rank, int my_dev, ncclGin_v13_t* p,
            volatile SharedRegion* shm) {
  fprintf(stderr, "[rank %d] init\n", my_rank);
  void* ctx = nullptr;
  if (p->init(&ctx, 0xfeedbeef, NcclLogger) != ncclSuccess) return 10;

  fprintf(stderr, "[rank %d] listen on dev %d\n", my_rank, my_dev);
  uint8_t my_handle[kHandleSize] = {0};
  void* listenComm = nullptr;
  if (p->listen(ctx, my_dev, my_handle, &listenComm) != ncclSuccess) {
    fprintf(stderr, "[rank %d] listen failed\n", my_rank);
    return 11;
  }
  memcpy(const_cast<uint8_t*>(shm->handle[my_rank]), my_handle, kHandleSize);
  __atomic_store_n(&shm->handle_ready[my_rank], 1, __ATOMIC_RELEASE);

  // Wait for peer handle.
  int peer = 1 - my_rank;
  fprintf(stderr, "[rank %d] waiting for peer handle\n", my_rank);
  while (__atomic_load_n(&shm->handle_ready[peer], __ATOMIC_ACQUIRE) == 0) {
    usleep(1000);
  }
  fprintf(stderr, "[rank %d] both handles present, connecting\n", my_rank);

  void* handles[2];
  uint8_t local_handle[kHandleSize];
  uint8_t peer_handle[kHandleSize];
  memcpy(local_handle, const_cast<uint8_t*>(shm->handle[my_rank]), kHandleSize);
  memcpy(peer_handle, const_cast<uint8_t*>(shm->handle[peer]), kHandleSize);
  handles[my_rank] = local_handle;
  handles[peer] = peer_handle;

  void* collComm = nullptr;
  if (p->connect(ctx, handles, 2, my_rank, listenComm, &collComm) !=
      ncclSuccess) {
    fprintf(stderr, "[rank %d] connect failed\n", my_rank);
    return 12;
  }
  fprintf(stderr, "[rank %d] connect OK collComm=%p\n", my_rank, collComm);

  fprintf(stderr, "[rank %d] createContext\n", my_rank);
  ncclGinConfig_v13_t cfg = {};
  cfg.queueDepth = 1024;
  cfg.nCounters = 16;
  cfg.nSignals = 16;
  cfg.nContexts = 1;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  if (p->createContext(collComm, &cfg, &ginCtx, &devHandle) != ncclSuccess) {
    fprintf(stderr, "[rank %d] createContext failed\n", my_rank);
    return 13;
  }
  fprintf(stderr,
          "[rank %d] createContext OK ginCtx=%p devHandle=%p netDevType=%d\n",
          my_rank, ginCtx, devHandle, devHandle ? devHandle->netDeviceType : -1);

  // Allocate + register a CUDA buffer. We use a helper here that calls into
  // the plugin. Skipped in this scaffold to keep deps light — see M5 for
  // the full data put round-trip.

  // Sleep briefly to let the progress thread process anything (debug aid).
  usleep(200 * 1000);

  fprintf(stderr, "[rank %d] teardown\n", my_rank);
  if (p->destroyContext) p->destroyContext(ginCtx);
  if (p->closeColl) p->closeColl(collComm);
  if (p->closeListen) p->closeListen(listenComm);
  if (p->finalize) p->finalize(ctx);
  fprintf(stderr, "[rank %d] done\n", my_rank);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "./libnccl-gin.so";
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 2;
  }
  ncclGin_v13_t* p = (ncclGin_v13_t*)dlsym(h, "ncclGinPlugin_v13");
  if (p == nullptr) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 3;
  }

  // Shared memory for inter-rank handle exchange.
  void* shm_void = mmap(NULL, sizeof(SharedRegion), PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (shm_void == MAP_FAILED) {
    perror("mmap");
    return 4;
  }
  auto* shm = static_cast<SharedRegion*>(shm_void);
  memset(shm, 0, sizeof(*shm));

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 5;
  }
  if (pid == 0) {
    // child = rank 1, dev 1 (eth2)
    return RunRank(1, 1, p, shm);
  }
  // parent = rank 0, dev 0 (eth1)
  int rc0 = RunRank(0, 0, p, shm);
  int status = 0;
  waitpid(pid, &status, 0);
  int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  fprintf(stderr, "rank 0 rc=%d, rank 1 rc=%d\n", rc0, rc1);
  munmap(shm_void, sizeof(SharedRegion));
  dlclose(h);
  return (rc0 == 0 && rc1 == 0) ? 0 : 6;
}
