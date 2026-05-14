/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Two-rank cross-host NCCL test that finally exercises the GIN PROXY
 * data path. Single-mode binary:
 *
 *   <rank> <nranks> <file>
 *     rank 0: ncclGetUniqueId, write to <file>, then ncclCommInitRank
 *             (the listener inside the UniqueId stays alive only while the
 *             generating process is alive, so we cannot split gen and run
 *             across processes)
 *     rank N>0: spin until <file> exists, read UniqueId, ncclCommInitRank
 */

#include <cuda_runtime.h>
#include <fcntl.h>
#include <nccl.h>
#include <unistd.h>

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK(x, msg)                                                          \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__,               \
              cudaGetErrorString(e));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_NCCL(x)                                                          \
  do {                                                                         \
    ncclResult_t r = (x);                                                      \
    if (r != ncclSuccess) {                                                    \
      fprintf(stderr, "NCCL err %s:%d %s\n", __FILE__, __LINE__,               \
              ncclGetErrorString(r));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int WriteId(const char* path, const ncclUniqueId& id) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CHECK(fd >= 0, "open(write)");
  ssize_t n = write(fd, &id, sizeof(id));
  CHECK(n == (ssize_t)sizeof(id), "write");
  close(fd);
  fprintf(stderr, "wrote ncclUniqueId to %s (%zu bytes)\n", path, sizeof(id));
  return 0;
}

static int ReadId(const char* path, ncclUniqueId* id) {
  int fd = open(path, O_RDONLY);
  CHECK(fd >= 0, "open(read)");
  ssize_t n = read(fd, id, sizeof(*id));
  CHECK(n == (ssize_t)sizeof(*id), "read");
  close(fd);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <rank> <nranks> <file>\n", argv[0]);
    return 2;
  }
  int rank = atoi(argv[1]);
  int nranks = atoi(argv[2]);
  const char* path = argv[3];

  fprintf(stderr, "[rank %d] CUDA setup\n", rank);
  CHECK_CUDA(cudaSetDevice(0));

  ncclUniqueId id;
  if (rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
    if (WriteId(path, id) != 0) return 1;
    fprintf(stderr, "[rank 0] generated uniqueId, wrote to %s\n", path);
  } else {
    fprintf(stderr, "[rank %d] waiting for uniqueId file %s\n", rank, path);
    auto deadline = time(nullptr) + 60;
    while (true) {
      int fd = open(path, O_RDONLY);
      if (fd >= 0) { close(fd); break; }
      if (time(nullptr) > deadline) {
        fprintf(stderr, "[rank %d] timeout waiting for %s\n", rank, path);
        return 1;
      }
      usleep(100000);
    }
    CHECK(ReadId(path, &id) == 0, "ReadId");
    fprintf(stderr, "[rank %d] read uniqueId from %s\n", rank, path);
  }

  fprintf(stderr, "[rank %d] ncclCommInitRank (nranks=%d)\n", rank, nranks);
  ncclComm_t comm = nullptr;
  CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));
  fprintf(stderr, "[rank %d] comm init OK\n", rank);

  // Use a large symmetric AllGather to push NCCL toward the GIN kernel
  // (all_gather_gin). Each rank contributes 1 MB; total recv = nranks MB.
  constexpr size_t kPerRank = 1 << 20;     // 1 MiB per rank
  constexpr size_t kTotal = kPerRank;       // we reuse the same window for send+recv
  void* sbuf = nullptr;
  void* rbuf = nullptr;
  CHECK_NCCL(ncclMemAlloc(&sbuf, kPerRank));
  CHECK_NCCL(ncclMemAlloc(&rbuf, kPerRank * nranks));
  ncclWindow_t swin = nullptr;
  ncclWindow_t rwin = nullptr;
  CHECK_NCCL(ncclCommWindowRegister(comm, sbuf, kPerRank, &swin, 0));
  CHECK_NCCL(ncclCommWindowRegister(comm, rbuf, kPerRank * nranks, &rwin, 0));
  fprintf(stderr, "[rank %d] symmetric windows OK (sbuf=%p rbuf=%p)\n",
          rank, sbuf, rbuf);

  // Fill send buffer with distinctive pattern: byte = (rank+1)
  CHECK_CUDA(cudaMemset(sbuf, rank + 1, kPerRank));
  CHECK_CUDA(cudaMemset(rbuf, 0, kPerRank * nranks));

  fprintf(stderr, "[rank %d] AllGather %zu bytes per rank\n", rank, kPerRank);
  CHECK_NCCL(ncclAllGather(sbuf, rbuf, kPerRank, ncclInt8, comm, 0));
  CHECK_CUDA(cudaStreamSynchronize(0));
  fprintf(stderr, "[rank %d] AllGather done, verifying\n", rank);

  // Pull back a slice from each rank's region and verify content.
  bool ok = true;
  uint8_t sample[16];
  for (int r = 0; r < nranks; ++r) {
    CHECK_CUDA(cudaMemcpy(sample, (uint8_t*)rbuf + r * kPerRank, sizeof(sample),
                          cudaMemcpyDeviceToHost));
    uint8_t want = (uint8_t)(r + 1);
    for (size_t i = 0; i < sizeof(sample); ++i) {
      if (sample[i] != want) {
        fprintf(stderr, "[rank %d] MISMATCH region %d byte %zu got %u want %u\n",
                rank, r, i, sample[i], want);
        ok = false;
        break;
      }
    }
  }

  CHECK_NCCL(ncclCommWindowDeregister(comm, swin));
  CHECK_NCCL(ncclCommWindowDeregister(comm, rwin));
  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_NCCL(ncclMemFree(sbuf));
  CHECK_NCCL(ncclMemFree(rbuf));
  fprintf(stderr, "[rank %d] %s\n", rank, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
