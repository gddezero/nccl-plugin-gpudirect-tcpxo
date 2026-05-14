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

  // Performance sweep: AllReduce + AllGather across geometric sizes.
  // For each size: warm-up 5, time 20 iterations, report avg latency + busbw.
  constexpr size_t kMaxBytes = 256ULL << 20;  // 256 MiB
  void* sbuf = nullptr;
  void* rbuf = nullptr;
  CHECK_NCCL(ncclMemAlloc(&sbuf, kMaxBytes));
  CHECK_NCCL(ncclMemAlloc(&rbuf, kMaxBytes * nranks));
  ncclWindow_t swin = nullptr;
  ncclWindow_t rwin = nullptr;
  CHECK_NCCL(ncclCommWindowRegister(comm, sbuf, kMaxBytes, &swin, 0));
  CHECK_NCCL(ncclCommWindowRegister(comm,
                                    rbuf, kMaxBytes * nranks, &rwin, 0));
  CHECK_CUDA(cudaMemset(sbuf, rank + 1, kMaxBytes));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  cudaEvent_t evb, eve;
  CHECK_CUDA(cudaEventCreate(&evb));
  CHECK_CUDA(cudaEventCreate(&eve));

  if (rank == 0) {
    fprintf(stderr,
            "\n%-12s | %-22s | %-22s\n"
            "%-12s | %-10s %-10s | %-10s %-10s\n",
            "size", "AllReduce", "AllGather",
            "bytes", "lat(us)", "busbw(GB/s)", "lat(us)", "busbw(GB/s)");
  }

  bool ok = true;
  const size_t kSizes[] = {1<<10, 1<<14, 1<<18, 1<<20, 1<<22, 4<<20,
                           16<<20, 64<<20, 256<<20};
  constexpr int kWarm = 5;
  constexpr int kIters = 20;
  for (size_t bytes : kSizes) {
    if (bytes > kMaxBytes) break;
    size_t count = bytes / sizeof(float);

    // Warm up.
    for (int i = 0; i < kWarm; ++i) {
      CHECK_NCCL(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum,
                               comm, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // AllReduce timing.
    CHECK_CUDA(cudaEventRecord(evb, stream));
    for (int i = 0; i < kIters; ++i) {
      CHECK_NCCL(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum,
                               comm, stream));
    }
    CHECK_CUDA(cudaEventRecord(eve, stream));
    CHECK_CUDA(cudaEventSynchronize(eve));
    float ms_ar = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_ar, evb, eve));
    double lat_ar_us = (double)ms_ar * 1000.0 / kIters;
    // AllReduce busbw factor for ring: 2 * (n-1) / n
    double bw_ar = (double)bytes * 2.0 * (nranks - 1) / nranks /
                   (lat_ar_us * 1e-6) / 1e9;

    // AllGather timing.
    for (int i = 0; i < kWarm; ++i) {
      CHECK_NCCL(ncclAllGather(sbuf, rbuf, bytes, ncclInt8, comm, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventRecord(evb, stream));
    for (int i = 0; i < kIters; ++i) {
      CHECK_NCCL(ncclAllGather(sbuf, rbuf, bytes, ncclInt8, comm, stream));
    }
    CHECK_CUDA(cudaEventRecord(eve, stream));
    CHECK_CUDA(cudaEventSynchronize(eve));
    float ms_ag = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_ag, evb, eve));
    double lat_ag_us = (double)ms_ag * 1000.0 / kIters;
    // AllGather busbw factor: (n-1) / n  -- per output byte
    double bw_ag = (double)bytes * (nranks - 1) / nranks /
                   (lat_ag_us * 1e-6) / 1e9;

    if (rank == 0) {
      fprintf(stderr,
              "%-12zu | %-10.1f %-10.2f | %-10.1f %-10.2f\n",
              bytes, lat_ar_us, bw_ar, lat_ag_us, bw_ag);
    }
  }

  CHECK_CUDA(cudaEventDestroy(evb));
  CHECK_CUDA(cudaEventDestroy(eve));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_NCCL(ncclCommWindowDeregister(comm, swin));
  CHECK_NCCL(ncclCommWindowDeregister(comm, rwin));
  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_NCCL(ncclMemFree(sbuf));
  CHECK_NCCL(ncclMemFree(rbuf));
  fprintf(stderr, "[rank %d] %s\n", rank, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
