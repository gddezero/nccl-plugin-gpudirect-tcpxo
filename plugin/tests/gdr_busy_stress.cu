// gdr_busy_stress.cu — launch a GPU kernel that busy-spins for ~10s (like
// DeepEP barrier kernel). While it spins, the host thread does
// gdr_copy_to_mapping repeatedly into GPU mem and measures latency.
// If each gdr_copy_to_mapping returns in <1ms, GDRCopy bypasses CUDA cmd-queue
// contention and is suitable for our plugin path. If it blocks like
// cudaMemcpyAsync did → GDRCopy ALSO doesn't help and we need a different plan.
//
// Build:
//   nvcc -O2 gdr_busy_stress.cu -o gdr_busy_stress \
//        -I/usr/local/gdrcopy/include -L/usr/local/gdrcopy/lib \
//        -lgdrapi -lcudart -lcuda
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>

#define BUF_SZ (16 * 1024 * 1024)
#define MSG_SZ (16 * 1024)
#define N_ITER 20

// Kernel: spin until *flag becomes non-zero, polling every cycle.
__global__ void busy_spin(volatile int* flag, long long max_cycles) {
  long long start = clock64();
  while (*flag == 0) {
    if (clock64() - start > max_cycles) break;
  }
}

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(void) {
  cudaError_t ce;

  // Pinned host buffer for the spin flag (so we can set it to 1 from host)
  int* spin_flag;
  ce = cudaMallocManaged(&spin_flag, sizeof(int));
  if (ce) { fprintf(stderr, "FAIL cudaMallocManaged: %d\n", ce); return 1; }
  *spin_flag = 0;

  // Device buffer we'll write to via GDRCopy
  void* d;
  ce = cudaMalloc(&d, BUF_SZ);
  if (ce) { fprintf(stderr, "FAIL cudaMalloc: %d\n", ce); return 1; }

  gdr_t g = gdr_open();
  if (!g) { fprintf(stderr, "FAIL gdr_open\n"); return 1; }

  gdr_mh_t mh;
  int rc = gdr_pin_buffer(g, (CUdeviceptr)(uintptr_t)d, BUF_SZ, 0, 0, &mh);
  if (rc) { fprintf(stderr, "FAIL gdr_pin_buffer: %d\n", rc); return 1; }

  void* host_map = nullptr;
  rc = gdr_map(g, mh, &host_map, BUF_SZ);
  if (rc) { fprintf(stderr, "FAIL gdr_map: %d\n", rc); return 1; }

  gdr_info_t info; gdr_get_info(g, mh, &info);
  size_t off = ((uintptr_t)d) - info.va;
  fprintf(stderr, "ok: gdr setup, host_map=%p off=%zu\n", host_map, off);

  // Warm-up: one write before kernel
  static char pat[MSG_SZ];
  for (int i = 0; i < MSG_SZ; ++i) pat[i] = (char)(i & 0xff);
  rc = gdr_copy_to_mapping(mh, (char*)host_map + off, pat, MSG_SZ);
  if (rc) { fprintf(stderr, "FAIL warm-up gdr_copy_to_mapping: %d\n", rc); return 1; }
  fprintf(stderr, "ok: warm-up gdr_copy_to_mapping OK\n");

  // Launch busy_spin kernel that runs for ~10s (1.7GHz × 10s = 17e9 cycles)
  cudaStream_t s;
  cudaStreamCreate(&s);
  busy_spin<<<1, 1, 0, s>>>(spin_flag, 17000000000LL);
  fprintf(stderr, "ok: launched busy_spin kernel (~10s)\n");

  // Sleep 1s to ensure kernel is actively spinning
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  fprintf(stderr, "ok: kernel running, now timing gdr_copy_to_mapping x %d ...\n", N_ITER);

  double max_us = 0, total_us = 0;
  for (int i = 0; i < N_ITER; ++i) {
    double t0 = now_ms();
    rc = gdr_copy_to_mapping(mh, (char*)host_map + off + i * MSG_SZ, pat, MSG_SZ);
    double dt_us = (now_ms() - t0) * 1000.0;
    if (rc) { fprintf(stderr, "  iter %d FAIL rc=%d\n", i, rc); return 1; }
    if (dt_us > max_us) max_us = dt_us;
    total_us += dt_us;
    fprintf(stderr, "  iter %d: gdr_copy_to_mapping %.1f us\n", i, dt_us);
  }
  fprintf(stderr, "ok: %d iters, avg=%.1f us, max=%.1f us\n",
          N_ITER, total_us / N_ITER, max_us);

  // Signal the kernel to exit
  *spin_flag = 1;
  cudaStreamSynchronize(s);
  fprintf(stderr, "ok: kernel exited\n");

  if (max_us < 1000.0) {
    fprintf(stderr, "PASS: GDRCopy not blocked by busy kernel (max %.1f us < 1ms)\n", max_us);
  } else {
    fprintf(stderr, "FAIL: GDRCopy WAS blocked by busy kernel (max %.1f us >= 1ms)\n", max_us);
    return 2;
  }

  gdr_unmap(g, mh, host_map, BUF_SZ);
  gdr_unpin_buffer(g, mh);
  gdr_close(g);
  cudaFree(d);
  cudaFree(spin_flag);
  return 0;
}
