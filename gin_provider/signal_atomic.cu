/*
 * v20: GPU-side atomicAdd kernel for plugin signal RMW emulation.
 *
 * Replaces all CPU-side signal_host_addr() + GDR-mapped store + sfence and
 * cudaMemcpy fallback with a single tiny CUDA kernel that does
 * atomicAdd((unsigned long long*)slot, val) directly on the GPU buffer.
 *
 * Why: TCPXO+DXS lacks RDMA write-with-immediate, so we have to host-emulate
 * the signal RMW. v13b-v19 tried various host-mapped paths (GDR pin, lazy
 * chunked pin, cudaMemcpy fallback, sfence/clflushopt) and all hit either
 * BAR1 contention (1.16 GB DeepEP elastic buffer pin disrupts DXS RegBuf)
 * or atomicity limits. Pushing the RMW back to GPU sidesteps both.
 *
 * Cost: ~10-20 μs per kernel launch + sync. For ~10K signal RMWs in a
 * dispatch this adds ~100-200 ms — fits in the 60s WaitRecvDone deadline.
 * Per-CollComm dedicated stream serializes launches so order matches
 * receiver wire_seq commit_seq gate.
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace fastrak {
namespace gin {

__global__ void GinSignalAddKernel(unsigned long long* slot,
                                   unsigned long long val) {
  atomicAdd(slot, val);
}

extern "C" cudaError_t LaunchSignalAdd(void* slot_dev_ptr, uint64_t val,
                                       cudaStream_t stream) {
  GinSignalAddKernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<unsigned long long*>(slot_dev_ptr),
      static_cast<unsigned long long>(val));
  return cudaGetLastError();
}

}  // namespace gin
}  // namespace fastrak
