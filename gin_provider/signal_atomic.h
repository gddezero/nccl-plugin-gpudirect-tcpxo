/*
 * v20: launch helper for GinSignalAddKernel. See signal_atomic.cu.
 */
#ifndef GIN_PROVIDER_SIGNAL_ATOMIC_H_
#define GIN_PROVIDER_SIGNAL_ATOMIC_H_

#include <cuda_runtime.h>
#include <stdint.h>

namespace fastrak {
namespace gin {

extern "C" cudaError_t LaunchSignalAdd(void* slot_dev_ptr, uint64_t val,
                                       cudaStream_t stream);

}  // namespace gin
}  // namespace fastrak

#endif  // GIN_PROVIDER_SIGNAL_ATOMIC_H_
