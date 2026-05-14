/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Lazy dlopen() wrapper around libgdrapi.so. The plugin uses GDRCopy to
 * obtain a CPU-side mapping of NCCL's signalsDev (cuMemAlloc'd device
 * memory) so the host proxy thread can do __atomic_fetch_add directly
 * — without per-signal cudaMemcpy round-trips that would dominate
 * barrier latency.
 *
 * We dlopen at first use rather than linking -lgdrapi so the plugin
 * still loads gracefully (with degraded perf) when GDRCopy is missing.
 */

#ifndef GIN_PROVIDER_GDR_HELPER_H_
#define GIN_PROVIDER_GDR_HELPER_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace fastrak::gin {

// libgdrapi has 64KB GPU page granularity for pin_buffer.
constexpr size_t kGdrPageSize = 1u << 16;

// Singleton-style availability check. Loads libgdrapi.so on first call;
// returns false (cached) if dlopen or dlsym failed.
bool GdrAvailable();

// Wraps a pinned + mapped GDRCopy region. Owned (RAII).
// Construction does the gdr_pin_buffer + gdr_map dance and returns
// host_map() which is a CPU VA pointing at the GPU memory.
class GdrPinnedRegion {
 public:
  // gpu_ptr / size do not have to be 64KB-aligned: the kernel module
  // aligns down for us, and we expose host_map() pointing to the
  // ORIGINAL gpu_ptr (i.e. with the alignment offset already applied).
  static absl::StatusOr<GdrPinnedRegion> Create(void* gpu_ptr, size_t size);

  GdrPinnedRegion() = default;
  GdrPinnedRegion(const GdrPinnedRegion&) = delete;
  GdrPinnedRegion& operator=(const GdrPinnedRegion&) = delete;
  GdrPinnedRegion(GdrPinnedRegion&& other) noexcept;
  GdrPinnedRegion& operator=(GdrPinnedRegion&& other) noexcept;
  ~GdrPinnedRegion();

  // CPU-accessible view of the GPU memory region. Valid until destroyed.
  // host writes via this pointer are visible to the GPU through the BAR.
  void* host_map() const { return host_map_; }
  size_t size() const { return size_; }
  bool ok() const { return host_map_ != nullptr; }

 private:
  GdrPinnedRegion(void* gdr, uint64_t handle_h, void* host_aligned,
                  size_t mapped_size, void* host_map, size_t size);

  void* gdr_ = nullptr;             // gdr_t handle (opaque)
  uint64_t handle_h_ = 0;            // gdr_mh_t.h
  void* host_aligned_ = nullptr;     // value returned by gdr_map (aligned-down)
  size_t mapped_size_ = 0;           // size used at gdr_pin_buffer / gdr_map
  void* host_map_ = nullptr;         // host_aligned_ + offset to match gpu_ptr
  size_t size_ = 0;
};

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_GDR_HELPER_H_
