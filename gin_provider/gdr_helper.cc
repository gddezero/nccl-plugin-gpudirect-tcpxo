/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "gin_provider/gdr_helper.h"

#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace fastrak::gin {

namespace {

struct GdrInfoV2 {
  uint64_t va;
  uint64_t mapped_size;
  uint32_t page_size;
  uint64_t tm_cycles;
  uint32_t cycles_per_ms;
  unsigned mapped : 1;
  unsigned wc_mapping : 1;
  int mapping_type;
};

struct GdrSyms {
  void* (*open)(void) = nullptr;
  int (*close)(void*) = nullptr;
  int (*pin_buffer)(void*, unsigned long, size_t, uint64_t, uint32_t,
                    uint64_t* /*gdr_mh_t*->h*/) = nullptr;
  int (*unpin_buffer)(void*, uint64_t /*gdr_mh_t.h*/) = nullptr;
  int (*map)(void*, uint64_t, void**, size_t) = nullptr;
  int (*unmap)(void*, uint64_t, void*, size_t) = nullptr;
  int (*get_info_v2)(void*, uint64_t, GdrInfoV2*) = nullptr;
  bool ok = false;
};

GdrSyms& gdr_syms() {
  static GdrSyms s;
  static std::atomic<bool> init_done{false};
  if (init_done.load(std::memory_order_acquire)) return s;
  static absl::Mutex mu;
  absl::MutexLock lock(&mu);
  if (init_done.load(std::memory_order_relaxed)) return s;

  void* h = dlopen("libgdrapi.so.2", RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libgdrapi.so", RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) {
    LOG(WARNING) << "GDRCopy: dlopen(libgdrapi.so) failed: " << dlerror()
                 << " — signal writes will fall back to cudaMemcpy";
    init_done.store(true, std::memory_order_release);
    return s;
  }

  s.open = reinterpret_cast<decltype(s.open)>(dlsym(h, "gdr_open"));
  s.close = reinterpret_cast<decltype(s.close)>(dlsym(h, "gdr_close"));
  s.pin_buffer = reinterpret_cast<decltype(s.pin_buffer)>(
      dlsym(h, "gdr_pin_buffer"));
  s.unpin_buffer = reinterpret_cast<decltype(s.unpin_buffer)>(
      dlsym(h, "gdr_unpin_buffer"));
  s.map = reinterpret_cast<decltype(s.map)>(dlsym(h, "gdr_map"));
  s.unmap = reinterpret_cast<decltype(s.unmap)>(dlsym(h, "gdr_unmap"));
  s.get_info_v2 = reinterpret_cast<decltype(s.get_info_v2)>(
      dlsym(h, "gdr_get_info_v2"));

  s.ok = (s.open && s.close && s.pin_buffer && s.unpin_buffer && s.map &&
          s.unmap && s.get_info_v2);
  if (!s.ok) {
    LOG(WARNING) << "GDRCopy: libgdrapi loaded but missing required symbols";
  } else {
    LOG(INFO) << "GDRCopy: libgdrapi.so loaded — host->GPU atomic path enabled";
  }
  init_done.store(true, std::memory_order_release);
  return s;
}

void* g_gdr_singleton = nullptr;
absl::Mutex g_gdr_singleton_mu;

void* GetGdrSingleton() {
  auto& s = gdr_syms();
  if (!s.ok) return nullptr;
  absl::MutexLock lock(&g_gdr_singleton_mu);
  if (g_gdr_singleton != nullptr) return g_gdr_singleton;
  g_gdr_singleton = s.open();
  if (g_gdr_singleton == nullptr) {
    LOG(ERROR) << "GDRCopy: gdr_open() returned NULL — is /dev/gdrdrv "
                  "accessible? gdrdrv module loaded?";
  }
  return g_gdr_singleton;
}

}  // namespace

bool GdrAvailable() { return gdr_syms().ok && GetGdrSingleton() != nullptr; }

absl::StatusOr<GdrPinnedRegion> GdrPinnedRegion::Create(void* gpu_ptr,
                                                       size_t size) {
  auto& s = gdr_syms();
  if (!s.ok) return absl::FailedPreconditionError("libgdrapi not available");
  void* g = GetGdrSingleton();
  if (g == nullptr) {
    return absl::InternalError("gdr_open singleton not available");
  }

  // Round addr down to 64KB page; round size up to next page.
  uintptr_t addr = reinterpret_cast<uintptr_t>(gpu_ptr);
  uintptr_t aligned_addr = addr & ~(kGdrPageSize - 1);
  size_t pre_offset = addr - aligned_addr;
  size_t pinned_size = (pre_offset + size + kGdrPageSize - 1) &
                       ~(kGdrPageSize - 1);

  uint64_t handle_h = 0;
  int rc = s.pin_buffer(g, aligned_addr, pinned_size, /*p2p_token=*/0,
                        /*va_space=*/0, &handle_h);
  if (rc != 0) {
    return absl::InternalError(absl::StrCat("gdr_pin_buffer failed: ", rc));
  }
  void* host_aligned = nullptr;
  rc = s.map(g, handle_h, &host_aligned, pinned_size);
  if (rc != 0) {
    s.unpin_buffer(g, handle_h);
    return absl::InternalError(absl::StrCat("gdr_map failed: ", rc));
  }

  GdrInfoV2 info{};
  rc = s.get_info_v2(g, handle_h, &info);
  if (rc == 0) {
    // info.va is the kernel-aligned start; recompute pre_offset from it
    pre_offset = addr - info.va;
  }
  void* host_map =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(host_aligned) +
                              pre_offset);
  return GdrPinnedRegion(g, handle_h, host_aligned, pinned_size, host_map,
                         size);
}

GdrPinnedRegion::GdrPinnedRegion(void* gdr, uint64_t handle_h,
                                 void* host_aligned, size_t mapped_size,
                                 void* host_map, size_t size)
    : gdr_(gdr),
      handle_h_(handle_h),
      host_aligned_(host_aligned),
      mapped_size_(mapped_size),
      host_map_(host_map),
      size_(size) {}

GdrPinnedRegion::GdrPinnedRegion(GdrPinnedRegion&& other) noexcept
    : gdr_(other.gdr_),
      handle_h_(other.handle_h_),
      host_aligned_(other.host_aligned_),
      mapped_size_(other.mapped_size_),
      host_map_(other.host_map_),
      size_(other.size_) {
  other.gdr_ = nullptr;
  other.handle_h_ = 0;
  other.host_aligned_ = nullptr;
  other.host_map_ = nullptr;
  other.size_ = 0;
  other.mapped_size_ = 0;
}

GdrPinnedRegion& GdrPinnedRegion::operator=(GdrPinnedRegion&& other) noexcept {
  if (this != &other) {
    this->~GdrPinnedRegion();
    new (this) GdrPinnedRegion(std::move(other));
  }
  return *this;
}

GdrPinnedRegion::~GdrPinnedRegion() {
  if (gdr_ != nullptr && host_aligned_ != nullptr) {
    auto& s = gdr_syms();
    if (s.unmap) s.unmap(gdr_, handle_h_, host_aligned_, mapped_size_);
    if (s.unpin_buffer) s.unpin_buffer(gdr_, handle_h_);
  }
  gdr_ = nullptr;
  host_aligned_ = nullptr;
  host_map_ = nullptr;
}

}  // namespace fastrak::gin
