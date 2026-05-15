/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Wire format for the bytes NCCL exchanges out-of-band between ranks during
 * GIN bootstrap (the `listen()` -> `connect()` handshake). NCCL allows up to
 * NCCL_GIN_HANDLE_MAXSIZE (128) bytes per rank.
 *
 * One handle per (rank, NIC) is constructed in our `listen()` callback and
 * gathered by NCCL via its bootstrap, then unpacked in our `connect()`
 * callback to learn each peer's DXS endpoint.
 *
 * Layout (POD, host byte order — collective is single-process today, so no
 * cross-arch concern; if that ever changes we'll byte-swap on encode/decode):
 *
 *   uint32_t   magic;            // 'FGIN' (0x4647494e BE)
 *   uint16_t   version;          // == 1
 *   uint16_t   addr_family;      // AF_INET (2) or AF_INET6 (10)
 *   uint8_t    addr[16];         // packed IPv4 (4 bytes) or IPv6 (16 bytes)
 *   uint16_t   port;             // dxs listen port
 *   uint8_t    fastrak_idx;      // local NIC index (0..7) — used as railId
 *   uint8_t    pad0;
 *   uint64_t   nonce;            // randomized per listen, sanity-check against MITM
 *   uint64_t   listen_token;     // local opaque cookie correlating Accept() to Connect()
 *   uint8_t    reserved[80];     // pad to 128 bytes
 */

#ifndef GIN_PROVIDER_LISTEN_HANDLE_H_
#define GIN_PROVIDER_LISTEN_HANDLE_H_

#include <cstdint>
#include <cstring>

namespace fastrak::gin {

constexpr uint32_t kListenHandleMagic = 0x4647494eu;  // 'F','G','I','N'
// v9: bumped from 1 → 2 to encode N NIC endpoints per ListenComm so peers can
// fan out across multiple receiver NICs. v1 layout is no longer wire-
// compatible; both peers must run the same plugin build.
constexpr uint16_t kListenHandleVersion = 2;
constexpr size_t kListenHandleSize = 128;  // == NCCL_GIN_HANDLE_MAXSIZE

constexpr int kListenHandleMaxNics = 8;

// One NIC endpoint inside a ListenHandle. IPv4 only — every fastrak NIC on
// a3-mega has an IPv4 address; if IPv6 is ever needed bump version again.
struct ListenHandleNic {
  uint8_t  addr[4];   // packed IPv4
  uint16_t port;      // dxs listen port for this NIC
  uint8_t  fastrak_idx;
  uint8_t  pad;
};
static_assert(sizeof(ListenHandleNic) == 8, "ListenHandleNic must be 8 bytes");

struct ListenHandle {
  uint32_t magic;
  uint16_t version;
  uint8_t  n_nics;
  uint8_t  primary_nic_idx;  // index INTO `nics`, not fastrak_idx
  uint64_t nonce;
  uint64_t listen_token;
  ListenHandleNic nics[kListenHandleMaxNics];   // 8 * 8 = 64 bytes
  uint8_t  reserved[40];
};
static_assert(sizeof(ListenHandle) == kListenHandleSize,
              "ListenHandle must be exactly NCCL_GIN_HANDLE_MAXSIZE bytes");

inline void EncodeListenHandle(void* dst, const ListenHandle& h) {
  std::memcpy(dst, &h, sizeof(h));
}

inline ListenHandle DecodeListenHandle(const void* src) {
  ListenHandle h;
  std::memcpy(&h, src, sizeof(h));
  return h;
}

}  // namespace fastrak::gin

#endif  // GIN_PROVIDER_LISTEN_HANDLE_H_
