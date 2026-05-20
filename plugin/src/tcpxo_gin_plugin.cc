// SPDX-License-Identifier: Apache-2.0
// TCPXO NCCL GIN Plugin — M2: real TCP-based data plane.
//
// Architecture:
//  - listen()      : bind ephemeral TCP port on configured iface (NCCL_SOCKET_IFNAME or first non-loopback IPv4); write (ip, port) into 128B handle.
//  - connect()     : collective. For each peer p != self: if p < self, connect; if p > self, accept. Disambiguate peer-rank via 4B handshake.
//  - regMrSym()    : append MR record to per-CollComm table; ginHandle = monotonic token starting at 1. SAME buffer registered in SAME order across ranks ⇒ token equal everywhere, no allgather.
//  - iput / iputSignal : sender stages CUDA payload to host pinned buf, writes WireMsgHdr + payload over TCP; mark request done immediately (TCP guarantees in-order + sendmsg copy-out).
//  - recv_thread   : per CollComm, poll()-multiplex peer sockets; on incoming hdr+payload, look up dst MR by token, cudaMemcpy → device, then if signal_op != 0 apply atomic add to signal MR (host RMW via cudaMemcpy — receiver thread is single-threaded so no race against itself; kernel only reads).
//  - test()        : poll request->done (set synchronously in iput/iputSignal — TCP send completed).
//  - ginProgress() : no-op (recv thread runs independently).
//
// Out of scope for M2:
//  - DXS / GPUDirect TCPX-O (perf-only; M4-post).
//  - rail-aware NIC steering (single NIC for now; all peers via NCCL_SOCKET_IFNAME).
//  - DMABUF.
//  - Multi-chunk for size > stage_buf (cap = 64 MiB per iput).

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>

#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>

#include "nccl.h"
#include "nccl_plugin_abi/nccl_common.h"
#include "nccl_plugin_abi/nccl_net.h"
#include "nccl_plugin_abi/nccl_gin.h"

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

static ncclDebugLogger_t g_log = nullptr;

#define LOG(level, flags, fmt, ...)                                            \
  do {                                                                         \
    if (g_log)                                                                 \
      g_log((level), (flags), __FILE__, __LINE__, "[tcpxo-gin] " fmt,          \
            ##__VA_ARGS__);                                                    \
    else                                                                       \
      fprintf(stderr, "[tcpxo-gin] " fmt "\n", ##__VA_ARGS__);                 \
  } while (0)

#define INFO(fmt, ...) LOG(NCCL_LOG_INFO, NCCL_NET, fmt, ##__VA_ARGS__)
#define WARN(fmt, ...) LOG(NCCL_LOG_WARN, NCCL_NET, fmt, ##__VA_ARGS__)
#define TRACE(fmt, ...) LOG(NCCL_LOG_TRACE, NCCL_NET, fmt, ##__VA_ARGS__)

static const char* env_or(const char* k, const char* fallback) {
  const char* v = std::getenv(k);
  return v ? v : fallback;
}

// ---------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t WIRE_MAGIC = 0x47494E58u; // "GINX"
constexpr uint8_t  OP_PUT        = 1;
constexpr uint8_t  OP_PUT_SIGNAL = 2;
// iget protocol: requester sends OP_GET with signal_token/signal_off pointing
// at the REMOTE source MR, and dst_token/dst_off at its OWN local dst. Remote
// recv reads its local mem and ships back as OP_PUT (with req_id echoed) so
// requester's recv_thread can mark the original Request done.
constexpr uint8_t  OP_GET        = 3;
constexpr uint8_t  SIGOP_NONE = 0;
constexpr uint8_t  SIGOP_INC  = 1;
constexpr uint8_t  SIGOP_ADD  = 2;

#pragma pack(push, 1)
struct WireMsgHdr {
  uint32_t magic;
  uint8_t  op;
  uint8_t  signal_op;
  uint16_t pad0;
  uint32_t pad1;
  uint64_t req_id;
  uint64_t dst_token;
  uint64_t dst_off;
  uint64_t size;
  uint64_t signal_token;
  uint64_t signal_off;
  uint64_t signal_val;
};
#pragma pack(pop)
static_assert(sizeof(WireMsgHdr) == 68, "WireMsgHdr layout");

constexpr size_t kStageBufBytes = 64ull * 1024 * 1024; // hard cap per iput

// ---------------------------------------------------------------------------
// Plugin state
// ---------------------------------------------------------------------------

struct PluginCtx {
  uint64_t commId{0};
  std::atomic<bool> initialized{false};
  std::string iface;
  in_addr_t local_ip{0}; // network byte order
  gdr_t gdr{nullptr};    // GDRCopy handle (opened lazily on first CUDA regMrSym)
  std::mutex gdr_mu;
};

struct MrRecord {
  void* addr{nullptr};
  size_t size{0};
  int type{0};
  uint64_t flags{0};
  uint64_t token{0};
  // GDRCopy mapping (for CUDA-type MRs only): map device VA → host BAR1 VA so
  // CPU can read/write GPU mem without going through cudaMemcpyAsync (which
  // gets blocked by spinning user kernels on H100).
  bool gdr_pinned{false};
  gdr_mh_t gdr_mh{};
  void* gdr_host_map{nullptr};
  size_t gdr_map_offset{0};  // (addr - mapping base VA) so caller can index
  size_t gdr_mapped_size{0};
};

struct ListenHandleV1 {
  uint32_t magic;
  uint32_t pad;
  uint32_t ip;   // network byte order (in_addr_t)
  uint16_t port; // network byte order
  uint16_t pad2;
  uint8_t reserved[112];
};
static_assert(sizeof(ListenHandleV1) == NCCL_NET_HANDLE_MAXSIZE,
              "ListenHandle must be 128 bytes");

struct ListenComm {
  int fd{-1};
  int dev{0};
};

struct PeerLink {
  int fd{-1};
  std::mutex send_mu;
};

struct Request {
  std::atomic<int> done{0};
};

struct CollComm {
  int rank{0};
  int nranks{0};
  std::vector<std::unique_ptr<PeerLink>> peers; // size nranks; self entry unused

  std::mutex mr_mu;
  std::vector<MrRecord*> mrs;
  uint64_t next_token{1};

  // iget pending request tracking. requester assigns req_id, stores Request*
  // here, then waits for remote to PUT-back with same req_id.
  std::mutex pending_gets_mu;
  std::unordered_map<uint64_t, Request*> pending_gets;
  std::atomic<uint64_t> next_req_id{1};

  // (Path-B experiment 2 reverted — caused std::system_error: Invalid argument
  //  at NCCL's first GIN-touching call. Even if it had run, it could not have
  //  unblocked NCCL because NCCL never calls our test() to begin with.)

  // recv side
  std::thread recv_thread;
  std::atomic<bool> running{false};
  int wakeup_pipe[2]{-1, -1}; // self-pipe to wake poll() during shutdown

  // staging buffers (host pinned)
  void* recv_stage{nullptr};
  void* send_stage{nullptr};
  size_t stage_bytes{kStageBufBytes};
  std::mutex send_stage_mu;

  // signal scratch (8B host pinned, single owner = recv thread)
  void* signal_scratch{nullptr};

  // Dedicated CUDA stream for plugin-side signal RMW. Must NOT be the default
  // stream — DeepEP barrier kernels busy-wait on user stream; cudaMemcpy on
  // default stream implicit-syncs with all streams and deadlocks.
  cudaStream_t signal_stream{nullptr};

  // DIAG: 16MB device buffer the plugin owns, used to test cudaMemcpyAsync
  // behaviour when dst is NOT a DeepEP/NCCL-allocated region. If memcpy to
  // this buffer also hangs while user kernel runs → kernel busy-wait blocks
  // ANY async memcpy. If it works → issue is dst-region specific.
  void* diag_dev_buf{nullptr};
  std::atomic<int> diag_done{0};
};

// Global plugin instance (process-wide PluginCtx).
PluginCtx* g_plugin = nullptr;
std::mutex g_plugin_mu;

// Path-B diagnostic counters: how often does NCCL actually invoke our
// data-plane entry points? If these stay at 0 while the barrier kernel runs,
// NCCL's proxy progress chain isn't reaching us at all.
std::atomic<uint64_t> g_iput_calls{0};
std::atomic<uint64_t> g_iputsignal_calls{0};
std::atomic<uint64_t> g_test_calls{0};
std::atomic<uint64_t> g_ginprogress_calls{0};

// ---------------------------------------------------------------------------
// Network helpers
// ---------------------------------------------------------------------------

static in_addr_t resolve_local_ip(const std::string& iface_pref) {
  struct ifaddrs* ifap = nullptr;
  if (getifaddrs(&ifap) != 0) return INADDR_ANY;
  in_addr_t pref_ip = 0, fallback_ip = 0;
  for (auto* p = ifap; p != nullptr; p = p->ifa_next) {
    if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
    if (!(p->ifa_flags & IFF_UP) || (p->ifa_flags & IFF_LOOPBACK)) continue;
    auto* sin = reinterpret_cast<sockaddr_in*>(p->ifa_addr);
    in_addr_t ip = sin->sin_addr.s_addr;
    if (!iface_pref.empty() && iface_pref == p->ifa_name) {
      pref_ip = ip;
      break;
    }
    if (fallback_ip == 0) fallback_ip = ip;
  }
  freeifaddrs(ifap);
  return pref_ip ? pref_ip : fallback_ip;
}

static int write_all(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  size_t left = n;
  while (left > 0) {
    ssize_t w = ::write(fd, p, left);
    if (w < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (w == 0) return -1;
    p += w;
    left -= w;
  }
  return 0;
}

static int read_all(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  size_t left = n;
  while (left > 0) {
    ssize_t r = ::read(fd, p, left);
    if (r < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (r == 0) return -2; // EOF
    p += r;
    left -= r;
  }
  return 0;
}

static void set_socket_bufs(int fd) {
  int sz = 64 * 1024 * 1024;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// ---------------------------------------------------------------------------
// Recv thread
// ---------------------------------------------------------------------------

// Forward decls for iget path
static ncclResult_t do_send(CollComm* cc, int dst_rank, const WireMsgHdr& h,
                            const void* src_addr, int src_type);
static int handle_get_request(CollComm* cc, const WireMsgHdr& h, int src_rank);

static int apply_recv(CollComm* cc, const WireMsgHdr& h, const void* payload,
                      int src_rank) {
  cudaError_t ce = cudaSuccess;
  static std::atomic<uint64_t> g_apply_recv_calls{0};
  uint64_t arc = ++g_apply_recv_calls;
  INFO("PATHB apply_recv #%lu cc=%p op=%u size=%lu dstTok=%lu dstOff=%lu sigOp=%u sigTok=%lu sigOff=%lu sigVal=%lu req_id=%lu src=%d",
       arc, cc, (unsigned)h.op, (unsigned long)h.size, (unsigned long)h.dst_token,
       (unsigned long)h.dst_off, (unsigned)h.signal_op,
       (unsigned long)h.signal_token, (unsigned long)h.signal_off,
       (unsigned long)h.signal_val, (unsigned long)h.req_id, src_rank);

  // OP_GET: a peer is asking us to ship back data from one of our MRs.
  if (h.op == OP_GET) {
    return handle_get_request(cc, h, src_rank);
  }

  // 1. Apply data payload if size > 0.
  if (h.size > 0) {
    if (h.dst_token == 0 || h.dst_token > cc->mrs.size()) {
      WARN("recv: bad dst_token=%lu (have %zu mrs)", h.dst_token,
           cc->mrs.size());
      return -1;
    }
    MrRecord* dm = cc->mrs[h.dst_token - 1];
    if (!dm) {
      WARN("recv: null dst mr at token=%lu", h.dst_token);
      return -1;
    }
    if (h.dst_off + h.size > dm->size) {
      WARN("recv: out-of-bounds dst_off=%lu size=%lu mr_size=%zu", h.dst_off,
           h.size, dm->size);
      return -1;
    }
    void* dst = static_cast<uint8_t*>(dm->addr) + h.dst_off;
    if (dm->type == NCCL_PTR_CUDA) {
      if (dm->gdr_pinned && g_plugin && g_plugin->gdr) {
        // GDRCopy path: host-mapped BAR1 write, bypasses CUDA cmd queue, ~7us
        // even when user kernel is spinning.
        void* hm_off = static_cast<uint8_t*>(dm->gdr_host_map)
                       + dm->gdr_map_offset + h.dst_off;
        int rc = gdr_copy_to_mapping(dm->gdr_mh, hm_off, payload, h.size);
        if (rc != 0) {
          WARN("recv: gdr_copy_to_mapping payload rc=%d", rc);
          return -1;
        }
      } else {
        ce = cudaMemcpyAsync(dst, payload, h.size, cudaMemcpyHostToDevice,
                             cc->signal_stream);
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(cc->signal_stream);
        if (ce != cudaSuccess) {
          WARN("recv: cudaMemcpyAsync H→D payload failed: %s", cudaGetErrorString(ce));
          return -1;
        }
      }
    } else {
      std::memcpy(dst, payload, h.size);
    }
  }

  // 2. If signal op present, apply RMW to signal MR.
  if (h.signal_op != SIGOP_NONE) {
    if (h.signal_token == 0 || h.signal_token > cc->mrs.size()) {
      WARN("recv: bad signal_token=%lu", h.signal_token);
      return -1;
    }
    MrRecord* sm = cc->mrs[h.signal_token - 1];
    if (!sm || h.signal_off + sizeof(uint64_t) > sm->size) {
      WARN("recv: bad signal MR or off");
      return -1;
    }
    void* sp = static_cast<uint8_t*>(sm->addr) + h.signal_off;
    uint64_t cur = 0;
    if (sm->type == NCCL_PTR_CUDA) {
      if (sm->gdr_pinned && g_plugin && g_plugin->gdr) {
        void* hm_off = static_cast<uint8_t*>(sm->gdr_host_map)
                       + sm->gdr_map_offset + h.signal_off;
        int rc = gdr_copy_from_mapping(sm->gdr_mh, &cur, hm_off, 8);
        if (rc != 0) { WARN("recv: gdr_copy_from_mapping signal rc=%d", rc); return -1; }
      } else {
        ce = cudaMemcpyAsync(cc->signal_scratch, sp, 8, cudaMemcpyDeviceToHost,
                             cc->signal_stream);
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(cc->signal_stream);
        if (ce != cudaSuccess) {
          WARN("recv: cudaMemcpyAsync D→H signal failed: %s",
               cudaGetErrorString(ce));
          return -1;
        }
        cur = *static_cast<uint64_t*>(cc->signal_scratch);
      }
    } else {
      cur = *static_cast<uint64_t*>(sp);
    }
    uint64_t next = (h.signal_op == SIGOP_INC) ? cur + 1 : cur + h.signal_val;
    if (sm->type == NCCL_PTR_CUDA) {
      if (sm->gdr_pinned && g_plugin && g_plugin->gdr) {
        void* hm_off = static_cast<uint8_t*>(sm->gdr_host_map)
                       + sm->gdr_map_offset + h.signal_off;
        int rc = gdr_copy_to_mapping(sm->gdr_mh, hm_off, &next, 8);
        if (rc != 0) { WARN("recv: gdr_copy_to_mapping signal rc=%d", rc); return -1; }
      } else {
        *static_cast<uint64_t*>(cc->signal_scratch) = next;
        ce = cudaMemcpyAsync(sp, cc->signal_scratch, 8, cudaMemcpyHostToDevice,
                             cc->signal_stream);
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(cc->signal_stream);
        if (ce != cudaSuccess) {
          WARN("recv: cudaMemcpyAsync H→D signal failed: %s",
               cudaGetErrorString(ce));
          return -1;
        }
      }
    } else {
      *static_cast<uint64_t*>(sp) = next;
    }
    TRACE("recv: signal tok=%lu off=%lu %lu → %lu (op=%u)",
          h.signal_token, h.signal_off, cur, next, h.signal_op);
    INFO("PATHB apply_recv: signal RMW ok tok=%lu off=%lu %lu->%lu type=%d",
         (unsigned long)h.signal_token, (unsigned long)h.signal_off,
         (unsigned long)cur, (unsigned long)next, sm->type);
  }

  // If this PUT was the response to an outstanding iget, complete it.
  if (h.req_id != 0) {
    Request* r = nullptr;
    {
      std::lock_guard<std::mutex> g(cc->pending_gets_mu);
      auto it = cc->pending_gets.find(h.req_id);
      if (it != cc->pending_gets.end()) {
        r = it->second;
        cc->pending_gets.erase(it);
      }
    }
    if (r) {
      r->done.store(1, std::memory_order_release);
      INFO("PATHB apply_recv: iget reply matched req_id=%lu req=%p done=1",
           (unsigned long)h.req_id, r);
    } else {
      WARN("apply_recv: orphan req_id=%lu (no pending iget)", (unsigned long)h.req_id);
    }
  }
  return 0;
}

// Service an incoming OP_GET: we are the SOURCE peer of an iget. Read our
// local mem at (signal_token, signal_off), package an OP_PUT reply that
// targets the requester's (dst_token, dst_off) with req_id echoed.
static int handle_get_request(CollComm* cc, const WireMsgHdr& greq, int src_rank) {
  if (src_rank < 0) {
    WARN("handle_get_request: unknown src_rank");
    return -1;
  }
  // Look up source MR with mr_mu held briefly, then release BEFORE calling
  // do_send (which itself acquires mr_mu via find_mr_for_range → recursive
  // deadlock with std::mutex).
  void* src_addr = nullptr;
  int src_type = 0;
  {
    std::lock_guard<std::mutex> mg(cc->mr_mu);
    if (greq.signal_token == 0 || greq.signal_token > cc->mrs.size()) {
      WARN("handle_get_request: bad src_token=%lu", (unsigned long)greq.signal_token);
      return -1;
    }
    MrRecord* sm = cc->mrs[greq.signal_token - 1];
    if (!sm || greq.signal_off + greq.size > sm->size) {
      WARN("handle_get_request: bad src MR or off");
      return -1;
    }
    src_addr = static_cast<uint8_t*>(sm->addr) + greq.signal_off;
    src_type = sm->type;
  }
  WireMsgHdr reply{};
  reply.magic = WIRE_MAGIC;
  reply.op = OP_PUT;
  reply.signal_op = SIGOP_NONE;
  reply.req_id = greq.req_id;      // echo
  reply.dst_token = greq.dst_token;
  reply.dst_off = greq.dst_off;
  reply.size = greq.size;
  INFO("PATHB handle_get_request: src_rank=%d req_id=%lu reading srcTok=%lu srcOff=%lu size=%lu → reply dstTok=%lu dstOff=%lu",
       src_rank, (unsigned long)greq.req_id, (unsigned long)greq.signal_token,
       (unsigned long)greq.signal_off, (unsigned long)greq.size,
       (unsigned long)greq.dst_token, (unsigned long)greq.dst_off);
  ncclResult_t rc = do_send(cc, src_rank, reply, src_addr, src_type);
  if (rc != ncclSuccess) {
    WARN("handle_get_request: do_send reply failed rc=%d", (int)rc);
    return -1;
  }
  return 0;
}

static void recv_thread_fn(CollComm* cc) {
  // Bind this thread to CUDA device 0. Required before any cudaMemcpyAsync —
  // a freshly spawned std::thread has no current CUDA context.
  cudaError_t set_ce = cudaSetDevice(0);
  if (set_ce != cudaSuccess) {
    WARN("recv_thread_fn: cudaSetDevice failed: %s", cudaGetErrorString(set_ce));
  }
  std::vector<pollfd> pfds;
  pfds.reserve(cc->nranks + 1);
  while (cc->running.load(std::memory_order_acquire)) {
    pfds.clear();
    pollfd wp = {cc->wakeup_pipe[0], POLLIN, 0};
    pfds.push_back(wp);
    for (int r = 0; r < cc->nranks; ++r) {
      if (r == cc->rank) continue;
      int fd = cc->peers[r]->fd;
      if (fd < 0) continue;
      pollfd pe = {fd, POLLIN, 0};
      pfds.push_back(pe);
    }
    int rc = ::poll(pfds.data(), pfds.size(), 500); // 500ms tick
    if (rc < 0) {
      if (errno == EINTR) continue;
      WARN("recv_thread: poll() failed: %s", std::strerror(errno));
      break;
    }
    if (rc == 0) continue;
    // Drain wakeup pipe (if any)
    if (pfds[0].revents & POLLIN) {
      char trash[64];
      while (::read(cc->wakeup_pipe[0], trash, sizeof(trash)) > 0) {
      }
    }
    for (size_t i = 1; i < pfds.size(); ++i) {
      if (!(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      int fd = pfds[i].fd;
      WireMsgHdr h;
      int r = read_all(fd, &h, sizeof(h));
      static std::atomic<uint64_t> g_rt_hdr_calls{0};
      uint64_t rtc = ++g_rt_hdr_calls;
      INFO("PATHB recv_thread #%lu fd=%d read_rc=%d magic=0x%x op=%u size=%lu sigOp=%u sigOff=%lu",
           rtc, fd, r, (unsigned)h.magic, (unsigned)h.op,
           (unsigned long)h.size, (unsigned)h.signal_op,
           (unsigned long)h.signal_off);
      if (r != 0) {
        if (cc->running.load()) {
          WARN("recv_thread: read hdr failed on fd=%d rc=%d (%s)",
               fd, r, std::strerror(errno));
        }
        // Close and forget. We don't try to repair connections in M2.
        ::close(fd);
        for (int p = 0; p < cc->nranks; ++p) {
          if (cc->peers[p]->fd == fd) cc->peers[p]->fd = -1;
        }
        continue;
      }
      if (h.magic != WIRE_MAGIC) {
        WARN("recv_thread: bad magic 0x%x", h.magic);
        ::close(fd);
        for (int p = 0; p < cc->nranks; ++p) {
          if (cc->peers[p]->fd == fd) cc->peers[p]->fd = -1;
        }
        continue;
      }
      if (h.size > cc->stage_bytes) {
        WARN("recv_thread: oversized msg size=%lu cap=%zu", h.size,
             cc->stage_bytes);
        ::close(fd);
        for (int p = 0; p < cc->nranks; ++p) {
          if (cc->peers[p]->fd == fd) cc->peers[p]->fd = -1;
        }
        continue;
      }
      if (h.size > 0 && h.op != OP_GET) {
        if (read_all(fd, cc->recv_stage, h.size) != 0) {
          WARN("recv_thread: read payload failed");
          ::close(fd);
          for (int p = 0; p < cc->nranks; ++p) {
            if (cc->peers[p]->fd == fd) cc->peers[p]->fd = -1;
          }
          continue;
        }
      }
      // Find which peer this fd belongs to (so OP_GET handlers know who to reply to).
      int src_rank = -1;
      for (int p = 0; p < cc->nranks; ++p) {
        if (cc->peers[p] && cc->peers[p]->fd == fd) { src_rank = p; break; }
      }
      apply_recv(cc, h, cc->recv_stage, src_rank);
    }
  }
}

// ---------------------------------------------------------------------------
// GIN plugin entry points
// ---------------------------------------------------------------------------

static ncclResult_t gin_init(void** ctx, uint64_t commId,
                             ncclDebugLogger_t logFunction) {
  g_log = logFunction;
  std::lock_guard<std::mutex> g(g_plugin_mu);
  if (!g_plugin) {
    g_plugin = new PluginCtx;
    g_plugin->iface = env_or("NCCL_SOCKET_IFNAME", "eth0");
    g_plugin->local_ip = resolve_local_ip(g_plugin->iface);
    char buf[64];
    inet_ntop(AF_INET, &g_plugin->local_ip, buf, sizeof(buf));
    INFO("gin_init: iface=%s local_ip=%s commId=%lu",
         g_plugin->iface.c_str(), buf, (unsigned long)commId);
  }
  g_plugin->commId = commId;
  g_plugin->initialized.store(true);
  *ctx = g_plugin;
  return ncclSuccess;
}

static ncclResult_t gin_devices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t gin_getProperties(int dev,
                                      ncclNetProperties_v11_t* props) {
  if (dev != 0) return ncclInvalidArgument;
  std::memset(props, 0, sizeof(*props));
  static char name_buf[] = "tcpxo-gin/0";
  static char pci_buf[] = "";
  props->name = name_buf;
  props->pciPath = pci_buf;
  props->guid = 0xC0FFEEull;
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 200000;
  props->port = 0;
  props->latency = 1.0f;
  props->maxComms = 1024;
  // Path-B experiment 1/2: bump maxRecvs to allow NCCL queue depth
  // 32*8=256 GFDs in flight, so a single stalled iputSignal doesn't backpressure
  // the entire barrier.
  props->maxRecvs = 8;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  props->netDeviceVersion = 100;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = 0;
  props->maxP2pBytes = kStageBufBytes;
  props->maxCollBytes = kStageBufBytes;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t gin_listen(void* /*ctx*/, int dev, void* opaqueHandle,
                               void** listenComm) {
  auto* h = reinterpret_cast<ListenHandleV1*>(opaqueHandle);
  std::memset(h, 0, sizeof(*h));
  h->magic = WIRE_MAGIC;
  // Create listen socket
  int s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) {
    WARN("gin_listen: socket() failed: %s", std::strerror(errno));
    return ncclSystemError;
  }
  int one = 1;
  ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  set_socket_bufs(s);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = g_plugin ? g_plugin->local_ip : INADDR_ANY;
  sa.sin_port = 0;
  if (::bind(s, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    WARN("gin_listen: bind() failed: %s", std::strerror(errno));
    ::close(s);
    return ncclSystemError;
  }
  if (::listen(s, 64) != 0) {
    WARN("gin_listen: listen() failed: %s", std::strerror(errno));
    ::close(s);
    return ncclSystemError;
  }
  sockaddr_in bound{};
  socklen_t bl = sizeof(bound);
  ::getsockname(s, reinterpret_cast<sockaddr*>(&bound), &bl);
  h->ip = bound.sin_addr.s_addr;
  h->port = bound.sin_port;
  auto* lc = new ListenComm;
  lc->fd = s;
  lc->dev = dev;
  *listenComm = lc;
  char ipbuf[64];
  inet_ntop(AF_INET, &h->ip, ipbuf, sizeof(ipbuf));
  INFO("gin_listen: dev=%d bound=%s:%u listenComm=%p", dev, ipbuf,
       ntohs(h->port), lc);
  return ncclSuccess;
}

static ncclResult_t gin_connect(void* /*ctx*/, void* handles[], int nranks,
                                int rank, void* listenComm, void** collComm) {
  auto* lc = reinterpret_cast<ListenComm*>(listenComm);
  if (!lc) return ncclInvalidArgument;

  auto* cc = new CollComm;
  cc->rank = rank;
  cc->nranks = nranks;
  cc->peers.reserve(nranks);
  for (int i = 0; i < nranks; ++i) cc->peers.emplace_back(new PeerLink());

  // Validate handles
  for (int r = 0; r < nranks; ++r) {
    auto* h = reinterpret_cast<ListenHandleV1*>(handles[r]);
    if (h->magic != WIRE_MAGIC) {
      WARN("gin_connect: bad magic from rank %d", r);
      delete cc;
      return ncclInternalError;
    }
  }

  // For each peer != self:
  //   if peer < self: we connect to peer (peer accepts).
  //   if peer > self: peer connects to us, we accept.
  int expected_accepts = nranks - 1 - rank; // count of peers > rank
  int connects_done = 0, accepts_done = 0;

  for (int peer = 0; peer < nranks; ++peer) {
    if (peer == rank) continue;
    if (peer >= rank) continue; // only do connects in pass 1
    auto* h = reinterpret_cast<ListenHandleV1*>(handles[peer]);
    sockaddr_in pa{};
    pa.sin_family = AF_INET;
    pa.sin_addr.s_addr = h->ip;
    pa.sin_port = h->port;
    int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
      WARN("gin_connect: socket() failed: %s", std::strerror(errno));
      delete cc;
      return ncclSystemError;
    }
    set_socket_bufs(s);
    // retry connect a few times in case peer not yet listening
    int attempts = 0;
    while (true) {
      if (::connect(s, reinterpret_cast<sockaddr*>(&pa), sizeof(pa)) == 0)
        break;
      if (++attempts > 100) {
        WARN("gin_connect: connect to rank %d failed: %s", peer,
             std::strerror(errno));
        ::close(s);
        delete cc;
        return ncclSystemError;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Send our rank so peer can route the accept.
    uint32_t my_rank_be = htonl((uint32_t)rank);
    if (write_all(s, &my_rank_be, sizeof(my_rank_be)) != 0) {
      WARN("gin_connect: handshake write to rank %d failed", peer);
      ::close(s);
      delete cc;
      return ncclSystemError;
    }
    cc->peers[peer]->fd = s;
    connects_done++;
  }

  // Accept pass: accept (nranks - 1 - rank) connections; each carries 4B rank.
  while (accepts_done < expected_accepts) {
    sockaddr_in pa{};
    socklen_t pl = sizeof(pa);
    int s = ::accept(lc->fd, reinterpret_cast<sockaddr*>(&pa), &pl);
    if (s < 0) {
      WARN("gin_connect: accept() failed: %s", std::strerror(errno));
      delete cc;
      return ncclSystemError;
    }
    set_socket_bufs(s);
    uint32_t remote_rank_be = 0;
    if (read_all(s, &remote_rank_be, sizeof(remote_rank_be)) != 0) {
      WARN("gin_connect: handshake read failed");
      ::close(s);
      continue; // try next accept
    }
    int remote_rank = (int)ntohl(remote_rank_be);
    if (remote_rank < 0 || remote_rank >= nranks || remote_rank == rank) {
      WARN("gin_connect: bad remote rank %d", remote_rank);
      ::close(s);
      continue;
    }
    if (cc->peers[remote_rank]->fd >= 0) {
      WARN("gin_connect: duplicate accept from rank %d", remote_rank);
      ::close(s);
      continue;
    }
    cc->peers[remote_rank]->fd = s;
    accepts_done++;
  }

  // Set up recv thread + staging.
  if (::pipe(cc->wakeup_pipe) != 0) {
    WARN("gin_connect: pipe() failed: %s", std::strerror(errno));
    delete cc;
    return ncclSystemError;
  }
  // Make wakeup-read non-blocking so we can drain.
  int fl = fcntl(cc->wakeup_pipe[0], F_GETFL, 0);
  fcntl(cc->wakeup_pipe[0], F_SETFL, fl | O_NONBLOCK);

  cudaError_t ce =
      cudaMallocHost(&cc->recv_stage, cc->stage_bytes);
  if (ce != cudaSuccess) {
    WARN("gin_connect: cudaMallocHost(recv) failed: %s",
         cudaGetErrorString(ce));
    delete cc;
    return ncclSystemError;
  }
  ce = cudaMallocHost(&cc->send_stage, cc->stage_bytes);
  if (ce != cudaSuccess) {
    WARN("gin_connect: cudaMallocHost(send) failed: %s",
         cudaGetErrorString(ce));
    cudaFreeHost(cc->recv_stage);
    delete cc;
    return ncclSystemError;
  }
  ce = cudaMallocHost(&cc->signal_scratch, 16);
  if (ce != cudaSuccess) {
    WARN("gin_connect: cudaMallocHost(scratch) failed: %s",
         cudaGetErrorString(ce));
    cudaFreeHost(cc->recv_stage);
    cudaFreeHost(cc->send_stage);
    delete cc;
    return ncclSystemError;
  }
  // Non-blocking signal stream so signal RMW doesn't implicit-sync with the
  // user kernel busy-waiting on signal memory.
  ce = cudaStreamCreateWithFlags(&cc->signal_stream, cudaStreamNonBlocking);
  if (ce != cudaSuccess) {
    WARN("gin_connect: cudaStreamCreateWithFlags failed: %s",
         cudaGetErrorString(ce));
    cudaFreeHost(cc->signal_scratch);
    cudaFreeHost(cc->recv_stage);
    cudaFreeHost(cc->send_stage);
    delete cc;
    return ncclSystemError;
  }
  // DIAG: allocate 16MB device buffer for the probing test
  ce = cudaMalloc(&cc->diag_dev_buf, 16 * 1024 * 1024);
  if (ce != cudaSuccess) {
    WARN("gin_connect: cudaMalloc(diag_dev_buf) failed: %s",
         cudaGetErrorString(ce));
    cc->diag_dev_buf = nullptr;
  } else {
    INFO("gin_connect: diag_dev_buf=%p (16MB plugin-owned)",
         cc->diag_dev_buf);
  }
  cc->running.store(true, std::memory_order_release);
  cc->recv_thread = std::thread(recv_thread_fn, cc);

  *collComm = cc;
  INFO("gin_connect: rank=%d/%d cc=%p connects=%d accepts=%d",
       rank, nranks, cc, connects_done, accepts_done);
  return ncclSuccess;
}

struct PluginGinCtx {
  void* collComm;
  int nSignals;
  int nCounters;
};

static ncclResult_t gin_createContext(void* collComm, int nSignals,
                                      int nCounters, void** ginCtx,
                                      ncclNetDeviceHandle_v11_t** devHandle) {
  auto* gc = new PluginGinCtx;
  gc->collComm = collComm;
  gc->nSignals = nSignals;
  gc->nCounters = nCounters;
  if (ginCtx) *ginCtx = gc;
  if (devHandle) {
    auto* dh = (ncclNetDeviceHandle_v11_t*)std::calloc(
        1, sizeof(ncclNetDeviceHandle_v11_t));
    dh->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    dh->netDeviceVersion = 100;
    dh->handle = nullptr;
    dh->size = 0;
    dh->needsProxyProgress = 1;
    *devHandle = dh;
  }
  INFO("gin_createContext: cc=%p sig=%d cnt=%d devHandle=%s",
        collComm, nSignals, nCounters, devHandle ? "set" : "NULL");
  return ncclSuccess;
}

// Lazily open GDRCopy handle on first CUDA MR registration. One handle is
// shared by all MRs in the process.
static gdr_t ensure_gdr() {
  if (!g_plugin) return nullptr;
  std::lock_guard<std::mutex> g(g_plugin->gdr_mu);
  if (!g_plugin->gdr) {
    g_plugin->gdr = gdr_open();
    if (!g_plugin->gdr)
      WARN("gdr_open() returned NULL — GDRCopy unavailable, will fallback to cudaMemcpyAsync");
    else
      INFO("gdr_open() OK gdr=%p", g_plugin->gdr);
  }
  return g_plugin->gdr;
}

static ncclResult_t gin_regMrSym(void* collComm, void* data, size_t size,
                                 int type, uint64_t mrFlags, void** mhandle,
                                 void** ginHandle) {
  auto* cc = reinterpret_cast<CollComm*>(collComm);
  if (!cc) return ncclInvalidArgument;
  if (type != NCCL_PTR_HOST && type != NCCL_PTR_CUDA) {
    WARN("gin_regMrSym: bad type=%d", type);
    return ncclInvalidArgument;
  }
  auto* m = new MrRecord;
  m->addr = data;
  m->size = size;
  m->type = type;
  m->flags = mrFlags;
  {
    std::lock_guard<std::mutex> g(cc->mr_mu);
    m->token = cc->next_token++;
    cc->mrs.push_back(m);
  }
  *mhandle = m;
  *ginHandle = reinterpret_cast<void*>((uintptr_t)m->token);

  // For CUDA MRs, try to pin+map via GDRCopy so apply_recv / do_send can
  // bypass cudaMemcpyAsync (which deadlocks under busy-wait kernels).
  if (type == NCCL_PTR_CUDA) {
    gdr_t g = ensure_gdr();
    if (g) {
      // gdr_pin_buffer requires 64KB-page-aligned base; round down.
      const uint64_t PG = 65536;
      uintptr_t base = ((uintptr_t)data) & ~(PG - 1);
      size_t end = ((uintptr_t)data + size + PG - 1) & ~(PG - 1);
      size_t pin_sz = end - base;
      gdr_mh_t mh{};
      int rc = gdr_pin_buffer(g, (CUdeviceptr)base, pin_sz, 0, 0, &mh);
      if (rc == 0) {
        void* hm = nullptr;
        rc = gdr_map(g, mh, &hm, pin_sz);
        if (rc == 0) {
          gdr_info_t info{};
          gdr_get_info(g, mh, &info);
          m->gdr_pinned = true;
          m->gdr_mh = mh;
          m->gdr_host_map = hm;
          m->gdr_map_offset = ((uintptr_t)data) - info.va;
          m->gdr_mapped_size = pin_sz;
          INFO("gin_regMrSym: GDR pinned token=%lu base=%p pin_sz=%zu "
               "host_map=%p map_off=%zu page=%u",
               (unsigned long)m->token, (void*)base, pin_sz, hm,
               m->gdr_map_offset, info.page_size);
        } else {
          WARN("gin_regMrSym: gdr_map rc=%d for token=%lu — falling back",
               rc, (unsigned long)m->token);
          gdr_unpin_buffer(g, mh);
        }
      } else {
        WARN("gin_regMrSym: gdr_pin_buffer rc=%d for token=%lu — falling back",
             rc, (unsigned long)m->token);
      }
    }
  }

  INFO("gin_regMrSym: addr=%p size=%zu type=%d flags=0x%lx → token=%lu gdr=%d",
        data, size, type, (unsigned long)mrFlags, (unsigned long)m->token,
        (int)m->gdr_pinned);
  return ncclSuccess;
}

static ncclResult_t gin_regMrSymDmaBuf(void* /*collComm*/, void* /*data*/,
                                       size_t /*size*/, int /*type*/,
                                       uint64_t /*offset*/, int /*fd*/,
                                       uint64_t /*mrFlags*/, void** mhandle,
                                       void** ginHandle) {
  if (mhandle) *mhandle = nullptr;
  if (ginHandle) *ginHandle = nullptr;
  return ncclInvalidUsage;
}

static ncclResult_t gin_deregMrSym(void* collComm, void* mhandle) {
  auto* cc = reinterpret_cast<CollComm*>(collComm);
  auto* m = reinterpret_cast<MrRecord*>(mhandle);
  if (!cc || !m) return ncclInvalidArgument;
  std::lock_guard<std::mutex> g(cc->mr_mu);
  // Keep slot to preserve token indexing.
  if (m->token > 0 && m->token <= cc->mrs.size()) {
    cc->mrs[m->token - 1] = nullptr;
  }
  if (m->gdr_pinned && g_plugin && g_plugin->gdr) {
    gdr_unmap(g_plugin->gdr, m->gdr_mh, m->gdr_host_map, m->gdr_mapped_size);
    gdr_unpin_buffer(g_plugin->gdr, m->gdr_mh);
  }
  delete m;
  return ncclSuccess;
}

static ncclResult_t gin_destroyContext(void* ginCtx) {
  delete reinterpret_cast<PluginGinCtx*>(ginCtx);
  return ncclSuccess;
}

static ncclResult_t gin_closeColl(void* collComm) {
  auto* cc = reinterpret_cast<CollComm*>(collComm);
  if (!cc) return ncclSuccess;
  cc->running.store(false, std::memory_order_release);
  if (cc->wakeup_pipe[1] >= 0) {
    char b = 1;
    (void)::write(cc->wakeup_pipe[1], &b, 1);
  }
  if (cc->recv_thread.joinable()) cc->recv_thread.join();
  for (auto& pl : cc->peers) {
    if (pl && pl->fd >= 0) ::close(pl->fd);
  }
  if (cc->wakeup_pipe[0] >= 0) ::close(cc->wakeup_pipe[0]);
  if (cc->wakeup_pipe[1] >= 0) ::close(cc->wakeup_pipe[1]);
  if (cc->recv_stage) cudaFreeHost(cc->recv_stage);
  if (cc->send_stage) cudaFreeHost(cc->send_stage);
  if (cc->signal_scratch) cudaFreeHost(cc->signal_scratch);
  if (cc->signal_stream) cudaStreamDestroy(cc->signal_stream);
  if (cc->diag_dev_buf) cudaFree(cc->diag_dev_buf);
  for (auto* m : cc->mrs) if (m) delete m;
  INFO("gin_closeColl: %p (mrs=%zu)", cc, cc->mrs.size());
  delete cc;
  return ncclSuccess;
}

static ncclResult_t gin_closeListen(void* listenComm) {
  auto* lc = reinterpret_cast<ListenComm*>(listenComm);
  if (!lc) return ncclSuccess;
  if (lc->fd >= 0) ::close(lc->fd);
  delete lc;
  return ncclSuccess;
}

// Forward decl: apply a "received" wire message locally (used by self-send).
static int apply_recv(CollComm* cc, const WireMsgHdr& h, const void* payload,
                      int src_rank);

// Reverse-lookup MrRecord whose [addr, addr+size) contains src. Used in
// do_send to find GDR mapping for a CUDA src pointer. Linear scan is fine
// since #MRs is small (~20) and called only on size>0 paths.
static MrRecord* find_mr_for_range(CollComm* cc, const void* p, size_t len) {
  if (!p) return nullptr;
  std::lock_guard<std::mutex> g(cc->mr_mu);
  for (auto* m : cc->mrs) {
    if (!m || !m->gdr_pinned) continue;
    const uint8_t* base = static_cast<const uint8_t*>(m->addr);
    const uint8_t* end = base + m->size;
    const uint8_t* q = static_cast<const uint8_t*>(p);
    if (q >= base && (q + len) <= end) return m;
  }
  return nullptr;
}

// Sender common path. Caller holds peer's send_mu indirectly via NCCL's
// single-threaded proxy progress loop, but we still take per-peer mutex for
// safety in case future versions parallelize.
static ncclResult_t do_send(CollComm* cc, int dst_rank, const WireMsgHdr& h,
                            const void* src_addr, int src_type) {
  static std::atomic<uint64_t> g_do_send_calls{0};
  uint64_t dsn = ++g_do_send_calls;
  INFO("PATHB do_send #%lu cc=%p myRank=%d dst=%d op=%u size=%lu sigOp=%u sigOff=%lu branch=%s",
       dsn, cc, cc->rank, dst_rank, (unsigned)h.op, (unsigned long)h.size,
       (unsigned)h.signal_op, (unsigned long)h.signal_off,
       (dst_rank == cc->rank ? "SELF" : "CROSS"));
  if (dst_rank < 0 || dst_rank >= cc->nranks) {
    WARN("do_send: invalid dst_rank=%d", dst_rank);
    return ncclInvalidArgument;
  }
  // Self-send: skip the wire, apply directly in-process. Barrier kernels do
  // `for (i in 0..nranks) signal(team, i, ...)` which always includes self.
  if (dst_rank == cc->rank) {
    // Stage CUDA payload into host buffer if needed (apply_recv expects host).
    const void* payload = src_addr;
    if (h.size > 0 && src_addr != nullptr && src_type == NCCL_PTR_CUDA) {
      std::lock_guard<std::mutex> sg(cc->send_stage_mu);
      MrRecord* gm = find_mr_for_range(cc, src_addr, h.size);
      if (gm) {
        void* hm_off = static_cast<uint8_t*>(gm->gdr_host_map)
                       + gm->gdr_map_offset
                       + (static_cast<const uint8_t*>(src_addr)
                          - static_cast<uint8_t*>(gm->addr));
        int rc = gdr_copy_from_mapping(gm->gdr_mh, cc->send_stage, hm_off, h.size);
        if (rc != 0) {
          WARN("do_send[self]: gdr_copy_from_mapping rc=%d", rc);
          return ncclSystemError;
        }
      } else {
        cudaError_t ce = cudaMemcpyAsync(cc->send_stage, src_addr, h.size,
                                         cudaMemcpyDeviceToHost,
                                         cc->signal_stream);
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(cc->signal_stream);
        if (ce != cudaSuccess) {
          WARN("do_send[self]: cudaMemcpyAsync D→H failed: %s",
               cudaGetErrorString(ce));
          return ncclSystemError;
        }
      }
      payload = cc->send_stage;
      if (apply_recv(cc, h, payload, cc->rank) != 0) return ncclInternalError;
    } else {
      if (apply_recv(cc, h, payload, cc->rank) != 0) return ncclInternalError;
    }
    return ncclSuccess;
  }
  if (h.size > cc->stage_bytes) {
    WARN("do_send: oversize msg %lu > cap %zu (M2 limit)", h.size,
         cc->stage_bytes);
    return ncclInvalidUsage;
  }
  PeerLink& pl = *cc->peers[dst_rank];
  if (pl.fd < 0) {
    WARN("do_send: no link to rank %d", dst_rank);
    return ncclInternalError;
  }
  std::lock_guard<std::mutex> g(pl.send_mu);
  std::lock_guard<std::mutex> sg(cc->send_stage_mu);

  // Stage payload (CUDA → host) if needed. size=0 is a valid signal-only op.
  const void* payload = src_addr;
  if (h.size > 0 && src_addr != nullptr && src_type == NCCL_PTR_CUDA) {
    MrRecord* gm = find_mr_for_range(cc, src_addr, h.size);
    if (gm) {
      void* hm_off = static_cast<uint8_t*>(gm->gdr_host_map)
                     + gm->gdr_map_offset
                     + (static_cast<const uint8_t*>(src_addr)
                        - static_cast<uint8_t*>(gm->addr));
      int rc = gdr_copy_from_mapping(gm->gdr_mh, cc->send_stage, hm_off, h.size);
      if (rc != 0) {
        WARN("do_send[cross]: gdr_copy_from_mapping rc=%d", rc);
        return ncclSystemError;
      }
      payload = cc->send_stage;
    } else {
    cudaError_t ce = cudaMemcpyAsync(cc->send_stage, src_addr, h.size,
                                     cudaMemcpyDeviceToHost,
                                     cc->signal_stream);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(cc->signal_stream);
    if (ce != cudaSuccess) {
      WARN("do_send: cudaMemcpyAsync D→H failed: %s", cudaGetErrorString(ce));
      return ncclSystemError;
    }
    payload = cc->send_stage;
    }  // close else (non-GDR path)
  }

  if (write_all(pl.fd, &h, sizeof(h)) != 0) {
    WARN("do_send: write hdr failed to rank %d: %s", dst_rank,
         std::strerror(errno));
    return ncclSystemError;
  }
  // OP_GET is hdr-only (request to be fulfilled remotely with a PUT reply).
  if (h.size > 0 && h.op != OP_GET) {
    if (write_all(pl.fd, payload, h.size) != 0) {
      WARN("do_send: write payload failed to rank %d: %s", dst_rank,
           std::strerror(errno));
      return ncclSystemError;
    }
  }
  INFO("PATHB do_send[cross]: TCP wrote hdr+%lu payload bytes to rank=%d ok",
       (unsigned long)h.size, dst_rank);
  return ncclSuccess;
}

// In PROXY mode, NCCL passes the ginHandle (= token uintptr) back to us as
// srcMhandle/dstMhandle/signalMhandle in iput/iputSignal — NOT the mhandle
// pointer we returned from regMrSym. Look up MR by token (=mrs[token-1]).
static MrRecord* mr_from_handle(CollComm* cc, void* h) {
  if (!h) return nullptr;
  uintptr_t tok = reinterpret_cast<uintptr_t>(h);
  std::lock_guard<std::mutex> g(cc->mr_mu);
  if (tok == 0 || tok > cc->mrs.size()) return nullptr;
  return cc->mrs[tok - 1];
}

static ncclResult_t gin_iput(void* collComm, uint64_t srcOff, void* srcMhandle,
                             size_t size, uint64_t dstOff, void* dstMhandle,
                             uint32_t rank, void** request) {
  uint64_t n = ++g_iput_calls;
  if (n <= 4 || (n & (n - 1)) == 0)
    INFO("PATHB iput #%lu rank=%u size=%zu", n, rank, size);
  auto* cc = reinterpret_cast<CollComm*>(collComm);
  if (!cc) return ncclInvalidArgument;
  MrRecord* srcMr = mr_from_handle(cc, srcMhandle);
  MrRecord* dstMr = mr_from_handle(cc, dstMhandle);
  if (size > 0 && (!srcMr || !dstMr)) {
    WARN("gin_iput: bad MR handles for size=%zu (srcTok=%lu dstTok=%lu)",
         size, (uintptr_t)srcMhandle, (uintptr_t)dstMhandle);
    return ncclInvalidArgument;
  }

  WireMsgHdr h{};
  h.magic = WIRE_MAGIC;
  h.op = OP_PUT;
  h.signal_op = SIGOP_NONE;
  h.dst_token = (uintptr_t)dstMhandle;
  h.dst_off = dstOff;
  h.size = size;
  void* src_addr = (size > 0 && srcMr) ?
      (static_cast<uint8_t*>(srcMr->addr) + srcOff) : nullptr;
  int src_type = srcMr ? srcMr->type : NCCL_PTR_HOST;
  ncclResult_t rc = do_send(cc, (int)rank, h, src_addr, src_type);
  if (rc != ncclSuccess) return rc;

  auto* r = new Request;
  r->done.store(1, std::memory_order_release);
  *request = r;
  TRACE("gin_iput: rank=%u size=%zu dstTok=%lu dstOff=%lu", rank, size,
        dstMr->token, dstOff);
  return ncclSuccess;
}

static ncclResult_t gin_iputSignal(void* collComm, uint64_t srcOff,
                                   void* srcMhandle, size_t size,
                                   uint64_t dstOff, void* dstMhandle,
                                   uint32_t rank, uint64_t signalOff,
                                   void* signalMhandle, uint64_t signalValue,
                                   uint32_t signalOp, void** request) {
  uint64_t n = ++g_iputsignal_calls;
  // log every iputSignal so we can see EXACTLY what NCCL dispatches
  INFO("PATHB iputSignal #%lu rank=%u size=%zu dstTok=%lu sigTok=%lu sigOff=%lu sigOp=%u sigVal=%lu cc=%p req_out=%p",
       n, rank, size, (uintptr_t)dstMhandle, (uintptr_t)signalMhandle,
       signalOff, signalOp, signalValue, collComm, (void*)request);
  auto* cc = reinterpret_cast<CollComm*>(collComm);
  if (!cc) return ncclInvalidArgument;
  MrRecord* srcMr = mr_from_handle(cc, srcMhandle);
  MrRecord* dstMr = mr_from_handle(cc, dstMhandle);
  MrRecord* sigMr = mr_from_handle(cc, signalMhandle);
  if (!sigMr) {
    WARN("gin_iputSignal: bad signal handle tok=%lu",
         (uintptr_t)signalMhandle);
    return ncclInvalidArgument;
  }
  if (size > 0 && (!srcMr || !dstMr)) {
    WARN("gin_iputSignal: bad MR handles for size=%zu (srcTok=%lu dstTok=%lu)",
         size, (uintptr_t)srcMhandle, (uintptr_t)dstMhandle);
    return ncclInvalidArgument;
  }

  uint8_t sop = SIGOP_NONE;
  if (signalOp == NCCL_NET_SIGNAL_OP_INC) sop = SIGOP_INC;
  else if (signalOp == NCCL_NET_SIGNAL_OP_ADD) sop = SIGOP_ADD;
  else {
    WARN("gin_iputSignal: unsupported signalOp=%u", signalOp);
    return ncclInvalidUsage;
  }

  WireMsgHdr h{};
  h.magic = WIRE_MAGIC;
  h.op = OP_PUT_SIGNAL;
  h.signal_op = sop;
  h.dst_token = (uintptr_t)dstMhandle;
  h.dst_off = dstOff;
  h.size = size;
  h.signal_token = (uintptr_t)signalMhandle;
  h.signal_off = signalOff;
  h.signal_val = signalValue;
  void* src_addr = (size > 0 && srcMr) ?
      (static_cast<uint8_t*>(srcMr->addr) + srcOff) : nullptr;
  int src_type = srcMr ? srcMr->type : NCCL_PTR_HOST;
  ncclResult_t rc = do_send(cc, (int)rank, h, src_addr, src_type);
  if (rc != ncclSuccess) return rc;

  auto* r = new Request;
  r->done.store(1, std::memory_order_release);
  *request = r;
  TRACE("gin_iputSignal: rank=%u size=%zu sigTok=%lu sigOff=%lu val=%lu op=%u",
        rank, size, sigMr->token, signalOff, signalValue, signalOp);
  return ncclSuccess;
}

static ncclResult_t gin_test(void* collComm, void* request, int* done) {
  uint64_t n = ++g_test_calls;
  INFO("PATHB test #%lu cc=%p req=%p", n, collComm, request);
  auto* r = reinterpret_cast<Request*>(request);
  if (!r) {
    *done = 1;
    return ncclSuccess;
  }
  *done = r->done.load(std::memory_order_acquire);
  if (*done) delete r;
  return ncclSuccess;
}

static ncclResult_t gin_ginProgress(void* /*collComm*/) {
  uint64_t n = ++g_ginprogress_calls;
  if (n <= 4 || (n & (n - 1)) == 0)
    INFO("PATHB ginProgress #%lu (counters iput=%lu iputSignal=%lu test=%lu)",
         n, g_iput_calls.load(), g_iputsignal_calls.load(),
         g_test_calls.load());
  return ncclSuccess;
}

static ncclResult_t gin_queryLastError(void* /*ginCtx*/, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

static ncclResult_t gin_finalize(void* ctx) {
  auto* p = reinterpret_cast<PluginCtx*>(ctx);
  if (p) {
    INFO("gin_finalize: ctx=%p", p);
    // We don't delete g_plugin since it's process-global and may be re-init'd.
  }
  return ncclSuccess;
}

// ===========================================================================
// Net plugin stub (ndev=0)
// ===========================================================================

static ncclResult_t net_init(void** ctx, uint64_t /*commId*/,
                             ncclNetCommConfig_v11_t* /*cfg*/,
                             ncclDebugLogger_t logFunction,
                             ncclProfilerCallback_t /*profFunction*/) {
  if (logFunction && !g_log) g_log = logFunction;
  *ctx = (void*)0x1;
  INFO("net_init: stub (ndev=0)");
  return ncclSuccess;
}

static ncclResult_t net_devices(int* ndev) {
  *ndev = 0;
  return ncclSuccess;
}

static ncclResult_t net_getProperties(int /*dev*/,
                                      ncclNetProperties_v11_t* /*props*/) {
  return ncclInternalError;
}

static ncclResult_t net_unreachable() { return ncclInternalError; }
#define UNREACH_AS(field)                                                      \
  reinterpret_cast<decltype(((ncclNet_v11_t*)0)->field)>(net_unreachable)

static ncclResult_t net_finalize(void* /*ctx*/) { return ncclSuccess; }

// ===========================================================================
// v13 GIN ABI wrappers — adapt newer (multi-context) signatures to our
// existing v11-shaped internal logic. ginCtx is our PluginGinCtx; it holds
// collComm so we can route iput/iputSignal/iget/iflush.
// ===========================================================================

static ncclResult_t gin_v13_getProperties(int dev, ncclNetProperties_v12_t* p) {
  if (dev != 0) return ncclInvalidArgument;
  std::memset(p, 0, sizeof(*p));
  static char name_buf[] = "tcpxo-gin/v13";
  static char pci_buf[] = "";
  p->name = name_buf;
  p->pciPath = pci_buf;
  p->guid = 0xC0FFEE13ull;
  p->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  p->regIsGlobal = 0;
  p->forceFlush = 0;
  p->speed = 200000;
  p->port = 0;
  p->latency = 1.0f;
  p->maxComms = 1024;
  p->maxRecvs = 8;
  p->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  p->netDeviceVersion = 100;
  p->vProps.ndevs = 1;
  p->vProps.devs[0] = 0;
  p->maxP2pBytes = kStageBufBytes;
  p->maxCollBytes = kStageBufBytes;
  p->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t gin_v13_createContext(void* collComm,
                                          ncclGinConfig_v13_t* config,
                                          void** ginCtxOut,
                                          ncclNetDeviceHandle_v11_t** devHandle) {
  INFO("v13 createContext: cc=%p sig=%d cnt=%d nCtx=%d qDepth=%d tc=%d "
       "ginCtxOut=%p devHandleOut=%p",
       collComm, config ? config->nSignals : -1,
       config ? config->nCounters : -1, config ? config->nContexts : -1,
       config ? config->queueDepth : -1, config ? config->trafficClass : -1,
       (void*)ginCtxOut, (void*)devHandle);
  auto* gc = new PluginGinCtx;
  gc->collComm = collComm;
  gc->nSignals = config ? config->nSignals : 0;
  gc->nCounters = config ? config->nCounters : 0;
  if (ginCtxOut) *ginCtxOut = gc;
  if (devHandle) {
    auto* dh = (ncclNetDeviceHandle_v11_t*)std::calloc(
        1, sizeof(ncclNetDeviceHandle_v11_t));
    dh->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    dh->netDeviceVersion = 100;
    dh->handle = nullptr;
    dh->size = 0;
    dh->needsProxyProgress = 1;
    *devHandle = dh;
  }
  return ncclSuccess;
}

static ncclResult_t gin_v13_iput(void* ginCtx, int context, uint64_t srcOff,
                                 void* srcMhandle, size_t size, uint64_t dstOff,
                                 void* dstMhandle, uint32_t rank,
                                 void** request) {
  auto* gc = reinterpret_cast<PluginGinCtx*>(ginCtx);
  if (!gc) return ncclInvalidArgument;
  uint64_t n = ++g_iput_calls;
  INFO("PATHB v13 iput #%lu ctx=%d rank=%u size=%zu dstTok=%lu cc=%p",
       n, context, rank, size, (uintptr_t)dstMhandle, gc->collComm);
  // Reuse v11 path — context value is independent of which collComm we use
  // because each ginCtx is bound to one collComm (1 ctx per ginComm in our setup).
  return gin_iput(gc->collComm, srcOff, srcMhandle, size, dstOff, dstMhandle,
                  rank, request);
}

static ncclResult_t gin_v13_iputSignal(void* ginCtx, int context,
                                       uint64_t srcOff, void* srcMhandle,
                                       size_t size, uint64_t dstOff,
                                       void* dstMhandle, uint32_t rank,
                                       uint64_t signalOff, void* signalMhandle,
                                       uint64_t signalValue, uint32_t signalOp,
                                       void** request) {
  auto* gc = reinterpret_cast<PluginGinCtx*>(ginCtx);
  if (!gc) return ncclInvalidArgument;
  uint64_t n = ++g_iputsignal_calls;
  INFO("PATHB v13 iputSignal #%lu ctx=%d rank=%u size=%zu sigTok=%lu sigOff=%lu "
       "sigOp=%u sigVal=%lu cc=%p req_out=%p",
       n, context, rank, size, (uintptr_t)signalMhandle, signalOff, signalOp,
       signalValue, gc->collComm, (void*)request);
  return gin_iputSignal(gc->collComm, srcOff, srcMhandle, size, dstOff,
                        dstMhandle, rank, signalOff, signalMhandle,
                        signalValue, signalOp, request);
}

static ncclResult_t gin_v13_iget(void* ginCtx, int context,
                                 uint64_t remoteOff, void* remoteMhandle,
                                 size_t size, uint64_t localOff,
                                 void* localMhandle, uint32_t rank,
                                 void** request) {
  auto* gc = reinterpret_cast<PluginGinCtx*>(ginCtx);
  if (!gc) return ncclInvalidArgument;
  auto* cc = reinterpret_cast<CollComm*>(gc->collComm);
  if (!cc || !request) return ncclInvalidArgument;
  MrRecord* dstMr = mr_from_handle(cc, localMhandle);   // local dst
  MrRecord* srcMr = mr_from_handle(cc, remoteMhandle);  // remote handle as token
  if (!dstMr) {
    WARN("gin_v13_iget: bad localMhandle tok=%lu", (uintptr_t)localMhandle);
    return ncclInvalidArgument;
  }
  auto* r = new Request;
  *request = r;

  // Self-iget: GDR-copy locally, no wire.
  if ((int)rank == cc->rank) {
    if (!srcMr) {
      WARN("gin_v13_iget[self]: bad remoteMhandle tok=%lu", (uintptr_t)remoteMhandle);
      r->done.store(1, std::memory_order_release);
      return ncclInternalError;
    }
    void* src_addr = static_cast<uint8_t*>(srcMr->addr) + remoteOff;
    void* dst_addr = static_cast<uint8_t*>(dstMr->addr) + localOff;
    if (dstMr->gdr_pinned && srcMr->gdr_pinned && g_plugin && g_plugin->gdr) {
      // Stage src GPU mem → host via gdr_copy_from_mapping, then host → dst GPU.
      std::lock_guard<std::mutex> sg(cc->send_stage_mu);
      void* hm_src = static_cast<uint8_t*>(srcMr->gdr_host_map)
                     + srcMr->gdr_map_offset + remoteOff;
      void* hm_dst = static_cast<uint8_t*>(dstMr->gdr_host_map)
                     + dstMr->gdr_map_offset + localOff;
      int rc = gdr_copy_from_mapping(srcMr->gdr_mh, cc->send_stage, hm_src, size);
      if (rc == 0) rc = gdr_copy_to_mapping(dstMr->gdr_mh, hm_dst, cc->send_stage, size);
      if (rc != 0) { WARN("gin_v13_iget[self]: gdr copy rc=%d", rc); r->done.store(1); return ncclInternalError; }
    } else {
      WARN("gin_v13_iget[self]: non-GDR fallback not implemented");
      r->done.store(1, std::memory_order_release);
      return ncclInternalError;
    }
    r->done.store(1, std::memory_order_release);
    return ncclSuccess;
  }

  // Cross-iget: build OP_GET hdr, register pending, send to peer who owns src.
  uint64_t rid = cc->next_req_id.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> g(cc->pending_gets_mu);
    cc->pending_gets[rid] = r;
  }
  WireMsgHdr h{};
  h.magic = WIRE_MAGIC;
  h.op = OP_GET;
  h.signal_op = SIGOP_NONE;
  h.req_id = rid;
  h.dst_token = (uintptr_t)localMhandle;   // requester's local dst
  h.dst_off = localOff;
  h.size = size;
  h.signal_token = (uintptr_t)remoteMhandle;  // remote src token to read
  h.signal_off = remoteOff;
  INFO("PATHB gin_v13_iget: ctx=%d peer=%u size=%lu local(tok=%lu off=%lu) remote(tok=%lu off=%lu) req_id=%lu",
       context, rank, (unsigned long)size, (uintptr_t)localMhandle,
       (unsigned long)localOff, (uintptr_t)remoteMhandle,
       (unsigned long)remoteOff, (unsigned long)rid);
  ncclResult_t rc = do_send(cc, (int)rank, h, nullptr, NCCL_PTR_HOST);
  if (rc != ncclSuccess) {
    std::lock_guard<std::mutex> g(cc->pending_gets_mu);
    cc->pending_gets.erase(rid);
    r->done.store(1, std::memory_order_release);
    return rc;
  }
  return ncclSuccess;
}

static ncclResult_t gin_v13_iflush(void* /*ginCtx*/, int /*context*/,
                                   void* /*mhandle*/, uint32_t /*rank*/,
                                   void** request) {
  // No-op: data path is TCP synchronous, no fence needed.
  if (request) *request = nullptr;
  return ncclSuccess;
}

static ncclResult_t gin_v13_ginProgress(void* ginCtx) {
  auto* gc = reinterpret_cast<PluginGinCtx*>(ginCtx);
  return gin_ginProgress(gc ? gc->collComm : nullptr);
}

static ncclResult_t gin_v13_queryLastError(void* /*ginCtx*/, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

} // namespace

// ===========================================================================
// Exports
// ===========================================================================

extern "C" {

ncclNet_v11_t ncclNetPlugin_v11 = {
    .name = "tcpxo-gin-stub-net",
    .init = net_init,
    .devices = net_devices,
    .getProperties = net_getProperties,
    .listen = UNREACH_AS(listen),
    .connect = UNREACH_AS(connect),
    .accept = UNREACH_AS(accept),
    .regMr = UNREACH_AS(regMr),
    .regMrDmaBuf = UNREACH_AS(regMrDmaBuf),
    .deregMr = UNREACH_AS(deregMr),
    .isend = UNREACH_AS(isend),
    .irecv = UNREACH_AS(irecv),
    .iflush = UNREACH_AS(iflush),
    .test = UNREACH_AS(test),
    .closeSend = UNREACH_AS(closeSend),
    .closeRecv = UNREACH_AS(closeRecv),
    .closeListen = UNREACH_AS(closeListen),
    .getDeviceMr = UNREACH_AS(getDeviceMr),
    .irecvConsumed = UNREACH_AS(irecvConsumed),
    .makeVDevice = UNREACH_AS(makeVDevice),
    .finalize = net_finalize,
    .setNetAttr = UNREACH_AS(setNetAttr),
};

ncclGin_v11_t ncclGinPlugin_v11 = {
    .name = "tcpxo-gin/m2",
    .init = gin_init,
    .devices = gin_devices,
    .getProperties = gin_getProperties,
    .listen = gin_listen,
    .connect = gin_connect,
    .createContext = gin_createContext,
    .regMrSym = gin_regMrSym,
    .regMrSymDmaBuf = gin_regMrSymDmaBuf,
    .deregMrSym = gin_deregMrSym,
    .destroyContext = gin_destroyContext,
    .closeColl = gin_closeColl,
    .closeListen = gin_closeListen,
    .iput = gin_iput,
    .iputSignal = gin_iputSignal,
    .test = gin_test,
    .ginProgress = gin_ginProgress,
    .queryLastError = gin_queryLastError,
    .finalize = gin_finalize,
};

// v13 plugin export — NCCL 2.30.4 dlsym order tries this first (newest ABI).
// Multi-context aware: iput/iputSignal/iget/iflush receive (ginCtx, int context, ...).
ncclGin_v13_t ncclGinPlugin_v13 = {
    .name = "tcpxo-gin/v13",
    .init = gin_init,
    .devices = gin_devices,
    .getProperties = gin_v13_getProperties,
    .listen = gin_listen,
    .connect = gin_connect,
    .createContext = gin_v13_createContext,
    .regMrSym = gin_regMrSym,
    .regMrSymDmaBuf = gin_regMrSymDmaBuf,
    .deregMrSym = gin_deregMrSym,
    .destroyContext = gin_destroyContext,
    .closeColl = gin_closeColl,
    .closeListen = gin_closeListen,
    .iput = gin_v13_iput,
    .iputSignal = gin_v13_iputSignal,
    .iget = gin_v13_iget,
    .iflush = gin_v13_iflush,
    .test = gin_test,
    .ginProgress = gin_v13_ginProgress,
    .queryLastError = gin_v13_queryLastError,
    .finalize = gin_finalize,
};

} // extern "C"
