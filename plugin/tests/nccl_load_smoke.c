// Force NCCL to dlopen our plugin and run plugin->init via a minimal flow.
// Build: gcc tests/nccl_load_smoke.c -lnccl -lcudart -o build_deploy/nccl_load_smoke \
//          -I/usr/local/lib/python3.10/dist-packages/nvidia/nccl/include \
//          -L/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib \
//          -Wl,-rpath,/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib
// Run:   NCCL_DEBUG=INFO NCCL_NET_PLUGIN=<path-to-our-so> ./build_deploy/nccl_load_smoke
//
// Success: NCCL log contains "Loaded gin plugin tcpxo-gin/m1 (v11)".
// (We don't try to bring up an actual collective — nranks=1 is enough to force
//  plugin discovery + init paths.)

#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
  // Force NCCL to load plugins by calling ncclGetUniqueId which initializes
  // the runtime including plugin discovery.
  ncclUniqueId id;
  ncclResult_t r = ncclGetUniqueId(&id);
  printf("ncclGetUniqueId rc=%d (%s)\n", (int)r, ncclGetErrorString(r));
  if (r != ncclSuccess) return 1;

  // Now create a 1-rank communicator. This forces the bootstrap → net plugin
  // → ginConnectOnce paths.
  ncclComm_t comm = NULL;
  int rank = 0, nranks = 1;
  r = ncclCommInitRank(&comm, nranks, id, rank);
  printf("ncclCommInitRank rc=%d (%s)\n", (int)r, ncclGetErrorString(r));
  if (r != ncclSuccess) return 2;

  // M1: skip QueryProperties (linker-weak; defer to Python E2E test).
  ncclCommDestroy(comm);
  printf("M1 smoke OK\n");
  return 0;
}
