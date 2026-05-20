// M1 smoke test: dlopen the plugin, dlsym both required symbols, call gin_devices.
// Build:  gcc -ldl tests/dlopen_smoke.c -o build_deploy/dlopen_smoke
// Run:    LD_LIBRARY_PATH=build_deploy ./build_deploy/dlopen_smoke

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*fn_t)();

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "build_deploy/libnccl-net-tcpxo-gin.so";
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror());
    return 1;
  }
  void* gin = dlsym(h, "ncclGinPlugin_v11");
  void* net = dlsym(h, "ncclNetPlugin_v11");
  printf("ncclGinPlugin_v11 = %p\n", gin);
  printf("ncclNetPlugin_v11 = %p\n", net);
  if (!gin || !net) {
    fprintf(stderr, "missing symbol\n");
    return 2;
  }
  // Read name (first field). It's a pointer to const char.
  const char* gin_name = *(const char**)gin;
  const char* net_name = *(const char**)net;
  printf("gin->name = %s\n", gin_name);
  printf("net->name = %s\n", net_name);
  // Try gin->devices (4th function pointer: skip name + init = 8+8 bytes)
  // Struct layout: char* name, init, devices, ...
  fn_t devices = *(fn_t*)((char*)gin + 8 /*name*/ + 8 /*init*/);
  int ndev = -1;
  int rc = ((int (*)(int*))devices)(&ndev);
  printf("gin->devices(&ndev) rc=%d ndev=%d\n", rc, ndev);
  dlclose(h);
  return (rc == 0 && ndev == 1) ? 0 : 3;
}
