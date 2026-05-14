/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Stand-alone smoke test for libnccl-gin.so:
 *   - dlopen the .so
 *   - dlsym ncclGinPlugin_v13
 *   - call ->init / ->devices / ->getProperties for every device
 *   - call ->finalize
 *
 * The point is to validate the GIN v13 ABI surface and FasTrak NIC discovery
 * (PluginCoreInit -> WaitForRxDM -> initializeNetIfs) without dragging in a
 * full NCCL communicator + bootstrap. Run inside a host context where RxDM
 * is up (e.g. on forrest-h100-01 with the rxdm container running).
 *
 * Usage:
 *   load_plugin_test /path/to/libnccl-gin.so
 */

#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gin_provider/nccl_gin_v13_abi.h"

static void NcclLogger(ncclDebugLogLevel level, unsigned long flags,
                       const char* file, int line, const char* fmt, ...) {
  (void)flags;
  va_list ap;
  fprintf(stderr, "[nccl-log lvl=%d %s:%d] ", level, file, line);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "./libnccl-gin.so";
  fprintf(stderr, "==> dlopen(%s)\n", path);
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (h == NULL) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 2;
  }

  fprintf(stderr, "==> dlsym(ncclGinPlugin_v13)\n");
  ncclGin_v13_t* p = (ncclGin_v13_t*)dlsym(h, "ncclGinPlugin_v13");
  if (p == NULL) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    dlclose(h);
    return 3;
  }
  fprintf(stderr, "    plugin name: %s\n", p->name ? p->name : "<null>");

  void* ctx = NULL;
  fprintf(stderr, "==> plugin->init(commId=0xfeedbeef)\n");
  ncclResult_t r = p->init(&ctx, 0xfeedbeef, NcclLogger);
  if (r != ncclSuccess) {
    fprintf(stderr, "init returned %d\n", (int)r);
    dlclose(h);
    return 4;
  }
  fprintf(stderr, "    init OK, ctx=%p\n", ctx);

  int ndev = -1;
  fprintf(stderr, "==> plugin->devices()\n");
  r = p->devices(&ndev);
  if (r != ncclSuccess) {
    fprintf(stderr, "devices returned %d\n", (int)r);
    if (p->finalize) p->finalize(ctx);
    dlclose(h);
    return 5;
  }
  fprintf(stderr, "    ndev=%d\n", ndev);

  for (int d = 0; d < ndev; ++d) {
    ncclNetProperties_v12_t props;
    memset(&props, 0, sizeof(props));
    r = p->getProperties(d, &props);
    if (r != ncclSuccess) {
      fprintf(stderr, "    getProperties[%d] returned %d\n", d, (int)r);
      continue;
    }
    fprintf(stderr,
            "    dev[%d] name=%-12s pci=%-30s speed=%dMbps "
            "ptrSupport=0x%x netDeviceType=%d railId=%d planeId=%d\n",
            d, props.name ? props.name : "<null>",
            props.pciPath ? props.pciPath : "<null>", props.speed,
            props.ptrSupport, props.netDeviceType, props.railId, props.planeId);
  }

  bool err = false;
  if (p->queryLastError) {
    p->queryLastError(ctx, &err);
    fprintf(stderr, "==> queryLastError: hasError=%d\n", (int)err);
  }

  fprintf(stderr, "==> plugin->finalize()\n");
  if (p->finalize) {
    r = p->finalize(ctx);
    fprintf(stderr, "    finalize returned %d\n", (int)r);
  }

  dlclose(h);
  fprintf(stderr, "==> done\n");
  return 0;
}
