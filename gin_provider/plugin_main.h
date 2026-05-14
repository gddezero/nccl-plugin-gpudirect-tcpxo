/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef GIN_PROVIDER_PLUGIN_MAIN_H_
#define GIN_PROVIDER_PLUGIN_MAIN_H_

#include "gin_provider/nccl_gin_v13_abi.h"

extern "C" {
extern ncclGin_v13_t ncclGinPlugin_v13;
}

#endif  // GIN_PROVIDER_PLUGIN_MAIN_H_
