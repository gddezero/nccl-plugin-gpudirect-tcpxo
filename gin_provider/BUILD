load("@rules_cc//cc:defs.bzl", "cc_library")

# BUILD for the GCP TCPXO NCCL GIN provider plugin (PROXY mode).
#
# Implements ncclGinPlugin_v13 (NCCL >= 2.30.4) backed by the existing
# DXS / Falcon transport. The plugin reports NCCL_NET_DEVICE_GIN_PROXY,
# meaning: device-side GFD posting code lives in NCCL itself; this
# plugin only provides host-side functions plus a host progress thread
# that drains GFDs and translates each one into a dxs::Send.
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "gpu_ctx_alloc",
    srcs = ["gpu_ctx_alloc.cc"],
    hdrs = ["gpu_ctx_alloc.h"],
    deps = [
        "//nccl_cuda:cuda_common",
        "//nccl_cuda:cuda_defs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nccl//:plugin_lib",
    ],
)

cc_library(
    name = "gfd_decoder",
    srcs = ["gfd_decoder.cc"],
    hdrs = ["gfd_decoder.h"],
    deps = [
        "@nccl//:plugin_lib",
    ],
)

# Merged: proxy_context + proxy_progress live in one library to avoid a
# circular dep (GinCtx owns a ProxyProgress; ProxyProgress reads GinCtx state).
cc_library(
    name = "proxy_runtime",
    srcs = [
        "proxy_context.cc",
        "proxy_progress.cc",
    ],
    hdrs = [
        "proxy_context.h",
        "proxy_progress.h",
    ],
    deps = [
        ":gfd_decoder",
        ":gpu_ctx_alloc",
        "//buffer_mgmt_daemon/client:buffer_mgr_client",
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//dxs/client:dxs-client",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nccl//:plugin_lib",
    ],
)

cc_library(
    name = "nccl_gin_v13_abi",
    hdrs = ["nccl_gin_v13_abi.h"],
    deps = ["@nccl//:plugin_lib"],
)

cc_library(
    name = "gin_plugin",
    srcs = ["plugin_main.cc"],
    hdrs = ["plugin_main.h", "listen_handle.h"],
    linkopts = ["-Wl,--export-dynamic-symbol=ncclGinPlugin_v13"],
    deps = [
        ":gfd_decoder",
        ":gpu_ctx_alloc",
        ":nccl_gin_v13_abi",
        ":proxy_runtime",
        "//tcpdirect_plugin/fastrak_offload:fastrak_common",
        "//tcpdirect_plugin/fastrak_offload:init",
        "//tcpdirect_plugin/fastrak_offload:nccl_params",
        "//tcpdirect_plugin/fastrak_offload:nic_client_router",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@nccl//:plugin_lib",
    ],
    alwayslink = 1,
)

# Standalone shared library exposing only the GIN provider symbol.
# Load via NCCL_GIN_PLUGIN=/path/to/libnccl-gin.so once we are ready
# to test against NCCL >= 2.30.4.
cc_binary(
    name = "libnccl-gin.so",
    features = ["-enable_relr"],
    linkshared = True,
    linkstatic = True,
    deps = [":gin_plugin"],
)
