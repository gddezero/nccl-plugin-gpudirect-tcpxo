load("@rules_cc//cc:defs.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_binary(
    name = "load_plugin_test",
    srcs = ["load_plugin_test.cc"],
    linkopts = ["-ldl"],
    deps = [
        "//gin_provider:nccl_gin_v13_abi",
    ],
)

cc_binary(
    name = "two_rank_loopback_test",
    srcs = ["two_rank_loopback_test.cc"],
    linkopts = ["-ldl"],
    deps = [
        "//gin_provider:nccl_gin_v13_abi",
    ],
)

cc_binary(
    name = "two_host_nccl_test",
    srcs = ["two_host_nccl_test.cc"],
    linkopts = [
        "-L/usr/local/nccl/build/lib",
        "-lnccl",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-Wl,-rpath,/usr/local/nccl/build/lib",
        "-Wl,-rpath,/usr/local/cuda/lib64",
    ],
    deps = [
        "@nccl//:plugin_lib",
    ],
    tags = ["manual"],
)

# NCCL >= 2.30.4 integration test. Built on demand via:
#   bazel build --define=enable_nccl_init_test=1 //gin_provider/test:nccl_init_test
# To skip when NCCL libs aren't present on the build machine, gate via tags.
cc_binary(
    name = "nccl_init_test",
    srcs = ["nccl_init_test.cc"],
    linkopts = [
        "-L/usr/local/nccl/build/lib",
        "-lnccl",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-Wl,-rpath,/usr/local/nccl/build/lib",
        "-Wl,-rpath,/usr/local/cuda/lib64",
    ],
    deps = [
        "@nccl//:plugin_lib",
    ],
    tags = ["manual"],
)

