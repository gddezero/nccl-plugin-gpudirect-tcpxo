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
