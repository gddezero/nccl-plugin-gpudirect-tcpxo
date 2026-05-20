# M1 TCPXO GIN plugin — single-file C++ build.
# Build inside deepep-dev container (has NCCL headers at NCCL_RUNTIME_INC).

NCCL_RUNTIME_INC ?= /usr/local/lib/python3.10/dist-packages/nvidia/nccl/include
CUDA_INC         ?= /usr/local/cuda/include
GDR_INC          ?= /usr/local/gdrcopy/include
GDR_LIB          ?= /usr/local/gdrcopy/lib
ABI_INC          := $(CURDIR)/include

CXX      ?= g++
CXXFLAGS += -std=c++17 -O2 -g -fPIC -fvisibility=default -Wall -Wextra \
            -Wno-unused-parameter -Wno-missing-field-initializers
INC      := -I$(ABI_INC) -I$(ABI_INC)/nccl_plugin_abi -I$(NCCL_RUNTIME_INC) -I$(CUDA_INC) -I$(GDR_INC)
LDFLAGS  += -shared -Wl,-z,defs -L/usr/local/cuda/lib64 -L$(GDR_LIB) -lcudart -lgdrapi -lpthread \
            -Wl,-rpath,/usr/local/cuda/lib64 -Wl,-rpath,$(GDR_LIB)

SRCS := src/tcpxo_gin_plugin.cc
OUT  := build_deploy/libnccl-net-tcpxo-gin.so

all: $(OUT)

$(OUT): $(SRCS) | build_deploy
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $(SRCS) $(LDFLAGS)
	@echo "[ok] built $@"
	@nm -D --defined-only $@ | grep -E 'ncclNetPlugin_v11|ncclGinPlugin_v11' || \
	  (echo "[fail] required symbols missing"; exit 1)

build_deploy:
	@mkdir -p build_deploy

clean:
	rm -rf build_deploy

.PHONY: all clean
