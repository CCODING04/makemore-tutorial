# Part 9 - 一键编译所有 CUDA 脚本
#
# 用法：
#   make            # 编译全部 → bin/
#   make run        # 编译并依次运行
#   make clean      # 清理
#   make NVCC=/usr/local/cuda-11.8/bin/nvcc   # 强制指定某个 CUDA 版本
#
# 多版本 CUDA 共存（本课的机器就是 11.8 + 12.4 两个版本并存）：
#   - 下面自动选择 /usr/local/cuda-* 里【版本最高】的一个；
#   - PATH 和 /usr/local/cuda 软链接保持指向旧版，不受影响（不会"覆盖"已有 CUDA）；
#   - 想换版本时 make NVCC=<路径> 即可。

# ── 工具链自动选择 ─────────────────────────────────────────────
DETECTED_CUDA_HOME := $(shell ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V | tail -1)
ifeq ($(DETECTED_CUDA_HOME),)
DETECTED_NVCC := nvcc
else
DETECTED_NVCC := $(DETECTED_CUDA_HOME)/bin/nvcc
endif
NVCC ?= $(DETECTED_NVCC)

ARCH = -arch=native

# 坑点：CUDA 对宿主 gcc 有版本上限——CUDA 11.x 最高 gcc-11（12.1+ 原生支持 gcc-12/13）。
# 系统 gcc 太新时会报 "unsupported GNU version"：
#   - 选中 CUDA 11.x → 自动加 -ccbin gcc-11（机器上得装了 gcc-11）
#   - 选中 CUDA 12.x → 不需要任何特殊参数
CUDA_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
ifeq ($(CUDA_MAJOR),11)
CCBIN = $(shell command -v gcc-11 >/dev/null 2>&1 && echo "-ccbin gcc-11")
else
CCBIN =
endif

# 坑点 2（多版本共存特有）：链接 libcublas 等 CUDA 库后，运行时也要能找到对应版本的
# lib64。系统 ldconfig 通常只配了旧版路径 → 这里把所用版本的 lib64 用 rpath 焊进二进制，
# 运行免配 LD_LIBRARY_PATH（用 `ldd bin/06_cublas_sgemm` 可以看到解析到了哪个版本）。
# 注意 $(dir) 对尾部带斜杠的路径不再向上跳（dir(a/b/) = a/b/），所以先去尾斜杠再取上级
CUDA_PARENT = $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(NVCC)))))
ifeq ($(CUDA_PARENT),.)
RPATH =
else
RPATH = -Xlinker -rpath,$(CUDA_PARENT)/lib64
endif

CFLAGS = -O2 $(ARCH) $(CCBIN) $(RPATH)

# 二进制名 = 源文件名（01_vector_add → bin/01_vector_add）
BINS := 01_vector_add 02_thread_hierarchy 03_naive_matmul 04_matmul_tiled \
        05_atomics_streams 06_cublas_sgemm

all: $(BINS:%=bin/%)

bin/%: %.cu
	mkdir -p bin
	$(NVCC) $(CFLAGS) $< -o $@ $(LDLIBS)

bin/06_cublas_sgemm: LDLIBS = -lcublas

run: all
	@echo ">> using NVCC: $$( $(NVCC) --version | tail -2 | head -1 )"
	@for b in $(BINS); do echo "===== running bin/$$b ====="; ./bin/$$b || echo "FAILED: bin/$$b"; done

clean:
	rm -rf bin

.PHONY: all run clean
