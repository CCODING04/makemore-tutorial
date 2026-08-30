/*
 * Part 9 - 脚本 01: 第一个 CUDA 内核 — vector add（CPU vs GPU）
 * 目标：跑通"CUDA 程序五步曲"（分配显存 → 拷贝 → 启动内核 → 拷回 → 验证），
 *       并对比 CPU 与 GPU 在 100 万个元素加法上的耗时。
 *
 * 对应 cuda-course: 05_Writing_your_First_Kernels/02 Kernels/00_vector_add_v1.cu
 * 编译运行（也可直接 make）:
 *   nvcc -O2 -arch=native 01_vector_add.cu -o bin/01_vector_add
 *   ./bin/01_vector_add
 *
 * 核心公式：全局线程索引  i = blockIdx.x * blockDim.x + threadIdx.x
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000000          /* 向量长度：100 万（原仓库是 1000 万，这里缩小保证 <30s） */
#define BLOCK_SIZE 256     /* 每个 block 里的线程数（必须是 32 的倍数 = warp 的整数倍） */

/* ---------- CPU 参照实现：永远是"标准答案" ---------- */
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/* ---------- GPU 内核：每个线程只算一个 c[i] ----------
 * __global__ = 这个函数从 CPU 启动、在 GPU 上执行
 * 一个 thread 的工作：
 *   1) 用 blockIdx/blockDim/threadIdx 算出"我是谁"（全局下标 i）
 *   2) 边界守卫：if (i < n) —— 多余的线程直接返回
 *   3) 读 a[i]、b[i]，写 c[i]
 */
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;   /* h_* = host（内存） */
    float *d_a, *d_b, *d_c;                 /* d_* = device（显存） */
    size_t size = N * sizeof(float);

    /* 1. 分配 host 内存并初始化 */
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);
    srand(42);                 /* 固定种子，方便复现 */
    init_vector(h_a, N);
    init_vector(h_b, N);

    /* 2. 分配 device 显存 */
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    /* 3. 把输入拷到显存（CPU 内存和显存是两个世界） */
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /* 4. 启动配置：多少个 block × 每个 block 多少线程 */
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    /* 例：N=1000000, BLOCK_SIZE=256 → 3907 个 block（向上取整，所以需要边界守卫） */

    printf("N = %d, grid = %d blocks x %d threads = %d threads\n",
           N, num_blocks, BLOCK_SIZE, num_blocks * BLOCK_SIZE);

    /* warm-up：第一次启动内核有初始化开销，不计入基准 */
    printf("Warm-up...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();   /* 等 GPU 真正做完（内核启动是异步的！） */
    }

    /* 5. 基准：CPU 10 次 vs GPU 10 次 */
    printf("Benchmarking CPU...\n");
    double cpu_total = 0.0;
    for (int i = 0; i < 10; i++) {
        double t0 = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        cpu_total += get_time() - t0;
    }

    printf("Benchmarking GPU...\n");
    double gpu_total = 0.0;
    for (int i = 0; i < 10; i++) {
        double t0 = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        gpu_total += get_time() - t0;
    }

    printf("CPU avg: %.3f ms\n", cpu_total / 10 * 1000);
    printf("GPU avg: %.3f ms (kernel only, no memcpy)\n", gpu_total / 10 * 1000);
    printf("Speedup: %.1fx\n", (cpu_total / 10) / (gpu_total / 10));

    /* 6. 验证：把 GPU 结果拷回来，逐元素对比 CPU 答案
     * —— cuda-course 的方法论：先写 CPU 版，GPU 版永远和它对照 */
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) errors++;
    }
    printf("Verification: %s (%d errors)\n", errors == 0 ? "CORRECT" : "WRONG", errors);

    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return errors == 0 ? 0 : 1;
}
