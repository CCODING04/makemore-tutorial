/*
 * Part 9 - 脚本 05: atomics（原子加法归约）与 streams（流重叠）
 * 目标：① 用 atomicAdd 做向量求和，理解"原子操作为什么对、为什么慢"；
 *       ② 用两个 stream 让"数据拷贝"和"内核计算"重叠，实测加速。
 *
 * 对应 cuda-course: 05_Writing_your_First_Kernels/04 Atomics/00_atomicAdd.cu
 *                   + 05 Streams/01_stream_basics.cu
 * 编译运行:
 *   nvcc -O2 -arch=native 05_atomics_streams.cu -o bin/05_atomics_streams
 *   ./bin/05_atomics_streams
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000000

/* ---------- Part A: 归约（求和） ----------
 * 想让 100 万个线程把结果加到一个数上：
 *   result += x[i]   ← 100 万个线程同时读改写同一个地址 → 竞争，结果错乱
 * atomicAdd 保证"读-改-写"三步不可被打断，但代价是冲突的线程被硬件串行化。
 * （真正快的归约要用树形 reduce：每层折半，log2(N) 步 —— 见教程 03 章） */
__global__ void sum_naive_wrong(float *x, float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[0] += x[i];              /* ❌ 竞争条件：结果每次都不一样 */
}

__global__ void sum_atomic(float *x, float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(result, x[i]);        /* ✅ 正确，但 100 万次原子写互相排队 */
}

/* 先在每个 block 内用 shared memory 做树形归约，每个 block 只对全局 result 做
 * 1 次 atomicAdd —— 原子操作次数从 100 万降到 3907（block 数）。
 * 这就是"分层归约"：tree reduce (block 内) + atomic (block 间)。 */
#define BLOCK 256
__global__ void sum_tree_atomic(float *x, float *result, int n) {
    __shared__ float sdata[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    /* 树形归约：每轮活跃线程数减半，共 log2(256)=8 轮 */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, sdata[0]);   /* 每 block 一次 */
}

/* ---------- Part B: streams ----------
 * 默认所有操作排在同一条"默认流"里串行执行。
 * 多条流 = 多条并行队列：stream1 拷贝第 2 块数据时，stream0 可以算第 1 块。
 * 下面把 4 块数据分给 2 条流，对比串行 vs 重叠的总耗时。 */

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) vec[i] = (float)rand() / RAND_MAX;
}
__global__ void scale_kernel(float *x, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    /* ======== Part A: 三种求和 ======== */
    size_t size = N * sizeof(float);
    float *h_x = (float*)malloc(size);
    init_vector(h_x, N);

    float *d_x, *d_result;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK - 1) / BLOCK;

    /* CPU 答案 */
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) cpu_sum += h_x[i];
    printf("CPU reference sum: %.2f\n", cpu_sum);

    /* 1) naive：错误示范（只跑一次看它错得多离谱） */
    float wrong = 0.0f;
    cudaMemcpy(d_result, &wrong, sizeof(float), cudaMemcpyHostToDevice);
    sum_naive_wrong<<<blocks, BLOCK>>>(d_x, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&wrong, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("naive  (racy)     : %.2f   <- WRONG, nondeterministic\n", wrong);

    /* 2) atomic：正确但慢 */
    cudaMemset(d_result, 0, sizeof(float));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    sum_atomic<<<blocks, BLOCK>>>(d_x, d_result);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_atomic;
    cudaEventElapsedTime(&ms_atomic, t0, t1);
    float sum_atomic_res;
    cudaMemcpy(&sum_atomic_res, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("atomic            : %.2f   (%.3f ms)\n", sum_atomic_res, ms_atomic);

    /* 3) tree + atomic：正确且快得多 */
    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(t0);
    sum_tree_atomic<<<blocks, BLOCK>>>(d_x, d_result, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_tree;
    cudaEventElapsedTime(&ms_tree, t0, t1);
    float sum_tree_res;
    cudaMemcpy(&sum_tree_res, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("tree + atomic     : %.2f   (%.3f ms)  <- correct AND fast\n",
           sum_tree_res, ms_tree);
    printf("  (reference sum in float32: %.2f, tiny diffs are rounding)\n\n", cpu_sum);

    /* ======== Part B: streams ======== */
    const int CHUNKS = 4;
    size_t chunk_size = size / CHUNKS;
    int chunk_n = N / CHUNKS;
    int chunk_blocks = (chunk_n + BLOCK - 1) / BLOCK;

    /* 串行：默认流里 4 轮"拷贝→计算→拷回" */
    double t_start = get_time();
    for (int c = 0; c < CHUNKS; c++) {
        float *h_chunk = h_x + c * chunk_n;
        cudaMemcpy(d_x, h_chunk, chunk_size, cudaMemcpyHostToDevice);
        scale_kernel<<<chunk_blocks, BLOCK>>>(d_x, 2.0f, chunk_n);
        cudaMemcpy(h_chunk, d_x, chunk_size, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    double serial_ms = (get_time() - t_start) * 1000;

    /* 重叠：两条流轮流干活（1、3 块在 stream[1]，2、4 块在 stream[0]） */
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    t_start = get_time();
    for (int c = 0; c < CHUNKS; c++) {
        cudaStream_t s = streams[c % 2];
        float *h_chunk = h_x + c * chunk_n;
        cudaMemcpyAsync(d_x, h_chunk, chunk_size, cudaMemcpyHostToDevice, s);
        scale_kernel<<<chunk_blocks, BLOCK, 0, s>>>(d_x, 2.0f, chunk_n);
        cudaMemcpyAsync(h_chunk, d_x, chunk_size, cudaMemcpyDeviceToHost, s);
    }
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    double overlap_ms = (get_time() - t_start) * 1000;

    printf("4 chunks, default stream (serial) : %.3f ms\n", serial_ms);
    printf("4 chunks, 2 streams   (overlapped): %.3f ms  (copy & compute overlap)\n",
           overlap_ms);
    printf("(ratio depends on PCIe vs kernel time; see tutorial 03)\n");

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(h_x);
    cudaFree(d_x); cudaFree(d_result);
    return 0;
}
