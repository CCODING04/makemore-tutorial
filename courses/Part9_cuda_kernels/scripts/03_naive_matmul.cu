/*
 * Part 9 - 脚本 03: naive matmul —— 每个 GPU 线程算输出矩阵的一个元素
 * 目标：写出 (M,K) @ (K,N) 的最朴素 GPU 版本，与 CPU 版对照验证，
 *       并算出第一个 GFLOPS —— 它会慢得离谱，这正是脚本 04 的存在意义。
 *
 * 对应 cuda-course: 05_Writing_your_First_Kernels/02 Kernels/02 matmul.cu
 * 编译运行:
 *   nvcc -O2 -arch=native 03_naive_matmul.cu -o bin/03_naive_matmul
 *   ./bin/03_naive_matmul
 *
 * 关键点：
 *   1) 矩阵按 row-major 存成一维数组：A[i][k] = A[i * K + k]
 *   2) 启动用 2D grid：一个 thread 负责 C 的一个 (row, col)
 *   3) 输出元素个数 M*N 是 2 的倍数关系 → 索引错一位结果就全错，务必用 CPU 版对照
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 512      /* A 的行数 */
#define K 512      /* A 的列数 = B 的行数 */
#define N 512      /* B 的列数 */
#define BLOCK_SIZE 32

/* ---------- CPU 参照实现（三重循环，标准答案） ---------- */
void matmul_cpu(const float *A, const float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ---------- GPU naive 版本 ----------
 * 每个 thread：
 *   1) 用 2D 索引算出自己负责的 (row, col)
 *   2) 沿 K 维做点积：C[row][col] = A[row,:] · B[:,col]
 * 内存访问形态（脚本 04 要优化的就是它）：
 *   一个 warp（32 个相邻 col 的线程）读 B[:, col..col+31] —— 跨行跳跃，不合并（uncoalesced）
 */
__global__ void matmul_gpu_naive(const float *A, const float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    float *h_A = (float*)malloc(a_size);
    float *h_B = (float*)malloc(b_size);
    float *h_C_cpu = (float*)malloc(c_size);
    float *h_C_gpu = (float*)malloc(c_size);

    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, a_size);
    cudaMalloc(&d_B, b_size);
    cudaMalloc(&d_C, c_size);
    cudaMemcpy(d_A, h_A, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice);

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    printf("matmul (%dx%d) @ (%dx%d), grid=(%d,%d), block=(%d,%d)\n",
           M, K, K, N, grid.x, grid.y, block.x, block.y);

    /* warm-up + 计时（GPU 内核时间，不含拷贝） */
    matmul_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    int iters = 5;
    double t0 = get_time();
    for (int i = 0; i < iters; i++) {
        matmul_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }
    double gpu_ms = (get_time() - t0) / iters * 1000;

    /* CPU 版计时（一次性，512^3 ≈ 1.3 亿次乘加，CPU 要几秒） */
    t0 = get_time();
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    double cpu_s = get_time() - t0;

    /* GFLOPS：matmul 的浮点运算次数 ≈ 2*M*N*K（乘+加各算一次） */
    double gflops = 2.0 * M * N * K / (gpu_ms / 1000.0) / 1e9;
    printf("GPU naive : %.3f ms  ->  %.1f GFLOPS\n", gpu_ms, gflops);
    printf("CPU       : %.0f ms   (one shot)\n", cpu_s * 1000);

    /* 验证 */
    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);
    double max_err = 0.0;
    for (int i = 0; i < M * N; i++) {
        double err = fabs(h_C_cpu[i] - h_C_gpu[i]);
        if (err > max_err) max_err = err;
    }
    printf("Verification: %s (max_err = %.2e)\n",
           max_err < 1e-3 ? "CORRECT" : "WRONG", max_err);

    /* 计算强度提示：为什么这么慢？见脚本 04 */
    double bytes = (double)(M * K + K * N + M * N) * sizeof(float);
    printf("Arithmetic intensity: %.2f FLOP/byte -> memory-bound! (see script 04)\n",
           2.0 * M * N * K / bytes);

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
