/*
 * Part 9 - 脚本 06: cuBLAS —— 用 NVIDIA 的库做 matmul，与手写内核对照
 * 目标：调用 cuBLAS 的 sgemm（单精度 GEMM），测出"库 vs 手写 L5"的差距，
 *       顺便理解 cuBLAS 的两个著名坑：列主序 + 行/列主序换算。
 *
 * 对应 cuda-course: 06_CUDA_APIs/01 CUBLAS/01 cuBLAS/01_Hgemm_Sgemm.cu
 * 编译运行（注意 -lcublas）:
 *   nvcc -O2 -arch=native 06_cublas_sgemm.cu -o bin/06_cublas_sgemm -lcublas
 *   ./bin/06_cublas_sgemm
 *
 * cuBLAS 坑点速记：
 *   cuBLAS 是 Fortran 血统 → 全部按列主序（column-major）思考。
 *   关键恒等式：一块"行主序存储"的缓冲区，按列主序去读，读到的正好是它的转置。
 *   我们要算行主序 C = A @ B（A: M×K, B: K×N）。按列主序读这三块缓冲区，
 *   得到的分别是 A^T (K×M)、B^T (N×K)、C^T (N×M)。而
 *       C^T = (A@B)^T = B^T @ A^T
 *   于是"行主序的 A@B"恰好等于"列主序的 B^T @ A^T"——
 *   不用真的转置任何数据，只是把 B、A 的指针换个顺序传给 cuBLAS（OP_N, OP_N）。
 *   对应调用：m=N, n=M, k=K，A 位传 d_B(ld=N)，B 位传 d_A(ld=K)，C 的 ldc=N。
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 512
#define N 512
#define K 512

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) mat[i] = (float)rand() / RAND_MAX;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    size_t as = M * K * sizeof(float), bs_ = K * N * sizeof(float), cs = M * N * sizeof(float);
    float *h_A = (float*)malloc(as), *h_B = (float*)malloc(bs_), *h_ref = (float*)malloc(cs);
    float *h_C = (float*)malloc(cs);
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    /* CPU 参照 */
    printf("Computing CPU reference...\n");
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int l = 0; l < K; l++) s += h_A[i * K + l] * h_B[l * N + j];
            h_ref[i * N + j] = s;
        }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, as); cudaMalloc(&d_B, bs_); cudaMalloc(&d_C, cs);
    cudaMemcpy(d_A, h_A, as, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bs_, cudaMemcpyHostToDevice);

    /* ---- cuBLAS 句柄：类似"库的上下文"，一个进程建一次就够 ---- */
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    /* warm-up + 计时
     * cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
     * 计算 C(m×n) = op(A)(m×k) @ op(B)(k×n)。代入推导出的参数：
     *   m=N, n=M, k=K；A 位 = d_B（ld=N，列主序读出 B^T），B 位 = d_A（ld=K，读出 A^T）
     *   两个 op 都是 OP_N（不转置）——"转置"已经藏在"行主序当列主序读"里了
     */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,                 /* 列主序视角：C^T 是 N x M */
                &alpha,
                d_B, N,                  /* B^T：ld = N */
                d_A, K,                  /* A^T：ld = K */
                &beta, d_C, N);          /* C^T：ldc = N */
    cudaDeviceSynchronize();

    int iters = 10;
    double t0 = get_time();
    for (int i = 0; i < iters; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    double ms = (get_time() - t0) / iters * 1000;

    cudaMemcpy(h_C, d_C, cs, cudaMemcpyDeviceToHost);
    double max_err = 0.0;
    for (int i = 0; i < M * N; i++) {
        double e = fabs(h_ref[i] - h_C[i]);
        if (e > max_err) max_err = e;
    }

    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    printf("cuBLAS sgemm: %.3f ms -> %.1f GFLOPS (max_err %.1e, %s)\n",
           ms, gflops, max_err, max_err < 1e-3 ? "OK" : "WRONG");
    printf("\nCompare with script 04: cuBLAS is the 'ceiling' of the ladder.\n");
    printf("Ladder to the ceiling: vectorized loads -> autotuning -> Tensor Cores.\n");
    printf("(cuBLAS also uses Tensor Cores on modern GPUs for fp16/bf16/fp32-TF32)\n");

    /* ---- 顺手演示：cuBLAS 也提供现成的激活/softmax 相关原语 ----
     * 教程 03 章会讲 cuDNN 覆盖 conv/RNN/activation；
     * 这里只提醒：PyTorch 里 @ 的背后就是这类库调用（ATen -> cuBLAS/cuDNN）。 */

    cublasDestroy(handle);
    free(h_A); free(h_B); free(h_ref); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return max_err < 1e-3 ? 0 : 1;
}
