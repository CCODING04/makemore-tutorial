/*
 * Part 9 - 脚本 04: matmul 优化阶梯 —— 从 naive 到 shared memory + block tiling
 * 目标：亲手走一遍"让 matmul 快起来"的完整阶梯，每登一级测一次 GFLOPS：
 *   L0 naive           （脚本 03 的写法，作为基准）
 *   L1 uncoalesced     （故意把线程映射写坏，感受"合并访存"的价值）
 *   L2 coalesced       （相邻线程读相邻内存 —— 现代寄存器 tiling 的起点）
 *   L3 shared memory   （tile 进 SMEM，减少全局内存重复读）
 *   L4 1D block tiling （每个线程算一列 8 个输出 → 寄存器复用）
 *   L5 2D block tiling （每个线程算 4x4 微tile → 寄存器复用最大化）
 *
 * 对应 cuda-course: 07_Faster_Matmul（优化阶梯源自 Simon Boehm 的
 * "How to Optimize a CUDA Matmul Kernel"：https://siboehm.com/articles/22/CUDA-MMM）
 *
 * 编译运行:
 *   nvcc -O2 -arch=native 04_matmul_tiled.cu -o bin/04_matmul_tiled
 *   ./bin/04_matmul_tiled
 *
 * 注：再往上还有 向量化访存(float4/128bit)、autotuning、double buffering、Tensor Core，
 *     原教程都有；本脚本走到 L5，剩下的见教程 02 章 + 脚本 06 的 cuBLAS 对照。
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 512
#define N 512
#define K 512

/* ============ L0/L1/L2: 每线程一个输出 ============ */

/* L1 故意写坏：x 方向映射到 row。一个 warp 里 32 个线程的 row 连续、col 相同，
 * 读 B[l*N+col] 时地址跨 N 相跳 —— 内存事务利用率暴跌。 */
__global__ void matmul_uncoalesced(const float *A, const float *B, float *C, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   /* ← x 给了 row（坏） */
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
            sum += A[row * k + l] * B[l * n + col];
        C[row * n + col] = sum;
    }
}

/* L2 正确映射：x 方向映射到 col。warp 内相邻线程读 B[l*N+col..col+31] —— 地址连续，
 * 一次内存事务服务 32 个线程（coalesced）；A[row*k+l] 对整个 warp 是同一个地址（广播）。 */
__global__ void matmul_coalesced(const float *A, const float *B, float *C, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   /* ← x 给 col（好） */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
            sum += A[row * k + l] * B[l * n + col];
        C[row * n + col] = sum;
    }
}

/* ============ L3: shared memory tiling，每线程仍 1 个输出 ============
 * 直觉：C 的一个 32x32 tile 需要 A 的 32 行 + B 的 32 列。
 * naive 下每个输出重复读一整行 A / 一整列 B；现在整个 block 把 tile 一次性
 * 搬进 SMEM，之后 K 循环全部命中 SMEM（比 L2 快一个档）。
 * block=(32,32)，tile 正好每线程搬 1 个 A 元素 + 1 个 B 元素。 */
#define BM 32
#define BN 32
#define BK 32

__global__ void matmul_smem(const float *A, const float *B, float *C, int m, int n, int k) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;
    float sum = 0.0f;

    for (int t = 0; t < k; t += BK) {
        /* 合作搬运：每个线程搬 1 个 A、1 个 B 到 SMEM（越界补 0） */
        int aRow = blockIdx.y * BM + ty;
        int aCol = t + tx;
        As[ty][tx] = (aRow < m && aCol < k) ? A[aRow * k + aCol] : 0.0f;
        int bRow = t + ty;
        int bCol = blockIdx.x * BN + tx;
        Bs[ty][tx] = (bRow < k && bCol < n) ? B[bRow * n + bCol] : 0.0f;
        __syncthreads();                     /* 等 tile 全部落位 */

        #pragma unroll
        for (int l = 0; l < BK; l++)
            sum += As[ty][l] * Bs[l][tx];
        __syncthreads();                     /* 算完再搬下一块，防踩踏 */
    }
    if (row < m && col < n) C[row * n + col] = sum;
}

/* ============ L4: 1D block tiling，每线程 8 个输出 ============
 * tile 64x64、BK=8。block 里 512 个线程排成 (64 x 8)：
 *   - 每个线程负责同一列的 TM=8 个输出（存进寄存器数组 sums[8]）
 *   - 内层循环读一次 Bs[l][tx]，喂给 8 次乘加 → SMEM 读摊薄 8 倍
 * 这一步把瓶颈从"SMEM 带宽"推向"寄存器/计算"。 */
#define BM1 64
#define BN1 64
#define BK1 8
#define TM 8

__global__ void matmul_1d(const float *A, const float *B, float *C, int m, int n, int k) {
    __shared__ float As[BM1][BK1];
    __shared__ float Bs[BK1][BN1];

    int tx = threadIdx.x;                 /* 0..63 —— 列坐标 */
    int ty = threadIdx.y;                 /* 0..7  —— 行方向第几组 */
    int col = blockIdx.x * BN1 + tx;
    int rowBase = blockIdx.y * BM1 + ty * TM;
    float sums[TM] = {0};                 /* 8 个结果全部活在寄存器里 */

    for (int t = 0; t < k; t += BK1) {
        /* 合作搬运：512 线程 = 64x8；As 64x8、Bs 8x64，每线程各搬 1 个元素 */
        int linear = ty * BN1 + tx;       /* 0..511 */
        int aRow = linear / BK1, aCol = linear % BK1;
        int gRow = blockIdx.y * BM1 + aRow, gCol = t + aCol;
        As[aRow][aCol] = (gRow < m && gCol < k) ? A[gRow * k + gCol] : 0.0f;
        int bRow = linear / BN1, bCol = linear % BN1;
        gRow = t + bRow; gCol = blockIdx.x * BN1 + bCol;
        Bs[bRow][bCol] = (gRow < k && gCol < n) ? B[gRow * n + gCol] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int l = 0; l < BK1; l++) {
            float bval = Bs[l][tx];       /* 1 次读，8 次复用 */
            #pragma unroll
            for (int r = 0; r < TM; r++)
                sums[r] += As[ty * TM + r][l] * bval;
        }
        __syncthreads();
    }
    for (int r = 0; r < TM; r++) {
        int row = rowBase + r;
        if (row < m && col < n) C[row * n + col] = sums[r];
    }
}

/* ============ L5: 2D block tiling，每线程 4x4 微tile ============
 * tile 64x64、BK=8、TM=TN=4。block 里 16x16=256 线程：
 *   - 每线程算自己的 4x4 微tile（16 个寄存器）
 *   - 搬运：As/Bs 各 512 元素，每线程搬 2 个
 * 这一版已经很接近 cuBLAS 的形态（还差向量化/autotune/Tensor Core）。 */
#define BM2 64
#define BN2 64
#define BK2 8
#define TM2 4
#define TN2 4

__global__ void matmul_2d(const float *A, const float *B, float *C, int m, int n, int k) {
    __shared__ float As[BM2][BK2];
    __shared__ float Bs[BK2][BN2];

    int tx = threadIdx.x, ty = threadIdx.y;      /* 0..15 x 0..15 */
    int colBase = blockIdx.x * BN2 + tx * TN2;
    int rowBase = blockIdx.y * BM2 + ty * TM2;
    float sums[TM2][TN2] = {{0}};                 /* 微tile 活在寄存器里 */

    for (int t = 0; t < k; t += BK2) {
        /* 合作搬运：256 线程，As/Bs 各 512 元素 → 每线程 2 个 */
        int linear = ty * blockDim.x + tx;        /* 0..255 */
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = linear + i * 256;
            int aRow = idx / BK2, aCol = idx % BK2;
            int gRow = blockIdx.y * BM2 + aRow, gCol = t + aCol;
            As[aRow][aCol] = (gRow < m && gCol < k) ? A[gRow * k + gCol] : 0.0f;
            int bRow = idx / BN2, bCol = idx % BN2;
            gRow = t + bRow; gCol = blockIdx.x * BN2 + bCol;
            Bs[bRow][bCol] = (gRow < k && gCol < n) ? B[gRow * n + gCol] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int l = 0; l < BK2; l++)
            #pragma unroll
            for (int r = 0; r < TM2; r++) {
                float aval = As[ty * TM2 + r][l];
                #pragma unroll
                for (int c = 0; c < TN2; c++)
                    sums[r][c] += aval * Bs[l][tx * TN2 + c];
            }
        __syncthreads();
    }
    for (int r = 0; r < TM2; r++)
        for (int c = 0; c < TN2; c++) {
            int row = rowBase + r, col = colBase + c;
            if (row < m && col < n) C[row * n + col] = sums[r][c];
        }
}

/* ============ 基准框架 ============ */
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) mat[i] = (float)rand() / RAND_MAX;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 跑一个内核：warm-up 1 次 + 计时 5 次，返回平均毫秒 */
double bench(void (*launch)(const float*, const float*, float*, int, int, int, dim3, dim3),
             const float *d_A, const float *d_B, float *d_C, dim3 grid, dim3 block) {
    launch(d_A, d_B, d_C, M, N, K, grid, block);
    cudaDeviceSynchronize();
    int iters = 5;
    double t0 = get_time();
    for (int i = 0; i < iters; i++) launch(d_A, d_B, d_C, M, N, K, grid, block);
    cudaDeviceSynchronize();
    return (get_time() - t0) / iters * 1000;
}

/* 每个内核包一层，统一启动配置 */
static void l1(const float*A,const float*B,float*C,int m,int n,int k,dim3 g,dim3 b){matmul_uncoalesced<<<g,b>>>(A,B,C,m,n,k);}
static void l2(const float*A,const float*B,float*C,int m,int n,int k,dim3 g,dim3 b){matmul_coalesced<<<g,b>>>(A,B,C,m,n,k);}
static void l3(const float*A,const float*B,float*C,int m,int n,int k,dim3 g,dim3 b){matmul_smem<<<g,b>>>(A,B,C,m,n,k);}
static void l4(const float*A,const float*B,float*C,int m,int n,int k,dim3 g,dim3 b){matmul_1d<<<g,b>>>(A,B,C,m,n,k);}
static void l5(const float*A,const float*B,float*C,int m,int n,int k,dim3 g,dim3 b){matmul_2d<<<g,b>>>(A,B,C,m,n,k);}

int main() {
    size_t as = M * K * sizeof(float), bs_ = K * N * sizeof(float), cs = M * N * sizeof(float);
    float *h_A = (float*)malloc(as), *h_B = (float*)malloc(bs_), *h_ref = (float*)malloc(cs);
    float *h_C = (float*)malloc(cs);
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    /* CPU 参照答案（512^3 慢，只算一次） */
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

    double flops = 2.0 * M * N * K;
    printf("\n%-18s %10s %12s %8s\n", "kernel", "time(ms)", "GFLOPS", "correct");
    printf("---------------------------------------------------------\n");

    struct { const char *name; void (*fn)(const float*,const float*,float*,int,int,int,dim3,dim3);
             dim3 grid, block; } kernels[] = {
        { "L1 uncoalesced", l1, dim3(M/32, N/32), dim3(32, 32) },
        { "L2 coalesced",   l2, dim3(N/32, M/32), dim3(32, 32) },
        { "L3 smem tile",   l3, dim3(N/BN, M/BM), dim3(BN, BM) },
        { "L4 1D blocktile",l4, dim3(N/BN1, M/BM1), dim3(BN1, BM1/TM) },
        { "L5 2D blocktile",l5, dim3(N/BN2, M/BM2), dim3(BN2/TN2, BM2/TM2) },
    };

    for (int i = 0; i < 5; i++) {
        double ms = bench(kernels[i].fn, d_A, d_B, d_C, kernels[i].grid, kernels[i].block);
        cudaMemcpy(h_C, d_C, cs, cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int j = 0; j < M * N; j++) {
            double e = fabs(h_ref[j] - h_C[j]);
            if (e > max_err) max_err = e;
        }
        printf("%-18s %10.3f %12.1f %8s (max_err %.1e)\n",
               kernels[i].name, ms, flops / (ms / 1000) / 1e9,
               max_err < 1e-3 ? "OK" : "WRONG", max_err);
    }
    printf("\nNext steps (see tutorial 02): vectorized loads (float4), autotuning,\n"
           "double buffering, Tensor Cores -> script 06 compares against cuBLAS.\n");

    free(h_A); free(h_B); free(h_ref); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
