/*
 * Part 9 - 脚本 02: 线程层级 —— grid / block / thread 的 1D、2D、3D 索引
 * 目标：肉眼看到"一个线程是谁"，掌握全局索引公式在各种维度下的写法，
 *       并写第一个"每个线程算自己那个元素"的 square 内核。
 *
 * 对应 cuda-course: 05_Writing_your_First_Kernels/01 CUDA Basics/01_idxing.cu
 *                   + 02 Kernels/01_vector_add_v2.cu（3D 版）
 * 编译运行:
 *   nvcc -O2 -arch=native 02_thread_hierarchy.cu -o bin/02_thread_hierarchy
 *   ./bin/02_thread_hierarchy
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* ---------- 内核 A：把"我是谁"打印出来 ----------
 * built-in 变量（都在内核里可直接用）：
 *   threadIdx.x/y/z  线程在 block 内的坐标
 *   blockIdx.x/y/z   block 在 grid 内的坐标
 *   blockDim.x/y/z   一个 block 有多少线程（启动时给定的）
 *   gridDim.x/y/z    一个 grid 有多少 block
 */
__global__ void whoami() {
    printf("block(%d,%d,%d) thread(%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

/* ---------- 内核 B：square —— 每个线程把自己的元素平方 ----------
 * 1D 情形：全局下标 = blockIdx.x * blockDim.x + threadIdx.x
 * 想象成：第 blockIdx 个"班"，班号乘以"每班人数"，加上"我在班里的学号"
 */
__global__ void square(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {           /* 边界守卫：grid 是向上取整的，多出来的线程要拦住 */
        x[i] = x[i] * x[i];
    }
}

/* ---------- 内核 C：2D 索引（下一步 matmul 的铺垫） ----------
 * 2D：先算行、列两个全局坐标，再折算成一维下标（row-major 行主序）
 *   col = blockIdx.x * blockDim.x + threadIdx.x
 *   row = blockIdx.y * blockDim.y + threadIdx.y
 *   idx = row * n_cols + col
 */
__global__ void square_2d(float *m, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        int idx = row * n_cols + col;
        m[idx] = m[idx] * m[idx];
    }
}

int main() {
    /* === 1) 观察 2D 线程层级：grid(2,2) x block(2,2) = 16 个线程 ===
     * 顺序不重要！GPU 不保证打印顺序 —— 线程是真的并行乱序执行的 */
    printf("=== whoami: grid(2,2) x block(2,2), 16 threads, order NOT guaranteed ===\n");
    dim3 grid(2, 2), block(2, 2);
    whoami<<<grid, block>>>();
    cudaDeviceSynchronize();

    /* === 2) 1D square：验证 CPU 和 GPU 结果一致 === */
    const int N = 1024;
    float h_x[N], expected[N];
    for (int i = 0; i < N; i++) { h_x[i] = i % 10 + 0.5f; expected[i] = h_x[i] * h_x[i]; }

    float *d_x;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + 255) / 256;   /* 向上取整 */
    square<<<blocks, 256>>>(d_x, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N; i++) if (fabsf(h_x[i] - expected[i]) > 1e-6f) errors++;
    printf("\n1D square: %s (%d errors)\n", errors == 0 ? "CORRECT" : "WRONG", errors);

    /* === 3) 同样的数据换 2D 启动（32x32 网格），结果必须相同 ===
     * 感受重点：数据是一维线性数组，"1D/2D/3D"只是我们给线程编号的方式 */
    for (int i = 0; i < N; i++) h_x[i] = i % 10 + 0.5f;   /* 复原输入 */
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid2d(1, 32);        /* 32 行 x 32 列 = 1024 个线程（列方向 1 个 block x 32 线程） */
    dim3 block2d(32, 1);
    square_2d<<<grid2d, block2d>>>(d_x, 32, 32);
    cudaDeviceSynchronize();

    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    errors = 0;
    for (int i = 0; i < N; i++) if (fabsf(h_x[i] - expected[i]) > 1e-6f) errors++;
    printf("2D square (same data, 2D launch): %s (%d errors)\n",
           errors == 0 ? "CORRECT" : "WRONG", errors);

    cudaFree(d_x);
    return errors == 0 ? 0 : 1;
}
