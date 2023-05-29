#include <stdio.h>

#include "../include/cuda_base.cuh"
#include "../include/tq_datatypes.h"
#include "../include/tq_op_gpu.cuh"

#define THR_PER_BLOCK 512
#define TILE_WIDTH 16

__global__ void cuda_vec_add(float *A, float *B, float *C, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void cuda_vec_sub(float *A, float *B, float *C, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] - B[i];
    }
}

__global__ void cuda_vec_scalar_prod(float *v, float *A, float *C, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = (*v) * A[i];
    }
}

__global__ void cuda_vec_dot_prod(float *A, float *B, float *C, int N)
{
    __shared__ float tmp[THR_PER_BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float suma = 0.0;

    if (i < N)
    {
        tmp[threadIdx.x] = A[i] * B[i];
        __syncthreads(); // Barrera

        if (threadIdx.x == 0)
        {
            for (int j = 0; j < THR_PER_BLOCK; j++)
            {
                suma += tmp[j];
            }

            atomicAdd(C, suma);
        }
    }
}

// __global__ void cuda_vec_prod(float *A, float *B, float *C, int N)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N)
//         C[i] = A[i] * B[i];
// }

/**
 * @brief Matrix product kernel for CUDA.
 * Extracted from https://github.com/debowin/cuda-tiled-matrix-multiplication/blob/master/matrixmul_kernel.cu,
 * blockIdx.y Debowin (based on CUDA docs blockIdx.y Nvidia).
 * https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
 *
 * @param M
 * @param Mw
 * @param Mh
 * @param N
 * @param Nw
 * @param Nh
 * @param P
 */
__global__ void cuda_mat_prod(float *M, int *Mw, int *Mh,
                              float *N, int *Nw, int *Nh,
                              float *P)
{
    __shared__ float tileMs[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileNs[TILE_WIDTH][TILE_WIDTH];

    int i, j;

    /**
     * P matrix, width & height
     */
    int M_height = *Mh;
    int M_width = *Mw;

    /**
     * N matrix, width & height
     */
    int N_height = *Nh;
    int N_width = *Nw;

    /**
     * P matrix, width & height
     */
    int P_height = M_height;
    int P_width = N_width;

    // target element coordinates
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int column = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // compute target element value
    float pValue = 0.0f;

    for (i = 0; i < (int)ceilf(M_width / (float)TILE_WIDTH); i++)
    {
        // move the tiles and update shared memory value for new tile positions
        if ((row < M_height) && ((i * TILE_WIDTH + threadIdx.x) < M_width))
        {
            tileMs[threadIdx.y][threadIdx.x] = M[row * M_width + (i * TILE_WIDTH + threadIdx.x)];
        }
        else
        {
            tileMs[threadIdx.y][threadIdx.x] = 0;
        }

        if ((column < N_width) && ((i * TILE_WIDTH + threadIdx.y) < N_height))
        {
            tileNs[threadIdx.y][threadIdx.x] = N[(i * TILE_WIDTH + threadIdx.y) * N_width + column];
        }
        else
        {
            tileNs[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (j = 0; j < TILE_WIDTH; j++)
        {
            pValue += (tileMs[threadIdx.y][j] * tileNs[j][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < P_height && column < P_width)
    {
        P[row * P_width + column] = pValue;
    }
}

void __TQ_GPUMat_Add(TQ_Matrix one,
                     TQ_Matrix other,
                     TQ_Matrix *result)
{
    /**
     * Time measurement, using CUDA events.
     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execution time, in ms.
    float t_exe;

    /**
     * Device mem.
     */
    // Device memory for matrix ONE
    float *d_one;

    // Device memory for matrix OTHER
    float *d_other;

    // Device memory for matrix RESULT (the sum of both matrices)
    float *d_result;

    // Execution env.
    int thread_per_block;
    int block_in_grid;
    long num_float = one.dims_prod;

    /**
     * Allocate device memory.
     */
    gpuErrchk(
        cudaMalloc((void **)(&d_one), one.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_other), other.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_result), result->length_bytes));

    /**
     * Copy host mem. ONE, OTHER to device.
     */
    gpuErrchk(
        cudaMemcpy((void *)(d_one), (const void *)(one.h_mem), one.length_bytes,
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_other), (const void *)(other.h_mem), other.length_bytes,
                   cudaMemcpyHostToDevice));

    /**
     * SET UP CUDA execution env.
     *      thr_per_blk: number of CUDA threads per grid block
     *      blk_in_grid: number of blocks in grid
     */
    thread_per_block = THR_PER_BLOCK;
    block_in_grid = (int)ceil((float)num_float / thread_per_block);

    // ! RUN - KERNEL
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_add<<<block_in_grid, thread_per_block>>>(d_one, d_other, d_result, num_float);
    gpuErrchk(cudaEventRecord(stop));
    // ! END - KERNEL

    /**
     * Recover DATA from DEVICE
     */
    gpuErrchk(
        cudaMemcpy((void *)(result->h_mem), (const void *)(d_result), result->length_bytes,
                   cudaMemcpyDeviceToHost));

    /**
     * Work out elapsed time and finish operation.
     */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_exe, start, stop);

    /**
     * CLEAN DEVICE mem.
     */
    cudaFree(d_one);
    cudaFree(d_other);
    cudaFree(d_result);

    // printf("Matrix ADD: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
}

void __TQ_GPUMat_Sub(TQ_Matrix one,
                     TQ_Matrix other,
                     TQ_Matrix *result)
{
    /**
     * Time measurement, using CUDA events.
     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execution time, in ms.
    float t_exe;

    /**
     * Device mem.
     */
    // Device memory for matrix ONE
    float *d_one;

    // Device memory for matrix OTHER
    float *d_other;

    // Device memory for matrix RESULT (the sum of both matrices)
    float *d_result;

    // Execution env.
    int thread_per_block;
    int block_in_grid;
    long num_float = one.dims_prod;

    /**
     * Allocate device memory.
     */
    gpuErrchk(
        cudaMalloc((void **)(&d_one), one.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_other), other.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_result), result->length_bytes));

    /**
     * Copy host mem. ONE, OTHER to device.
     */
    gpuErrchk(
        cudaMemcpy((void *)(d_one), (const void *)(one.h_mem), one.length_bytes,
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_other), (const void *)(other.h_mem), other.length_bytes,
                   cudaMemcpyHostToDevice));

    /**
     * SET UP CUDA execution env.
     *      thr_per_blk: number of CUDA threads per grid block
     *      blk_in_grid: number of blocks in grid
     */
    thread_per_block = THR_PER_BLOCK;
    block_in_grid = (int)ceil((float)num_float / thread_per_block);

    // ! RUN - KERNEL
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_sub<<<block_in_grid, thread_per_block>>>(d_one, d_other, d_result, num_float);
    gpuErrchk(cudaEventRecord(stop));
    // ! END - KERNEL

    /**
     * Recover DATA from DEVICE
     */
    gpuErrchk(
        cudaMemcpy((void *)(result->h_mem), (const void *)(d_result), result->length_bytes,
                   cudaMemcpyDeviceToHost));

    /**
     * Work out elapsed time and finish operation.
     */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_exe, start, stop);

    /**
     * CLEAN DEVICE mem.
     */
    cudaFree(d_one);
    cudaFree(d_other);
    cudaFree(d_result);

    // printf("Matrix SUB: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
}

void __TQ_GPUMat_ProdNum(TQ_Matrix one,
                         float factor,
                         TQ_Matrix *result)
{
    /**
     * Time measurement, using CUDA events.
     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execution time, in ms.
    float t_exe;

    /**
     * Device mem.
     */
    // Device memory for matrix ONE
    float *d_one;

    // Device memory for matrix OTHER
    float *d_factor;

    // Device memory for matrix RESULT (the sum of both matrices)
    float *d_result;

    // Execution env.
    int thread_per_block;
    int block_in_grid;
    long num_float = one.dims_prod;

    /**
     * Allocate device memory.
     */
    gpuErrchk(
        cudaMalloc((void **)(&d_one), one.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_factor), sizeof(float)));
    gpuErrchk(
        cudaMalloc((void **)(&d_result), result->length_bytes));

    /**
     * Copy host mem. ONE, OTHER to device.
     */
    gpuErrchk(
        cudaMemcpy((void *)(d_one), (const void *)(one.h_mem), one.length_bytes,
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_factor), (const void *)(&factor), sizeof(float),
                   cudaMemcpyHostToDevice));

    /**
     * SET UP CUDA execution env.
     *      thr_per_blk: number of CUDA threads per grid block
     *      blk_in_grid: number of blocks in grid
     */
    thread_per_block = THR_PER_BLOCK;
    block_in_grid = (int)ceil((float)num_float / thread_per_block);

    // ! RUN - KERNEL
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_scalar_prod<<<block_in_grid, thread_per_block>>>(d_factor, d_one, d_result, num_float);
    gpuErrchk(cudaEventRecord(stop));
    // ! END - KERNEL

    /**
     * Recover DATA from DEVICE
     */
    gpuErrchk(
        cudaMemcpy((void *)(result->h_mem), (const void *)(d_result), result->length_bytes,
                   cudaMemcpyDeviceToHost));

    /**
     * Work out elapsed time and finish operation.
     */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_exe, start, stop);

    /**
     * CLEAN DEVICE mem.
     */
    cudaFree(d_one);
    cudaFree(d_factor);
    cudaFree(d_result);

    // printf("Matrix PROD by FACTOR: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
}

void __TQ_GPUMat_Prod(TQ_Matrix one,
                      TQ_Matrix other,
                      TQ_Matrix *result)
{
    /**
     * Time measurement, using CUDA events.
     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execution time, in ms.
    float t_exe;

    /**
     * Device mem.
     */
    // Device memory for matrix ONE
    float *d_one;
    int *d_one_w;
    int *d_one_h;

    // Device memory for matrix OTHER
    float *d_other;
    int *d_other_w;
    int *d_other_h;

    // Device memory for matrix RESULT (the sum of both matrices)
    float *d_result;

    // Execution env.
    dim3 block_size;
    dim3 grid_size;

    long num_float = result->dims_prod;

    /**
     * Allocate device memory.
     */
    gpuErrchk(
        cudaMalloc((void **)(&d_one), one.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_one_w), sizeof(int)));
    gpuErrchk(
        cudaMalloc((void **)(&d_one_h), sizeof(int)));

    gpuErrchk(
        cudaMalloc((void **)(&d_other), other.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_other_w), sizeof(int)));
    gpuErrchk(
        cudaMalloc((void **)(&d_other_h), sizeof(int)));

    gpuErrchk(
        cudaMalloc((void **)(&d_result), result->length_bytes));

    /**
     * Copy host mem. ONE, OTHER to device.
     */
    gpuErrchk(
        cudaMemcpy((void *)(d_one), (const void *)(one.h_mem), one.length_bytes,
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_one_h), (const void *)(&(one.dimensions[0])), sizeof(int),
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_one_w), (const void *)(&(one.dimensions[1])), sizeof(int),
                   cudaMemcpyHostToDevice));

    gpuErrchk(
        cudaMemcpy((void *)(d_other), (const void *)(other.h_mem), other.length_bytes,
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_other_h), (const void *)(&(other.dimensions[0])), sizeof(int),
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_other_w), (const void *)(&(other.dimensions[1])), sizeof(int),
                   cudaMemcpyHostToDevice));

    /**
     * SET UP CUDA execution env.
     *      block_size: number of CUDA threads per grid block
     *      grid_size: number of blocks in grid
     */
    block_size.x = TILE_WIDTH;
    block_size.y = TILE_WIDTH;
    block_size.z = 1;

    grid_size.x = ceil(result->dimensions[0] / (float)block_size.x);
    grid_size.y = ceil(result->dimensions[1] / (float)block_size.y);
    grid_size.z = 1;

    // ! RUN - KERNEL
    gpuErrchk(cudaEventRecord(start));
    cuda_mat_prod<<<grid_size, block_size>>>(d_one, d_one_w, d_one_h,
                                             d_other, d_other_w, d_other_h,
                                             d_result);
    gpuErrchk(cudaEventRecord(stop));
    // ! END - KERNEL

    /**
     * Recover DATA from DEVICE
     */
    gpuErrchk(
        cudaMemcpy((void *)(result->h_mem), (const void *)(d_result), result->length_bytes,
                   cudaMemcpyDeviceToHost));

    /**
     * Work out elapsed time and finish operation.
     */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_exe, start, stop);

    /**
     * CLEAN DEVICE mem.
     */
    cudaFree(d_one);
    cudaFree(d_one_w);
    cudaFree(d_one_h);
    cudaFree(d_other);
    cudaFree(d_other_w);
    cudaFree(d_other_h);
    cudaFree(d_result);

    // printf("Matrix MATRIX PROD: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
}

void __TQ_GPUVec_Dot(TQ_Matrix one,
                     TQ_Matrix other,
                     float *result)
{
    /**
     * Time measurement, using CUDA events.
     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execution time, in ms.
    float t_exe;

    /**
     * Device mem.
     */
    // Device memory for matrix ONE
    float *d_one;

    // Device memory for matrix OTHER
    float *d_other;

    // Device memory for matrix RESULT (the sum of both matrices)
    float *d_result;

    // Execution env.
    int thread_per_block;
    int block_in_grid;
    long num_float = one.dims_prod;

    /**
     * Allocate device memory.
     */
    gpuErrchk(
        cudaMalloc((void **)(&d_one), one.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_other), other.length_bytes));
    gpuErrchk(
        cudaMalloc((void **)(&d_result), sizeof(float)));

    /**
     * Copy host mem. ONE, OTHER to device.
     */
    gpuErrchk(
        cudaMemcpy((void *)(d_one), (const void *)(one.h_mem), one.length_bytes,
                   cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy((void *)(d_other), (const void *)(other.h_mem), other.length_bytes,
                   cudaMemcpyHostToDevice));

    /**
     * SET result mem. to 0
     */
    gpuErrchk(
        cudaMemset((void *)(d_result), 0x0, sizeof(float)));

    /**
     * SET UP CUDA execution env.
     *      thr_per_blk: number of CUDA threads per grid block
     *      blk_in_grid: number of blocks in grid
     */
    thread_per_block = THR_PER_BLOCK;
    block_in_grid = (int)ceil((float)num_float / thread_per_block);

    // ! RUN - KERNEL
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_dot_prod<<<block_in_grid, thread_per_block>>>(d_one, d_other, d_result, num_float);
    gpuErrchk(cudaEventRecord(stop));
    // ! END - KERNEL

    /**
     * Recover DATA from DEVICE
     */
    gpuErrchk(
        cudaMemcpy((void *)(result), (const void *)(d_result), sizeof(float),
                   cudaMemcpyDeviceToHost));

    /**
     * Work out elapsed time and finish operation.
     */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_exe, start, stop);

    /**
     * CLEAN DEVICE mem.
     */
    cudaFree(d_one);
    cudaFree(d_other);
    cudaFree(d_result);

    // printf("Matrix VecDot PROD: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
}