#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../include/__tq_datatypes.h"
#include "../include/__tq_op_gpu.cuh"

#define THR_PER_BLOCK 1024

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

__global__ void cuda_vec_prod(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] * B[i];
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

void __TQ_GPUMat_Add(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result)
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

    printf("Matrix ADD: %ld float(s) -- Elapsed time: %1.3fms\n",
           num_float, t_exe);
}

void __TQ_GPUMat_Sub(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result)
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

    printf("Matrix SUB: %ld float(s) -- Elapsed time: %1.3fms\n",
           num_float, t_exe);
}

void __TQ_GPUMat_ProdNum(struct TQ_Matrix one,
                         float factor,
                         struct TQ_Matrix *result)
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

    printf("Matrix PROD by FACTOR: %ld float(s) -- Elapsed time: %1.3fms\n",
           num_float, t_exe);
}

void __TQ_GPUMat_Prod(struct TQ_Matrix one,
                      struct TQ_Matrix other,
                      struct TQ_Matrix *result)
{
}