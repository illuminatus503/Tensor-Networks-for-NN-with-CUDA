#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../include/__tq_datatypes.h"
#include "../include/__tq_op_gpu.cuh"

#include "__tq_op_gpu_kernels.cu"

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

    // printf("Matrix ADD: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
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

    // printf("Matrix SUB: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
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

    // printf("Matrix PROD by FACTOR: %ld float(s) -- Elapsed time: %1.3fms\n",
    //        num_float, t_exe);
}

void __TQ_GPUMat_Prod(struct TQ_Matrix one,
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

void __TQ_GPUVec_Dot(struct TQ_Matrix one,
                     struct TQ_Matrix other,
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