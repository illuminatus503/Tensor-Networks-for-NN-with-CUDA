#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "codeGPU.cuh"

#define THR_PER_BLOCK 1024

__global__ void cuda_vec_add(float *A, float *B, float *C, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
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

double add_vectors_GPU(float *A, float *B, float *C, size_t N)
{
    // Medida de tiempos en GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    float *d_A, *d_B, *d_C;
    int thr_per_blk, blk_in_grid;

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((float)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_add<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    gpuErrchk(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (milliseconds);
}

double prod_vectors_GPU(float *A, float *B, float *C, size_t N)
{
    // Medida de tiempos en GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    float *d_A, *d_B, *d_C;
    int thr_per_blk, blk_in_grid;

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((float)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_prod<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    gpuErrchk(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (milliseconds);
}

double dot_prod_vectors_GPU(float *A, float *B, float *C, size_t N)
{
    // Medida de tiempos en GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    float *d_A, *d_B, *d_C;
    int thr_per_blk, blk_in_grid;

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C, sizeof(float)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    *C = 0.0;
    // gpuErrchk(cudaMemset(C, 0x0, sizeof(float))); // Inicializar C
    gpuErrchk(cudaMemcpy(d_C, C, sizeof(float), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((float)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_dot_prod<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    gpuErrchk(cudaMemcpy(C, d_C, sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (milliseconds);
}