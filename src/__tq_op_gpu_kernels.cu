#ifndef _CUDA_OP_KERNELS_
#define _CUDA_OP_KERNELS_

#include <cuda.h>
#include <cuda_runtime.h>

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

#endif