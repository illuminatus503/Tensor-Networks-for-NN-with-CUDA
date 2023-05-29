#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "include/cuda_errchk.cuh"

#include "include/tq_mem.h"

#define BUFF_SIZE 256 // In bytes

int main(int argc, char **argv)
{
    const unsigned int NUM_TEST = 3;

    const int dx_val = 123;
    const float df_val = 987.0;

    unsigned int i;

    /**
     * CPU arena.
     */
    int *h_x;
    float *h_f;

    unsigned char h_mem[BUFF_SIZE];

    Arena cpu_arena = {0};
    arena_init(&cpu_arena, h_mem, BUFF_SIZE);

    printf("\n***TEST CPU\n");
    for (i = 0; i < NUM_TEST; i++)
    {
        // Reset all arena offsets for each loop
        arena_free_all(&cpu_arena);

        // ! INT test
        h_x = (int *)arena_alloc(&cpu_arena, sizeof(int));
        *h_x = dx_val;
        printf("x: %p: %d\n", h_x, *h_x);

        // ! FLOAT test
        h_f = (float *)arena_alloc(&cpu_arena, sizeof(float));
        *h_f = df_val;
        printf("f: %p: %f\n", h_f, *h_f);
    }

    /**
     * GPU arena.
     */
    int *d_x;
    float *d_f;

    unsigned char *d_mem;
    gpuErrchk(
        cudaMalloc((void **)(&d_mem), BUFF_SIZE));

    Arena gpu_arena = {0};
    CUDA_arena_init(&gpu_arena, d_mem, BUFF_SIZE);

    printf("\n***TEST GPU\n");
    for (i = 0; i < NUM_TEST; i++)
    {
        // Reset all arena offsets for each loop
        arena_free_all(&gpu_arena);

        // ! INT test
        d_x = (int *)CUDA_arena_alloc(&gpu_arena, sizeof(int));
        gpuErrchk(cudaMemcpy((void *)d_x, (const void *)&dx_val, sizeof(int),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy((void *)h_x, (const void *)d_x, sizeof(int),
                             cudaMemcpyDeviceToHost));

        printf("x: %p (%d)\n", d_x, *h_x);

        // ! FLOAT test
        d_f = (float *)CUDA_arena_alloc(&gpu_arena, sizeof(float));
        gpuErrchk(cudaMemcpy((void *)d_f, (const void *)&df_val, sizeof(float),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy((void *)h_f, (const void *)d_f, sizeof(float),
                             cudaMemcpyDeviceToHost));

        printf("f: %p (%f)\n", d_f, *h_f);
    }

    arena_free_all(&cpu_arena);
    arena_free_all(&gpu_arena);
    cudaFree(d_mem);

    return 0;
}