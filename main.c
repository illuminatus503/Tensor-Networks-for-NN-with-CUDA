#include <stdio.h>
#include <string.h>

#include "include/tq_global_mem.h"
#include "include/cuda_base.cuh"

Arena TQ_CPU_ARENA;
Arena TQ_GPU_ARENA;

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
    arena_init(&TQ_CPU_ARENA, h_mem, BUFF_SIZE);

    printf("\n***TEST CPU\n");
    for (i = 0; i < NUM_TEST; i++)
    {
        // Reset all arena offsets for each loop
        arena_free_all(&TQ_CPU_ARENA);

        // ! INT test
        h_x = (int *)arena_alloc(&TQ_CPU_ARENA, sizeof(int));
        *h_x = dx_val;
        printf("x: %p: %d\n", h_x, *h_x);

        // ! FLOAT test
        h_f = (float *)arena_alloc(&TQ_CPU_ARENA, sizeof(float));
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

    CUDA_arena_init(&TQ_GPU_ARENA, d_mem, BUFF_SIZE);

    printf("\n***TEST GPU\n");
    for (i = 0; i < NUM_TEST; i++)
    {
        // Reset all arena offsets for each loop
        arena_free_all(&TQ_GPU_ARENA);

        // ! INT test
        d_x = (int *)CUDA_arena_alloc(&TQ_GPU_ARENA, sizeof(int));
        gpuErrchk(cudaMemcpy((void *)d_x, (const void *)&dx_val, sizeof(int),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy((void *)h_x, (const void *)d_x, sizeof(int),
                             cudaMemcpyDeviceToHost));

        printf("x: %p (%d)\n", d_x, *h_x);

        // ! FLOAT test
        d_f = (float *)CUDA_arena_alloc(&TQ_GPU_ARENA, sizeof(float));
        gpuErrchk(cudaMemcpy((void *)d_f, (const void *)&df_val, sizeof(float),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy((void *)h_f, (const void *)d_f, sizeof(float),
                             cudaMemcpyDeviceToHost));

        printf("f: %p (%f)\n", d_f, *h_f);
    }

    arena_free_all(&TQ_CPU_ARENA);
    arena_free_all(&TQ_GPU_ARENA);
    cudaFree(d_mem);

    return 0;
}