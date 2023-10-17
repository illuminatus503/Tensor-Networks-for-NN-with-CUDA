#include <stdio.h>
#include <stdlib.h>

#include "include/tq_matrix.h"

void init_args(int argc, char **argv,
               unsigned int *ndims,
               unsigned int **dims)
{
    unsigned int i;
    if (argc >= 2)
    {
        (*ndims) = atoi(argv[1]);
    }
    else
    {
        (*ndims) = 1;
    }

    *dims = (unsigned int *)malloc(sizeof(int) * (*ndims));

    if (argc >= 3)
    {
        if (argc == 3)
        {
            for (i = 0; i < (*ndims); i++)
            {
                (*dims)[i] = atoi(argv[2]);
            }
        }
        else
        {
            for (i = 0; i < (*ndims); i++)
            {
                (*dims)[i] = atoi(argv[2 + i]);
            }
        }
    }
    else
    {
        for (i = 0; i < (*ndims); i++)
        {
            (*dims)[i] = 1;
        }
    }
}

int main(int argc, char **argv)
{
    unsigned int NDIMS;
    unsigned int *dims;

    TQ_Matrix A_cpu, B_cpu, C_cpu;
    TQ_Matrix A_gpu, B_gpu, C_gpu;
    TQ_Matrix DELTA_C;

    // Initialise params.
    init_args(argc, argv, &NDIMS, &dims);

    // Create & initialise base matrices A (nxm), B (mxm)
    // Both random uniform matrices.
    TQ_Matrix_Create(&A_cpu,
                     dims, NDIMS,
                     TQ_CPU_Matrix);
    TQ_Matrix_Create(&A_gpu,
                     dims, NDIMS,
                     TQ_GPU_Matrix);
    TQ_Matrix_Unif(&A_cpu);
    TQ_Matrix_CopyData(A_cpu, &A_gpu);

    printf("A (cpu) = \n");
    TQ_Matrix_Print(A_cpu);
    printf("A (gpu) = \n");
    TQ_Matrix_Print(A_gpu);

    dims[0] = dims[1];
    TQ_Matrix_Create(&B_cpu,
                     dims, NDIMS,
                     TQ_CPU_Matrix);
    TQ_Matrix_Create(&B_gpu,
                     dims, NDIMS,
                     TQ_GPU_Matrix);
    TQ_Matrix_Unif(&B_cpu);
    TQ_Matrix_CopyData(B_cpu, &B_gpu);

    printf("B (cpu) = \n");
    TQ_Matrix_Print(B_cpu);
    printf("B (gpu) = \n");
    TQ_Matrix_Print(B_gpu);

    // Apply matrix product.
    TQ_Matrix_Prod(A_cpu, B_cpu, &C_cpu);
    printf("A · B (cpu) = \n");
    TQ_Matrix_Print(C_cpu);

    TQ_Matrix_Prod(A_gpu, B_gpu, &C_gpu);
    printf("A · B (gpu) = \n");
    TQ_Matrix_Print(C_gpu);

    // Difference
    TQ_Matrix_Sub(C_cpu, C_gpu, &DELTA_C);
    printf("Difference: \n");
    TQ_Matrix_Print(DELTA_C);

    // Free heap mem.
    TQ_Matrix_Free(&A_cpu);
    TQ_Matrix_Free(&B_cpu);
    TQ_Matrix_Free(&C_cpu);
    TQ_Matrix_Free(&A_gpu);
    TQ_Matrix_Free(&B_gpu);
    TQ_Matrix_Free(&C_gpu);
    TQ_Matrix_Free(&DELTA_C);
    free(dims);

    return 0;
}