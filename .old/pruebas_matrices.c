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
    TQ_Matrix A, B, C;
    float c_value;

    /**
     * Crea las matrices
     */
    init_args(argc, argv, &NDIMS, &dims);
    TQ_Matrix_Create(&A,
                     dims, NDIMS,
                     TQ_CPU_Matrix);
    TQ_Matrix_Create(&B,
                     dims, NDIMS,
                     TQ_GPU_Matrix);
    // TQ_Matrix_Create(&C,
    //                  dims, NDIMS,
    //                  TQ_GPU_Matrix);

    // TQ_Matrix_Init(&A, 6.998f);
    // TQ_Matrix_Ones(&A);
    // TQ_Matrix_Zeros(&A);
    // TQ_Matrix_Eyes(&A);
    // TQ_Matrix_Eyes(&B);

    TQ_Matrix_Unif(&A);
    TQ_Matrix_Rand(&B, -10.0, 10.0);

    printf("A = \n");
    TQ_Matrix_Print(A);
    printf("B = \n");
    TQ_Matrix_Print(B);
    TQ_Vec_Dot(A, B, &c_value);
    printf("A dot B = %1.3f\n", c_value);

    // TQ_Matrix_Add(A, B, &C);
    // TQ_Matrix_Print(C);
    // TQ_Matrix_ProdNum(C, 2.0f, &C);
    // TQ_Matrix_Print(C);
    // TQ_Matrix_Prod(C, C, &C);
    // TQ_Matrix_Print(C);

    // int coords[] = {2, 1, 2};
    // printf("Posición (%d, %d, %d) = %lu en la matriz\n",
    //        coords[0], coords[1], coords[2], __TQ_Matrix_Pos(matrix, coords, 3));

    TQ_Matrix_Free(&A);
    TQ_Matrix_Free(&B);
    // TQ_Matrix_Free(&C);

    free(dims);
    return 0;
}