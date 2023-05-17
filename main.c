#include <stdio.h>
#include <stdlib.h>

#include "include/tq_matrix.h"

int main(int argc, char **argv)
{
    unsigned int i;

    unsigned int NDIMS;
    unsigned int *dims;

    TQ_Matrix A, B, C;

    if (argc >= 3)
    {
        NDIMS = atoi(argv[1]);
        dims = (unsigned int *)malloc(sizeof(int) * NDIMS);

        if (argc == 3)
        {
            for (i = 0; i < NDIMS; i++)
            {
                dims[i] = atoi(argv[2]);
            }
        }
        else
        {
            for (i = 0; i < NDIMS; i++)
            {
                dims[i] = atoi(argv[2 + i]);
            }
        }
    }
    else
    {
        NDIMS = 2;
        dims = (unsigned int *)malloc(sizeof(int) * NDIMS);

        for (i = 0; i < NDIMS; i++)
        {
            dims[i] = 2;
        }
    }

    /**
     * Crea las matrices
     */
    TQ_Matrix_Create(&A,
                     dims, NDIMS,
                     TQ_GPU_Matrix);
    // TQ_Matrix_Create(&B,
    //                  dims, NDIMS,
    //                  TQ_GPU_Matrix);
    // TQ_Matrix_Create(&C,
    //                  dims, NDIMS,
    //                  TQ_GPU_Matrix);

    // TQ_Matrix_Init(&A, 6.998f);
    // TQ_Matrix_Ones(&A);
    // TQ_Matrix_Zeros(&A);
    // TQ_Matrix_Eyes(&A);
    // TQ_Matrix_Eyes(&B);

    TQ_Matrix_Unif(&A);
    // TQ_Matrix_Rand(&B, -10.0, 10.0);

    // Inicializamos las matrices a 1.0f (N x N)
    // for (i = 0; i < N; i++)
    // {
    //     coords[0] = i;

    //     for (j = 0; j < N; j++)
    //     {
    //         coords[1] = j;

    //         if (i == j)
    //         {
    //             TQ_Matrix_SetElem(&A, 1.0f, coords, NDIMS);
    //             TQ_Matrix_SetElem(&B, 0.10f, coords, NDIMS);
    //         }
    //         else
    //         {
    //             TQ_Matrix_SetElem(&A, 0.0f, coords, NDIMS);
    //             TQ_Matrix_SetElem(&B, 0.0f, coords, NDIMS);
    //         }
    //     }
    // }

    printf("A = \n");
    TQ_Matrix_Print(A);
    // printf("B = \n");
    // TQ_Matrix_Print(B);
    // TQ_Matrix_Prod(A, B, &C);

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
    // TQ_Matrix_Free(&B);
    // TQ_Matrix_Free(&C);

    free(dims);
    return 0;
}