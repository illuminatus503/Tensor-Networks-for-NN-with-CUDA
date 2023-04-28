#include <stdio.h>
#include <stdlib.h>

#include "include/tq_matrix.h"

int main(int argc, char **argv)
{
    TQ_Matrix A, B, C;
    unsigned int dims[] = {2, 2};

    TQ_Matrix_Create(&A,
                     dims, 2,
                     TQ_CPU_Matrix);
    TQ_Matrix_Create(&B,
                     dims, 2,
                     TQ_CPU_Matrix);
    TQ_Matrix_Create(&C,
                     dims, 2,
                     TQ_CPU_Matrix);

    // Inicializamos las matrices a 1.0f
    unsigned int i, j;
    unsigned int coords[2];
    for (i = 0; i < 2; i++)
    {
        coords[0] = i;

        for (j = 0; j < 2; j++)
        {
            coords[1] = j;

            if (i == j)
            {
                TQ_Matrix_SetElem(&A, 1.0f, coords, 2);
                TQ_Matrix_SetElem(&B, 0.10f, coords, 2);
            }
            else
            {
                TQ_Matrix_SetElem(&A, 0.0f, coords, 2);
                TQ_Matrix_SetElem(&B, 0.0f, coords, 2);
            }
        }
    }

    TQ_Matrix_Prod(A, B, &C);

    // TQ_Matrix_Add(A, B, &C);
    // TQ_Matrix_ProdNum(C, 2.0f, &C);

    // Imprimir el resultado
    for (i = 0; i < 2; i++)
    {
        coords[0] = i;

        for (j = 0; j < 2; j++)
        {
            coords[1] = j;
            printf("%.2f ", TQ_Matrix_GetElem(C, coords, 2));
        }
        printf("\n");
    }

    // int coords[] = {2, 1, 2};
    // printf("Posición (%d, %d, %d) = %lu en la matriz\n",
    //        coords[0], coords[1], coords[2], __TQ_Matrix_Pos(matrix, coords, 3));

    TQ_Matrix_Free(&A);
    TQ_Matrix_Free(&B);
    TQ_Matrix_Free(&C);

    return 0;
}