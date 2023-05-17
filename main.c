#include <stdio.h>
#include <stdlib.h>

#include "include/tq_matrix.h"
#include "include/tq_perceptron.h"

void init_args(int argc, char **argv,
               unsigned int *n_elems)
{
    if (argc >= 2)
    {
        (*n_elems) = atoi(argv[1]);
    }
    else
    {
        (*n_elems) = 1;
    }
}

int main(int argc, char **argv)
{
    unsigned int num_elems;
    unsigned int dims[2];

    TQ_Perceptron P;
    TQ_Matrix X, A;

    init_args(argc, argv, &num_elems);

    // Input matrix: random data.
    dims[0] = num_elems;
    dims[1] = 1;
    TQ_Matrix_Create(&X,
                     dims, 2,
                     TQ_GPU_Matrix);
    TQ_Matrix_Unif(&X);

    printf("X = \n");
    TQ_Matrix_Print(X);

    /**
     * Perceptron create & launch
     */
    printf("Creating perceptron...\n");
    TQ_Perceptron_Create(&P, num_elems, TQ_GPU_Matrix);
    printf("Pass forward\n");
    TQ_Perceptron_Forward(X, P, &A);
    printf("Activation = \n");
    TQ_Matrix_Print(A);

    TQ_Matrix_Free(&X);
    TQ_Matrix_Free(&A);

    return 0;
}