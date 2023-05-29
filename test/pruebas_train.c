#include <stdio.h>
#include <stdlib.h>

#include "include/tq_matrix.h"
#include "include/tq_perceptron.h"

#define EPOCHS 10
#define GENERAL_TYPE TQ_CPU_Matrix

void init_args(int argc, char **argv,
               unsigned int *n_elems)
{
    if (argc == 3)
    {
        n_elems[0] = atoi(argv[1]);
        n_elems[1] = atoi(argv[2]);
    }
    else
    {
        fprintf(stderr,
                "ERROR: es necesario pasar [1] número de ejemplos y [2] número de características\n");
        exit(1);
    }
}

int main(int argc, char **argv)
{
    unsigned int epoch;

    unsigned int input_dims[2];
    unsigned int dims[2];

    TQ_Perceptron P;

    TQ_Matrix X;
    TQ_Matrix Y;

    init_args(argc, argv, input_dims);

    // Input matrix: random data.
    dims[0] = input_dims[0];
    dims[1] = input_dims[1];
    TQ_Matrix_Create(&X,
                     dims, 2,
                     GENERAL_TYPE);
    TQ_Matrix_Unif(&X);

    printf("X = \n");
    TQ_Matrix_Print(X);

    // Output matrix: random data.
    dims[1] = 1;
    TQ_Matrix_Create(&Y,
                     dims, 2,
                     GENERAL_TYPE);
    TQ_Matrix_Unif(&Y);

    printf("Y = \n");
    TQ_Matrix_Print(Y);

    /**
     * Perceptron create & launch
     */
    printf("Creating perceptron...\n");
    TQ_Perceptron_Create(&P, input_dims[1],
                         GENERAL_TYPE);

    for (epoch = 1; epoch <= EPOCHS; epoch++)
    {
        printf("\n(%d / %d epochs)\n",
               epoch, EPOCHS);

        // Forward pass
        TQ_Perceptron_Forward(X, &P);
        printf("Activation = \n");
        TQ_Matrix_Print(P.activation_v);

        // Backward pass
        TQ_Perceptron_Backward(&P, Y);
        printf("dW = \n");
        TQ_Matrix_Print(P.dW);
    }

    TQ_Matrix_Free(&X);
    TQ_Matrix_Free(&Y);
    TQ_Perceptron_Free(&P);
    return 0;
}