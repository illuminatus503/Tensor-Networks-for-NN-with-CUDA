#include <stdio.h>
#include <stdlib.h>

#include "include/tq_matrix.h"

int main(int argc, char **argv)
{
    TQ_Matrix *matrix = TQ_emptymat(TQ_newtuple((unsigned long[]){2, 1}, 2, TQ_ULONG), // inline init of a tuple
                                    TQ_FLOAT);
    printf("Se ha creado una matriz\n");

    printf("Imprimimos la matrix: \n");

    printf("Forma vector: \n");
    TQ_print_vector(matrix->data);

    printf("Forma tensor: \n");
    TQ_print_matrix(matrix);

    TQ_delete_matrix(&matrix);
    printf("Se ha eliminado una matriz\n");

    matrix = TQ_newmatrix((double[]){3.141592, 2.718281, 2.718281, 3.141592}, // inline init of an array
                          TQ_newtuple((unsigned long[]){2, 2}, 2, TQ_ULONG),  // inline init of a tuple: shape
                          TQ_DOUBLE);
    printf("Creamos una matriz a partir de un vector plano:\n");
    TQ_print_matrix(matrix);

    double value = 6.0;
    TQ_set_value_matrix(matrix,
                        TQ_newtuple((unsigned long[]){0, 1}, 2, TQ_ULONG),
                        (void *)&value);

    printf("Se ha insertado un valor en (0, 1) de la matriz:\n");
    TQ_print_matrix(matrix);

    value = -7.83;
    TQ_set_value_matrix(matrix,
                        TQ_newtuple((unsigned long[]){1, 1}, 2, TQ_ULONG),
                        (void *)&value);

    printf("Se ha insertado un valor en (1, 1) de la matriz:\n");
    TQ_print_matrix(matrix);

    double new_val = *(double *)TQ_get_value_matrix(matrix,
                                                    TQ_newtuple((unsigned long[]){0, 0}, 2, TQ_ULONG));
    printf("Se ha recogido el valor en (0, 0) de la matriz: %3.3lf\n", new_val);

    new_val = *(double *)TQ_get_value_matrix(matrix,
                                             TQ_newtuple((unsigned long[]){0, 1}, 2, TQ_ULONG));
    printf("Se ha recogido el valor en (0, 1) de la matriz: %3.3lf\n", new_val);

    new_val = *(double *)TQ_get_value_matrix(matrix,
                                             TQ_newtuple((unsigned long[]){1, 0}, 2, TQ_ULONG));
    printf("Se ha recogido el valor en (1, 0) de la matriz: %3.3lf\n", new_val);

    TQ_delete_matrix(&matrix);
    printf("Se ha eliminado una matriz\n");

    return 0;
}