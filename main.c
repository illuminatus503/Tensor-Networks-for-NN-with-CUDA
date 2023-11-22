#include <stdio.h>
#include <stdlib.h>

#include "include/tq_matrix.h"

int main(int argc, char **argv)
{
    TQ_Matrix *matrix = TQ_create_empty_matrix(TQ_create_tuple_from_array((long[]){2, 1}, 2, TQ_LONG), // inline init of a tuple
                                               TQ_FLOAT);
    printf("Se ha creado una matriz\n");

    printf("Imprimimos la matrix: \n");

    printf("Forma vector: \n");
    TQ_print_vector(matrix->data);

    printf("Forma tensor: \n");
    TQ_print_matrix(matrix);

    TQ_delete_matrix(&matrix);
    printf("Se ha eliminado una matriz\n");

    matrix = TQ_create_matrix_from_array((double[]){3.141592, 2.718281, 2.718281, 3.141592},     // inline init of an array
                                         TQ_create_tuple_from_array((long[]){2, 2}, 2, TQ_LONG), // inline init of a tuple: shape
                                         TQ_DOUBLE);
    printf("Creamos una matriz a partir de un vector plano:\n");
    TQ_print_matrix(matrix);

    TQ_delete_matrix(&matrix);
    printf("Se ha eliminado una matriz\n");

    matrix = TQ_create_matrix_from_array((double[]){3.141592, 2.718281, 2.718281, 3.141592, 3.141592, 2.718281, 2.718281, 3.141592}, // inline init of an array
                                         TQ_create_tuple_from_array((long[]){2, 2, 2}, 3, TQ_LONG),                                  // inline init of a tuple: shape
                                         TQ_DOUBLE);
    printf("Creamos una matriz a partir de un vector plano:\n");
    TQ_print_matrix(matrix);

    TQ_delete_matrix(&matrix);
    printf("Se ha eliminado una matriz\n");

    matrix = TQ_create_matrix_from_array((double[]){3.141592, 2.718281},                            // inline init of an array
                                         TQ_create_tuple_from_array((long[]){1, 2, 1}, 3, TQ_LONG), // inline init of a tuple: shape
                                         TQ_DOUBLE);
    printf("Creamos una matriz a partir de un vector plano:\n");
    TQ_print_matrix(matrix);

    TQ_delete_matrix(&matrix);
    printf("Se ha eliminado una matriz\n");

    // TQ_Vector *vector = TQ_create_empty_vector(10, TQ_INT);
    // printf("Se ha generado un vector\n");

    // TQ_print_vector(vector);

    // // insertamos 3 en el índice 3
    // int integer_val = 3;
    // TQ_set_value_vector(vector, 3, (void *)&integer_val);
    // TQ_print_vector(vector);

    // printf("Se ha insertado el valor %d en el vector\n",
    //        *(int *)TQ_get_value_vector(vector, 3));

    // TQ_delete_vector(&vector);
    // printf("Se ha borrado un vector\n");

    // float values[5] = {0.3, 0.2, 0.1, 0.0, -0.1};
    // vector = TQ_create_vector_from_array((void *)values, 5, TQ_FLOAT);
    // TQ_print_vector(vector);
    // TQ_delete_vector(&vector);

    return 0;
}