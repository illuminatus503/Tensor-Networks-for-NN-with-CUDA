#include "../include/tq_matrix.h"
#include "../include/tq_print_tensor.h"

TQ_Matrix *TQ_create_empty_matrix(TQ_Tuple *shape, TQ_DTYPE dtype)
{
    size_t i;
    size_t n_size;
    TQ_Matrix *new_matrix = (TQ_Matrix *)malloc(sizeof(TQ_Matrix));

    // Calculate the size of the data vector, from that shape.
    n_size = 1;
    for (i = 0; i < shape->n_size; i++)
    {
        n_size *= (size_t)(*((int *)TQ_get_value_tuple(shape, i)));
    }

    // Declare & init the new matrix
    new_matrix->data = TQ_create_empty_vector(n_size, dtype);
    new_matrix->shape = shape;
    new_matrix->dtype = dtype;

    return new_matrix;
}

TQ_Matrix *TQ_create_matrix_from_array(void *values, TQ_Tuple *shape, TQ_DTYPE dtype)
{
    size_t i;
    size_t n_size;
    TQ_Matrix *new_matrix = (TQ_Matrix *)malloc(sizeof(TQ_Matrix));

    // Calculate the size of the data vector, from that shape.
    n_size = 1;
    for (i = 0; i < shape->n_size; i++)
    {
        n_size *= (size_t)(*((int *)TQ_get_value_tuple(shape, i)));
    }

    // Declare & init the new matrix
    new_matrix->data = TQ_create_vector_from_array(values, n_size, dtype);
    new_matrix->shape = shape;
    new_matrix->dtype = dtype;

    return new_matrix;
}

void *TQ_get_value_matrix(TQ_Matrix *matrix, TQ_Tuple *indexes)
{
    return NULL;
}

void TQ_set_value_matrix(TQ_Matrix *matrix, TQ_Tuple *indexes, void *value)
{
}

void TQ_print_matrix(TQ_Matrix *matrix)
{
    TQ_print_tensor(matrix, BEGIN_MATRIX, END_MATRIX);
}

void TQ_delete_matrix(TQ_Matrix **matrix)
{
    TQ_delete_vector(&((*matrix)->data));
    TQ_delete_tuple(&((*matrix)->shape));
    free(*matrix);
}
