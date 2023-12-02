#include "../include/tq_matrix.h"
#include "../include/tq_print_tensor.h"
#include "../include/tq_indexation.h"

TQ_Matrix *TQ_emptymat(TQ_Tuple *shape, TQ_DTYPE dtype)
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
    new_matrix->data = TQ_emptyvec(n_size, dtype);
    new_matrix->shape = shape;
    new_matrix->dtype = dtype;

    return new_matrix;
}

TQ_Matrix *TQ_newmatrix(void *values, TQ_Tuple *shape, TQ_DTYPE dtype)
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
    new_matrix->data = TQ_newvec(values, n_size, dtype);
    new_matrix->shape = shape;
    new_matrix->dtype = dtype;

    return new_matrix;
}

void *TQ_get_value_matrix(TQ_Matrix *matrix, TQ_Tuple *indices)
{
    size_t flattened_idx;

    // Get the position of the element in the matrix
    flattened_idx = TQ_vect_position_fromtensor_(matrix, indices);

    return TQ_get_value_vector(matrix->data, flattened_idx);
}

void TQ_set_value_matrix(TQ_Matrix *matrix, TQ_Tuple *indices, void *value)
{
    size_t flattened_idx;

    // Get the position of the element in the matrix
    flattened_idx = TQ_vect_position_fromtensor_(matrix, indices);

    // Set the new value in the matrix
    TQ_set_value_vector(matrix->data, flattened_idx, value);
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
