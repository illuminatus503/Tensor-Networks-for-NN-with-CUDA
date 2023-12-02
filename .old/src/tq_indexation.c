#include "../include/tq_indexation.h"

#include <stdio.h>

size_t TQ_vect_position_fromtensor_(TQ_Matrix *matrix, TQ_Tuple *indices)
{
    size_t i;
    size_t tensor_dim_size;
    size_t tensor_index;

    size_t n_elems = matrix->data->n_size;
    size_t position = 0;

    for (i = 0; i < matrix->shape->n_size; i++)
    {
        // Calculate the offset in the dimension
        tensor_dim_size = *(size_t *)TQ_get_value_tuple(matrix->shape, i);
        n_elems /= tensor_dim_size;

        // Calculate the index in the dimension
        tensor_index = *(size_t *)TQ_get_value_tuple(indices, i);
        position += (n_elems * tensor_index);
    }

    // printf("DEBUG: position -  %ld\n", position);

    return position;
}

TQ_Tuple *TQ_indices_fromvect_pos_(TQ_Matrix *matrix, size_t position)
{
    size_t i;
    size_t tensor_dim_size;

    size_t n_elems = matrix->data->n_size;
    size_t auxiliar_position = position;

    // Allocate space for the new indices tuple
    TQ_Tuple *indices;
    size_t *indices_array = (size_t *)malloc(sizeof(size_t) * matrix->shape->n_size);

    for (i = 0; i < matrix->shape->n_size - 1; i++)
    {
        // Calculate the offset in the dimension
        tensor_dim_size = *(size_t *)TQ_get_value_tuple(matrix->shape, i);
        n_elems /= tensor_dim_size;

        // Update positions and get the new coordinate
        indices_array[i] = n_elems / n_elems;
        auxiliar_position = auxiliar_position % n_elems;
    }
    indices_array[matrix->shape->n_size - 1] = auxiliar_position;

    indices = TQ_newtuple((void *)indices_array, matrix->shape->n_size, TQ_ULONG);
    free(indices_array);

    return indices;
}

// unsigned char __TQ_Matrix_Pos_Is_Valid(struct TQ_Matrix matrix,
//                                        unsigned long pos)
// {
//     unsigned char is_valid;
//     unsigned int i;
//     unsigned int indices[matrix.num_dims];
//     __TQ_Matrix_PosToIndex(matrix, pos, indices);

//     // La posición dada está dentro del rango de la matriz?
//     is_valid = 1;

//     i = 0;
//     while ((i < matrix.num_dims) && is_valid)
//     {
//         is_valid = (indices[i] >= 0 && indices[i] < matrix.dimensions[i]);
//         i++;
//     }
//     return is_valid;
// }

// float TQ_Matrix_GetElem(struct TQ_Matrix matrix,
//                         unsigned int *indices,
//                         unsigned int num_ind)
// {
//     // Comprobación de tamaño
//     if (num_ind != matrix.num_dims)
//     {
//         fprintf(stderr,
//                 "<TQ Index length ERROR> %d != %d.\n",
//                 num_ind, matrix.num_dims);
//         exit(1);
//     }

//     // Comprobación de índices
//     unsigned long i;
//     for (i = 0; i < num_ind; i++)
//     {
//         if (indices[i] >= matrix.dimensions[i])
//         {
//             fprintf(stderr,
//                     "<TQ Index ERROR> %d >= %d.\n",
//                     indices[i], matrix.dimensions[i]);
//             exit(1);
//         }
//     }

//     return matrix.h_mem[__TQ_Matrix_IndexToPos(matrix, indices, num_ind)];
// }

// void TQ_Matrix_SetElem(struct TQ_Matrix *matrix,
//                        float value,
//                        unsigned int *indices,
//                        unsigned int num_ind)
// {
//     // Comprobación de tamaño
//     if (num_ind != matrix->num_dims)
//     {
//         fprintf(stderr,
//                 "<TQ Index length ERROR> %d != %d.\n",
//                 num_ind, matrix->num_dims);
//         exit(1);
//     }

//     // Comprobación de índices
//     unsigned long i;
//     for (i = 0; i < num_ind; i++)
//     {
//         if (indices[i] >= matrix->dimensions[i])
//         {
//             fprintf(stderr,
//                     "<TQ Index ERROR> %d >= %d.\n",
//                     indices[i], matrix->dimensions[i]);
//             exit(1);
//         }
//     }

//     // Colocar un valor en la matriz.
//     matrix->h_mem[__TQ_Matrix_IndexToPos(*matrix, indices, num_ind)] = value;
// }