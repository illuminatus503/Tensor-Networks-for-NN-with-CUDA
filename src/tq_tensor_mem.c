#include <stdio.h>
#include <stdlib.h>

#include "../include/tq_tensor_mem.h"

unsigned long __TQ_Tensor_IndexToPos(TQ_Tensor matrix,
                                     unsigned int *indices,
                                     unsigned int num_ind)
{
    unsigned long dims_prod = matrix.length;
    unsigned long i, position = 0;
    for (i = 0; i < matrix.num_dims; i++)
    {
        dims_prod = dims_prod / matrix.dimensions[i];
        position += (dims_prod * indices[i]);
    }

    return position;
}

void __TQ_Tensor_PosToIndex(TQ_Tensor matrix,
                            unsigned int position,
                            unsigned int *indices)
{
    unsigned long i;
    unsigned long dims_prod = matrix.length;
    unsigned int pos = position;

    for (i = 0; i < matrix.num_dims - 1; i++)
    {
        dims_prod = dims_prod / matrix.dimensions[i];
        indices[i] = pos / dims_prod;
        pos = pos % dims_prod;
    }
    indices[matrix.num_dims - 1] = pos;
}

unsigned char __TQ_Tensor_Pos_Is_Valid(TQ_Tensor matrix,
                                       unsigned long pos)
{
    unsigned char is_valid;
    unsigned int i;
    unsigned int indices[matrix.num_dims];
    __TQ_Tensor_PosToIndex(matrix, pos, indices);

    // La posición dada está dentro del rango de la matriz?
    is_valid = 1;

    i = 0;
    while ((i < matrix.num_dims) && is_valid)
    {
        is_valid = (indices[i] >= 0 && indices[i] < matrix.dimensions[i]);
        i++;
    }
    return is_valid;
}

float TQ_Tensor_GetElem(TQ_Tensor matrix,
                        unsigned int *indices,
                        unsigned int num_ind)
{
    // Comprobación de tamaño
    if (num_ind != matrix.num_dims)
    {
        fprintf(stderr,
                "<TQ Index length ERROR> %d != %d.\n",
                num_ind, matrix.num_dims);
        exit(1);
    }

    // Comprobación de índices
    unsigned long i;
    for (i = 0; i < num_ind; i++)
    {
        if (indices[i] >= matrix.dimensions[i])
        {
            fprintf(stderr,
                    "<TQ Index ERROR> %d >= %d.\n",
                    indices[i], matrix.dimensions[i]);
            exit(1);
        }
    }

    return matrix.mem[__TQ_Tensor_IndexToPos(matrix, indices, num_ind)];
}

void TQ_Tensor_SetElem(TQ_Tensor *matrix,
                       float value,
                       unsigned int *indices,
                       unsigned int num_ind)
{
    // Comprobación de tamaño
    if (num_ind != matrix->num_dims)
    {
        fprintf(stderr,
                "<TQ Index length ERROR> %d != %d.\n",
                num_ind, matrix->num_dims);
        exit(1);
    }

    // Comprobación de índices
    unsigned long i;
    for (i = 0; i < num_ind; i++)
    {
        if (indices[i] >= matrix->dimensions[i])
        {
            fprintf(stderr,
                    "<TQ Index ERROR> %d >= %d.\n",
                    indices[i], matrix->dimensions[i]);
            exit(1);
        }
    }

    // Colocar un valor en la matriz.
    matrix->mem[__TQ_Tensor_IndexToPos(*matrix, indices, num_ind)] = value;
}

/**
 * LA CREACIÓN & DESTRUCCIÓN
 */
void TQ_Matrix_Create(TQ_Tensor *matrix,
                      unsigned int *dimensions,
                      unsigned int num_dims,
                      TQ_Tensor_type type)
{
    size_t i;
    size_t total_size = 1;

    // Guardamos el tipo de la nueva matriz.
    matrix->type = type;

    // Guardar las dimensiones de la nueva matriz.
    matrix->dimensions = (unsigned int *)malloc(num_dims * sizeof(unsigned int));
    matrix->num_dims = num_dims;
    for (i = 0; i < num_dims; i++)
    {
        matrix->dimensions[i] = dimensions[i];
        total_size *= (unsigned long)dimensions[i];
    }

    // Guardamos el tamaño total de la matriz en bytes (float).
    // Esto será útil cuando haya que reservar memoria en Device o Host.
    matrix->length_bytes = total_size * sizeof(float);

    // prod = PROD. (i = 0..k) matrix.dims[i] (para el cálculo de índices)
    matrix->length = total_size;

    // Reservamos la memoria para los datos de la matriz.
    matrix->mem = (float *)malloc(matrix->length_bytes);
}

void TQ_Matrix_Clone(TQ_Tensor input,
                     TQ_Tensor *output)
{
    unsigned int i;

    TQ_Matrix_Create(output, input.dimensions, input.num_dims, input.type);
    for (i = 0; i < input.length; i++)
    {
        output->mem[i] = input.mem[i];
    }
}

void TQ_Matrix_CopyData(TQ_Tensor input,
                        TQ_Tensor *output)
{
    unsigned int i;

    if (input.length != output->length)
    {
        fprintf(stderr,
                "<TQ CopyData> INPUT (num_elemns) != OUTPUT\n");
        exit(1);
    }

    for (i = 0; i < input.length; i++)
    {
        output->mem[i] = input.mem[i];
    }
}

void TQ_Matrix_Extend(TQ_Tensor input,
                      TQ_Tensor *output,
                      unsigned int *new_dims,
                      unsigned int num_dims,
                      float fill_val)
{
    unsigned long i;

    unsigned long new_dims_prod = 1;
    for (i = 0; i < num_dims; i++)
    {
        new_dims_prod *= new_dims[i];
    }

    if (new_dims_prod < input.length)
    {
        fprintf(stderr,
                "<TQ Matrix Reshape> Unable to reshape matrix: %lu != %lu\n",
                input.length, new_dims_prod);
        exit(1);
    }

    TQ_Matrix_Create(output, new_dims, num_dims, input.type);

    for (i = 0; i < new_dims_prod; i++)
    {
        if (__TQ_Tensor_Pos_Is_Valid(input, i))
        {
            output->mem[i] = input.mem[i];
        }
        else
        {
            output->mem[i] = fill_val;
        }
    }
}

void __TQ_Matrix_Print(float *tensor,
                       unsigned int *dims,
                       unsigned int ndims,
                       unsigned int depth)
{
    unsigned int i, j;

    if (depth == ndims - 1)
    {
        printf("[");
        for (i = 0; i < dims[depth]; i++)
        {
            printf("%1.3f ", *tensor);
            if (i < dims[depth] - 1)
            {
                printf(", ");
            }

            tensor++;
        }
        printf("]");
    }
    else
    {
        printf("[");
        for (i = 0; i < dims[depth]; i++)
        {
            __TQ_Matrix_Print(tensor, dims, ndims, depth + 1);
            if (i < dims[depth] - 1)
            {
                printf(",\n");
                for (j = 0; j <= depth; j++)
                {
                    printf(" ");
                }
            }

            tensor += dims[depth + 1];
        }
        printf("]");
    }
}

void TQ_Matrix_Print(TQ_Tensor matrix)
{
    __TQ_Matrix_Print((float *)matrix.mem,
                      matrix.dimensions, matrix.num_dims, 0);
    printf("\n");
}

void TQ_Matrix_Free(TQ_Tensor *matrix)
{
    free(matrix->mem);
    free(matrix->dimensions);
}