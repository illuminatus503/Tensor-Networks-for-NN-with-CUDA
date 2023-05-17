#include <stdio.h>
#include <stdlib.h>

#include "../include/__tq_op_cpu.h"

unsigned long __TQ_Matrix_IndexToPos(struct TQ_Matrix matrix,
                                     unsigned int *indices,
                                     unsigned int num_ind)
{
    unsigned long dims_prod = matrix.dims_prod;
    unsigned long i, position = 0;
    for (i = 0; i < matrix.num_dims; i++)
    {
        dims_prod = dims_prod / matrix.dimensions[i];
        position += (dims_prod * indices[i]);
    }

    return position;
}

void __TQ_Matrix_PosToIndex(struct TQ_Matrix matrix,
                            unsigned int position,
                            unsigned int *indices)
{
    unsigned long i;
    unsigned long dims_prod = matrix.dims_prod;
    unsigned int pos = position;

    for (i = 0; i < matrix.num_dims - 1; i++)
    {
        dims_prod = dims_prod / matrix.dimensions[i];
        indices[i] = pos / dims_prod;
        pos = pos % dims_prod;
    }
    indices[matrix.num_dims - 1] = pos;
}

float TQ_Matrix_GetElem(struct TQ_Matrix matrix,
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

    return matrix.h_mem[__TQ_Matrix_IndexToPos(matrix, indices, num_ind)];
}

void TQ_Matrix_SetElem(struct TQ_Matrix *matrix,
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
    matrix->h_mem[__TQ_Matrix_IndexToPos(*matrix, indices, num_ind)] = value;
}

void __TQ_CPUMat_Add(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result)
{
    unsigned int i;
    for (i = 0; i < one.dims_prod; i++)
    {
        result->h_mem[i] = one.h_mem[i] + other.h_mem[i];
    }
}

void __TQ_CPUMat_Sub(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result)
{
    unsigned int i;
    for (i = 0; i < one.dims_prod; i++)
    {
        result->h_mem[i] = one.h_mem[i] - other.h_mem[i];
    }
}

void __TQ_CPUMat_ProdNum(struct TQ_Matrix one,
                         float factor,
                         struct TQ_Matrix *result)
{
    unsigned int i;
    for (i = 0; i < one.dims_prod; i++)
    {
        result->h_mem[i] = factor * one.h_mem[i];
    }
}

void __TQ_CPUMat_Prod(struct TQ_Matrix one,
                      struct TQ_Matrix other,
                      struct TQ_Matrix *result)
{
    unsigned int i, j, k;

    float suma, elem1, elem2;
    unsigned int idx[2];

    for (i = 0; i < result->dimensions[0]; i++)
    {
        for (j = 0; j < result->dimensions[1]; j++)
        {
            suma = 0;
            for (k = 0; k < one.dimensions[1]; k++)
            {
                idx[0] = i;
                idx[1] = k;
                elem1 = TQ_Matrix_GetElem(one, idx, 2);

                idx[0] = k;
                idx[1] = j;
                elem2 = TQ_Matrix_GetElem(other, idx, 2);

                // Producto + Suma
                suma += (elem1 * elem2);
            }

            // Guardamos el resultado en la matriz.
            idx[0] = i;
            idx[1] = j;
            TQ_Matrix_SetElem(result, suma, idx, 2);
        }
    }
}

void __TQ_CPUVec_Dot(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     float *result)
{
    unsigned int i;
    result[0] = 0.0f;

    for (i = 0; i < one.dims_prod; i++) // Una dimensión = longitud
    {
        result[0] += (one.h_mem[i] * other.h_mem[i]);
    }
}
