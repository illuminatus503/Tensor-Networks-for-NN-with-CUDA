#include <stdio.h>
#include <stdlib.h>

#include "../include/tq_tensor_cpu.h"

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

void __TQ_CPUMat_Add(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result)
{
    unsigned int i;
    for (i = 0; i < one.length; i++)
    {
        result->mem[i] = one.mem[i] + other.mem[i];
    }
}

void __TQ_CPUMat_Sub(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result)
{
    unsigned int i;
    for (i = 0; i < one.length; i++)
    {
        result->mem[i] = one.mem[i] - other.mem[i];
    }
}

void __TQ_CPUMat_ProdNum(TQ_Tensor one,
                         float factor,
                         TQ_Tensor *result)
{
    unsigned int i;
    for (i = 0; i < one.length; i++)
    {
        result->mem[i] = factor * one.mem[i];
    }
}

void __TQ_CPUMat_Prod(TQ_Tensor one,
                      TQ_Tensor other,
                      TQ_Tensor *result)
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
                elem1 = TQ_Tensor_GetElem(one, idx, 2);

                idx[0] = k;
                idx[1] = j;
                elem2 = TQ_Tensor_GetElem(other, idx, 2);

                // Producto + Suma
                suma += (elem1 * elem2);
            }

            // Guardamos el resultado en la matriz.
            idx[0] = i;
            idx[1] = j;
            TQ_Tensor_SetElem(result, suma, idx, 2);
        }
    }
}

void __TQ_CPUVec_Dot(TQ_Tensor one,
                     TQ_Tensor other,
                     float *result)
{
    unsigned int i;
    result[0] = 0.0f;

    for (i = 0; i < one.length; i++) // Una dimensión = longitud
    {
        result[0] += (one.mem[i] * other.mem[i]);
    }
}
