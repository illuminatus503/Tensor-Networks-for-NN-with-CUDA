#include <stdio.h>
#include <stdlib.h>

#include "../include/tq_tensor_cpu.h"
#include "../include/tq_tensor_mem.h"

void TQ_CPUMat_Add(TQ_Tensor one,
                   TQ_Tensor other,
                   TQ_Tensor *result)
{
    unsigned int i;
    for (i = 0; i < one.length; i++)
    {
        result->mem[i] = one.mem[i] + other.mem[i];
    }
}

void TQ_CPUMat_Sub(TQ_Tensor one,
                   TQ_Tensor other,
                   TQ_Tensor *result)
{
    unsigned int i;
    for (i = 0; i < one.length; i++)
    {
        result->mem[i] = one.mem[i] - other.mem[i];
    }
}

void TQ_CPUMat_ProdNum(TQ_Tensor one,
                       float factor,
                       TQ_Tensor *result)
{
    unsigned int i;
    for (i = 0; i < one.length; i++)
    {
        result->mem[i] = factor * one.mem[i];
    }
}

void TQ_CPUMat_Prod(TQ_Tensor one,
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
