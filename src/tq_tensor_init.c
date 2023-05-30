#include <stdlib.h>

#include "../include/tq_tensor_init.h"
#include "../include/tq_tensor_mem.h"
#include "../include/tq_math.h"

void TQ_Matrix_ValueInit(TQ_Tensor *matrix, float value)
{
    unsigned int i;
    for (i = 0; i < matrix->length; i++)
    {
        matrix->mem[i] = value;
    }
}

void TQ_Matrix_Eyes(TQ_Tensor *matrix)
{
    unsigned int i, j;
    unsigned int indices[matrix->num_dims];

    for (i = 0; i < matrix->length; i++)
    {
        __TQ_Tensor_PosToIndex(*matrix, i, indices);
        
        j = 1;
        while ((j < matrix->num_dims) && (indices[j] == indices[0]))
        {
            j++;
        }

        if (j == matrix->num_dims)
        {
            matrix->mem[i] = 1.0f;
        }
        else
        {
            matrix->mem[i] = 0.0f;
        }
    }
}

void TQ_Matrix_Ones(TQ_Tensor *matrix)
{
    unsigned int i;
    for (i = 0; i < matrix->length; i++)
    {
        matrix->mem[i] = 1.0f;
    }
}

void TQ_Matrix_Zeros(TQ_Tensor *matrix)
{
    unsigned int i;
    for (i = 0; i < matrix->length; i++)
    {
        matrix->mem[i] = 0.0f;
    }
}

void TQ_Matrix_Rand(TQ_Tensor *matrix, float min, float max)
{
    unsigned int i;
    for (i = 0; i < matrix->length; i++)
    {
        matrix->mem[i] = TQ_rand_unif(min, max);
    }
}

void TQ_Matrix_Unif(TQ_Tensor *matrix)
{
    TQ_Matrix_Rand(matrix, 0.0, 1.0);
}

void TQ_Matrix_Normal(TQ_Tensor *matrix,
                      float mu,
                      float sigma)
{
    unsigned int i;
    for (i = 0; i < matrix->length; i++)
    {
        matrix->mem[i] = TQ_rand_norm(mu, sigma);
    }
}