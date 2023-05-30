#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/tq_tensor.h"

unsigned char __TQ_Send_To_CPU(TQ_Tensor one,
                               TQ_Tensor other)
{
    return ((one.type != TQ_GPU_Matrix) || (other.type != TQ_GPU_Matrix));
}

void __TQ_MatAdd_TEST(TQ_Tensor one,
                      TQ_Tensor other)
{
    // Mismo número de elementos?
    if (one.num_dims != other.num_dims)
    {
        fprintf(stderr,
                "<TQ Matrix Dims SIZE ERROR> %ul != %ul\n",
                one.num_dims, other.num_dims);
        exit(1);
    }

    // Mismas dimensiones?
    unsigned int i;
    for (i = 0; i < one.num_dims; i++)
    {
        if (one.dimensions[i] != other.dimensions[i])
        {
            fprintf(stderr,
                    "<TQ Matrix Dims ERROR> %ul != %ul\n",
                    one.dimensions[i], other.dimensions[i]);
            exit(1);
        }
    }

    // Everything OK!
}

void TQ_Matrix_Add(TQ_Tensor one,
                   TQ_Tensor other,
                   TQ_Tensor *result)
{
    __TQ_MatAdd_TEST(one, other);
    TQ_Matrix_Create(result, one.dimensions, one.num_dims, one.type);

    if (__TQ_Send_To_CPU(one, other))
    {
        TQ_CPUMat_Add(one, other, result);
    }
    else
    {
        TQ_GPUMat_Add(one, other, result);
    }
}

void TQ_Matrix_Sub(TQ_Tensor one,
                   TQ_Tensor other,
                   TQ_Tensor *result)
{
    __TQ_MatAdd_TEST(one, other);
    TQ_Matrix_Create(result, one.dimensions, one.num_dims, one.type);

    if (__TQ_Send_To_CPU(one, other))
    {
        TQ_CPUMat_Sub(one, other, result);
    }
    else
    {
        TQ_GPUMat_Sub(one, other, result);
    }
}

void __TQ_MatProd_TEST(TQ_Tensor one,
                       TQ_Tensor other)
{
    if (one.num_dims != 2)
    {
        fprintf(stderr,
                "<TQ MatProd (!2) ERROR> ONE dims.\n");
        exit(1);
    }

    if (other.num_dims != 2)
    {
        fprintf(stderr,
                "<TQ MatProd (!2) ERROR> OTHER dims.\n");
        exit(1);
    }

    // Comprobación de las dimensiones:
    if (one.dimensions[1] != other.dimensions[0])
    {
        fprintf(stderr,
                "<TQ MatProd ERROR> (%d, %d) & (%d, %d)\n",
                one.dimensions[0], one.dimensions[1],
                other.dimensions[0], other.dimensions[1]);
        exit(1);
    }

    // OK!
}

void TQ_Matrix_Prod(TQ_Tensor one,
                    TQ_Tensor other,
                    TQ_Tensor *result)
{
    enum TQ_Tensor_type type;
    unsigned char to_cpu;

    __TQ_MatProd_TEST(one, other);
    to_cpu = __TQ_Send_To_CPU(one, other);

    if (to_cpu)
    {
        type = TQ_CPU_Matrix;
    }
    else
    {
        type = TQ_GPU_Matrix;
    }

    // Reservamos memoria.
    unsigned int dimensions[] = {one.dimensions[0], other.dimensions[1]};
    TQ_Matrix_Create(result, dimensions, 2, type);

    // Aplicamos el producto de matrices
    if (to_cpu)
    {
        TQ_CPUMat_Prod(one, other, result);
    }
    else
    {
        TQ_GPUMat_Prod(one, other, result);
    }
}

void TQ_Matrix_ProdNum(TQ_Tensor one,
                       float factor,
                       TQ_Tensor *result)
{
    TQ_Matrix_Create(result, one.dimensions, one.num_dims, one.type);

    if (one.type == TQ_GPU_Matrix)
    {
        TQ_GPUMat_ProdNum(one, factor, result);
    }
    else
    {
        TQ_CPUMat_ProdNum(one, factor, result); // Por defecto, será más rápido.
    }
}

void TQ_Matrix_T(TQ_Tensor input,
                 TQ_Tensor *output)
{
    if (input.num_dims != 2)
    {
        fprintf(stderr,
                "<TQ TQ_matrix_T> Num dims %d != 2\n",
                input.num_dims);
        exit(1);
    }

    unsigned int i, j;
    unsigned int dims[2] = {input.dimensions[1], input.dimensions[0]};

    // TODO Traspuesta en GPU
    // Traspuesta en CPU (localidad)
    TQ_Matrix_Create(output, dims, 2, input.type);

    for (j = 0; j < input.dimensions[0]; j++)
    {
        for (i = 0; i < input.dimensions[1]; i++)
        {
            output->mem[i * dims[0] + j] = input.mem[j * dims[0] + i];
        }
    }
}

void TQ_Matrix_Apply(TQ_Tensor input,
                     float (*function)(float),
                     TQ_Tensor *output)
{
    unsigned int i;
    TQ_Matrix_Create(output, input.dimensions, input.num_dims, input.type);

    for (i = 0; i < input.length; i++)
    {
        output->mem[i] = function(input.mem[i]);
    }
}