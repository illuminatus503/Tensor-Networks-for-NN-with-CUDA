#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/tq_matrix.h"

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

void TQ_Matrix_Init(TQ_Tensor *matrix, float value)
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
    srand(time(NULL));

    float rand_val;
    float range = max - min;

    for (i = 0; i < matrix->length; i++)
    {
        rand_val = (float)rand() / (float)RAND_MAX;
        matrix->mem[i] = rand_val * range + min;
    }
}

void TQ_Matrix_Unif(TQ_Tensor *matrix)
{
    unsigned int i;
    srand(time(NULL));

    for (i = 0; i < matrix->length; i++)
    {
        matrix->mem[i] = (float)rand() / (float)RAND_MAX;
    }
}

void TQ_Matrix_Free(TQ_Tensor *matrix)
{
    free(matrix->mem);
    free(matrix->dimensions);
}

/**
 * OPERACIONES CON MATRICES
 */

unsigned char __TQ_Send_To_CPU(TQ_Tensor one,
                               TQ_Tensor other)
{
    unsigned char to_cpu;
    to_cpu = ((one.type != TQ_GPU_Matrix) || (other.type != TQ_GPU_Matrix));
    return to_cpu;
}

void __TQ_VecDot_TEST(TQ_Tensor one,
                      TQ_Tensor other)
{
    if (one.num_dims != 1)
    {
        fprintf(stderr,
                "<TQ VecDot (!1) ERROR> ONE dims.\n");
        exit(1);
    }

    if (other.num_dims != 1)
    {
        fprintf(stderr,
                "<TQ VecDot (!1) ERROR> OTHER dims.\n");
        exit(1);
    }

    // Comprobación de las dimensiones:
    if (one.dimensions[0] != other.dimensions[0])
    {
        fprintf(stderr,
                "<TQ VecDot ERROR> %d != %d\n",
                one.dimensions[0], other.dimensions[0]);
        exit(1);
    }
}

void TQ_Vec_Dot(TQ_Tensor one,
                TQ_Tensor other,
                float *result)
{
    __TQ_VecDot_TEST(one, other);

    if (__TQ_Send_To_CPU(one, other))
    {
        __TQ_CPUVec_Dot(one, other, result);
    }
    else
    {
        __TQ_GPUVec_Dot(one, other, result);
    }
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
        __TQ_CPUMat_Add(one, other, result);
    }
    else
    {
        __TQ_GPUMat_Add(one, other, result);
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
        __TQ_CPUMat_Sub(one, other, result);
    }
    else
    {
        __TQ_GPUMat_Sub(one, other, result);
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
        __TQ_CPUMat_Prod(one, other, result);
    }
    else
    {
        __TQ_GPUMat_Prod(one, other, result);
    }
}

void TQ_Matrix_ProdNum(TQ_Tensor one,
                       float factor,
                       TQ_Tensor *result)
{
    TQ_Matrix_Create(result, one.dimensions, one.num_dims, one.type);

    if (one.type == TQ_GPU_Matrix)
    {
        __TQ_GPUMat_ProdNum(one, factor, result);
    }
    else
    {
        __TQ_CPUMat_ProdNum(one, factor, result); // Por defecto, será más rápido.
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