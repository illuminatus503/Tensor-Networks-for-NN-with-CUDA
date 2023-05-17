#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/tq_matrix.h"

/**
 * LA CREACIÓN & DESTRUCCIÓN
 */
void TQ_Matrix_Create(struct TQ_Matrix *matrix,
                      unsigned int *dimensions,
                      unsigned int num_dims,
                      TQ_Matrix_t type)
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
    matrix->dims_prod = total_size;

    // Reservamos la memoria para los datos de la matriz.
    matrix->h_mem = (float *)malloc(matrix->length_bytes);
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

void TQ_Matrix_Print(struct TQ_Matrix matrix)
{
    __TQ_Matrix_Print((float *)matrix.h_mem,
                      matrix.dimensions, matrix.num_dims, 0);
    printf("\n");
}

void TQ_Matrix_Init(struct TQ_Matrix *matrix, float value)
{
    unsigned int i;
    for (i = 0; i < matrix->dims_prod; i++)
    {
        matrix->h_mem[i] = value;
    }
}

void TQ_Matrix_Eyes(struct TQ_Matrix *matrix)
{
    unsigned int i, j;
    unsigned int indices[matrix->num_dims];

    for (i = 0; i < matrix->dims_prod; i++)
    {
        __TQ_Matrix_PosToIndex(*matrix, i, indices);
        j = 1;
        while ((j < matrix->num_dims) && (indices[j] == indices[0]))
        {
            j++;
        }

        if (j == matrix->num_dims)
        {
            matrix->h_mem[i] = 1.0f;
        }
        else
        {
            matrix->h_mem[i] = 0.0f;
        }
    }
}

void TQ_Matrix_Ones(struct TQ_Matrix *matrix)
{
    unsigned int i;
    for (i = 0; i < matrix->dims_prod; i++)
    {
        matrix->h_mem[i] = 1.0f;
    }
}

void TQ_Matrix_Zeros(struct TQ_Matrix *matrix)
{
    unsigned int i;
    for (i = 0; i < matrix->dims_prod; i++)
    {
        matrix->h_mem[i] = 0.0f;
    }
}

void TQ_Matrix_Rand(struct TQ_Matrix *matrix, float min, float max)
{
    unsigned int i;
    srand(time(NULL));

    float rand_val;
    float range = max - min;

    for (i = 0; i < matrix->dims_prod; i++)
    {
        rand_val = (float)rand() / (float)RAND_MAX;
        matrix->h_mem[i] = rand_val * range + min;
    }
}

void TQ_Matrix_Unif(struct TQ_Matrix *matrix)
{
    unsigned int i;
    srand(time(NULL));

    for (i = 0; i < matrix->dims_prod; i++)
    {
        matrix->h_mem[i] = (float)rand() / (float)RAND_MAX;
    }
}

void TQ_Matrix_Free(struct TQ_Matrix *matrix)
{
    free(matrix->h_mem);
    free(matrix->dimensions);
}

/**
 * OPERACIONES CON MATRICES
 */

unsigned char __TQ_Send_To_CPU(struct TQ_Matrix one,
                               struct TQ_Matrix other)
{
    unsigned char to_cpu;
    to_cpu = ((one.type != TQ_GPU_Matrix) || (other.type != TQ_GPU_Matrix));
    return to_cpu;
}

void __TQ_MatAdd_TEST(struct TQ_Matrix one,
                      struct TQ_Matrix other)
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

void TQ_Matrix_Add(struct TQ_Matrix one,
                   struct TQ_Matrix other,
                   struct TQ_Matrix *result)
{
    __TQ_MatAdd_TEST(one, other);
    if (__TQ_Send_To_CPU(one, other))
    {
        __TQ_CPUMat_Add(one, other, result);
    }
    else
    {
        __TQ_GPUMat_Add(one, other, result);
    }
}

void TQ_Matrix_Sub(struct TQ_Matrix one,
                   struct TQ_Matrix other,
                   struct TQ_Matrix *result)
{
    __TQ_MatAdd_TEST(one, other);
    if (__TQ_Send_To_CPU(one, other))
    {
        __TQ_CPUMat_Sub(one, other, result);
    }
    else
    {
        __TQ_GPUMat_Sub(one, other, result);
    }
}

void __TQ_MatProd_TEST(struct TQ_Matrix one,
                       struct TQ_Matrix other)
{
    if (one.num_dims != 2)
    {
        fprintf(stderr,
                "<TQ MatProd (!2) ERROR> ONE dims.");
        exit(1);
    }

    if (other.num_dims != 2)
    {
        fprintf(stderr,
                "<TQ MatProd (!2) ERROR> OTHER dims.");
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

void TQ_Matrix_Prod(struct TQ_Matrix one,
                    struct TQ_Matrix other,
                    struct TQ_Matrix *result)
{
    enum TQ_Matrix_type type;
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

void TQ_Matrix_ProdNum(struct TQ_Matrix one,
                       float factor,
                       struct TQ_Matrix *result)
{
    if (one.type == TQ_GPU_Matrix)
    {
        __TQ_GPUMat_ProdNum(one, factor, result);
    }
    else
    {
        __TQ_CPUMat_ProdNum(one, factor, result); // Por defecto, será más rápido.
    }
}

void __TQ_VecDot_TEST(struct TQ_Matrix one,
                      struct TQ_Matrix other)
{
    if (one.num_dims != 1)
    {
        fprintf(stderr,
                "<TQ VecDot (!1) ERROR> ONE dims.");
        exit(1);
    }

    if (other.num_dims != 1)
    {
        fprintf(stderr,
                "<TQ VecDot (!1) ERROR> OTHER dims.");
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

void TQ_Vec_Dot(struct TQ_Matrix one,
                struct TQ_Matrix other,
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

void TQ_Matrix_T(struct TQ_Matrix input,
                 struct TQ_Matrix *output)
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
            output->h_mem[i * dims[0] + j] = input.h_mem[j * dims[0] + i];
        }
    }
}

void TQ_Matrix_Apply(struct TQ_Matrix input,
                     float (*function)(float),
                     struct TQ_Matrix *output)
{
    unsigned int i;
    TQ_Matrix_Create(output, input.dimensions, input.num_dims, input.type);

    for (i = 0; i < input.dims_prod; i++)
    {
        output->h_mem[i] = function(input.h_mem[i]);
    }
}