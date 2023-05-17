#ifndef TQMAT_H_
#define TQMAT_H_

#include "__tq_datatypes.h"
#include "__tq_op_cpu.h"
#include "__tq_op_gpu.cuh"

/**
 * Matrix CONSTRUCTION/DESTRUCTION
 */

void TQ_Matrix_Create(struct TQ_Matrix *matrix,
                      unsigned int *dimensions,
                      unsigned int num_dims,
                      TQ_Matrix_t type);

void TQ_Matrix_Free(struct TQ_Matrix *matrix);

void TQ_Matrix_Print(struct TQ_Matrix matrix);

/**
 * Matrix initialisation
 */

void TQ_Matrix_Init(struct TQ_Matrix *matrix, float value);

void TQ_Matrix_Ones(struct TQ_Matrix *matrix);
void TQ_Matrix_Zeros(struct TQ_Matrix *matrix);
void TQ_Matrix_Eyes(struct TQ_Matrix *matrix);

void TQ_Matrix_Unif(struct TQ_Matrix *matrix);
void TQ_Matrix_Rand(struct TQ_Matrix *matrix, float min, float max);

/**
 * Vector OP.
 */
void TQ_Vec_Dot(struct TQ_Matrix one,
                struct TQ_Matrix other,
                float *result);

/**
 * Matrix OP.
 */

void TQ_Matrix_Add(struct TQ_Matrix one,
                   struct TQ_Matrix other,
                   struct TQ_Matrix *result);

void TQ_Matrix_Sub(struct TQ_Matrix one,
                   struct TQ_Matrix other,
                   struct TQ_Matrix *result);

void TQ_Matrix_ProdNum(struct TQ_Matrix one,
                       float factor,
                       struct TQ_Matrix *result);

void TQ_Matrix_Prod(struct TQ_Matrix one,
                    struct TQ_Matrix other,
                    struct TQ_Matrix *result);

void TQ_Matrix_T(struct TQ_Matrix input,
                 struct TQ_Matrix *output);

void TQ_Matrix_Apply(struct TQ_Matrix input,
                     float (*function)(float),
                     struct TQ_Matrix *output);

#endif