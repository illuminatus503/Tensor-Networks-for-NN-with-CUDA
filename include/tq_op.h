#ifndef _TQ_OP_H_
#define _TQ_OP_H_

#include "tq_datatypes.h"

#include "tq_op_cpu.h"
#include "tq_op_gpu.cuh"

void TQ_Matrix_Create(TQ_Matrix *matrix,
                      unsigned int *dimensions,
                      unsigned int num_dims,
                      TQ_Matrix_type type);

void TQ_Matrix_Clone(TQ_Matrix input,
                     TQ_Matrix *output);

void TQ_Matrix_CopyData(TQ_Matrix input,
                        TQ_Matrix *output);

void TQ_Matrix_Extend(TQ_Matrix input,
                      TQ_Matrix *output,
                      unsigned int *new_dims,
                      unsigned int num_dims,
                      float fill_val);

void TQ_Matrix_Free(TQ_Matrix *matrix);

void TQ_Matrix_Print(TQ_Matrix matrix);

void TQ_Matrix_Init(TQ_Matrix *matrix, float value);

void TQ_Matrix_Ones(TQ_Matrix *matrix);
void TQ_Matrix_Zeros(TQ_Matrix *matrix);
void TQ_Matrix_Eyes(TQ_Matrix *matrix);

void TQ_Matrix_Unif(TQ_Matrix *matrix);
void TQ_Matrix_Rand(TQ_Matrix *matrix, float min, float max);

void TQ_Vec_Dot(TQ_Matrix one,
                TQ_Matrix other,
                float *result);

void TQ_Matrix_Add(TQ_Matrix one,
                   TQ_Matrix other,
                   TQ_Matrix *result);

void TQ_Matrix_Sub(TQ_Matrix one,
                   TQ_Matrix other,
                   TQ_Matrix *result);

void TQ_Matrix_ProdNum(TQ_Matrix one,
                       float factor,
                       TQ_Matrix *result);

void TQ_Matrix_Prod(TQ_Matrix one,
                    TQ_Matrix other,
                    TQ_Matrix *result);

void TQ_Matrix_T(TQ_Matrix input,
                 TQ_Matrix *output);

void TQ_Matrix_Apply(TQ_Matrix input,
                     float (*function)(float),
                     TQ_Matrix *output);

#endif