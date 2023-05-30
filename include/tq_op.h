#ifndef _TQ_OP_H_
#define _TQ_OP_H_

#include "tq_datatypes.h"

#include "tq_op_cpu.h"
#include "tq_op_gpu.cuh"

void TQ_Matrix_Create(TQ_Tensor *matrix,
                      unsigned int *dimensions,
                      unsigned int num_dims,
                      TQ_Tensor_type type);

void TQ_Matrix_Clone(TQ_Tensor input,
                     TQ_Tensor *output);

void TQ_Matrix_CopyData(TQ_Tensor input,
                        TQ_Tensor *output);

void TQ_Matrix_Extend(TQ_Tensor input,
                      TQ_Tensor *output,
                      unsigned int *new_dims,
                      unsigned int num_dims,
                      float fill_val);

void TQ_Matrix_Free(TQ_Tensor *matrix);

void TQ_Matrix_Print(TQ_Tensor matrix);

void TQ_Matrix_Init(TQ_Tensor *matrix, float value);

void TQ_Matrix_Ones(TQ_Tensor *matrix);
void TQ_Matrix_Zeros(TQ_Tensor *matrix);
void TQ_Matrix_Eyes(TQ_Tensor *matrix);

void TQ_Matrix_Unif(TQ_Tensor *matrix);
void TQ_Matrix_Rand(TQ_Tensor *matrix, float min, float max);

void TQ_Vec_Dot(TQ_Tensor one,
                TQ_Tensor other,
                float *result);

void TQ_Matrix_Add(TQ_Tensor one,
                   TQ_Tensor other,
                   TQ_Tensor *result);

void TQ_Matrix_Sub(TQ_Tensor one,
                   TQ_Tensor other,
                   TQ_Tensor *result);

void TQ_Matrix_ProdNum(TQ_Tensor one,
                       float factor,
                       TQ_Tensor *result);

void TQ_Matrix_Prod(TQ_Tensor one,
                    TQ_Tensor other,
                    TQ_Tensor *result);

void TQ_Matrix_T(TQ_Tensor input,
                 TQ_Tensor *output);

void TQ_Matrix_Apply(TQ_Tensor input,
                     float (*function)(float),
                     TQ_Tensor *output);

#endif