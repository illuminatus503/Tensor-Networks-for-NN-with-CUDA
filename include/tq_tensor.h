#ifndef _TQ_MAT_H_
#define _TQ_MAT_H_

#include "tq_tensor_dtypes.h"
#include "tq_tensor_init.h"
#include "tq_tensor_mem.h"
#include "tq_tensor_cpu.h"
#include "tq_tensor_gpu.cuh"

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