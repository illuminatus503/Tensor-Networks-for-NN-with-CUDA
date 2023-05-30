#ifndef GPU_TQMAT_H_
#define GPU_TQMAT_H_

#include "tq_datatypes.h"

void __TQ_GPUMat_Add(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result);

void __TQ_GPUMat_Sub(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result);

void __TQ_GPUMat_ProdNum(TQ_Tensor one,
                         float factor,
                         TQ_Tensor *result);

void __TQ_GPUMat_Prod(TQ_Tensor one,
                      TQ_Tensor other,
                      TQ_Tensor *result);

void __TQ_GPUVec_Dot(TQ_Tensor one,
                     TQ_Tensor other,
                     float *result);

#endif