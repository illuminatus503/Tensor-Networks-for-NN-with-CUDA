#ifndef GPU_TQMAT_H_
#define GPU_TQMAT_H_

#include "tq_datatypes.h"

void __TQ_GPUMat_Add(TQ_Matrix one,
                     TQ_Matrix other,
                     TQ_Matrix *result);

void __TQ_GPUMat_Sub(TQ_Matrix one,
                     TQ_Matrix other,
                     TQ_Matrix *result);

void __TQ_GPUMat_ProdNum(TQ_Matrix one,
                         float factor,
                         TQ_Matrix *result);

void __TQ_GPUMat_Prod(TQ_Matrix one,
                      TQ_Matrix other,
                      TQ_Matrix *result);

void __TQ_GPUVec_Dot(TQ_Matrix one,
                     TQ_Matrix other,
                     float *result);

#endif