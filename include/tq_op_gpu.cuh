#ifndef GPU_TQMAT_H_
#define GPU_TQMAT_H_

#include "tq_datatypes.h"

void __TQ_GPUMat_Add(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result);

void __TQ_GPUMat_Sub(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result);

void __TQ_GPUMat_ProdNum(struct TQ_Matrix one,
                         float factor,
                         struct TQ_Matrix *result);

void __TQ_GPUMat_Prod(struct TQ_Matrix one,
                      struct TQ_Matrix other,
                      struct TQ_Matrix *result);

void __TQ_GPUVec_Dot(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     float *result);

#endif