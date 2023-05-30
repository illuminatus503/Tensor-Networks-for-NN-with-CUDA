#ifndef CPU_TQMAT_H_
#define CPU_TQMAT_H_

#include "tq_tensor_dtypes.h"

/**
 * Matrix Operation on CPU
 */

void TQ_CPUMat_Add(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result);

void TQ_CPUMat_Sub(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result);

void TQ_CPUMat_ProdNum(TQ_Tensor one,
                         float factor,
                         TQ_Tensor *result);

void TQ_CPUMat_Prod(TQ_Tensor one,
                      TQ_Tensor other,
                      TQ_Tensor *result);

#endif