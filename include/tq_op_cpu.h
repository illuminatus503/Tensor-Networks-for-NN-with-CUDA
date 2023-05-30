#ifndef CPU_TQMAT_H_
#define CPU_TQMAT_H_

#include "tq_datatypes.h"

/**
 * Matrix Index operations.
 */

unsigned long __TQ_Tensor_IndexToPos(TQ_Tensor matrix,
                                     unsigned int *indices,
                                     unsigned int num_ind);
void __TQ_Tensor_PosToIndex(TQ_Tensor matrix,
                            unsigned int position,
                            unsigned int *indices);

unsigned char __TQ_Tensor_Pos_Is_Valid(TQ_Tensor matrix,
                                       unsigned long pos);

float TQ_Tensor_GetElem(TQ_Tensor matrix,
                        unsigned int *indices,
                        unsigned int num_ind);

void TQ_Tensor_SetElem(TQ_Tensor *matrix,
                       float value,
                       unsigned int *indices,
                       unsigned int num_ind);

/**
 * Matrix Operation on CPU
 */

void __TQ_CPUMat_Add(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result);

void __TQ_CPUMat_Sub(TQ_Tensor one,
                     TQ_Tensor other,
                     TQ_Tensor *result);

void __TQ_CPUMat_ProdNum(TQ_Tensor one,
                         float factor,
                         TQ_Tensor *result);

void __TQ_CPUMat_Prod(TQ_Tensor one,
                      TQ_Tensor other,
                      TQ_Tensor *result);

void __TQ_CPUVec_Dot(TQ_Tensor one,
                     TQ_Tensor other,
                     float *result);

#endif