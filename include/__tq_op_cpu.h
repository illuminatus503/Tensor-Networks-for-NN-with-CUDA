#ifndef CPU_TQMAT_H_
#define CPU_TQMAT_H_

#include "__tq_datatypes.h"

/**
 * Matrix Index operations.
 */

unsigned long __TQ_Matrix_IndexToPos(struct TQ_Matrix matrix,
                                     unsigned int *indices,
                                     unsigned int num_ind);
void __TQ_Matrix_PosToIndex(struct TQ_Matrix matrix,
                            unsigned int position,
                            unsigned int *indices);

float TQ_Matrix_GetElem(struct TQ_Matrix matrix,
                        unsigned int *indices,
                        unsigned int num_ind);

void TQ_Matrix_SetElem(struct TQ_Matrix *matrix,
                       float value,
                       unsigned int *indices,
                       unsigned int num_ind);

/**
 * Matrix Operation on CPU
 */

void __TQ_CPUMat_Add(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result);

void __TQ_CPUMat_Sub(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result);

void __TQ_CPUMat_ProdNum(struct TQ_Matrix one,
                         float factor,
                         struct TQ_Matrix *result);

void __TQ_CPUMat_Prod(struct TQ_Matrix one,
                      struct TQ_Matrix other,
                      struct TQ_Matrix *result);

void __TQ_CPUVec_Dot(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     float *result);

#endif