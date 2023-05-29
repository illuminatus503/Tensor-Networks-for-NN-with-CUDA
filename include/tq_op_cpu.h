#ifndef CPU_TQMAT_H_
#define CPU_TQMAT_H_

#include "tq_datatypes.h"

/**
 * Matrix Index operations.
 */

unsigned long __TQ_Matrix_IndexToPos(TQ_Matrix matrix,
                                     unsigned int *indices,
                                     unsigned int num_ind);
void __TQ_Matrix_PosToIndex(TQ_Matrix matrix,
                            unsigned int position,
                            unsigned int *indices);
                            
unsigned char __TQ_Matrix_Pos_Is_Valid(TQ_Matrix matrix,
                                       unsigned long pos);

float TQ_Matrix_GetElem(TQ_Matrix matrix,
                        unsigned int *indices,
                        unsigned int num_ind);

void TQ_Matrix_SetElem(TQ_Matrix *matrix,
                       float value,
                       unsigned int *indices,
                       unsigned int num_ind);

/**
 * Matrix Operation on CPU
 */

void __TQ_CPUMat_Add(TQ_Matrix one,
                     TQ_Matrix other,
                     TQ_Matrix *result);

void __TQ_CPUMat_Sub(TQ_Matrix one,
                     TQ_Matrix other,
                     TQ_Matrix *result);

void __TQ_CPUMat_ProdNum(TQ_Matrix one,
                         float factor,
                         TQ_Matrix *result);

void __TQ_CPUMat_Prod(TQ_Matrix one,
                      TQ_Matrix other,
                      TQ_Matrix *result);

void __TQ_CPUVec_Dot(TQ_Matrix one,
                     TQ_Matrix other,
                     float *result);

#endif