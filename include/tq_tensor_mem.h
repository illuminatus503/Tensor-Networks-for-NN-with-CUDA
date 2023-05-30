#ifndef _TQ_TMEM_H_
#define _TQ_TMEM_H_

#include "tq_tensor_dtypes.h"

/**
 * TENSOR alloc. / dealloc.
 */

void TQ_Matrix_Create(TQ_Tensor *matrix,
                      unsigned int *dimensions,
                      unsigned int num_dims,
                      TQ_Tensor_type type);

void TQ_Matrix_Free(TQ_Tensor *matrix);

/**
 * TENSOR index access
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
 * TENSOR mem. clone/reshape
 */
void TQ_Matrix_Clone(TQ_Tensor input,
                     TQ_Tensor *output);

void TQ_Matrix_CopyData(TQ_Tensor input,
                        TQ_Tensor *output);

void TQ_Matrix_Extend(TQ_Tensor input,
                      TQ_Tensor *output,
                      unsigned int *new_dims,
                      unsigned int num_dims,
                      float fill_val);

/**
 * TENSOR log.
 */

void TQ_Matrix_Print(TQ_Tensor matrix);

#endif