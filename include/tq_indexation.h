#ifndef __TQ_MATRIX_INDEX_H_
#define __TQ_MATRIX_INDEX_H_

#include <stdlib.h>

#include "tq_matrix.h"

/**
 * @brief ** INTERNAL ** return a vector position for tensor
 * indexation.
 *
 * @param matrix A generic matrix structure.
 * @param indices An index tuple (ULONG type).
 * @return size_t The vector index of the flattened tensor.
 */
size_t TQ_vect_position_fromtensor_(TQ_Matrix *matrix, TQ_Tuple *indices);

/**
 * @brief ** INTERNAL ** return a tuple of indices from flattened
 * tensor index.
 *
 * @param matrix A generic matrix structure.
 * @param position A position of the flattened tensor.
 * @return TQ_Tuple* The tuple of indices of the flattened tensor, in the
 * unflattened tensor.
 */
TQ_Tuple *TQ_indices_fromvect_pos_(TQ_Matrix *matrix, size_t position);

#endif