#ifndef _TQ_TINIT_H_
#define _TQ_TINIT_H_

#include "tq_tensor_dtypes.h"

/**
 * TENSOR Fixed Init.
 */

/**
 * @brief Initialise TENSOR to all 0.
 *
 * @param matrix Inout matrix.
 */
void TQ_Matrix_Zeros(TQ_Tensor *matrix);

/**
 * @brief Initialise TENSOR to all 1.
 *
 * @param matrix Inout matrix.
 */
void TQ_Matrix_Ones(TQ_Tensor *matrix);

/**
 * @brief Initialise TENSOR all to some value.
 *
 * @param matrix Inout matrix.
 * @param value (float) The value to be set.
 */
void TQ_Matrix_ValueInit(TQ_Tensor *matrix, float value);

/**
 * @brief Initialise TENSOR to Identity TENSOR.
 * 1 iff all index are equal; else 0.
 *
 * @param matrix Inout matrix.
 */
void TQ_Matrix_Eyes(TQ_Tensor *matrix);

/**
 * TENSOR Random Init.
 */

/**
 * @brief Initialise TENSOR with values extracted from
 * a rand. unif. distribution in [0, 1) interv.
 *
 * @param matrix Inout matrix.
 */
void TQ_Matrix_Unif(TQ_Tensor *matrix);

/**
 * @brief Initialise TENSOR with values extracted from
 * a rand. unif. distribution in [min, max) interv.
 *
 * @param matrix Inout matrix.
 * @param min Lower bound.
 * @param max Upper bound.
 */
void TQ_Matrix_Rand(TQ_Tensor *matrix, float min, float max);

/**
 * @brief Initialise TENSOR with values drawn from a rand.
 * normal distribution N(mu, sigma).
 *
 * @param matrix Inout matrix.
 * @param mu Real mean.
 * @param sigma Real std.
 */
void TQ_Matrix_Normal(TQ_Tensor *matrix,
                      float mu,
                      float sigma);

#endif