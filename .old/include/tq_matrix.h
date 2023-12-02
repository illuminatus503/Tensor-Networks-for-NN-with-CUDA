#ifndef __TQ_MATRIX_H_
#define __TQ_MATRIX_H_

#include "tq_vector.h"
#include "tq_tuple.h"
#include "tq_dtype.h"

#define BEGIN_MATRIX "["
#define END_MATRIX "]"

struct TQ_Matrix
{
    TQ_Vector *data; // Vector que mantiene la estructura matriz
    TQ_Tuple *shape; // Tupla de dimensiones
    TQ_DTYPE dtype;  // Tipo de datos de la matriz
} typedef TQ_Matrix;

/**
 * @brief Create an empty matrix given its size.
 * Initialized to all 0s.
 *
 * @param shape The dimensions of the matrix.
 * @param dtype The datatype of the contents of the matrix.
 * @return TQ_Matrix* The new matrix itself.
 */
TQ_Matrix *TQ_emptymat(TQ_Tuple *shape, TQ_DTYPE dtype);

/**
 * @brief Create a matrix from an array. Copy the contents
 * of the array into a matrix.
 *
 * @param values The value array to cast into a matrix, as a flat array.
 * @param shape The dimensions of the matrix.
 * @param dtype The datatype of the contents of the matrix.
 * @return TQ_Matrix* The new matrix itself.
 */
TQ_Matrix *TQ_newmatrix(void *values, TQ_Tuple *shape, TQ_DTYPE dtype);

/**
 * @brief Get an indexed value from the matrix.
 *
 * @param matrix The matrix itself.
 * @param indexes The indexes to access to.
 * @return long The indexed value from the matrix.
 */
void *TQ_get_value_matrix(TQ_Matrix *matrix, TQ_Tuple *indexes);

/**
 * @brief Set a value in the matrix, at some index.
 *
 * @param matrix The matrix itself.
 * @param indexes The indexes in which to write.
 * @param value The value to be writen in the matrix.
 */
void TQ_set_value_matrix(TQ_Matrix *matrix, TQ_Tuple *indexes, void *value);

/**
 * @brief Print a matrix to display.
 *
 * @param matrix The matrix itself.
 */
void TQ_print_matrix(TQ_Matrix *matrix);

/**
 * @brief Delete a matrix structure from memory, with all its
 * contents.
 *
 * @param matrix The matrix itself.
 */
void TQ_delete_matrix(TQ_Matrix **matrix);

#endif