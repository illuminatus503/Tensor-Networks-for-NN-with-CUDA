#ifndef __TQ_VECTOR_H_
#define __TQ_VECTOR_H_

#include <stdlib.h>

#include "tq_dtype.h"

// String repr. of a vector.
#define BEGIN_VECT "<"
#define END_VECT ">\n"

struct TQ_Vector
{
    TQ_DTYPE dtype;
    size_t dtype_bytes;

    size_t n_size;
    size_t n_size_bytes;

    void *data; // Memory
} typedef TQ_Vector;

/**
 * @brief Create an empty vector given its size.
 *
 * @param n_size The size of the vector.
 * @param dtype The datatype of the contents of the vector.
 * @return TQ_Vector* The new vector itself.
 */
TQ_Vector *TQ_create_empty_vector(size_t n_size, TQ_DTYPE dtype);

/**
 * @brief Create a vector from an array. Copy the contents
 * of the array into a vector.
 *
 * @param values The value array to cast into a vector.
 * @param n_size The size of the vector.
 * @param dtype The datatype of the contents of the vector.
 * @return TQ_Vector* The new vector itself.
 */
TQ_Vector *TQ_create_vector_from_array(void *values, size_t n_size, TQ_DTYPE dtype);

/**
 * @brief Get an indexed value from the vector.
 *
 * @param vector The vector itself.
 * @param index The index to access to.
 * @return long The indexed value from the vector.
 */
void *TQ_get_value_vector(TQ_Vector *vector, size_t index);

/**
 * @brief Set a value in the vector, at some index.
 *
 * @param vector The vector itself.
 * @param index The index in which to write.
 * @param value The value to be writen in the vector.
 */
void TQ_set_value_vector(TQ_Vector *vector, size_t index, void *value);

/**
 * @brief Print a vector to display.
 *
 * @param vector The vector itself.
 */
void TQ_print_vector(TQ_Vector *vector);

/**
 * @brief Delete a vector structure from memory, with all its
 * contents.
 *
 * @param vector The vector itself.
 */
void TQ_delete_vector(TQ_Vector **vector);

#endif