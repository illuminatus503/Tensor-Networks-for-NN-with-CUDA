#ifndef __TQ_TUPLE_H_
#define __TQ_TUPLE_H_

#include <stdlib.h>

#include "tq_dtype.h"

// String repr. of a TUPLE.
#define BEGIN_TUPLE "("
#define END_TUPLE ")\n"

struct TQ_Tuple
{
    TQ_DTYPE dtype;
    size_t dtype_bytes;

    size_t n_size;
    size_t n_size_bytes;

    void *data; // Memory
} typedef TQ_Tuple;

/**
 * @brief Create a TUPLE from an array. Copy the contents
 * of the array into a TUPLE.
 *
 * @param values The value array to cast into a TUPLE.
 * @param n_size The size of the TUPLE.
 * @param dtype The datatype of the contents of the TUPLE.
 * @return TQ_Tuple* The new TUPLE itself.
 */
TQ_Tuple *TQ_newtuple(void *values, size_t n_size, TQ_DTYPE dtype);

/**
 * @brief Get an indexed value from the tuple.
 *
 * @param tuple The tuple itself.
 * @param index The index to access to.
 * @return long The indexed value from the tuple.
 */
void *TQ_get_value_tuple(TQ_Tuple *tuple, size_t index);

/**
 * @brief Print a TUPLE to display.
 *
 * @param tuple The TUPLE itself.
 */
void TQ_print_tuple(TQ_Tuple *tuple);

/**
 * @brief Delete a TUPLE structure from memory, with all its
 * contents.
 *
 * @param tuple The TUPLE itself.
 */
void TQ_delete_tuple(TQ_Tuple **tuple);

#endif