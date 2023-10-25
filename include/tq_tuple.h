#ifndef __TQ_TUPLE_H_
#define __TQ_TUPLE_H_

#include <stdlib.h>

// String repr. of a TUPLE.
#define BEGIN_TUPLE "( "
#define END_TUPLE ")\n"
#define TUPLE_FMT_DTYPE_INT "%d "
#define TUPLE_FMT_DTYPE_LONG "%ld "
#define TUPLE_FMT_DTYPE_FLOAT "%3.6f "
#define TUPLE_FMT_DTYPE_DOUBLE "%3.15lf "

enum TQ_DTYPE_TUPLE
{
    TQ_INT,
    TQ_LONG,
    TQ_FLOAT,
    TQ_DOUBLE
} typedef TQ_DTYPE_TUPLE;

struct TQ_Tuple
{
    TQ_DTYPE_TUPLE dtype;
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
TQ_Tuple *TQ_create_tuple_from_array(void *values, size_t n_size, TQ_DTYPE_TUPLE dtype);

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