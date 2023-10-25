#ifndef __TQ_MATRIX_H_
#define __TQ_MATRIX_H_

#include "tq_vector.h"
#include "tq_tuple.h"

enum TQ_DTYPE_MATRIX
{
    TQ_INT,
    TQ_LONG,
    TQ_FLOAT,
    TQ_DOUBLE
} typedef TQ_DTYPE_MATRIX;

struct TQ_matrix
{
    TQ_Vector *data;

    TQ_Tuple *shape;

    TQ_DTYPE_MATRIX dtype;
    size_t dtype_bytes;
} typedef TQ_matrix;

#endif