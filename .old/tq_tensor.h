#ifndef TQ_DATATYPES_H_
#define TQ_DATATYPES_H_

#include <stdlib.h>
#include "tq_vector.h"

enum TQ_Matrix_type
{
    TQ_GPU_Matrix,
    TQ_CPU_Matrix
} typedef TQ_Matrix_t;

struct TQ_Matrix
{
    size_t length_bytes;
    size_t n_elems;

    TQ_Matrix_t type;

    TQ_Int_Vector *dims;    
    float *h_mem;
} typedef TQ_Matrix;

#endif