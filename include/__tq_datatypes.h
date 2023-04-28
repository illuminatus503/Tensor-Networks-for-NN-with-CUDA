#ifndef TQ_DATATYPES_H_
#define TQ_DATATYPES_H_

enum TQ_Matrix_type
{
    TQ_GPU_Matrix,
    TQ_CPU_Matrix
} typedef TQ_Matrix_t;

struct TQ_Matrix
{
    TQ_Matrix_t type;
    unsigned int num_dims;
    unsigned int *dimensions;
    unsigned long length_bytes, dims_prod;
    float *h_mem;
} typedef TQ_Matrix;

#endif