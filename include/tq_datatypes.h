#ifndef TQ_DATATYPES_H_
#define TQ_DATATYPES_H_

typedef enum TQ_Matrix_type
{
    TQ_GPU_Matrix,
    TQ_CPU_Matrix
} TQ_Matrix_type;

typedef struct TQ_Matrix
{
    TQ_Matrix_type type;
    unsigned int num_dims;
    unsigned int *dimensions;
    unsigned long length_bytes, dims_prod;
    float *h_mem;
} TQ_Matrix;

#endif