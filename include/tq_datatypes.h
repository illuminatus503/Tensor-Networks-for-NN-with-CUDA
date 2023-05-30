#ifndef TQ_DATATYPES_H_
#define TQ_DATATYPES_H_

typedef enum TQ_Tensor_type
{
    TQ_GPU_Matrix,
    TQ_CPU_Matrix
} TQ_Tensor_type;

/**
 * @brief Tensor datatype.
 *
 */
typedef struct TQ_Tensor
{
    TQ_Tensor_type type;        // CPU or GPU matrix?
    unsigned int num_dims;      // Number of dimnesions
    unsigned int *dimensions;   // Dimension array
    unsigned long length;       // Length of the mem. in float
    unsigned long length_bytes; // Length of the mem., in bytes
    float *mem;                 // Inner memory buffer: mem. in CPU or in GPU
    float *h_mem;               // Specific host mem.: for retrieving from GPU to CPU
} TQ_Tensor;

#endif