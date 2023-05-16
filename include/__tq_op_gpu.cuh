#ifndef GPU_TQMAT_H_
#define GPU_TQMAT_H_

#include <stdbool.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "__tq_datatypes.h"

/* Macro for checking cuda errors following a cuda launch or api call
 Taken from: https://gist.github.com/jefflarkin/5390993 */
#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(0);                                                                         \
        }                                                                                    \
    }

#define gpuErrchk(call)                                             \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

void __TQ_GPUMat_Add(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result);

void __TQ_GPUMat_Sub(struct TQ_Matrix one,
                     struct TQ_Matrix other,
                     struct TQ_Matrix *result);

void __TQ_GPUMat_ProdNum(struct TQ_Matrix one,
                         float factor,
                         struct TQ_Matrix *result);

void __TQ_GPUMat_Prod(struct TQ_Matrix one,
                      struct TQ_Matrix other,
                      struct TQ_Matrix *result);

#endif