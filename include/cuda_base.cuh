#ifndef _CUDA_CHKERR_H_
#define _CUDA_CHKERR_H_

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

/* Macro for checking cuda errors following a cuda launch or api call
 Taken from: https://gist.github.com/jefflarkin/5390993 */
#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_SUCCESS);                                                              \
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

#endif