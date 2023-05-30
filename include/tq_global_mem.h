/**
 * @file tq_global_mem.h
 * @author David Cuenca Marcos
 * @brief
 * Custom memory allocation , based on
 * https://www.gingerbill.org/article/2019/02/01/memory-allocation-strategies-001/
 * https://www.gingerbill.org/article/2019/02/08/memory-allocation-strategies-002/#using-the-allocator
 *
 * @version 0.1
 * @date 2023-05-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _TQ_GLOBAL_H_
#define _TQ_GLOBAL_H_

#include "tq_mem_datatype.h"
#include "tq_mem_cpu.h"
#include "tq_mem_gpu.cuh"
#include "tq_mem_tmp.h"

/**
 * Global CPU memory
 */
extern Arena TQ_CPU_ARENA;

/**
 * Global GPU memory
 */
extern Arena TQ_GPU_ARENA;

#endif