#ifndef _TQ_MEM_TMP_H_
#define _TQ_MEM_TMP_H_

#include "tq_mem_datatype.h"

/**
 * @brief (Wrapper) Initialise some temporal memory.
 * Temp. cplx. O(1)
 * @param a Non-temp. arena ptr.
 * @return Temp_Arena_Memory The temporal arena (lifespan)
 */
Temp_Arena_Memory temp_arena_memory_begin(Arena *a);

/**
 * @brief End lifespan of temp. arena.
 *
 * @param temp Lifespan struct. To be ended.
 */
void temp_arena_memory_end(Temp_Arena_Memory temp);

#endif