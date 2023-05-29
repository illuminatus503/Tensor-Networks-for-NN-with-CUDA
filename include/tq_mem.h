/**
 * @file tq_mem.h
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

#ifndef _TQ_MEM_H_
#define _TQ_MEM_H_

#include <stdlib.h>

typedef struct Arena Arena;
struct Arena
{
    unsigned char *buf;
    size_t buf_len;
    size_t prev_offset;
    size_t curr_offset;
};

typedef struct Temp_Arena_Memory Temp_Arena_Memory;
struct Temp_Arena_Memory
{
    Arena *arena;
    size_t prev_offset;
    size_t curr_offset;
};

/**
 * @brief Initialise a Linear Memory allocator.
 * Based on a backing buffer.
 *
 * @param a Arena ptr
 * @param backing_buffer Backing mem.
 * @param backing_buffer_length Total length (bytes) of backing mem.
 */
void arena_init(Arena *a,
                void *backing_buffer,
                size_t backing_buffer_length);

/**
 * @brief Allocate some mem. (size, in bytes) on the backing mem.
 * Temp. complx. O(1).
 *
 * @param a Arena ptr
 * @param size Size (bytes) to be alloc.
 * @return void*
 */
void *arena_alloc(Arena *a,
                  size_t size);
void *arena_resize(Arena *a,
                   void *old_memory,
                   size_t old_size,
                   size_t new_size);
void arena_free_all(Arena *a);

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