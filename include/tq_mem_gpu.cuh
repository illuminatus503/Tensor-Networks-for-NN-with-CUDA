#ifndef _TQ_MEM_GPU_H_
#define _TQ_MEM_GPU_H_

/**
 * @brief Initialise a Linear Memory allocator.
 * Based on a backing buffer.
 *
 * @param a Arena ptr
 * @param backing_buffer Backing mem.
 * @param backing_buffer_length Total length (bytes) of backing mem.
 */
void CUDA_arena_init(Arena *a,
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
void *CUDA_arena_alloc(Arena *a,
                       size_t size);
void *CUDA_arena_resize(Arena *a,
                        void *old_memory,
                        size_t old_size,
                        size_t new_size);
void CUDA_arena_free_all(Arena *a);

#endif