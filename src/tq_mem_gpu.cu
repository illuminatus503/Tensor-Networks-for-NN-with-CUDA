#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/cuda_errchk.cuh"

#include "../include/tq_mem_datatype.h"
#include "../include/tq_mem_gpu.cuh"

#define is_power_of_two(x) ((x & (x - 1)) == 0)

#ifndef DEFAULT_ALIGNMENT
#define DEFAULT_ALIGNMENT (2 * sizeof(void *))
#endif

uintptr_t CUDA_align_forward(uintptr_t ptr, size_t align)
{
    uintptr_t p, a, modulo;

    assert(is_power_of_two(align));

    p = ptr;
    a = (uintptr_t)align;

    // Same as (p % a) but faster as 'a' is a power of two
    modulo = p & (a - 1);

    if (modulo != 0)
    {
        // If 'p' address is not aligned, push the address to the
        // next value which is aligned
        p += a - modulo;
    }

    return p;
}

void *CUDA_arena_alloc_align(Arena *a,
                             size_t size,
                             size_t align)
{
    // Return NULL if the arena is out of memory (or handle differently)
    void *ptr = NULL;

    // Align 'curr_offset' forward to the specified alignment
    uintptr_t curr_ptr = (uintptr_t)a->buf + (uintptr_t)a->curr_offset;
    uintptr_t offset = CUDA_align_forward(curr_ptr, align);
    offset -= (uintptr_t)a->buf; // Change to relative offset

    // Check to see if the backing memory has space left
    if (offset + size <= a->buf_len)
    {
        ptr = &(a->buf[offset]);
        a->prev_offset = offset;
        a->curr_offset = offset + size;

        // Zero new memory by default
        gpuErrchk(
            cudaMemset((void *)ptr, 0, size));
    }

    return ptr;
}

// Because C doesn't have default parameters
void *CUDA_arena_alloc(Arena *a, size_t size)
{
    return CUDA_arena_alloc_align(a, size, DEFAULT_ALIGNMENT);
}

void CUDA_arena_init(Arena *a,
                     void *backing_buffer,
                     size_t backing_buffer_length)
{
    a->buf = (unsigned char *)backing_buffer;
    a->buf_len = backing_buffer_length;
    a->curr_offset = 0;
    a->prev_offset = 0;
}

void *CUDA_arena_resize_align(Arena *a,
                              void *old_memory,
                              size_t old_size,
                              size_t new_size,
                              size_t align)
{
    unsigned char *old_mem = (unsigned char *)old_memory;

    assert(is_power_of_two(align));

    if (old_mem == NULL || old_size == 0)
    {
        return CUDA_arena_alloc_align(a, new_size, align);
    }
    else if (a->buf <= old_mem && old_mem < a->buf + a->buf_len)
    {
        if (a->buf + a->prev_offset == old_mem)
        {
            a->curr_offset = a->prev_offset + new_size;
            if (new_size > old_size)
            {
                // Zero the new memory by default
                gpuErrchk(
                    cudaMemset(&a->buf[a->curr_offset], 0, new_size - old_size));
            }
            return old_memory;
        }
        else
        {
            void *new_memory = CUDA_arena_alloc_align(a, new_size, align);
            size_t copy_size = old_size < new_size ? old_size : new_size;

            // Copy across old memory to the new memory
            gpuErrchk(
                cudaMemcpy((void *)new_memory, (const void *)old_memory,
                           copy_size,
                           cudaMemcpyDeviceToDevice));

            return new_memory;
        }
    }
    else
    {
        assert(0 && "Memory is out of bounds of the buffer in this arena");
        return NULL;
    }
}

// Because C doesn't have default parameters
void *CUDA_arena_resize(Arena *a,
                        void *old_memory, size_t old_size,
                        size_t new_size)
{
    return CUDA_arena_resize_align(a, old_memory, old_size, new_size, DEFAULT_ALIGNMENT);
}

void CUDA_arena_free_all(Arena *a)
{
    a->curr_offset = 0;
    a->prev_offset = 0;
}
