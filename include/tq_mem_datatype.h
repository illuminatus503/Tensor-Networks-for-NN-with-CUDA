#ifndef _TQ_MEM_DTYPE_H_
#define _TQ_MEM_DTYPE_H_

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

#endif