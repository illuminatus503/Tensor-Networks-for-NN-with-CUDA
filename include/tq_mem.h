#ifndef _TQ_MEM_H_
#define _TQ_MEM_H_

#include <stdlib.h>

typedef struct Arena Arena;
struct Arena
{
    unsigned char *buf;
    size_t buf_len;
    size_t prev_offset; // This will be useful for later on
    size_t curr_offset;
};

#endif