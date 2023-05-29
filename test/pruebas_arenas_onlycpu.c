#include <stdio.h>
#include <string.h>

#include "include/tq_mem.h"

int main(int argc, char **argv)
{
    int i;

    unsigned char backing_buffer[256];
    Arena a = {0};
    arena_init(&a, backing_buffer, 256);

    for (i = 0; i < 10; i++)
    {
        int *x;
        float *f;
        char *str;

        // Reset all arena offsets for each loop
        arena_free_all(&a);

        x = (int *)arena_alloc(&a, sizeof(int));
        f = (float *)arena_alloc(&a, sizeof(float));
        str = (char *)arena_alloc(&a, 10);

        *x = 123;
        *f = 987;
        memmove(str, "Hellope", 7);

        printf("%p: %d\n", x, *x);
        printf("%p: %f\n", f, *f);
        printf("%p: %s\n", str, str);

        str = (char *)arena_resize(&a, (void *)str, 10, 16);
        memmove(str + 7, " world!", 7);
        printf("%p: %s\n", str, str);
    }

    arena_free_all(&a);

    return 0;
}