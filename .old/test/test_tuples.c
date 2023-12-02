#include <stdio.h>
#include <stdlib.h>

#include "../include/tq_tuple.h"

int main(int argc, char **argv)
{
    TQ_Tuple *tuple;

    float values[5] = {0.3, 0.2, 0.1, 0.0, -0.1};
    tuple = TQ_newtuple((void *)values, 5, TQ_FLOAT);
    printf("Se ha generado una tupla desde un array\n");
    TQ_print_tuple(tuple);
    printf("Se ha borrado una tupla\n");
    TQ_delete_tuple(&tuple);

    return 0;
}