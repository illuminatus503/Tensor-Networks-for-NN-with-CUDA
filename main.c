#include <stdio.h>
#include <stdlib.h>

#include "include/tq_vector.h"

int main(int argc, char **argv)
{
    TQ_Vector *vector = TQ_create_empty_vector(10, TQ_INT);
    printf("Se ha generado un vector\n");

    TQ_print_vector(vector);

    // insertamos 3 en el índice 3
    int integer_val = 3;
    TQ_set_value_vector(vector, 3, (void *)&integer_val);
    TQ_print_vector(vector);

    // Kquita

    printf("Se ha insertado el valor %d en el vector\n",
           *(int *)TQ_get_value_vector(vector, 3));

    TQ_delete_vector(&vector);
    printf("Se ha borrado un vector\n");

    float values[5] = {0.3, 0.2, 0.1, 0.0, -0.1};
    vector = TQ_create_from_array_vector((void *)values, 5, TQ_FLOAT);
    TQ_print_vector(vector);
    TQ_delete_vector(&vector);

    return 0;
}