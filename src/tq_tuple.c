#include <stdio.h>
#include <string.h>

#include "../include/tq_tuple.h"

/**
 * @brief Crea una tupla vacía. Solo con fines de ser modificada internamente.
 * 
 * @param n_size Tamaño de la tupla, en número de datos.
 * @param dtype Tipo de datos de la tupla
 * @return TQ_Tuple* La tupla en sí misma
 */
TQ_Tuple *__TQ_create_empty_tuple(size_t n_size, TQ_DTYPE dtype)
{
    size_t dtype_bytes = 0;
    TQ_Tuple *new_tuple;

    switch (dtype)
    {
    case TQ_INT:
        dtype_bytes = sizeof(int);
        break;

    case TQ_FLOAT:
        dtype_bytes = sizeof(float);
        break;

    case TQ_DOUBLE:
        dtype_bytes = sizeof(double);
        break;

    case TQ_LONG:
        dtype_bytes = sizeof(long);
        break;
    }

    // Initialization
    new_tuple = (TQ_Tuple *)malloc(sizeof(TQ_Tuple));
    new_tuple->dtype = dtype;
    new_tuple->dtype_bytes = dtype_bytes;
    new_tuple->n_size = n_size;
    new_tuple->n_size_bytes = dtype_bytes * n_size;
    new_tuple->data = calloc(n_size, dtype_bytes);

    return new_tuple;
}

TQ_Tuple *TQ_create_tuple_from_array(void *values, size_t n_size, TQ_DTYPE dtype)
{
    TQ_Tuple *new_tuple;

    if (values == NULL)
    {
        fprintf(stderr, "InitError (in line %d): the array you introduced is empty.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    new_tuple = __TQ_create_empty_tuple(n_size, dtype);

    memcpy(new_tuple->data, (void *)values, new_tuple->n_size_bytes);

    return new_tuple;
}

void *TQ_get_value_tuple(TQ_Tuple *tuple, size_t index)
{
    if (index < 0 || index >= tuple->n_size)
    {
        fprintf(stderr, "IndexError (in line %d): Index out of bounds.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    return (char *)(tuple->data) + index * tuple->dtype_bytes;
}

void __TQ_print_tuple(TQ_Tuple *tuple)
{
    size_t i;
    int *idata;
    long *ldata;
    float *fdata;
    double *ddata;

    printf(BEGIN_TUPLE);

    switch (tuple->dtype)
    {
    case TQ_INT:
        idata = (int *)tuple->data;
        for (i = 0; i < tuple->n_size; i++)
        {
            printf(FMT_DTYPE_INT, idata[i]);
        }

        break;

    case TQ_LONG:
        ldata = (long *)tuple->data;
        for (i = 0; i < tuple->n_size; i++)
        {
            printf(FMT_DTYPE_LONG, ldata[i]);
        }

        break;

    case TQ_FLOAT:
        fdata = (float *)tuple->data;
        for (i = 0; i < tuple->n_size; i++)
        {
            printf(FMT_DTYPE_FLOAT, fdata[i]);
        }
        break;

    case TQ_DOUBLE:
        ddata = (double *)tuple->data;
        for (i = 0; i < tuple->n_size; i++)
        {
            printf(FMT_DTYPE_DOUBLE, ddata[i]);
        }

        break;
    }

    printf(END_TUPLE);
}

void TQ_print_tuple(TQ_Tuple *tuple)
{
    if (tuple == NULL)
    {
        fprintf(stderr, "FreeError (in line %d): You are trying to print an uninitialized tuple.", __LINE__ + 10);
        exit(EXIT_FAILURE);
    }

    if (tuple->data == NULL || tuple->n_size_bytes == 0)
    {
        fprintf(stderr, "FreeError (in line %d): The contents of this tuple are freed.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    // Print the tuple to display
    __TQ_print_tuple(tuple);
}

void TQ_delete_tuple(TQ_Tuple **tuple)
{
    if (*tuple == NULL || (*tuple)->n_size_bytes == 0)
    {
        fprintf(stderr, "FreeError (in line %d): You are trying to delete an uninitialized tuple.", __LINE__ + 11);
        exit(EXIT_FAILURE);
    }

    if ((*tuple)->data == NULL)
    {
        fprintf(stderr, "FreeError (in line %d): The contents of this tuple are already freed.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    free((*tuple)->data);
    free(*tuple);
}