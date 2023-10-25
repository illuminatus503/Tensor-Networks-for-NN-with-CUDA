#include <stdio.h>
#include <string.h>

#include "../include/tq_vector.h"

TQ_Vector *TQ_create_empty_vector(size_t n_size, TQ_DTYPE dtype)
{
    size_t dtype_bytes = 0;
    TQ_Vector *new_vector;

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
    new_vector = (TQ_Vector *)malloc(sizeof(TQ_Vector));
    new_vector->dtype = dtype;
    new_vector->dtype_bytes = dtype_bytes;
    new_vector->n_size = n_size;
    new_vector->n_size_bytes = dtype_bytes * n_size;
    new_vector->data = calloc(n_size, dtype_bytes);

    return new_vector;
}

TQ_Vector *TQ_create_vector_from_array(void *values, size_t n_size, TQ_DTYPE dtype)
{
    TQ_Vector *new_vector;

    if (values == NULL)
    {
        fprintf(stderr, "InitError (in line %d): the array you introduced is empty.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    new_vector = TQ_create_empty_vector(n_size, dtype);

    memcpy(new_vector->data, (void *)values, new_vector->n_size_bytes);

    return new_vector;
}

void *TQ_get_value_vector(TQ_Vector *vector,
                          size_t index)
{
    if (index < 0 || index >= vector->n_size)
    {
        fprintf(stderr, "IndexError (in line %d): Index out of bounds.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    return (char *)(vector->data) + index * vector->dtype_bytes;
}

void TQ_set_value_vector(TQ_Vector *vector,
                         size_t index,
                         void *value)
{
    if (index < 0 || index >= vector->n_size)
    {
        fprintf(stderr, "IndexError (in line %d): Index out of bounds.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    // Copy some bytes to the memory
    *((char *)vector->data + index * vector->dtype_bytes) = *(char *)value;
}

void __TQ_print_vector(TQ_Vector *vector)
{
    size_t i;
    int *idata;
    long *ldata;
    float *fdata;
    double *ddata;

    printf(BEGIN_VECT);

    switch (vector->dtype)
    {
    case TQ_INT:
        idata = (int *)vector->data;
        for (i = 0; i < vector->n_size; i++)
        {
            printf(FMT_DTYPE_INT, idata[i]);
            if (i< vector->n_size-1)
            {
                printf(", ");
            }
        }

        break;

    case TQ_LONG:
        ldata = (long *)vector->data;
        for (i = 0; i < vector->n_size; i++)
        {
            printf(FMT_DTYPE_LONG, ldata[i]);
            if (i< vector->n_size-1)
            {
                printf(", ");
            }
        }

        break;

    case TQ_FLOAT:
        fdata = (float *)vector->data;
        for (i = 0; i < vector->n_size; i++)
        {
            printf(FMT_DTYPE_FLOAT, fdata[i]);
            if (i< vector->n_size-1)
            {
                printf(", ");
            }
        }
        break;

    case TQ_DOUBLE:
        ddata = (double *)vector->data;
        for (i = 0; i < vector->n_size; i++)
        {
            printf(FMT_DTYPE_DOUBLE, ddata[i]);
            if (i< vector->n_size-1)
            {
                printf(", ");
            }
        }

        break;
    }

    printf(END_VECT);
}

void TQ_print_vector(TQ_Vector *vector)
{
    if (vector == NULL)
    {
        fprintf(stderr, "FreeError (in line %d): You are trying to print an uninitialized vector.", __LINE__ + 10);
        exit(EXIT_FAILURE);
    }

    if (vector->data == NULL || vector->n_size_bytes == 0)
    {
        fprintf(stderr, "FreeError (in line %d): The contents of this vector are freed.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    // Print the vector to display
    __TQ_print_vector(vector);
}

void TQ_delete_vector(TQ_Vector **vector)
{
    if (*vector == NULL || (*vector)->n_size_bytes == 0)
    {
        fprintf(stderr, "FreeError (in line %d): You are trying to delete an uninitialized vector.", __LINE__ + 11);
        exit(EXIT_FAILURE);
    }

    if ((*vector)->data == NULL)
    {
        fprintf(stderr, "FreeError (in line %d): The contents of this vector are already freed.", __LINE__ + 4);
        exit(EXIT_FAILURE);
    }

    free((*vector)->data);
    free(*vector);
}