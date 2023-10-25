#include "../include/tq_print_tensor.h"

void TQ_print_tensor_element(void *data, TQ_DTYPE dtype)
{
    switch (dtype)
    {
    case TQ_INT:
        printf(FMT_DTYPE_INT, *((int *)data));
        break;
    case TQ_LONG:
        printf(FMT_DTYPE_LONG, *((long *)data));
        break;
    case TQ_FLOAT:
        printf(FMT_DTYPE_FLOAT, *((float *)data));
        break;
    case TQ_DOUBLE:
        printf(FMT_DTYPE_DOUBLE, *((double *)data));
        break;
    }
}

void TQ_print_tensor_recursive(void *data,
                               TQ_DTYPE dtype,
                               size_t dtype_bytes,
                               size_t *dims,
                               size_t ndims,
                               size_t depth,
                               const char *begin,
                               const char *end)
{
    size_t i, j;

    if (depth == ndims - 1)
    {
        printf("%s", begin);
        for (i = 0; i < dims[depth]; i++)
        {
            TQ_print_tensor_element(data, dtype);
            if (i < dims[depth] - 1)
            {
                printf(", ");
            }

            data = (void *)((char *)data + dtype_bytes);
        }
        printf("%s", end);
    }
    else
    {
        printf("%s", begin);
        for (i = 0; i < dims[depth]; i++)
        {
            TQ_print_tensor_recursive(data,
                                      dtype, dtype_bytes,
                                      dims, ndims,
                                      depth + 1,
                                      begin, end);
            if (i < dims[depth] - 1)
            {
                printf(",\n");
                for (j = 0; j <= depth; j++)
                {
                    printf(" ");
                }
            }
            data = (void *)((char *)data + dtype_bytes);
        }
        printf("%s", end);
    }
}

void TQ_print_tensor(TQ_Matrix *tensor, const char *begin, const char *end)
{
    size_t *dims = (size_t *)(tensor->shape->data);
    size_t ndims = tensor->shape->n_size;

    void *tensor_data = tensor->data->data;
    TQ_DTYPE dtype = tensor->dtype;
    size_t dtype_bytes = tensor->data->dtype_bytes;

    TQ_print_tensor_recursive(tensor_data,
                              dtype, dtype_bytes,
                              dims, ndims,
                              0,
                              begin, end);

    printf("\n");
}
