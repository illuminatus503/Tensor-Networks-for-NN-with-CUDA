#ifndef __TQ_TENSOR_PRINT_H_
#define __TQ_TENSOR_PRINT_H_

#include <stdlib.h>
#include <stdio.h>

#include "tq_matrix.h"

void TQ_print_tensor(TQ_Matrix *tensor,
                     const char *begin,
                     const char *end);

#endif