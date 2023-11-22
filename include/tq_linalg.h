#ifndef __TQ_LINALG_H_
#define __TQ_LINALG_H_

#include <stdlib.h>

#include "../include/tq_matrix.h"

TQ_Vector *TQ_vec_sum(TQ_Vector *A, TQ_Vector *B);
TQ_Vector *TQ_vec_sub(TQ_Vector *A, TQ_Vector *B);
TQ_Vector *TQ_vec_elemprod(TQ_Vector *A, TQ_Vector *B);
TQ_Vector *TQ_vec_scale(TQ_Vector *A, void *factor);

#endif