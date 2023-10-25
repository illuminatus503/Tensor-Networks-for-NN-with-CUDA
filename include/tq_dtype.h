#ifndef __TQ_GLOBAL_DTYPE_H_
#define __TQ_GLOBAL_DTYPE_H_

#define FMT_DTYPE_INT "%d "
#define FMT_DTYPE_LONG "%ld "
#define FMT_DTYPE_FLOAT "%3.6f "
#define FMT_DTYPE_DOUBLE "%3.15lf "

enum TQ_DTYPE
{
    TQ_INT,
    TQ_LONG,
    TQ_FLOAT,
    TQ_DOUBLE
} typedef TQ_DTYPE;

#endif