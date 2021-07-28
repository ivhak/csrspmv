#ifndef ARGS_H
#define ARGS_H
#include <stdbool.h>

#define STR(a) #a

typedef enum {
    NO_FORMAT = 0,
    ALL       = 1,
    CSR       = 2,
    ELLPACK   = 4,
} matrix_format;

typedef struct {
    int iterations;
    bool verbose;
    matrix_format format;
    char *matrix_market_path;
    int max_nonzeros_per_row;
    bool benchmark_cpu;
} args_t;

void parse_args(int argc, char *argv[], args_t *args, bool *help);

#endif

