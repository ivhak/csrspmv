#ifndef ARGS_H
#define ARGS_H

#define STR(a) #a

typedef enum {
    NO_FORMAT = 0,
    CSR       = 1,
    ELLPACK   = 2,
} matrix_format;

typedef struct {
    int iterations;
    int verbose;
    matrix_format format;
    char *matrix_market_path;
    int max_nonzeros_per_row;
    int benchmark_cpu;
} args_t;

void parse_args(int argc, char *argv[], args_t *args, int *help);

#endif

