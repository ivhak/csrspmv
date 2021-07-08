#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

inline void set_vector_double(double *v, int c, double val)
{
#pragma omp parallel for
    for (int i = 0; i < c; i++)
        v[i] = val;
}

void parse_args(int argc, char *argv[], char **matrix_market_path, int *max_nonzeros_per_row)
{
    int opt;
    extern int optind;
    extern char *optarg;
    while ((opt=getopt(argc, argv, "m:")) != -1) {
        switch (opt) {
            case 'm':
                *max_nonzeros_per_row = atoi(optarg);
            default:
                break;
        }
    }
    *matrix_market_path=argv[optind];
}

void log_execution(const char *matrix_format, int num_rows, int num_cols, int num_nonzeros, float time)
{
    printf("%s: %d rows %d columns, %d non-zero elements: %.6f seconds\n", matrix_format, num_rows, num_cols, num_nonzeros, time);
}

float time_spent(struct timespec t0, struct timespec t1)
{
    return ((t1.tv_sec - t0.tv_sec)) + ((t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}
