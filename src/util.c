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

void parse_args(int argc, char *argv[], char **matrix_market_path, int *max_nonzeros_per_row, int *verbose)
{
    int opt;
    extern int optind;
    extern char *optarg;
    while ((opt=getopt(argc, argv, "vm:")) != -1) {
        switch (opt) {
            case 'm':
                *max_nonzeros_per_row = atoi(optarg);
                break;
            case 'v':
                *verbose = 1;
                break;
            default:
                break;
        }
    }
    *matrix_market_path=argv[optind];
}

void log_execution(const char *matrix_format, matrix_info_t mi, float time)
{
    printf("%-14s  %d rows %d columns %d non-zero elements: %.6f seconds\n", matrix_format, mi.num_rows, mi.num_columns, mi.num_nonzeros, time);
}

float time_spent(struct timespec t0, struct timespec t1)
{
    return ((t1.tv_sec - t0.tv_sec)) + ((t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}
