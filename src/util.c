#include "util.h"
#include <stdio.h>

inline void set_vector_double(double *v, int c, double val)
{
#pragma omp parallel for
    for (int i = 0; i < c; i++)
        v[i] = val;
}


void log_execution(const char *matrix_format, matrix_info_t mi, int iterations, float time)
{
    printf("%-14s  %d rows %d columns %d non-zero elements, %d iterations: %.6f seconds, %.6f seconds per iteration\n",
            matrix_format, mi.num_rows, mi.num_columns, mi.num_nonzeros, iterations, time, time / (float)iterations);
}

float time_spent(struct timespec t0, struct timespec t1)
{
    return ((t1.tv_sec - t0.tv_sec)) + ((t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}
