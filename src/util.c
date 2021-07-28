#include "util.h"
#include <stdio.h>


inline void set_vector_double(double *v, int c, double val)
{
#pragma omp parallel for
    for (int i = 0; i < c; i++)
        v[i] = val;
}

inline void print_vector(double *y, const matrix_info_t mi)
{
    // Write the results to standard output.
    for (int i = 0; i < MIN(PRINT_MAX_ROWS,mi.num_rows); i++)
        fprintf(stdout, "%6g ", y[i]);
    fprintf(stdout, "\n");
}


void log_execution(const char *matrix_format, int iterations, float time)
{
    printf("%-34s  %.6f seconds, %.6f sec/it\n", matrix_format, time, time / (float)iterations);
}

void print_header(matrix_info_t mi, int iterations)
{
    printf("%dx%d matrix, %d nonzero elements, %d max nonzeroes per row, %d iterations\n",
            mi.num_rows, mi.num_columns, mi.num_nonzeros, mi.max_nonzeros_per_row, iterations);
}

float time_spent(struct timespec t0, struct timespec t1)
{
    return ((t1.tv_sec - t0.tv_sec)) + ((t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}
