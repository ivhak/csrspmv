#include "util.h"
#include <stdio.h>

void log_execution(const char *matrix_format, int num_rows, int num_cols, int num_nonzeros, float time)
{
    printf("%s: %d rows %d columns, %d non-zero elements: %.6f seconds\n", matrix_format, num_rows, num_cols, num_nonzeros, time);
}

float time_spent(struct timespec t0, struct timespec t1)
{
    return ((t1.tv_sec - t0.tv_sec)) + ((t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}
