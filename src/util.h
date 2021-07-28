#ifndef UTIL_H
#define UTIL_H
#include <time.h>
#include "matrix_info.h"

#define PRINT_MAX_ROWS 10
#define MIN(a,b) (a < b ? a : b)

void  set_vector_double(double *v, int c, double val);
void  print_vector(double *y, const matrix_info_t mi);
void  print_header(matrix_info_t mi, int iterations);
void  log_execution(const char *matrix_format, int iterations, float time);
float time_spent(struct timespec t0, struct timespec t1);
#endif
