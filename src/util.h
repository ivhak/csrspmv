#ifndef UTIL_H
#define UTIL_H
#include <time.h>
#include "matrix_info.h"

void set_vector_double(double *v, int c, double val);

void  parse_args(int argc, char *argv[], char **matrix_market_path, int *max_nonzeros_per_row, int *verbose);
void  log_execution(const char *matrix_format, matrix_info_t mi, float time);
float time_spent(struct timespec t0, struct timespec t1);
#endif
