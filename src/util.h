#ifndef UTIL_H
#define UTIL_H
#include <time.h>

void set_vector_double(double *v, int c, double val);

void  parse_args(int argc, char *argv[], char **matrix_market_path, int *max_nonzeros_per_row);
void  log_execution(const char *matrix_format, int num_rows, int num_cols, int num_nonzeros, float time);
float time_spent(struct timespec t0, struct timespec t1);
#endif
