#ifndef UTIL_H
#define UTIL_H
#include <time.h>

void  log_execution(const char *matrix_format, int num_rows, int num_cols, int num_nonzeros, float time);
float time_spent(struct timespec t0, struct timespec t1);
#endif
