#ifndef ELLPACK_H
#define ELLPACK_H

#include <float.h>
#include <limits.h>
#include "matrix_market.h"
#include "matrix_info.h"

#define ELLPACK_SENTINEL_INDEX INT_MAX
#define ELLPACK_SENTINEL_VALUE DBL_MAX

typedef struct {
    int *indices;
    double *data;
} ellpack_matrix_t;

void print_ellpack_matrix(ellpack_matrix_t ellpack, int num_rows, int max_nonzeros_per_row);

int ellpack_matrix_from_matrix_market(ellpack_matrix_t *ellpack,
                                      const matrix_market_t *mm,
                                      const matrix_info_t mi);
#endif
