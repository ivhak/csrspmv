#ifndef ELLPACK_H
#define ELLPACK_H

#include "matrix_market.h"

#define ELLPACK_SENTINEL_INDEX -100000000
#define ELLPACK_SENTINEL_VALUE -100000000.0

typedef struct {
    int **indices;
    double **data;
} ellpack_matrix_t;

void print_ellpack_matrix(ellpack_matrix_t ellpack, int num_rows, int max_nonzeros_per_row);

int ellpack_matrix_from_matrix_market(
    int num_rows,
    int num_columns,
    int num_nonzeros,
    int max_nonzeros_per_row,
    const matrix_market_t *mm,
    ellpack_matrix_t *ellpack);
#endif
