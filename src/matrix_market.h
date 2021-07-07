#ifndef MATRIX_MARKET_H
#define MATRIX_MARKET_H
#include <stdlib.h>
typedef struct {
    int *row_indices;
    int *column_indices;
    double *values;
} matrix_market_t;

void free_matrix_market(matrix_market_t mm);
#endif
