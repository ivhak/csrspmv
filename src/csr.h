#ifndef CSR_H
#define CSR_H
#include "matrix_market.h"

typedef struct {
    int *row_ptr;
    int *column_indices;
    double *values;
} csr_matrix_t;

void free_csr_matrix(csr_matrix_t csr);

int csr_matrix_from_matrix_market(
    int num_rows,
    int num_columns,
    int num_nonzeros,
    const matrix_market_t *mm,
    csr_matrix_t *csr);

#endif
