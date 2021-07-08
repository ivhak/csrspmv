#ifndef CSR_H
#define CSR_H
#include "matrix_market.h"

typedef struct {
    int *row_ptr;
    int *column_indices;
    double *values;
} csr_matrix_t;

void free_csr_matrix(csr_matrix_t csr);

void print_csr_matrix(csr_matrix_t csr, int num_rows, int num_nonzeros);

int csr_matrix_from_matrix_market(csr_matrix_t *csr,
                                  const matrix_market_t *mm,
                                  const int num_rows,
                                  const int num_columns,
                                  const int num_nonzeros);

#endif
