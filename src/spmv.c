#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "mmio.h"
#include "csr.h"
#include "matrix_market.h"

// `spmv_csr()` computes the multiplication of a sparse vector in the
// compressed sparse row (CSR) format with a dense vector, referred to as the
// source vector, to produce another dense vector, called the destination
// vector.
void spmv_csr(
    int num_rows, int num_columns, int num_nonzeros,
    csr_matrix_t *csr, double *x, double *y)
{
#pragma omp for
    for (int i = 0; i < num_rows; i++) {
        double z = 0.0;
        for (int k = csr->row_ptr[i]; k < csr->row_ptr[i+1]; k++)
            z += csr->values[k] * x[csr->column_indices[k]];
        y[i] += z;
    }
}

int main(int argc, char *argv[])
{
    int err;

    int num_rows, num_columns, num_nonzeros;

    // Struct to hold the matrix read in COO form.
    matrix_market_t mm;

    // Struct to hold the matrix as sparse vector in the compressed sparse row
    // (CSR) format.
    csr_matrix_t csr;

    // `x` is the dense vector multiplied with the sparse matrix
    // `y` is the dense vector holding the result
    double *x, *y;

    // The only argument should be an .mtx file
    if (argc < 2) {
        fprintf(stderr, "Usage: %s FILE\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Read in the sparse matrix in COO form
    err = mm_read_unsymmetric_sparse(
        argv[1], &num_rows, &num_columns, &num_nonzeros,
        &mm.values, &mm.row_indices, &mm.column_indices);

    if (err) return err;

    // Convert from COO to CSR
    err = csr_matrix_from_matrix_market(num_rows, num_columns, num_nonzeros, &mm, &csr);
    free_matrix_market(mm);

    if (err) return err;

    // Generate some sparse vector to use as the source vector for a
    // matrix-vector multiplication.
    x = (double *) malloc(num_columns * sizeof(double));
    if (!x) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(csr.values);
        free(csr.row_ptr);
        free(csr.column_indices);
        return errno;
    }

#pragma omp parallel for
    for (int j = 0; j < num_columns; j++)
        x[j] = 1.;

    // Allocate space for the result vector
    y = (double *) malloc(num_rows * sizeof(double));
    if (!y) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(x);
        free(csr.values);
        free(csr.row_ptr);
        free(csr.column_indices);
        return errno;
    }

#pragma omp parallel for
    for (int i = 0; i < num_rows; i++)
        y[i] = 0.;

    // Compute the sparse matrix-vector multiplication.
    spmv_csr(num_rows, num_columns, num_nonzeros, &csr, x, y);

#if 1
    // Write the results to standard output.
    for (int i = 0; i < num_rows-1; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    free_csr_matrix(csr);
    return 0;
}
