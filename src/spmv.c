#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "mmio.h"
#include "csr.h"
#include "matrix_market.h"
#include "ellpack.h"

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
// `spmv_ellpack()` computes the multiplication of a sparse vector in the
// ELLPACK format with a dense vector, referred to as the source vector, to
// produce another dense vector, called the destination vector.
//
// It is assumed that the sparse matrix has a maximum of `max_nonzeros_per_row`
// nonzeros per row
void spmv_ellpack(
    int num_rows, int num_columns, int num_nonzeros, int max_nonzeros_per_row,
    ellpack_matrix_t *ellpack, double *x, double *y)
{
    for (int i = 0; i < num_rows; i++) {
        double z = 0.0;
        for (int j = 0; j < max_nonzeros_per_row; j++) {

            size_t col_index = ellpack->indices[i][j];
            if (col_index == ELLPACK_SENTINEL_INDEX)
                break;

            z += ellpack->data[i][j] * x[col_index];
        }
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
    //
    // Struct to hold the matrix as sparse vector in the ELLPACK format.
    ellpack_matrix_t ellpack;

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
    // err = ellpack_matrix_from_matrix_market(num_rows, num_columns, num_nonzeros, 4, &mm, &ellpack);
    if (err) return err;

#if 0
    print_matrix_market(mm, num_nonzeros);
    print_csr_matrix(csr, num_rows, num_nonzeros);
    print_ellpack_matrix(ellpack, num_rows, 4);
#endif
    free_matrix_market(mm);

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
    printf("CSR\n");
    // Write the results to standard output.
    for (int i = 0; i < num_rows; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

#pragma omp parallel for
    for (int i = 0; i < num_rows; i++)
        y[i] = 0.;

    spmv_ellpack(num_rows, num_columns, num_nonzeros, 4, &ellpack, x, y);

#if 1
    printf("ELLPACK\n");
    // Write the results to standard output.
    for (int i = 0; i < num_rows; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    free_csr_matrix(csr);
    return 0;
}
