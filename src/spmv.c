#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "mmio.h"
#include "csr.h"
#include "matrix_market.h"
#include "ellpack.h"
#include "util.h"

// `spmv_csr()` computes the multiplication of a sparse vector in the
// compressed sparse row (CSR) format with a dense vector, referred to as the
// source vector, to produce another dense vector, called the destination
// vector.
void spmv_csr(const int num_rows,
              const int num_columns,
              const int num_nonzeros,
              const csr_matrix_t *csr,
              const double *x,
              double *y)
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
void spmv_ellpack(const int num_rows,
                  const int num_columns,
                  const int num_nonzeros,
                  const int max_nonzeros_per_row,
                  const ellpack_matrix_t *ellpack,
                  const double *x,
                  double *y)
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
    int max_nonzeros_per_row = 4;
    char *matrix_market_path = NULL;

    parse_args(argc, argv, &matrix_market_path, &max_nonzeros_per_row);

    if (matrix_market_path == NULL) {
        fprintf(stderr, "Usage: %s FILE\n", argv[0]);
        return EXIT_FAILURE;
    }

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

    // Read in the sparse matrix in COO form
    err = mm_read_unsymmetric_sparse(
        matrix_market_path, &num_rows, &num_columns, &num_nonzeros,
        &mm.values, &mm.row_indices, &mm.column_indices);

    if (err) return err;

    if (max_nonzeros_per_row > num_columns) {
        fprintf(stderr,
                "Maximum number of nonzero elements per row specified by -m (%d) "
                "is larger than the number of columns (%d) in %s\n",
                max_nonzeros_per_row, num_columns, matrix_market_path);
        free_matrix_market(mm);
        exit(1);
    }

    // Convert from COO to CSR
    err = csr_matrix_from_matrix_market(&csr, &mm, num_rows, num_columns, num_nonzeros);
    if (err) return err;

    err = ellpack_matrix_from_matrix_market(&ellpack, &mm, num_rows, num_columns, num_nonzeros, max_nonzeros_per_row);
    if (err) return err;

#if 1
    print_matrix_market(mm, num_nonzeros);
    print_csr_matrix(csr, num_rows, num_nonzeros);
    print_ellpack_matrix(ellpack, num_rows, max_nonzeros_per_row);
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
    set_vector_double(x, num_columns, 1.);

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

    // Zero out the result vector
    set_vector_double(y, num_rows, 0.);

    // Compute the sparse matrix-vector multiplication.
    spmv_csr(num_rows, num_columns, num_nonzeros, &csr, x, y);

#if 1
    printf("CSR\n");
    // Write the results to standard output.
    for (int i = 0; i < num_rows; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    // Zero out the result vector
    set_vector_double(y, num_rows, 0.);
    spmv_ellpack(num_rows, num_columns, num_nonzeros, max_nonzeros_per_row, &ellpack, x, y);

#if 1
    printf("ELLPACK\n");
    // Write the results to standard output.
    for (int i = 0; i < num_rows; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    free_csr_matrix(csr);
    return 0;
}
