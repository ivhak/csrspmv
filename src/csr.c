#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "csr.h"
#include "matrix_market.h"

void free_csr_matrix(csr_matrix_t csr)
{
    free(csr.row_ptr);
    free(csr.column_indices);
    free(csr.values);
}

void print_csr_matrix(csr_matrix_t csr, int num_rows, int num_nonzeros)
{
    printf("CSR\n");
    printf("values:\n");
    for (int i = 0; i < num_nonzeros; i++)
        printf(" %6.6lf", csr.values[i]);

    printf("\nrow_ptr:\n");
    for (int i = 0; i < num_rows+1; i++)
        printf(" %d", csr.row_ptr[i]);
    printf("\ncolumn_indices:\n");
    for (int i = 0; i < num_nonzeros; i++)
        printf(" %d", csr.column_indices[i]);
    printf("\n");

}

// `csr_matrix_from_matrix_market()` converts a matrix in the
// coordinate (COO) format, that is used in the Matrix Market file
// format, to a sparse matrix in the compressed sparse row (CSR)
// storage format.
int csr_matrix_from_matrix_market(
    csr_matrix_t *csr,
    const matrix_market_t *mm,
    const int num_rows,
    const int num_columns,
    const int num_nonzeros)
{
    int *row_ptr;
    int *column_indices;
    double *values;

    /* Allocate storage for row pointers. */
    row_ptr = (int *) malloc((num_rows+1) * sizeof(int));
    if (!row_ptr) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        return errno;
    }

    /* Allocate storage for the column indices of each non-zero. */
    column_indices = (int *) malloc(num_nonzeros * sizeof(int));
    if (!column_indices) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(row_ptr);
        return errno;
    }

    /* Allocate storage for the value of each non-zero. */
    values = (double *) malloc(num_nonzeros * sizeof(double));
    if (!values) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(row_ptr);
        free(column_indices);
        return errno;
    }

    /* Initialise the allocated arrays with zeros. */
#pragma omp parallel for
    for (int i = 0; i <= num_rows; i++)
        row_ptr[i] = 0;
#pragma omp parallel for
    for (int k = 0; k < num_nonzeros; k++) {
        column_indices[k] = 0;
        values[k] = 0;
    }

    /* Count the number of non-zeros in each row. */
    for (int k = 0; k < num_nonzeros; k++)
        row_ptr[mm->row_indices[k]+1]++;
    for (int i = 1; i <= num_rows; i++)
        row_ptr[i] += row_ptr[i-1];

    /* Sort column indices and non-zero values by their rows. */
    for (int k = 0; k < num_nonzeros; k++) {
        int i = mm->row_indices[k];
        column_indices[row_ptr[i]] = mm->column_indices[k];
        values[row_ptr[i]] = mm->values[k];
        row_ptr[i]++;
    }

    /* Adjust the row pointers after sorting. */
    for (int i = num_rows; i > 0; i--)
        row_ptr[i] = row_ptr[i-1];
    row_ptr[0] = 0;

    /*
     * Sort the non-zeros within each row by their column indices.
     * Here, a simple insertion sort algorithm is used.
     */
    for (int i = 0; i < num_rows; i++) {
        int num_nonzeros = row_ptr[i+1] - row_ptr[i];
        for (int k = 0; k < num_nonzeros; k++) {
            int column_index = column_indices[row_ptr[i]+k];
            double value = values[row_ptr[i]+k];
            int j = k-1;
            while (j >= 0 && column_indices[row_ptr[i]+j] > column_index) {
                column_indices[row_ptr[i]+j+1] = column_indices[row_ptr[i]+j];
                values[row_ptr[i]+j+1] = values[row_ptr[i]+j];
                j--;
            }
            column_indices[row_ptr[i]+j+1] = column_index;
            values[row_ptr[i]+j+1] = value;
        }
    }

    csr->row_ptr = row_ptr;
    csr->column_indices = column_indices;
    csr->values = values;
    return 0;
}
