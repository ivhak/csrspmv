#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

extern "C" {
#include "mmio.h"
}

#define HIP_CHECK(command) {     \
    hipError_t status = command; \
    if (status!=hipSuccess) {     \
        printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status)); \
        exit(1); \
    } \
}

/**
 * `csr_matrix_from_matrix_market()` converts a matrix in the
 * coordinate (COO) format, that is used in the Matrix Market file
 * format, to a sparse matrix in the compressed sparse row (CSR)
 * storage format.
 */
int csr_matrix_from_matrix_market(
    int num_rows,
    int num_columns,
    int num_nonzeros,
    const int * unsorted_row_indices,
    const int * unsorted_column_indices,
    const double * unsorted_values,
    int ** out_row_ptr,
    int ** out_column_indices,
    double ** out_values)
{
    /* Allocate storage for row pointers. */
    int * row_ptr = (int *) malloc((num_rows+1) * sizeof(int));
    if (!row_ptr) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        return errno;
    }

    /* Allocate storage for the column indices of each non-zero. */
    int * column_indices = (int *) malloc(num_nonzeros * sizeof(int));
    if (!column_indices) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(row_ptr);
        return errno;
    }

    /* Allocate storage for the value of each non-zero. */
    double * values = (double *) malloc(num_nonzeros * sizeof(double));
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
        row_ptr[unsorted_row_indices[k]+1]++;
    for (int i = 1; i <= num_rows; i++)
        row_ptr[i] += row_ptr[i-1];

    /* Sort column indices and non-zero values by their rows. */
    for (int k = 0; k < num_nonzeros; k++) {
        int i = unsorted_row_indices[k];
        column_indices[row_ptr[i]] = unsorted_column_indices[k];
        values[row_ptr[i]] = unsorted_values[k];
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

    *out_column_indices = column_indices;
    *out_row_ptr = row_ptr;
    *out_values = values;
    return 0;
}

// csr kernel
__global__ void csr_matrix_spmv(
            int num_rows, int num_columns, int num_nonzeros,
            int *row_ptr, int *column_indices, double *values, double *x, double *y)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= num_rows) return;

    double z = 0.0;
    for (int k = row_ptr[i]; k < row_ptr[i+1]; k++)
        z += values[k] * x[column_indices[k]];
    y[i] += z;
}

int benchmark_csr_matrix_spmv(const char *matrix_market_path)
{
    int err;

    int num_rows, num_columns, num_nonzeros;

    int *unsorted_row_indices, *unsorted_column_indices;

    double *unsorted_values;

    err = mm_read_unsymmetric_sparse(
        matrix_market_path, &num_rows, &num_columns, &num_nonzeros,
        &unsorted_values, &unsorted_row_indices, &unsorted_column_indices);

    if (err) return err;

    /* Convert to a compressed sparse row format. */
    int * row_ptr;
    int * column_indices;
    double * values;
    err = csr_matrix_from_matrix_market(
        num_rows, num_columns, num_nonzeros,
        unsorted_row_indices, unsorted_column_indices, unsorted_values,
        &row_ptr, &column_indices, &values);
    if (err) {
        free(unsorted_values);
        free(unsorted_column_indices);
        free(unsorted_row_indices);
        return err;
    }

    free(unsorted_values);
    free(unsorted_column_indices);
    free(unsorted_row_indices);

    /* Generate some sparse vector to use as the source vector for a
     * matrix-vector multiplication. */
    double * x = (double *) malloc(num_columns * sizeof(double));
    if (!x) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(values);
        free(row_ptr);
        free(column_indices);
        return errno;
    }

#pragma omp parallel for
    for (int j = 0; j < num_columns; j++)
        x[j] = 1.;

    /* Allocate storage for a destination vector for a matrix-vector
     * multiplication. */
    double * y = (double *) malloc(num_rows * sizeof(double));
    if (!y) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(x);
        free(values);
        free(row_ptr);
        free(column_indices);
        return errno;
    }

#pragma omp parallel for
    for (int i = 0; i < num_rows; i++)
        y[i] = 0.;

    int *d_row_ptr, *d_column_indices;

    HIP_CHECK(hipMalloc((void **)&d_row_ptr,        num_rows    * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_column_indices, num_columns * sizeof(int)));

    HIP_CHECK(hipMemcpy(row_ptr,        d_row_ptr,        num_rows    * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(column_indices, d_column_indices, num_columns * sizeof(int), hipMemcpyHostToDevice));


    double *d_values, *d_x, *d_y;

    HIP_CHECK(hipMalloc((void **)&d_values, num_rows    * sizeof(double)));
    HIP_CHECK(hipMalloc((void **)&d_x,      num_columns * sizeof(double)));
    HIP_CHECK(hipMalloc((void **)&d_y,      num_rows    * sizeof(double)));

    HIP_CHECK(hipMemcpy(values, d_values, num_rows    * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(x,      d_x,      num_columns * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(y,      d_y,      num_rows    * sizeof(double), hipMemcpyHostToDevice));


    dim3 block_size(1024);
    dim3 grid_size((num_rows + block_size.x - 1)/block_size.x);

    /* Compute the sparse matrix-vector multiplication. */
    {
        hipLaunchKernelGGL(csr_matrix_spmv, grid_size, block_size, 0, 0,
                           num_rows, num_columns, num_nonzeros,
                           d_row_ptr, d_column_indices, d_values, d_x, d_y);

    }

    HIP_CHECK(hipMemcpy(d_y, y, num_rows * sizeof(double), hipMemcpyDeviceToHost));

#if 1
    /* Write the results to standard output. */
    for (int i = 0; i < num_rows-1; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    hipFree(d_values);
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_row_ptr);
    hipFree(d_column_indices);

    free(y);
    free(x);
    free(values);
    free(column_indices);
    free(row_ptr);
    return 0;
}

int main(int argc, char *argv[])
{
    int err;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s FILE\n", argv[0]);
        return EXIT_FAILURE;
    }

    err = benchmark_csr_matrix_spmv(argv[1]);
}
