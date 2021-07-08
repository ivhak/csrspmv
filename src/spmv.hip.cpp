#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

extern "C" {
#include "mmio.h"
#include "csr.h"
#include "matrix_market.h"
#include "ellpack.h"
#include "util.h"
}

#define HIP_CHECK(command) {     \
    hipError_t status = command; \
    if (status!=hipSuccess) {     \
        printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status)); \
        exit(1); \
    } \
}

// `spmv_csr_kernel()` computes the multiplication of a sparse vector in the
// compressed sparse row (CSR) format with a dense vector, referred to as the
// source vector, to produce another dense vector, called the destination
// vector.
__global__ void spmv_csr_kernel(const int num_rows,
                                const int num_columns,
                                const int num_nonzeros,
                                const csr_matrix_t *csr,
                                const double *x,
                                double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;

    double z = 0.0;
    for (int k = csr->row_ptr[i]; k < csr->row_ptr[i+1]; k++)
        z += csr->values[k] * x[csr->column_indices[k]];
    y[i] += z;
}

int benchmark_csr(matrix_market_t *mm,
                  const int num_rows,
                  const int num_columns,
                  const int num_nonzeros,
                  const double *x,
                  double *y) {
    int err;
    struct timespec start_time, end_time;

    // Host side CSR
    csr_matrix_t csr;

    // Device side CSR
    csr_matrix_t *d_csr;

    // Device side in and out vectors
    // `x` is the dense vector multiplied with the sparse matrix
    // `y` is the dense vector holding the result
    double *d_x, *d_y;

    // Convert from COO to CSR
    err = csr_matrix_from_matrix_market(&csr, mm, num_rows, num_columns, num_nonzeros);
    if (err) return err;

    // Allocate device arrays
    HIP_CHECK(hipMalloc((void **)&d_csr,                            1 * sizeof(csr_matrix_t)));
    HIP_CHECK(hipMalloc((void **)&d_csr->row_ptr,        (num_rows+1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_csr->column_indices, num_nonzeros * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_csr->values,         num_nonzeros * sizeof(double)));
    HIP_CHECK(hipMalloc((void **)&d_x,                   num_columns  * sizeof(double)));
    HIP_CHECK(hipMalloc((void **)&d_y,                   num_rows     * sizeof(double)));

    // Transfer data to device
    HIP_CHECK(hipMemcpy(d_csr->row_ptr,       csr.row_ptr,        (num_rows+1) * sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr->column_indices,csr.column_indices, num_nonzeros * sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr->values,        csr.values,         num_nonzeros * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,                  x,                  num_columns  * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y,                  y,                  num_rows     * sizeof(double), hipMemcpyHostToDevice));

    // Setup work dimensions
    dim3 block_size(1024);
    dim3 grid_size((num_rows + block_size.x - 1)/block_size.x);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Compute the sparse matrix-vector multiplication.
    hipLaunchKernelGGL(spmv_csr_kernel, grid_size, block_size, 0, 0,
                       num_rows, num_columns, num_nonzeros, d_csr, d_x, d_y);

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    log_execution("CSR", num_rows, num_columns, num_nonzeros, time_spent(start_time, end_time));

    // Copy back the resulting vector to host.
    HIP_CHECK(hipMemcpy(y, d_y, num_rows * sizeof(double), hipMemcpyDeviceToHost));

#if 1
    // Write the results to standard output.
    for (int i = 0; i < num_rows; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    hipFree(d_csr->values);
    hipFree(d_csr->row_ptr);
    hipFree(d_csr->column_indices);
    hipFree(d_csr);
    hipFree(d_x);
    hipFree(d_y);
    return 1;
}

// `spmv_ellpack_kernel()` computes the multiplication of a sparse vector in the
// ELLPACK format with a dense vector, referred to as the source vector, to
// produce another dense vector, called the destination vector.
//
// It is assumed that the sparse matrix has a maximum of `max_nonzeros_per_row`
// nonzeros per row
__global__ void spmv_ellpack_kernel(const int num_rows,
                                    const int num_columns,
                                    const int num_nonzeros,
                                    const int max_nonzeros_per_row,
                                    const ellpack_matrix_t *ellpack,
                                    const double *x,
                                    double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_rows)             return;
    if (j >= max_nonzeros_per_row) return;

    size_t col_index = ellpack->indices[i][j];
    if (col_index == ELLPACK_SENTINEL_INDEX)
        return;

    atomicAdd(&y[i], ellpack->data[i][j] * x[col_index]);
}

int benchmark_ellpack(matrix_market_t *mm,
                      const int num_rows,
                      const int num_columns,
                      const int num_nonzeros,
                      const int max_nonzeros_per_row,
                      const double *x,
                      double *y)
{
    int err;
    struct timespec start_time, end_time;

    // Host side ELLPACK
    ellpack_matrix_t ellpack;

    // Device side ELLPACK
    ellpack_matrix_t *d_ellpack;

    // Device side in and out vectors
    // `x` is the dense vector multiplied with the sparse matrix
    // `y` is the dense vector holding the result
    double *d_x, *d_y;

    // Convert from COO to ELLPACK
    err = ellpack_matrix_from_matrix_market(&ellpack, mm, num_rows, num_columns, num_nonzeros, max_nonzeros_per_row);
    if (err) return err;

    // Allocate device arrays
    HIP_CHECK(hipMalloc((void **)&d_ellpack, sizeof(ellpack_matrix_t)));

    HIP_CHECK(hipMalloc((void **)&d_ellpack->data, num_rows * sizeof(double *)));
    for (int i = 0; i < num_rows; i++)
        HIP_CHECK(hipMalloc((void **)&d_ellpack->data[i], max_nonzeros_per_row * sizeof(double)));

    HIP_CHECK(hipMalloc((void **)&d_ellpack->indices, num_rows * sizeof(int *)));
    for (int i = 0; i < num_rows; i++)
        HIP_CHECK(hipMalloc((void **)&d_ellpack->indices[i], max_nonzeros_per_row * sizeof(int)));

    HIP_CHECK(hipMalloc((void **)&d_x,                  num_columns  * sizeof(double)));
    HIP_CHECK(hipMalloc((void **)&d_y,                  num_rows     * sizeof(double)));

    // Copy data over to device
    for (int i = 0; i < num_rows; i++)
        HIP_CHECK(hipMemcpy(d_ellpack->data[i], ellpack.data[i], max_nonzeros_per_row * sizeof(double), hipMemcpyHostToDevice));

    for (int i = 0; i < num_rows; i++)
        HIP_CHECK(hipMemcpy(d_ellpack->indices[i], ellpack.indices[i], max_nonzeros_per_row * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(d_x, x, num_columns  * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, y, num_rows     * sizeof(double), hipMemcpyHostToDevice));

    // Setup work dimensions
    // Each block (most likely) works on one whole row.
    dim3 block_size(1024);
    dim3 grid_size((max_nonzeros_per_row + block_size.x - 1)/block_size.x, num_rows);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Compute the sparse matrix-vector multiplication.
    hipLaunchKernelGGL(spmv_ellpack_kernel, grid_size, block_size, 0, 0,
                       num_rows, num_columns, num_nonzeros, max_nonzeros_per_row, d_ellpack, d_x, d_y);

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    log_execution("ELLPACK", num_rows, num_columns, num_nonzeros, time_spent(start_time, end_time));

    // Copy back the resulting vector to host.
    HIP_CHECK(hipMemcpy(y, d_y, num_rows * sizeof(double), hipMemcpyDeviceToHost));

#if 1
    // Write the results to standard output.
    for (int i = 0; i < num_rows; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    for (int i = 0; i < max_nonzeros_per_row; i++) {
        hipFree(d_ellpack->data[i]);
        hipFree(d_ellpack->indices[i]);
    }
    hipFree(d_ellpack->data);
    hipFree(d_ellpack->indices);
    hipFree(d_ellpack);
    hipFree(d_x);
    hipFree(d_y);
    return 1;
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

    // Host side in/out vectors
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


    // Generate some sparse vector to use as the source vector for a
    // matrix-vector multiplication.
    x = (double *) malloc(num_columns * sizeof(double));
    if (!x) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        return errno;
    }
    set_vector_double(x, num_columns, 1.);

    // Allocate space for the result vector
    y = (double *) malloc(num_rows * sizeof(double));
    if (!y) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(x);
        return errno;
    }

    // Zero out the result vector
    set_vector_double(y, num_rows, 0.);

    // Run CSR benchmark
    benchmark_csr(&mm, num_rows, num_columns, num_nonzeros, x, y);

    // Zero out the result vector
    set_vector_double(y, num_rows, 0.);

    // Run ELLPACK benchmark
    benchmark_ellpack(&mm, num_rows, num_columns, num_nonzeros, 4, x, y);
    return 0;
}
