#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

extern "C" {
#include "mmio.h"
#include "csr.h"
#include "matrix_market.h"
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
__global__ void spmv_csr_kernel(
    int num_rows, int num_columns, int num_nonzeros,
    csr_matrix_t *csr, double *x, double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;

    double z = 0.0;
    for (int k = csr->row_ptr[i]; k < csr->row_ptr[i+1]; k++)
        z += csr->values[k] * x[csr->column_indices[k]];
    y[i] += z;
}

int main(int argc, char *argv[])
{
    int err;

    int num_rows, num_columns, num_nonzeros;

    // Struct to hold the matrix read in COO form.
    matrix_market_t mm;

    // Host side CSR
    csr_matrix_t csr;

    // Device side CSR
    csr_matrix_t *d_csr;

    // Host and device side in and out vectors
    // `x` is the dense vector multiplied with the sparse matrix
    // `y` is the dense vector holding the result
    double *x, *y;
    double *d_x, *d_y;

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
    if (err) return err;
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


    // Allocate device arrays
    HIP_CHECK(hipMalloc((void **)&d_csr,                           1 * sizeof(csr_matrix_t)));
    HIP_CHECK(hipMalloc((void **)&d_csr->row_ptr,       (num_rows+1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_csr->column_indices,num_nonzeros * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_csr->values,        num_nonzeros * sizeof(double)));
    HIP_CHECK(hipMalloc((void **)&d_x,                  num_columns  * sizeof(double)));
    HIP_CHECK(hipMalloc((void **)&d_y,                  num_rows     * sizeof(double)));

    // Transfer data to device
    HIP_CHECK(hipMemcpy(d_csr->row_ptr,       csr.row_ptr,        (num_rows+1) * sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr->column_indices,csr.column_indices, num_nonzeros * sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr->values,        csr.values,         num_nonzeros * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,                  x,                  num_columns  * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y,                  y,                  num_rows     * sizeof(double), hipMemcpyHostToDevice));

    // Setup work dimensions
    dim3 block_size(1024);
    dim3 grid_size((num_rows + block_size.x - 1)/block_size.x);

    // Compute the sparse matrix-vector multiplication.
    hipLaunchKernelGGL(spmv_csr_kernel, grid_size, block_size, 0, 0,
                       num_rows, num_columns, num_nonzeros, d_csr, d_x, d_y);

    // Copy back the resulting vector to host.
    HIP_CHECK(hipMemcpy(y, d_y, num_rows * sizeof(double), hipMemcpyDeviceToHost));

#if 1
    // Write the results to standard output.
    for (int i = 0; i < num_rows-1; i++)
        fprintf(stdout, "%12g\n", y[i]);
#endif

    hipFree(d_csr->values);
    hipFree(d_csr->row_ptr);
    hipFree(d_csr->column_indices);
    hipFree(d_csr);
    hipFree(d_x);
    hipFree(d_y);

    free_csr_matrix(csr);
    return 0;
}
