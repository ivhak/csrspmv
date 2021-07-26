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
#include "matrix_info.h"
#include "args.h"
}

// Defualt settings
args_t args = {
    .iterations           = 1,         // Number of iterations to run each benchmark
    .verbose              = false,     // If true, print up to PRINT_MAX_ROWS rows of output
    .format               = NO_FORMAT, // Which formats to benchmark
    .matrix_market_path   = NULL,      // Filename of the input file
    .max_nonzeros_per_row = 4,         // Max number of nonzeros per row in the matrix
    .benchmark_cpu        = false      // If true, run benchmarks on CPU as well
};

#define PRINT_MAX_ROWS 10
#define MIN(a,b) (a < b ? a : b)

#define HC(command) {     \
    hipError_t status = command; \
    if (status!=hipSuccess) {     \
        printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status)); \
        exit(1); \
    } \
}


__global__ void spmv_csr_kernel    (const matrix_info_t mi, const csr_matrix_t csr,         const double *x, double *y);
__global__ void spmv_ellpack_kernel(const matrix_info_t mi, const ellpack_matrix_t ellpack, const double *x, double *y);

__host__   void spmv_csr_serial    (const matrix_info_t mi, const csr_matrix_t csr,         const double *x, double *y);
__host__   void spmv_ellpack_serial(const matrix_info_t mi, const ellpack_matrix_t ellpack, const double *x, double *y);

int benchmark_csr    (const matrix_market_t *mm, const matrix_info_t mi, const double *x, double *y);
int benchmark_ellpack(const matrix_market_t *mm, const matrix_info_t mi, const double *x, double *y);

void
usage(FILE *stream, char *filename) {
    fprintf(stream,
        "Usage: %1$s [-f FORMAT] [-c] [-v] [-i NUM] [-m NUM] [-h] INPUT_FILE\n"
        "Options:\n"
        "   -h         Show usage.\n\n"
        "   -c         Run the benchmarks on the CPU as well as the GPU.\n\n"
        "   -i  NUM    Run the benchmarks for NUM iterations.\n\n"
        "   -v         Be verbose, show the output of the SpMV calculation(s).\n\n"
        "   -f  FORMAT Run the benchmarks using the format FORMAT, where FORMAT\n"
        "              is either CSR or ELLPACK. More than one format can be specified\n"
        "              by repeating the option. Defaults to CSR if no format is specified.\n\n"
        "   -m  NUM    Sets the maximum number of nonzero values per row for the\n"
        "              matrix in INPUT_FILE. Must be less than or equal to the\n"
        "              number of columns in the matrix.\n\n"
        "Example:\n\n"
        "   Run ELLPACK on both the GPU and CPU, on the matrix in mat.mtx,\n"
        "   which has 16 nonzeros per row:\n\n"
        "       %1$s -f ELLPACK -m 16 -c mat.mtx\n",
        filename);
}

int
main(int argc, char *argv[])
{
    int err;

    // Struct to keep track of general information about the matrix
    matrix_info_t mi = {
        .num_rows             = 0,
        .num_columns          = 0,
        .num_nonzeros         = 0,
        .max_nonzeros_per_row = 4
    };

    bool help = false;
    parse_args(argc, argv, &args, &help);

    if (help) {
        usage(stdout, argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.matrix_market_path == NULL) {
        usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    // Struct to hold the matrix read in COO form.
    matrix_market_t mm;

    // Host side in/out vectors
    double *x, *y;

    if (args.format == NO_FORMAT) {
        printf("No matrix format supplied, defaulting to CSR...\n");
        args.format = CSR;
    }

    // Read in the sparse matrix in COO form
    err = mm_read_unsymmetric_sparse(
        args.matrix_market_path,
        &mi.num_rows, &mi.num_columns, &mi.num_nonzeros,
        &mm.values, &mm.row_indices, &mm.column_indices);
    if (err) return err;

    // Copy the maxmimum number of nonzeros per row into the matrix_info struct
    mi.max_nonzeros_per_row = args.max_nonzeros_per_row;

    if (mi.max_nonzeros_per_row > mi.num_columns) {
        fprintf(stderr,
                "Maximum number of nonzero elements per row specified by -m (%d) "
                "is larger than the number of columns (%d) in %s\n",
                mi.max_nonzeros_per_row, mi.num_columns, args.matrix_market_path);
        free_matrix_market(mm);
        exit(1);
    }


    // Generate some sparse vector to use as the source vector for a
    // matrix-vector multiplication.
    x = (double *) malloc(mi.num_columns * sizeof(double));
    if (!x) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        return errno;
    }
    set_vector_double(x, mi.num_columns, 1.);

    // Allocate space for the result vector
    y = (double *) malloc(mi.num_rows * sizeof(double));
    if (!y) {
        fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
        free(x);
        return errno;
    }


    if (args.format & CSR) {
        // Zero out the result vector
        set_vector_double(y, mi.num_rows, 0.);

        // Run CSR benchmark
        benchmark_csr(&mm, mi, x, y);
    }

    if (args.format & ELLPACK) {
        // Zero out the result vector
        set_vector_double(y, mi.num_rows, 0.);

        // Run ELLPACK benchmark
        benchmark_ellpack(&mm, mi, x, y);
    }
    return 0;
}

// `spmv_csr_kernel()` computes the multiplication of a sparse vector in the compressed sparse row
// (CSR) format with a dense vector, referred to as the source vector, to produce another dense
// vector, called the destination vector.
__global__ void
spmv_csr_kernel(const matrix_info_t mi,
                const csr_matrix_t csr,
                const double *x,
                double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mi.num_rows) return;

    double z = 0.0;
    for (int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++)
        z += csr.values[k] * x[csr.column_indices[k]];
    y[i] += z;
}

// `spmv_csr_serial()` computes the multiplication of a sparse vector in the compressed sparse row
// (CSR) format with a dense vector, referred to as the source vector, to produce another dense
// vector, called the destination vector.
__host__ void
spmv_csr_serial(const matrix_info_t mi,
                const csr_matrix_t csr,
                const double *x,
                double *y)
{
#pragma omp for
    for (int i = 0; i < mi.num_rows; i++) {
        double z = 0.0;
        for (int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++)
            z += csr.values[k] * x[csr.column_indices[k]];
        y[i] += z;
    }
}

// `benchmark_csr()` converts the matrix in COO form to CSR, and the performs SpMV on the resulting
// CSR matrix with the dense vector `x`. This is done on both the GPU (with `spmv_ellpack_kernel`)
// and on the CPU (with `spmv_ellpack_serial`)
int
benchmark_csr(const matrix_market_t *mm,
              const matrix_info_t mi,
              const double *x,
              double *y)
{
    int err;
    struct timespec start_time, end_time;

    // Host side CSR
    csr_matrix_t csr;

    // Device side CSR
    csr_matrix_t d_csr;

    // Device side in and out vectors
    // `x` is the dense vector multiplied with the sparse matrix
    // `y` is the dense vector holding the result
    double *d_x, *d_y;

    // Convert from COO to CSR
    err = csr_matrix_from_matrix_market(&csr, mm, mi);
    if (err) return err;

    // Allocate device side structs and arrays
    HC(hipMalloc((void **)&d_csr.row_ptr,        (mi.num_rows+1) * sizeof(int)));
    HC(hipMalloc((void **)&d_csr.column_indices, mi.num_nonzeros * sizeof(int)));
    HC(hipMalloc((void **)&d_csr.values,         mi.num_nonzeros * sizeof(double)));
    HC(hipMalloc((void **)&d_x,                  mi.num_columns  * sizeof(double)));
    HC(hipMalloc((void **)&d_y,                  mi.num_rows     * sizeof(double)));

    // Transfer data to device
    HC(hipMemcpy(d_csr.row_ptr,        csr.row_ptr,        (mi.num_rows+1) * sizeof(int),    hipMemcpyHostToDevice));
    HC(hipMemcpy(d_csr.column_indices, csr.column_indices, mi.num_nonzeros * sizeof(int),    hipMemcpyHostToDevice));
    HC(hipMemcpy(d_csr.values,         csr.values,         mi.num_nonzeros * sizeof(double), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_x,                  x,                  mi.num_columns  * sizeof(double), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_y,                  y,                  mi.num_rows     * sizeof(double), hipMemcpyHostToDevice));

    // Setup work dimensions
    dim3 block_size(1024);
    dim3 grid_size((mi.num_rows + block_size.x - 1)/block_size.x);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Compute the sparse matrix-vector multiplication.
    for (int i = 0; i < args.iterations; i++) {
        hipLaunchKernelGGL(spmv_csr_kernel, grid_size, block_size, 0, 0, mi, d_csr, d_x, d_y);
    }
    hipDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    log_execution("CSR (GPU)", mi, args.iterations, time_spent(start_time, end_time));

    // Copy back the resulting vector to host.
    HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

    if (args.verbose) {
        // Write the results to standard output.
        for (int i = 0; i < MIN(PRINT_MAX_ROWS,mi.num_rows); i++)
            fprintf(stdout, "%6g ", y[i]);
        fprintf(stdout, "\n");
    }

    // CPU
    if (args.benchmark_cpu) {
        set_vector_double(y, mi.num_rows, 0.);
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        for (int i = 0; i < args.iterations; i++)
            spmv_csr_serial(mi, csr, x, y);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        log_execution("CSR (CPU)", mi, args.iterations, time_spent(start_time, end_time));

        if (args.verbose) {
            // Write the results to standard output.
            for (int i = 0; i < MIN(PRINT_MAX_ROWS,mi.num_rows); i++)
                fprintf(stdout, "%6g ", y[i]);
            fprintf(stdout, "\n");
        }
    }

    HC(hipFree(d_csr.values));
    HC(hipFree(d_csr.row_ptr));
    HC(hipFree(d_csr.column_indices));
    HC(hipFree(d_x));
    HC(hipFree(d_y));
    return 1;
}

// `spmv_ellpack_kernel()` computes the multiplication of a sparse vector in the ELLPACK format with
// a dense vector, referred to as the source vector, to produce another dense vector, called the
// destination vector.
//
// It is assumed that the sparse matrix has a maximum of `max_nonzeros_per_row` nonzeros per row
__global__ void
spmv_ellpack_kernel(const matrix_info_t mi,
                    const ellpack_matrix_t ellpack,
                    const double *x,
                    double *y)
{
    // This kernel uses one block per row, one thread per element.
    // If blockDim.x > mi.max_nonzeros_per_row, there number of blocks per row
    // is scaled to fit the problem.

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= mi.max_nonzeros_per_row) return;
    if (row >= mi.num_rows)             return;

    size_t col_index = ellpack.indices[row*mi.max_nonzeros_per_row+col];
    if (col_index == ELLPACK_SENTINEL_INDEX)
        return;

    atomicAdd(&y[row], ellpack.data[row*mi.max_nonzeros_per_row+col] * x[col_index]);
}

// `spmv_ellpack_kernel2()` computes the multiplication of a sparse vector in the ELLPACK format with
// a dense vector, referred to as the source vector, to produce another dense vector, called the
// destination vector.
//
// It is assumed that the sparse matrix has a maximum of `max_nonzeros_per_row` nonzeros per row
__global__ void
spmv_ellpack_kernel2(const matrix_info_t mi,
                     const ellpack_matrix_t ellpack,
                     const double *x,
                     double *y)
{
    // This kernel uses one thread per row.

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= mi.num_rows) return;

    double z = 0.;
    for (int i = 0; i < mi.max_nonzeros_per_row; i++) {
        size_t col_index = ellpack.indices[row*mi.max_nonzeros_per_row+i];
        if (col_index == ELLPACK_SENTINEL_INDEX)
            break;

        z += ellpack.data[row*mi.max_nonzeros_per_row+i] * x[col_index];
    }

    y[row] += z;
}

__global__ void
spmv_ellpack_transposed_kernel(const matrix_info_t mi,
                               const ellpack_matrix_t ellpack,
                               const double *x,
                               double *y)
{
    // This kernel uses one thread per row.

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= mi.num_rows) return;

#if 1
    double z = 0.;
    for (int i = 0; i < mi.max_nonzeros_per_row; i++) {
        size_t col_index = ellpack.indices[i*mi.num_rows+row];
        if (col_index == ELLPACK_SENTINEL_INDEX)
            break;

        z += ellpack.data[i*mi.num_rows+row] * x[col_index];
    }

    y[row] += z;
#endif
}

// `spmv_ellpack_serial()` computes the multiplication of a sparse vector in the ELLPACK format with
// a dense vector, referred to as the source vector, to produce another dense vector, called the
// destination vector.
//
// It is assumed that the sparse matrix has a maximum of `max_nonzeros_per_row` nonzeros per row
__host__ void
spmv_ellpack_serial(const matrix_info_t mi,
                    const ellpack_matrix_t ellpack,
                    const double *x,
                    double *y)
{
#pragma omp parallel for
    for (int i = 0; i < mi.num_rows; i++) {
        double z = 0.0;
        for (int j = 0; j < mi.max_nonzeros_per_row; j++) {

            size_t col_index = ellpack.indices[i*mi.max_nonzeros_per_row+j];

            if (col_index == ELLPACK_SENTINEL_INDEX)
                break;

            z += ellpack.data[i*mi.max_nonzeros_per_row+j] * x[col_index];
        }
        y[i] += z;
    }
}

// `benchmark_ellpack()` converts the matrix in COO form to ELLPACK, and the performs SpMV on the
// resulting ELL matrix with the dense vector `x`. This is done on both the GPU (with
// `spmv_ellpack_kernel`) and on the CPU (with `spmv_ellpack_serial`)
int
benchmark_ellpack(const matrix_market_t *mm,
                  const matrix_info_t mi,
                  const double *x,
                  double *y)
{
    int err;
    struct timespec start_time, end_time;

    // host side ellpack
    ellpack_matrix_t ellpack;

    // host side ellpack transposed
    ellpack_matrix_t ellpack_trans;

    // Device side ELLPACK
    ellpack_matrix_t d_ellpack;

    // Device side ELLPACK transposed
    ellpack_matrix_t d_ellpack_trans;


    // Device side in and out vectors
    // `x` is the dense vector multiplied with the sparse matrix
    // `y` is the dense vector holding the result
    double *d_x, *d_y;

    // Convert from COO to ELLPACK
    err = ellpack_matrix_from_matrix_market(&ellpack, mm, mi);
    if (err) return err;


    const size_t num_elems = mi.num_rows * mi.max_nonzeros_per_row;

    ellpack_trans.data    = (double *)malloc(num_elems*sizeof(double));
    ellpack_trans.indices = (int *)   malloc(num_elems*sizeof(int));
    tranpose_ellpack(&ellpack, &ellpack_trans, mi);

    // Allocate device arrays
    HC(hipMalloc((void **)&d_ellpack.data,    num_elems * sizeof(double)));
    HC(hipMalloc((void **)&d_ellpack.indices, num_elems * sizeof(int)));

    HC(hipMalloc((void **)&d_ellpack_trans.data,    num_elems * sizeof(double)));
    HC(hipMalloc((void **)&d_ellpack_trans.indices, num_elems * sizeof(int)));

    // Allocate the device side in/out vectors
    HC(hipMalloc((void **)&d_x, mi.num_columns * sizeof(double)));
    HC(hipMalloc((void **)&d_y, mi.num_rows    * sizeof(double)));

    // Copy data over to device
    HC(hipMemcpy(d_ellpack.data,    ellpack.data,    num_elems * sizeof(double), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ellpack.indices, ellpack.indices, num_elems * sizeof(int),    hipMemcpyHostToDevice));

    HC(hipMemcpy(d_ellpack_trans.data,    ellpack_trans.data,    num_elems * sizeof(double), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ellpack_trans.indices, ellpack_trans.indices, num_elems * sizeof(int),    hipMemcpyHostToDevice));

    HC(hipMemcpy(d_x, x, mi.num_columns * sizeof(double), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_y, y, mi.num_rows    * sizeof(double), hipMemcpyHostToDevice));

    { // ELLPACK: One block per row
        dim3 block_size(64); // one warp per row
        dim3 grid_size((mi.max_nonzeros_per_row + block_size.x - 1)/block_size.x, mi.num_rows);

        HC(hipMemset(d_y, 0, mi.num_rows*sizeof(double)));

        clock_gettime(CLOCK_MONOTONIC, &start_time);


        // Compute the sparse matrix-vector multiplication.
        for (int i = 0; i < args.iterations; i++) {
            hipLaunchKernelGGL(spmv_ellpack_kernel, grid_size, block_size, 0, 0, mi, d_ellpack, d_x, d_y);
        }
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        log_execution("ELLPACK (ONE BLOCK PER ROW)  (GPU)", mi, args.iterations, time_spent(start_time, end_time));

        // Copy back the resulting vector to host.
        HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

        if (args.verbose) {
            // Write the results to standard output.
            for (int i = 0; i < MIN(PRINT_MAX_ROWS,mi.num_rows); i++)
                fprintf(stdout, "%6g ", y[i]);
            fprintf(stdout, "\n");
        }
    }


    { // ELLPACK: One thread per row
        dim3 block_size(64); // One warp per block
        dim3 grid_size((mi.num_rows+block_size.x-1)/block_size.x);

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Compute the sparse matrix-vector multiplication.
        for (int i = 0; i < args.iterations; i++) {
            hipLaunchKernelGGL(spmv_ellpack_kernel2, grid_size, block_size, 0, 0, mi, d_ellpack, d_x, d_y);
        }
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        log_execution("ELLPACK (ONE THREAD PER ROW) (GPU)", mi, args.iterations, time_spent(start_time, end_time));

        // Copy back the resulting vector to host.
        HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

        if (args.verbose) {
            // Write the results to standard output.
            for (int i = 0; i < MIN(PRINT_MAX_ROWS,mi.num_rows); i++)
                fprintf(stdout, "%6g ", y[i]);
            fprintf(stdout, "\n");
        }
    }


    { // ELLPACK: Transposed
        dim3 block_size(64); // one warp per block
        dim3 grid_size((mi.num_rows+block_size.x-1)/block_size.x);

        HC(hipMemset(d_y, 0, mi.num_rows*sizeof(double)));

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Compute the sparse matrix-vector multiplication.
        for (int i = 0; i < args.iterations; i++) {
            hipLaunchKernelGGL(spmv_ellpack_transposed_kernel, grid_size, block_size, 0, 0, mi, d_ellpack_trans, d_x, d_y);
        }
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        log_execution("ELLPACK (TRANSPOSED)         (GPU)", mi, args.iterations, time_spent(start_time, end_time));

        // Copy back the resulting vector to host.
        HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

        if (args.verbose) {
            // Write the results to standard output.
            for (int i = 0; i < MIN(PRINT_MAX_ROWS,mi.num_rows); i++)
                fprintf(stdout, "%6g ", y[i]);
            fprintf(stdout, "\n");
        }
    }


    // CPU
    if (args.benchmark_cpu) {
        // Zero out the result vector and rerun the benchmark on the CPU
        set_vector_double(y, mi.num_rows, 0.);
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        for (int i = 0; i < args.iterations; i++)
            spmv_ellpack_serial(mi, ellpack, x, y);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        log_execution("ELLPACK (CPU)", mi, args.iterations, time_spent(start_time, end_time));

        if (args.verbose) {
            // Write the results to standard output.
            for (int i = 0; i < MIN(PRINT_MAX_ROWS,mi.num_rows); i++)
                fprintf(stdout, "%6g ", y[i]);
            fprintf(stdout, "\n");
        }
    }

    HC(hipFree(d_ellpack.data));
    HC(hipFree(d_ellpack.indices));
    HC(hipFree(d_ellpack_trans.data));
    HC(hipFree(d_ellpack_trans.indices));
    HC(hipFree(d_x));
    HC(hipFree(d_y));
    return 1;
}

