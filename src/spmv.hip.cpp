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
    .format               = ALL,       // Which formats to benchmark
    .matrix_market_path   = NULL,      // Filename of the input file
    .max_nonzeros_per_row = -1,        // Max number of nonzeros per row in the matrix
    .benchmark_cpu        = false      // If true, run benchmarks on CPU as well
};

#define HC(command) {     \
    hipError_t status = command; \
    if (status!=hipSuccess) {     \
        printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status)); \
        exit(1); \
    } \
}

__global__ void spmv_csr_kernel (const matrix_info_t mi, const csr_matrix_t csr,         const double *x, double *y);
__host__   void spmv_csr_serial (const matrix_info_t mi, const csr_matrix_t csr,         const double *x, double *y);

__global__ void spmv_ellpack_kernel                   (const matrix_info_t mi, const ellpack_matrix_t ellpack, const double *x, double *y);
__global__ void spmv_ellpack_column_major_kernel      (const matrix_info_t mi, const ellpack_matrix_t ellpack, const double *x, double *y);
__global__ void spmv_ellpack_column_major_tiled_kernel(const matrix_info_t mi, const ellpack_matrix_t ellpack, const double *x, double *y);
__host__   void spmv_ellpack_serial                   (const matrix_info_t mi, const ellpack_matrix_t ellpack, const double *x, double *y);

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
        "              by repeating the option. Defaults to running all formats if no\n"
        "              format is specified.\n\n"
        "   -m  NUM    Sets the maximum number of nonzero values per row for the\n"
        "              matrix in INPUT_FILE. Must be less than or equal to the\n"
        "              number of columns in the matrix. Required when using the\n"
        "              ELLPACK format.\n\n"
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
    matrix_info_t mi = { 0 };

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

    // Read in the sparse matrix in COO form
    err = mm_read_unsymmetric_sparse(
        args.matrix_market_path,
        &mi.num_rows, &mi.num_columns, &mi.num_nonzeros,
        &mm.values, &mm.row_indices, &mm.column_indices);
    if (err) return err;

    if (args.max_nonzeros_per_row == -1 && args.format & ELLPACK) {
        fprintf(stderr,
                "It is required to set the maximal number of nonzeros per row when "
                "using the ELLPACK format. Set with the -m flag.\n");
        exit(1);
    }

    // Copy the maxmimum number of nonzeros per row into the matrix_info struct
    mi.max_nonzeros_per_row = args.max_nonzeros_per_row;

    if (mi.max_nonzeros_per_row > mi.num_columns) {
        fprintf(stderr,
                "Maximum number of nonzero elements per row specified by -m (%d) "
                "is larger than the number of columns (%d) in %s\n",
                mi.max_nonzeros_per_row, mi.num_columns, args.matrix_market_path);
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
        return errno;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    print_header(mi, args.iterations);

    if (args.format & (CSR | ALL)) {
        // Zero out the result vector
        set_vector_double(y, mi.num_rows, 0.);

        // Run CSR benchmark
        benchmark_csr(&mm, mi, x, y);
    }

    if (args.format & (ELLPACK | ALL)) {
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

    // Host and device side CSR matrices
    csr_matrix_t csr, d_csr;

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

    {   // CSR, GPU
        dim3 block_size(1024);
        dim3 grid_size((mi.num_rows + block_size.x - 1)/block_size.x);

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Compute the sparse matrix-vector multiplication.
        for (int i = 0; i < args.iterations; i++)
            hipLaunchKernelGGL(spmv_csr_kernel, grid_size, block_size, 0, 0, mi, d_csr, d_x, d_y);
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        log_execution("CSR (GPU)", args.iterations, time_spent(start_time, end_time));

        // Copy back the resulting vector to host.
        HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

        if (args.verbose) print_vector(y, mi);
    }

    // CSR, CPU
    if (args.benchmark_cpu) {
        set_vector_double(y, mi.num_rows, 0.);
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        for (int i = 0; i < args.iterations; i++)
            spmv_csr_serial(mi, csr, x, y);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        log_execution("CSR (CPU)", args.iterations, time_spent(start_time, end_time));

        if (args.verbose) print_vector(y, mi);
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

// `spmv_ellpack_column_major_kernel()` computes the multiplication of a sparse
// vector in the ELLPACK format, stored as column major, with a dense vector,
// referred to as the source vector, to produce another dense vector, called
// the destination vector.
//
// It is assumed that the sparse matrix has a maximum of `max_nonzeros_per_row` nonzeros per row
__global__ void
spmv_ellpack_column_major_kernel(const matrix_info_t mi,
                                 const ellpack_matrix_t ellpack,
                                 const double *x,
                                 double *y)
{
    // This kernel uses one thread per row.

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= mi.num_rows) return;

    double z = 0.;
    for (int i = 0; i < mi.max_nonzeros_per_row; i++) {
        size_t col_index = ellpack.indices[i*mi.num_rows+row];
        if (col_index == ELLPACK_SENTINEL_INDEX)
            continue;

        z += ellpack.data[i*mi.num_rows+row] * x[col_index];
    }

    y[row] += z;
}

__global__ void
spmv_ellpack_column_major_tiled_kernel(const matrix_info_t mi,
                                       const ellpack_matrix_t ellpack,
                                       const double *x,
                                       double *y,
                                       int tile_size)
{
    // This kernel uses one thread per row.
    const int N = mi.max_nonzeros_per_row;
    const int T = tile_size;

    int col     = threadIdx.x;
    int row_off = blockIdx.x*N*T;

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= mi.num_rows) return;

    double z = 0.;
    for (int i = 0; i < N; i++) {
        size_t col_index = ellpack.indices[row_off + i*T + col];
        if (col_index == ELLPACK_SENTINEL_INDEX)
            break;

        z += ellpack.data[row_off + i*T + col] * x[col_index];
    }

    y[row] += z;
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
// `spmv_ellpack_*_kernel`) and on the CPU (with `spmv_ellpack_serial`)
int
benchmark_ellpack(const matrix_market_t *mm,
                  const matrix_info_t mi,
                  const double *x,
                  double *y)
{
    int err;
    struct timespec start_time, end_time;

    // Host and device side ELLPACK matrices
    ellpack_matrix_t ellpack, ellpack_col_major, ellpack_col_major_tiled;
    ellpack_matrix_t d_ellpack;

    // Device side in and out vectors
    // `x` is the dense vector multiplied with the sparse matrix
    // `y` is the dense vector holding the result
    double *d_x, *d_y;

    // Convert from COO to ELLPACK
    err = ellpack_matrix_from_matrix_market(&ellpack, mm, mi);
    if (err) return err;

    const int num_elems = mi.num_rows * mi.max_nonzeros_per_row;

    const int tile_size = 256;

    // Setup the column major ellpack, and transpose the regular ELLPACK matrix into it.
    init_ellpack(&ellpack_col_major, num_elems);
    transpose_ellpack(&ellpack, &ellpack_col_major, mi);

    // Setup the column major, tiled ellpack.
    //
    // For the tiled version of the column major ELLPACK to work, the tile size
    // has to divide the number of rows evenly. Since we choose the tile size
    // to be a number that makes the data coalesce well, we align the data to
    // fit the tile size rather than choosing a tile size that fits the data.
    const int aligned_num_rows = tile_size*((mi.num_rows + tile_size - 1)/tile_size);
    const int tiled_num_elems = aligned_num_rows *  mi.max_nonzeros_per_row;
    init_ellpack(&ellpack_col_major_tiled, tiled_num_elems);
    tiled_transpose_ellpack(&ellpack, &ellpack_col_major_tiled, mi, tile_size);

    // Allocate device arrays. The same device side ellpack_matrixt_t is used
    // for all the different benchmarks, so it has to fit the largest of the
    // setups, i.e., `tiled_num_elems`. `tiled_num_elems` will be in the range
    // [`num_elems`, `num_elems`+`tile_size`-1], so it is not like we are
    // allocating a lot more than we need for the other ones.
    HC(hipMalloc((void **)&d_ellpack.data,    tiled_num_elems * sizeof(double)));
    HC(hipMalloc((void **)&d_ellpack.indices, tiled_num_elems * sizeof(int)));

    // Allocate the device side in/out vectors
    HC(hipMalloc((void **)&d_x, mi.num_columns * sizeof(double)));
    HC(hipMalloc((void **)&d_y, mi.num_rows    * sizeof(double)));

    // Copy over the input/output vectors
    HC(hipMemcpy(d_x, x, mi.num_columns * sizeof(double), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_y, y, mi.num_rows    * sizeof(double), hipMemcpyHostToDevice));

    // Run the ELLPACK benchmarks:
    //  * The first version runs with one thread per row, row major
    //  * The second version runs with one thread per row, with the matrix stored column major.
    //  * The third and last version runs on the CPU.

    {   // ELLPACK: One thread per row
        dim3 block_size(64);
        dim3 grid_size((mi.num_rows+block_size.x-1)/block_size.x);

        // Copy the ELLPACK for the first two benchmarks; normal row major
        HC(hipMemcpy(d_ellpack.data,    ellpack.data,    num_elems * sizeof(double), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_ellpack.indices, ellpack.indices, num_elems * sizeof(int),    hipMemcpyHostToDevice));

        // Zero out the result vector
        HC(hipMemset(d_y, 0, mi.num_rows*sizeof(double)));

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Compute the sparse matrix-vector multiplication.
        for (int i = 0; i < args.iterations; i++)
            hipLaunchKernelGGL(spmv_ellpack_kernel, grid_size, block_size, 0, 0, mi, d_ellpack, d_x, d_y);
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        log_execution("ELLPACK (GPU) (Row major)", args.iterations, time_spent(start_time, end_time));

        // Copy back the resulting vector to host.
        HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

        if (args.verbose) print_vector(y, mi);
    }

    {   // ELLPACK: One thread per row, column major

        // Copy over the column major host side ELLPACK
        HC(hipMemcpy(d_ellpack.data,    ellpack_col_major.data,    num_elems * sizeof(double), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_ellpack.indices, ellpack_col_major.indices, num_elems * sizeof(int),    hipMemcpyHostToDevice));

        dim3 block_size(256);
        dim3 grid_size((mi.num_rows+block_size.x-1)/block_size.x);

        // Zero out the result vector
        HC(hipMemset(d_y, 0, mi.num_rows*sizeof(double)));

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Compute the sparse matrix-vector multiplication.
        for (int i = 0; i < args.iterations; i++)
            hipLaunchKernelGGL(spmv_ellpack_column_major_kernel, grid_size, block_size, 0, 0, mi, d_ellpack, d_x, d_y);
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        log_execution("ELLPACK (GPU) (Column major)", args.iterations, time_spent(start_time, end_time));

        // Copy back the resulting vector to host.
        HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

        if (args.verbose) print_vector(y, mi);
    }

    {   // ELLPACK: One thread per row, column major, tiled

        // Copy over the column major, tiled host side ELLPACK
        HC(hipMemcpy(d_ellpack.data,    ellpack_col_major_tiled.data,    tiled_num_elems * sizeof(double), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_ellpack.indices, ellpack_col_major_tiled.indices, tiled_num_elems * sizeof(int),    hipMemcpyHostToDevice));

        dim3 block_size(tile_size);
        dim3 grid_size((mi.num_rows+block_size.x-1)/block_size.x);

        // Zero out the result vector
        HC(hipMemset(d_y, 0, mi.num_rows*sizeof(double)));

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Compute the sparse matrix-vector multiplication.
        for (int i = 0; i < args.iterations; i++)
            hipLaunchKernelGGL(spmv_ellpack_column_major_tiled_kernel, grid_size, block_size, 0, 0, mi, d_ellpack, d_x, d_y, tile_size);
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        log_execution("ELLPACK (GPU) (Column major,tiled)", args.iterations, time_spent(start_time, end_time));

        // Copy back the resulting vector to host.
        HC(hipMemcpy(y, d_y, mi.num_rows * sizeof(double), hipMemcpyDeviceToHost));

        if (args.verbose) print_vector(y, mi);
    }

    // CPU
    if (args.benchmark_cpu) {
        // Zero out the result vector and rerun the benchmark on the CPU
        set_vector_double(y, mi.num_rows, 0.);
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        for (int i = 0; i < args.iterations; i++)
            spmv_ellpack_serial(mi, ellpack, x, y);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        log_execution("ELLPACK (CPU)", args.iterations, time_spent(start_time, end_time));

        if (args.verbose) print_vector(y, mi);
    }

    HC(hipFree(d_ellpack.data));
    HC(hipFree(d_ellpack.indices));
    HC(hipFree(d_x));
    HC(hipFree(d_y));
    return 1;
}
