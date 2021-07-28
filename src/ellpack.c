#include <errno.h>
#include <stdio.h>

#include "ellpack.h"

void print_ellpack_matrix(ellpack_matrix_t ellpack, int num_rows, int width)
{
    printf("\nELLPACK matrix\n");
    printf("data\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < width; j++) {
            if (ellpack.data[i*width+j] == ELLPACK_SENTINEL_VALUE)
                printf(" *     ");
            else
                printf(" %.3lf", ellpack.data[i*width+j]);
        }
        printf("\n");
    }

    printf("\ndata\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < width; j++) {
            if (ellpack.indices[i*width+j] == ELLPACK_SENTINEL_VALUE)
                printf("* ");
            else
                printf("%d ", ellpack.indices[i*width+j]);
        }
        printf("\n");
    }

}

void init_ellpack(ellpack_matrix_t *ellpack, int num_elems)
{
    ellpack->data    = malloc(num_elems*sizeof(double));
    ellpack->indices = malloc(num_elems*sizeof(int));
}

// `ellpack_matrix_from_matrix_market()` converts a matrix in the
// coordinate (COO) format, that is used in the Matrix Market file
// format, to a sparse matrix in the ELLPACK storage format.
int ellpack_matrix_from_matrix_market(ellpack_matrix_t *ellpack,
                                      const matrix_market_t *mm,
                                      const matrix_info_t mi)
{

    if (mi.max_nonzeros_per_row > mi.num_columns) return EINVAL;

    size_t width = mi.max_nonzeros_per_row;

    double *data = malloc(mi.num_rows*width*sizeof(double *));
    int *indices = malloc(mi.num_rows*width*sizeof(int *));

    // Preset both indices and data to the sentinel values.
#pragma omp parallel for
    for (int i = 0; i < mi.num_rows; i++)
        for (int j = 0; j < mi.max_nonzeros_per_row; j++) {
            indices[i*width + j] = ELLPACK_SENTINEL_INDEX;
            data[i*width + j]    = ELLPACK_SENTINEL_VALUE;
        }

#pragma omp for
    for (int i = 0; i < mi.num_nonzeros; i++) {
        size_t row = mm->row_indices[i];

        // Find the first column not used, i.e., the first column containing a
        // sentinel value.
        int col = 0;
        while (indices[row*width+col] != ELLPACK_SENTINEL_INDEX && col < width)
            col++;

        indices[row*width+col] = mm->column_indices[i];
        data[row*width+col] = mm->values[i];

    }

    ellpack->data = data;
    ellpack->indices = indices;
    return 0;
}

void transpose_ellpack(ellpack_matrix_t *in, ellpack_matrix_t *out, const matrix_info_t mi)
{
    for (int i = 0; i < mi.num_rows; i++) {
        for (int j = 0; j < mi.max_nonzeros_per_row; j++) {
            out->data[j*mi.num_rows + i] = in->data[i*mi.max_nonzeros_per_row + j];
            out->indices[j*mi.num_rows + i] = in->indices[i*mi.max_nonzeros_per_row + j];
        }
    }
}

void tiled_transpose_ellpack(ellpack_matrix_t *in,
                            ellpack_matrix_t *out,
                            const matrix_info_t mi,
                            int tile_size)
{
    // Take an M x N matrix and transpose submatrices of dimensions T x N to
    // submatrices of dimenions N x T, resulting in a matrix of dimenions
    // (M*N/T)xT
    //
    // Example:
    //
    //            N=3                           T=2
    //       +---+---+---+                   +---+---+
    //       | a | b | c |                   | a | d |
    //       +---+---+---+                   +---+---+
    //       | d | e | f |   T=2             | b | e |
    // M=4   +---+---+---+   --->    M*N/T=6 +---+---+
    //       | g | h | i |                   | c | f |
    //       +---+---+---+                   +---+---+
    //       | j | k | l |                   | g | j |
    //       +---+---+---+                   +---+---+
    //                                       | h | k |
    //                                       +---+---+
    //                                       | i | l |
    //                                       +---+---+
    //
    // For this to work, the tile size has to divide the number of rows evenly.

    const int n = mi.max_nonzeros_per_row;

    for (int i = 0; i < mi.num_rows; i++) {
        for (int j = 0; j < mi.max_nonzeros_per_row; j++) {
            size_t out_row = j + n*(i/tile_size);
            size_t out_col = i % tile_size;

            out->data   [out_row*tile_size + out_col] = in->data   [i*n+j];
            out->indices[out_row*tile_size + out_col] = in->indices[i*n+j];
        }
    }
}
