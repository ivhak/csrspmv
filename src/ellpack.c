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
