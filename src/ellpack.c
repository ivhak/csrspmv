#include "ellpack.h"
#include <errno.h>
#include <stdio.h>

void print_ellpack_matrix(ellpack_matrix_t ellpack, int num_rows, int max_nonzeros_per_row)
{
    printf("\nELLPACK matrix\n");
    printf("data\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < max_nonzeros_per_row; j++) {
            if (ellpack.data[i][j] == ELLPACK_SENTINEL_VALUE)
                printf(" *     ");
            else
                printf(" %.3lf", ellpack.data[i][j]);
        }
        printf("\n");
    }

    printf("\ndata\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < max_nonzeros_per_row; j++) {
            if (ellpack.indices[i][j] == ELLPACK_SENTINEL_VALUE)
                printf("* ");
            else
                printf("%d ", ellpack.indices[i][j]);
        }
        printf("\n");
    }

}

int ellpack_matrix_from_matrix_market(
    int num_rows,
    int num_columns,
    int num_nonzeros,
    int max_nonzeros_per_row,
    const matrix_market_t *mm,
    ellpack_matrix_t *ellpack)
{

    if (max_nonzeros_per_row > num_columns) return EINVAL;

    double **data = malloc(num_rows*sizeof(double *));
    for (int i = 0; i < num_rows; i++)
        data[i] = malloc(max_nonzeros_per_row*sizeof(double));

    int **indices = malloc(num_rows*sizeof(int *));
    for (int i = 0; i < num_rows; i++)
        indices[i] = malloc(max_nonzeros_per_row*sizeof(int));

#pragma omp parallel for
    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < max_nonzeros_per_row; j++) {
            indices[i][j] = ELLPACK_SENTINEL_INDEX;
            data[i][j]    = ELLPACK_SENTINEL_VALUE;
        }

#pragma omp for
    for (int i = 0; i < num_nonzeros; i++) {
        size_t row = mm->row_indices[i];

        int col = 0;
        while (indices[row][col] != ELLPACK_SENTINEL_INDEX)
            col++;

        indices[row][col] = mm->column_indices[i];
        data[row][col] = mm->values[i];

    }

    ellpack->data = data;
    ellpack->indices = indices;
    return 0;
}
