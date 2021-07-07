#include "matrix_market.h"
#include <stdio.h>

void print_matrix_market(matrix_market_t mm, int num_nonzeros)
{
    printf ("matrix market\n");
    printf ("values:\n");
    for (int i = 0; i < num_nonzeros; i++)
        printf(" %lf", mm.values[i]);
    printf ("\nrow indices:\n");
    for (int i = 0; i < num_nonzeros; i++)
        printf(" %d", mm.row_indices[i]);
    printf ("\ncolumn indices:\n");
    for (int i = 0; i < num_nonzeros; i++)
        printf(" %d", mm.column_indices[i]);
    printf("\n");
}

void free_matrix_market(matrix_market_t mm)
{
    free(mm.row_indices);
    free(mm.column_indices);
    free(mm.values);
}
