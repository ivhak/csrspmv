#include "matrix_market.h"

void free_matrix_market(matrix_market_t mm)
{
    free(mm.row_indices);
    free(mm.column_indices);
    free(mm.values);
}
