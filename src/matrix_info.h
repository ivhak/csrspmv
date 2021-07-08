#ifndef SPMV_H
#define SPMV_H

typedef struct {
    int num_rows;
    int num_columns;
    int num_nonzeros;
    int max_nonzeros_per_row;
} matrix_info_t;

#endif
