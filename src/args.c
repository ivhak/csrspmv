#include "args.h"
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

void parse_args(int argc, char *argv[], args_t *args, int *help)
{
    int opt;
    extern int optind;
    extern char *optarg;
    while ((opt=getopt(argc, argv, "vm:f:ch")) != -1) {
        switch (opt) {
        case 'h': *help = 1; return;
        case 'm': args->max_nonzeros_per_row = atoi(optarg); break;
        case 'v': args->verbose = 1;                         break;
        case 'c': args->benchmark_cpu = 1;                   break;
        case 'f':
            if      (strncmp("ELLPACK", optarg, sizeof("ELLPACK")) == 0) args->format |= ELLPACK;
            else if (strncmp("CSR",     optarg, sizeof("CSR")) == 0)     args->format |= CSR;
            break;
        default:
            break;
        }
    }

    // The last argument should be the filename of the matrix market file
    args->matrix_market_path=argv[optind];
}
