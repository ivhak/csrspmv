#include "args.h"
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

void parse_args(int argc, char *argv[], args_t *args, bool *help)
{
    int opt;
    extern int optind;
    extern char *optarg;
    while ((opt=getopt(argc, argv, "vm:f:chi:")) != -1) {
        switch (opt) {
        case 'h': *help = true; return;
        case 'm': args->max_nonzeros_per_row = atoi(optarg); break;
        case 'i': args->iterations = atoi(optarg);           break;
        case 'v': args->verbose = true;                      break;
        case 'c': args->benchmark_cpu = true;                break;
        case 'f':
            if (args->format & ALL) // Format is explicitly specified; do not run all the benchmarkss
                args->format ^= ALL;

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
