##
## Example program for sparse matrix-vector multiplication with the
## compressed sparse row (CSR) format.
##

hip:    spmv-hip
serial: spmv


HIPCC := hipcc

CFLAGS += -ggdb3 -Wall
LDFLAGS :=

ifndef NO_OPENMP
CFLAGS += -fopenmp
endif

ifdef DEBUG
CFLAGS += -O0
else
CFLAGS += -O3
endif


C_SRC := src/mmio.c src/csr.c src/matrix_market.c src/ellpack.c src/util.c src/args.c
C_OBJ := $(patsubst %.c,%.o,$(C_SRC))


$(C_OBJ): %.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@

spmv-hip: ${C_OBJ} src/spmv.hip.cpp
	$(HIPCC) $(CFLAGS) $^ $(LDFLAGS) -o $@

clean:
	$(RM) spmv-hip spmv $(C_OBJ)

.PHONY: all clean hip
