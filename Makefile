##
## Example program for sparse matrix-vector multiplication with the
## compressed sparse row (CSR) format.
##

hip:    spmv-hip
serial: spmv

clean: $(clean-programs)

.PHONY: all clean hip

HIPCC := hipcc

CFLAGS += -ggdb3 -Wall
LDFLAGS += -lm

ifndef NO_OPENMP
CFLAGS += -fopenmp
endif

ifdef DEBUG
CFLAGS += -O0
else
CFLAGS += -O3
endif


C_SRC := src/mmio.c src/csr.c src/matrix_market.c src/ellpack.c src/util.c
C_OBJ := $(patsubst %.c,%.o,$(C_SRC))


$(C_OBJ): %.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

spmv-hip: ${C_OBJ} src/spmv.hip.cpp
	$(HIPCC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

spmv: ${C_OBJ} src/spmv.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

clean:
	$(RM) spmv-hip spmv $(C_OBJ)
