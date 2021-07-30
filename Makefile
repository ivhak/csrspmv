##
## Example program for sparse matrix-vector multiplication with the
## compressed sparse row (CSR) format.
##

HIPCC := hipcc

CFLAGS += -g -Wall
LDFLAGS :=
CPPFALGS := -03

ifndef NO_OPENMP
CFLAGS += -fopenmp
CPPFLAGS += -Xcompiler -fopenmp
endif

ifdef DEBUG
CFLAGS += -O0
else
CFLAGS += -O3
endif

C_SRC := src/mmio.c \
		 src/csr.c \
		 src/matrix_market.c \
		 src/ellpack.c \
		 src/util.c \
		 src/args.c

C_OBJ := $(patsubst %.c,%.o,$(C_SRC))

amd: export HIP_PLATFORM=amd
amd: spmv-hip-amd

nvidia: export HIP_PLATFORM=nvidia
nvidia: export CUDA_PATH=/cm/shared/apps/cuda11.0/toolkit/11.0.3
nvidia: CPPFLAGS += -arch=sm_70
nvidia: spmv-hip-nvidia

$(C_OBJ): %.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@

src/spmv.hip.o: src/spmv.hip.cpp
	$(HIPCC) -c $(CPPFLAGS) $^ $(LDFLAGS) -o $@

spmv-hip-%: ${C_OBJ} src/spmv.hip.o
	$(HIPCC) $(CPPFLAGS) $^ $(LDFLAGS) -o $@

clean:
	$(RM) src/spmv.hip.o spmv-hip-nvidia spmv-hip-amd $(C_OBJ)

.PHONY: all clean hip
