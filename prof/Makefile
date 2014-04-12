all: ../profile gpuprof cpuprof
.PHONY: gpuprof cpuprof

include ../make.inc

DEBUG=-O3 -Xcompiler -Wall -Xcompiler -Wextra
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm

../profile: main.c
	nvcc $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS)

gpuprof:
	$(MAKE) -C gpu/

cpuprof:
	$(MAKE) -C cpu/

clean:
	$(MAKE) -C gpu/ clean
	$(MAKE) -C cpu/ clean
	$(MAKE) -C dlaed4/ clean
	rm -fv ../profile
