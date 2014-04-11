all: ../profile gpuprof cpuprof
.PHONY: gpuprof cpuprof

DEBUG=-O3 -Xcompiler -Wall -Xcompiler -Wextra
LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS_PATH=/usr/local/atlas3.10.1/lib/
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
