all: tester

MATIO=$(HOME)/lib/libmatio.a
LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS_PATH=/usr/local/atlas3.10.1/lib/
CUDA_PATH=/usr/local/cuda/lib64
DEBUG=-arch=sm_35 -rdc=true -O3 --compiler-options -Wall
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt

TPB=128

tester: dstedc.o dlaed0.o dlaed1.o dlaed2.o  \
		 dlaed3.o dlaed4.o dlamrg.o dlapy2.o matio.o tester.o
	nvcc $(DEBUG) -o $@ $^ $(MATIO) $(LAPACKE) $(ATLAS) $(CUDA)

dstedc.o: dstedc.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed0.o: dlaed0.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed1.o: dlaed1.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed2.o: dlaed2.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed3.o: dlaed3.cu
	nvcc $(DEBUG) -c $< -o $@ -DTPB=$(TPB)
dlaed4.o: dlaed4.cu
	nvcc $(DEBUG) -c $< -o $@
dlamrg.o: dlamrg.cu
	nvcc $(DEBUG) -c $< -o $@
dlapy2.o: dlapy2.cu
	nvcc $(DEBUG) -c $< -o $@
matio.o: matio.cu
	nvcc $(DEBUG) -c $< -o $@ -I$(HOME)/include
tester.o: tester.cu
	nvcc $(DEBUG) -c $< -o $@

clean:
	rm -fv tester *.o
