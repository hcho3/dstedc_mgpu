all: tester

DEBUG=-arch=sm_35 -rdc=true -O3 -Xcompiler -Wall -Xcompiler -Wextra \
	-Xcompiler -fopenmp

ATLAS_PATH=/usr/local/atlas3.10.1/lib/
CUDA_PATH=/usr/local/cuda/lib64

LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt

TPB=128

tester: dlaed0_m.o dlaed1.o dlaed1_ph2.o dlaed2.o dlaed3.o dlaed3_ph2.o  \
		 dlaed4.o dlaed4_ph2.o dlamrg.o dlapy2.o matio.o tester.o workspace.o
	nvcc $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) $(CUDA)

dlaed0_m.o: dlaed0_m.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed1.o: dlaed1.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed1_ph2.o: dlaed1_ph2.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed2.o: dlaed2.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed3.o: dlaed3.cu
	nvcc $(DEBUG) -c $< -o $@ -DTPB=$(TPB)
dlaed3_ph2.o: dlaed3_ph2.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed4.o: dlaed4.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed4_ph2.o: dlaed4_ph2.cu
	nvcc $(DEBUG) -c $< -o $@
dlamrg.o: dlamrg.cu
	nvcc $(DEBUG) -c $< -o $@
dlapy2.o: dlapy2.cu
	nvcc $(DEBUG) -c $< -o $@
matio.o: matio.cu
	nvcc $(DEBUG) -c $< -o $@
tester.o: tester.cu
	nvcc $(DEBUG) -c $< -o $@
workspace.o: workspace.cu
	nvcc $(DEBUG) -c $< -o $@

clean:
	rm -fv tester *.o
