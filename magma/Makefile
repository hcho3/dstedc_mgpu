all: tester

MATIO=$(HOME)/lib/libmatio.a
LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS_PATH=/usr/local/atlas3.10.1/lib/
CUDA_PATH=/usr/local/cuda/lib64
DEBUG=-arch=sm_35 -rdc=true -O3 -Xcompiler -Wall -Xcompiler -fopenmp
MAGMA=-L$(HOME)/magma-1.4.1-beta2/lib -lmagma
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt

tester: matio.o tester.o
	nvcc $(DEBUG) -o $@ $^ $(MAGMA) $(MATIO) $(LAPACKE) $(ATLAS) $(CUDA)

matio.o: matio.cu
	nvcc $(DEBUG) -c $< -o $@ -I$(HOME)/include
tester.o: tester.cu
	nvcc $(DEBUG) -c $< -o $@ -I$(HOME)/magma-1.4.1-beta2/include

clean:
	rm -fv tester *.o
