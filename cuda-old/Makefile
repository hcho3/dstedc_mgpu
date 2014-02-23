all: tester

STATIC=$(HOME)/lib/libmatio.a \
       /usr/local/plasma2.5.0/lib/liblapacke.a
SHARED=-lm -lz -llapack -lblas
CUDA=-L/usr/local/cuda/lib64 -lcudadevrt -lcublas -lcublas_device
DEBUG=-O3

TPB=128

tester: dlaed0.o dlaed4.o matio.o tester.o
	nvcc -arch=sm_35 -rdc=true $(DEBUG) -o $@ $^ $(STATIC) $(SHARED) $(CUDA)

#dstedc.o: dstedc.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
dlaed0.o: dlaed0.cu dlaed1.cu dlaed2.cu dlaed3.cu dlamrg.cu dlacpy.cu dlapy2.cu
	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@ -DTPB=$(TPB)
#dlaed1.o: dlaed1.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
#dlaed2.o: dlaed2.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
#dlaed3.o: dlaed3.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
dlaed4.o: dlaed4.cu
	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
#initial_guess.o: initial_guess.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
#middle_way.o: middle_way.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
#dlamrg.o: dlamrg.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
#dlapy2.o: dlapy2.cu
#	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@
matio.o: matio.cu
	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@ -I$(HOME)/include
tester.o: tester.cu
	nvcc -arch=sm_35 -rdc=true  $(DEBUG) -c $< -o $@

clean:
	rm -fv tester *.o
