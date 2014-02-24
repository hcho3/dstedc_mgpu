all: gpu cpu

DEBUG=-arch=sm_35 -rdc=true -O3 -Xcompiler -Wall -Xcompiler -Wextra
OBJDIR=obj

ATLAS_PATH=/usr/local/atlas3.10.1/lib/
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm

_GOBJ= gpu.o timer.o matio.o
GOBJ=$(patsubst %,$(OBJDIR)/%,$(_GOBJ))
_COBJ= cpu.o timer.o matio.o
COBJ=$(patsubst %,$(OBJDIR)/%,$(_COBJ))

gpu: $(GOBJ)
	nvcc $(DEBUG) -o $@ $^ -lcublas -lcudadevrt

cpu: $(COBJ)
	g++ -O3 -Wall -Wextra -o $@ $^ $(ATLAS)

$(OBJDIR)/gpu.o: gpu.cu
	nvcc $(DEBUG) -o $@ -c $^

$(OBJDIR)/cpu.o: cpu.c
	g++ -O3 -Wall -Wextra -o $@ -c $^

$(OBJDIR)/matio.o: ../../matio/matio.c
	g++ -O3 -Wall -Wextra -c $< -o $@

$(OBJDIR)/timer.o: timer.c
	g++ -O3 -Wall -Wextra -c $< -o $@

clean:
	rm -fv cpu gpu $(OBJDIR)/*.o
