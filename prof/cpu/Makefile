all: cpuprof

include ../../make.inc

DEBUG=-O3 -Wall -Wextra -pedantic -fopenmp -std=gnu99
OBJDIR=obj

ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm

_OBJ= dlaed0_m.o dlaed1.o dlaed2.o dlaed3.o dlaed4.o  \
		dlamrg.o dlapy2.o matio.o main.o workspace.o \
		timer.o initial_guess.o middle_way.o nvtx.o
OBJ=$(patsubst %,$(OBJDIR)/%,$(_OBJ))

cpuprof: $(OBJ)
	gcc $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) -lm

$(OBJDIR)/%.o: %.c
	gcc $(DEBUG) -c $< -o $@
$(OBJDIR)/matio.o: ../../matio/matio.c
	gcc $(DEBUG) -c $< -o $@

clean:
	rm -fv cpuprof $(OBJDIR)/*.o
