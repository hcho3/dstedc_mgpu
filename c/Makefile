all: tester

MATIO=$(HOME)/lib/libmatio.a
LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS_PATH=/usr/local/atlas3.10.1/lib/
DEBUG=-O3

tester: dstedc.o dlaed0.o dlaed1.o dlaed2.o dlaed3.o dlaed4.o \
		initial_guess.o middle_way.o dlamrg.o dlapy2.o matio.o tester.o
	gcc -Wall -o $@ $^ $(MATIO) --static $(LAPACKE) -L$(ATLAS_PATH) \
		-llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm -lpthread -I.

dstedc.o: dstedc.c
	gcc -Wall $(DEBUG) -c $< -o $@
dlaed0.o: dlaed0.c
	gcc -Wall $(DEBUG) -c $< -o $@
dlaed1.o: dlaed1.c
	gcc -Wall $(DEBUG) -c $< -o $@
dlaed2.o: dlaed2.c
	gcc -Wall $(DEBUG) -c $< -o $@
dlaed3.o: dlaed3.c
	gcc -Wall $(DEBUG) -c $< -o $@
dlaed4.o: dlaed4.c
	gcc -Wall $(DEBUG) -c $< -o $@
initial_guess.o: initial_guess.c
	gcc -Wall $(DEBUG) -c $< -o $@
middle_way.o: middle_way.c
	gcc -Wall $(DEBUG) -c $< -o $@
dlamrg.o: dlamrg.c
	gcc -Wall $(DEBUG) -c $< -o $@
dlapy2.o: dlapy2.c
	gcc -Wall $(DEBUG) -c $< -o $@
matio.o: matio.c
	gcc -Wall $(DEBUG) -c $< -o $@ -I$(HOME)/include
tester.o: tester.c
	gcc -Wall $(DEBUG) -c $< -o $@

clean:
	rm -fv tester *.o
