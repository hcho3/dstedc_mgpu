#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dstedc.h"
#include "nvtx.h"

double *read_mat(const char *filename, long *dims)
{
    char temp[BUFSIZ];
    strcpy(temp, "read_mat(");
    strcat(temp, filename);
    strcat(temp, ")");
    RANGE_START(temp, 1, 5);

    FILE *fp;
    double *array;
    long M, N, nelem;

    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error opening matrix file \"%s\"!\n", filename);
        exit(1);
    }

    // matrix dimensions
    if (fread(&M, sizeof(long), 1, fp) < 1 ||
        fread(&N, sizeof(long), 1, fp) < 1 ||
        M <= 0 || N <= 0) {
        fprintf(stderr, "Invalid matrix dimensions\n");
        fclose(fp);
        exit(2);
    }

    dims[0] = M;
    dims[1] = N;
    nelem = M * N;
    array = (double *)malloc(nelem * sizeof(double));

    // matrix
    if (fread(array, sizeof(double), nelem, fp) < (size_t)nelem) {
        fprintf(stderr, "The matrix contains fewer entries than "
                        "the specified dimensions %ldx%ld.\n", M, N);
        free(array);
        fclose(fp);
        exit(3);
    }

    fclose(fp);

    RANGE_END(1);

    return array;
}

void write_mat(const char *filename, double *array, long *dims)
{
    char temp[BUFSIZ];
    strcpy(temp, "write_mat(");
    strcat(temp, filename);
    strcat(temp, ")");
    RANGE_START(temp, 1, 6);

    FILE *fp;
    long M, N, nelem;

    fp = fopen(filename, "w");

    if (!fp) {
        fprintf(stderr, "Error creating matrix file \"%s\"\n", filename);
        exit(1);
    }

    M = dims[0];
    N = dims[1];
    nelem = M * N;

    // matrix dimensions
    if (fwrite(&M, sizeof(long), 1, fp) < 1 ||
        fwrite(&N, sizeof(long), 1, fp) < 1) {
        fprintf(stderr, "Error recording the matrix dimensions to file.\n");
        fclose(fp);
        exit(4);
    }

    // matrix
    if (fwrite(array, sizeof(double), nelem, fp) < (size_t)nelem) {
        fprintf(stderr, "Error saving the matrix to file.\n");
        fclose(fp);
        exit(5);
    }

    fclose(fp);

    RANGE_END(1);
}
