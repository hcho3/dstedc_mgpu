#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mat.h>
#include "dstedc.h"

double *read_mat(const char *filename, const char *varname, size_t *dims)
{
    MATFile *pmat;
    mxArray *pa;
    double *array;
    size_t nbytes;

    pmat = matOpen(filename, "r");
    if (!pmat) {
        fprintf(stderr, "Error opening MAT file \"%s\"!\n", filename);
        exit(1);
    }

    pa = matGetVariable(pmat, varname);

    if (!pa) {
        fprintf(stderr, "Error reading the variable %s.\n", varname);
        matClose(pmat);
        exit(1);
    }

    if (!mxIsDouble(pa) || mxGetNumberOfDimensions(pa) != 2) {
        fprintf(stderr, "%s is not a double-precision matrix.\n", varname);
        mxDestroyArray(pa);
        matClose(pmat);
        exit(1);
    }

    nbytes = mxGetNumberOfElements(pa) * sizeof(double);
    array = (double *)malloc(nbytes);
    memcpy(array, mxGetPr(pa), nbytes);
    memcpy(dims, mxGetDimensions(pa), 2 * sizeof(size_t));

    mxDestroyArray(pa);
    matClose(pmat);

    return array;
}

void write_mat(const char *filename, const char *varname,
    double *array, size_t *dims)
{
    MATFile *pmat;
    mxArray *pa;
    int status;

    pmat = matOpen(filename, "w");
    if (!pmat) {
        fprintf(stderr, "Error creating MAT file \"%s\"\n", filename);
        exit(1);
    }

    pa = mxCreateDoubleMatrix(dims[0], dims[1], mxREAL);
    if (!pa) {
        fprintf(stderr, "Error creating variable %s.\n", varname);
        matClose(pmat);
        exit(1);
    }
    memcpy(mxGetPr(pa), array, dims[0]*dims[1]*sizeof(double));
    status = matPutVariable(pmat, varname, pa);
    if (status != 0) {
        fprintf(stderr, "Error writing variable %s.\n", varname);
        mxDestroyArray(pa);
        matClose(pmat);
        exit(1);
    }

    mxDestroyArray(pa);
    matClose(pmat);
}
